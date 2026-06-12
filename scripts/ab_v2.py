"""
scripts/ab_v2.py — pooled per-trade attribution for the EMA Pullback V2 filters.

The old version of this script ran four *columns* (V1 / +BTC gate / +RS / full
V2) and compared their aggregate stats. With ~20 trades per column on a year of
30m data, the deltas were noise: twelve small samples pointing in different
directions. This version pools instead.

Approach
--------
Run ONE baseline configuration per symbol with every V2 filter DISABLED, so
*every* regime + pullback setup becomes a real trade (~hundreds per symbol).
The strategy still evaluates each V2 filter's raw verdict at every entry and
records it (see EMAPullbackStrategy.entry_filter_log) — so for each executed
trade we know what btc_gate / rs / volume / pullback_memory *would* have said,
without any of them actually blocking the trade.

We then pool all trades across all symbols into one CSV, one row per trade, with
its realized outcome (pnl, R) and its four filter-pass flags. With that table we
can bucket and regress the outcome on filter state — answering "what is each
filter worth" with real n instead of anecdotes.

R is computed PER ROUND-TRIP (entry to final exit), not per exit leg. The
strategy scales out (tp1 / tp2), so one entry produces several TradeRecord legs;
averaging R over legs double-counts winners (which scale out into 2-3 legs)
against losers (a single stop leg), which is what made avg R read positive while
profit factor sat at ~1. Grouping legs back into round-trips fixes that.

Usage:
    python scripts/ab_v2.py --symbols ETH/USDT SOL/USDT BNB/USDT XRP/USDT \
        --source ccxt --interval 30m --start 2024-01-01 --end 2025-01-01

Run `python scripts/ab_v2.py --help` for all options.
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np
import pandas as pd

# Allow running as `python scripts/ab_v2.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli


# The four optional V2 filters, in funnel order.
FILTERS = ["btc_gate", "rs", "volume", "pullback_memory"]

# pullback_memory is a count threshold, not an on/off flag. The strategy records
# the raw consecutive-close count at entry; we threshold it here into a boolean
# so it lines up with the other three filters. 3 = the strategy's own default.
PB_MEM_REF = 3


def _baseline_overrides() -> dict:
    """
    All V2 filters OFF so every setup trades — but btc_gate_mode stays at a real
    mode (not 'off') so the gate's *raw* verdict is still computable per entry.
    """
    return dict(
        ep_btc_gate_enabled=False, ep_btc_gate_mode="ema_stack",
        ep_rs_filter_enabled=False, ep_volume_filter_enabled=False,
        ep_pullback_memory_bars=0, ep_btc_flatten_on_break=False,
    )


def build_base_cfg(args) -> dict:
    """Start from the CLI CONFIG, apply shared run settings + baseline overrides."""
    cfg = copy.deepcopy(cli.CONFIG)
    cfg["strategy"]         = "ema_pullback"
    cfg["data_source"]      = args.source
    cfg["symbols"]          = list(args.symbols)
    cfg["interval"]         = args.interval
    cfg["start"]            = args.start
    cfg["end"]              = args.end
    cfg["ep_direction"]     = args.direction
    cfg["ep_btc_symbol"]    = args.btc_symbol
    cfg["periods_per_year"] = args.periods_per_year
    cfg["refresh_cache"]    = args.refresh_cache
    cfg.update(_baseline_overrides())
    return cfg


# ---------------------------------------------------------------------------
# Lean runner (no result folders / charts / Monte Carlo)
# ---------------------------------------------------------------------------

def run_one(cfg: dict):
    """
    Run a single backtest. Returns (portfolio, strategies).

    BTC is forced into the feed (built from [btc_symbol, *symbols]) while
    strategies are built from `symbols` only — so BTC drives the gate/RS state
    for every alt but is never traded.
    """
    from core.portfolio import Portfolio
    from core.broker import Broker
    from core.engine import Engine

    real_symbols = list(cfg["symbols"])
    btc = cfg.get("ep_btc_symbol", "BTC/USDT")
    feed_syms = real_symbols if btc in real_symbols else [btc] + real_symbols

    feed_cfg = copy.deepcopy(cfg)
    feed_cfg["symbols"] = feed_syms
    feed = cli.build_data_handler(feed_cfg)

    strategies = [cli.build_strategy(cfg, s) for s in real_symbols]
    broker = Broker(
        fill_model=cli.build_fill_model(cfg),
        cost_model=cli.build_cost_model(cfg),
        fill_ratio=cfg["fill_ratio"],
        min_fill_volume=cfg.get("min_fill_volume", 0.0),
    )
    portfolio = Portfolio(
        initial_capital=cfg["initial_capital"],
        position_size_pct=cfg["position_size_pct"],
        risk_pct=cfg.get("risk_pct", 0.02),
        short_borrow_rate=cfg.get("short_borrow_rate", 0.0),
        short_initial_margin=cfg.get("short_initial_margin", 0.50),
    )
    Engine(data_handler=feed, strategies=strategies,
           portfolio=portfolio, broker=broker).run()
    return portfolio, strategies


# ---------------------------------------------------------------------------
# Round-trip reconstruction (the avg-R fix)
# ---------------------------------------------------------------------------

def round_trips(trades: pd.DataFrame, symbol: str) -> list:
    """
    Collapse a symbol's exit legs back into round-trips, in entry-time order.

    All legs of one entry share (symbol, side, entry_time, entry_price,
    stop_price); we sum pnl/qty/hold across them and compute ONE R per trade:
        R = total_pnl / (|entry_price - stop_price| * total_qty)
    """
    if trades is None or trades.empty:
        return []
    sym = trades[trades["symbol"] == symbol]
    if sym.empty:
        return []

    out = []
    for (entry_time, side), grp in sym.groupby(["entry_time", "side"], sort=False):
        first   = grp.iloc[0]
        total_q = float(grp["quantity"].sum())
        pnl     = float(grp["pnl"].sum())
        entry_p = float(first["entry_price"])
        stop_p  = first.get("stop_price")
        risk    = (abs(entry_p - stop_p) * total_q
                   if stop_p is not None and not pd.isna(stop_p) else np.nan)
        out.append({
            "symbol":      symbol,
            "side":        side,
            "entry_time":  entry_time,
            "exit_time":   grp["exit_time"].max(),
            "entry_price": entry_p,
            "stop_price":  float(stop_p) if stop_p is not None and not pd.isna(stop_p) else np.nan,
            "quantity":    total_q,
            "pnl":         pnl,
            "R":           (pnl / risk) if (risk and risk > 0) else np.nan,
            "hold_bars":   int(grp["hold_bars"].max()),
            "win":         pnl > 0,
        })
    out.sort(key=lambda r: r["entry_time"])
    return out


# ---------------------------------------------------------------------------
# Pool trades with their filter-pass flags
# ---------------------------------------------------------------------------

def build_rows(portfolio, strategies) -> list:
    """One row per executed round-trip, across all symbols, with filter flags."""
    trades = portfolio.trade_dataframe()
    rows = []
    for strat in strategies:
        sym  = strat.symbol
        rts  = round_trips(trades, sym)
        log  = strat.entry_filter_log()
        if len(rts) != len(log):
            logging.warning(
                "%s: %d round-trips but %d entry-verdict records — matching the "
                "first %d in order.", sym, len(rts), len(log), min(len(rts), len(log)))
        for rt, v in zip(rts, log):
            consec = v["pullback_consec"]
            rows.append({
                **rt,
                "pass_btc_gate":        bool(v["btc_gate"]),
                "pass_rs":              bool(v["rs"]),
                "pass_volume":          bool(v["volume"]),
                "pullback_consec":      int(consec),
                "pass_pullback_memory": consec >= PB_MEM_REF,
            })
    return rows


def save_csv(rows: list, base_cfg: dict) -> str:
    """Write the pooled per-trade table to scripts/results/<Coin>.csv; return the path.

    Filename is just the coin name(s), e.g. NEAR/USDT → Near.csv. Multiple symbols
    are joined with '_' (Eth_Sol.csv). Reruns overwrite the same file.
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    coins = [s.split("/")[0].capitalize() for s in base_cfg["symbols"]]
    fname = "_".join(coins) + ".csv"
    path  = os.path.join(results_dir, fname)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Analysis: bucket + regress outcome by filter state
# ---------------------------------------------------------------------------

def _subset_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "n":      n,
        "win":    float(df["win"].mean()) if n else float("nan"),
        "mean_r": float(df["R"].mean(skipna=True)) if n else float("nan"),
        "pnl":    float(df["pnl"].sum()) if n else 0.0,
        "exp":    float(df["pnl"].mean()) if n else float("nan"),
    }


def _fmt_stats(s: dict) -> str:
    return (f"n={s['n']:>4}  win={s['win']*100:5.1f}%  "
            f"R̄={s['mean_r']:+5.2f}  exp=${s['exp']:+7.2f}")


def print_bucket_report(df: pd.DataFrame):
    print("\n" + "=" * 86)
    print("  EMA Pullback V2 — pooled per-trade filter attribution")
    print("=" * 86)

    overall = _subset_stats(df)
    print(f"\n  ALL TRADES         {_fmt_stats(overall)}")
    by_sym = df.groupby("symbol").size().to_dict()
    print("  per symbol         " + "  ".join(f"{k}:{v}" for k, v in by_sym.items()))
    by_side = df.groupby("side").size().to_dict()
    print("  per side           " + "  ".join(f"{k}:{v}" for k, v in by_side.items()))

    print("\n  ── Bucketed by single-filter state ──"
          "  (Δ = pass − fail; positive ⇒ filter selects better trades)")
    for f in FILTERS:
        col = f"pass_{f}"
        p, q = _subset_stats(df[df[col]]), _subset_stats(df[~df[col]])
        d_win = (p["win"] - q["win"]) * 100
        d_r   = p["mean_r"] - q["mean_r"]
        d_exp = p["exp"] - q["exp"]
        print(f"\n  {f}")
        print(f"      pass   {_fmt_stats(p)}")
        print(f"      fail   {_fmt_stats(q)}")
        print(f"      Δ      win {d_win:+5.1f}pp   R̄ {d_r:+5.2f}   exp ${d_exp:+7.2f}")

    # All-pass intersection vs. the full pool.
    all_pass = df
    for f in FILTERS:
        all_pass = all_pass[all_pass[f"pass_{f}"]]
    print("\n  ── All four filters pass simultaneously (≈ full V2 selection) ──")
    print(f"      all-pass {_fmt_stats(_subset_stats(all_pass))}")
    print(f"      full pool{_fmt_stats(overall)}")

    _print_regression(df)
    print("=" * 86 + "\n")


def _print_regression(df: pd.DataFrame):
    """OLS of per-trade R on the four filter flags. Coefs = marginal R effect."""
    reg = df.dropna(subset=["R"])
    if len(reg) < len(FILTERS) + 2:
        print("\n  ── Regression skipped (too few trades with a defined R) ──")
        return

    y = reg["R"].to_numpy(dtype=float)
    names, cols = ["intercept"], [np.ones(len(reg))]
    dropped = []
    for f in FILTERS:
        v = reg[f"pass_{f}"].to_numpy(dtype=float)
        if v.min() == v.max():          # constant column → not identifiable
            dropped.append(f)
            continue
        names.append(f)
        cols.append(v)
    X = np.column_stack(cols)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof   = len(reg) - X.shape[1]
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        sigma2  = (resid @ resid) / dof if dof > 0 else np.nan
        se      = np.sqrt(np.diag(xtx_inv) * sigma2)
        tstat   = beta / se
    except np.linalg.LinAlgError:
        se = tstat = np.full_like(beta, np.nan)

    print(f"\n  ── OLS: R ~ filter flags   (n={len(reg)}, R-with-stop only) ──")
    print(f"      {'term':<18}{'coef':>9}{'std err':>10}{'t':>8}")
    for nm, b, s, t in zip(names, beta, se, tstat):
        print(f"      {nm:<18}{b:>9.3f}{s:>10.3f}{t:>8.2f}")
    if dropped:
        print(f"      (dropped — constant in sample: {', '.join(dropped)})")
    print("      coef = mean change in R when that filter passes, others held fixed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Pooled per-trade attribution for EMA Pullback V2 filters.")
    p.add_argument("--symbols", nargs="+",
                   default=["ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
                   help="alt symbols to trade and pool")
    p.add_argument("--btc-symbol", dest="btc_symbol", default="BTC/USDT",
                   help="symbol driving the BTC gate / RS filter")
    p.add_argument("--source", default="ccxt",
                   choices=["yfinance", "alpaca", "ccxt", "forex"])
    p.add_argument("--interval", default="30m")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--direction", default="both", choices=["long", "short", "both"],
                   help="trade direction(s); 'both' pools longs and shorts (side recorded per trade)")
    p.add_argument("--periods-per-year", dest="periods_per_year", type=int, default=17520,
                   help="for Sharpe/CAGR annualization (30m crypto ≈ 365*48 = 17520)")
    p.add_argument("--refresh-cache", dest="refresh_cache", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show engine INFO logs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    base_cfg = build_base_cfg(args)
    all_rows = []
    for sym in args.symbols:
        cfg = copy.deepcopy(base_cfg)
        cfg["symbols"] = [sym]
        print(f"Running {sym} ({args.direction}) ...", flush=True)
        portfolio, strategies = run_one(cfg)
        all_rows.extend(build_rows(portfolio, strategies))

    if not all_rows:
        print("No trades produced — nothing to pool.")
        return

    csv_path = save_csv(all_rows, base_cfg)
    df = pd.DataFrame(all_rows)
    print_bucket_report(df)
    print(f"Pooled {len(df)} trades across {len(args.symbols)} symbols → {csv_path}\n")


if __name__ == "__main__":
    main()
