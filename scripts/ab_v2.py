"""
scripts/ab_v2.py — A/B harness for the EMA Pullback V2 filters.

Runs identical data through four configurations and tabulates the deltas:

    (a) V1 baseline      : all V2 filters OFF
    (b) + BTC gate       : BTC regime gate only
    (c) + RS             : BTC gate + relative-strength-vs-BTC
    (d) full V2          : BTC gate + RS + volume + pullback memory

For each it reports trade count, win rate, avg R, expectancy, profit factor,
max drawdown, Sharpe, final equity, P&L bucketed by exit_reason, and the
per-filter rejection funnel (from strategy.filter_stats()). The deltas between
columns are the deliverable — they say what each filter is worth.

To keep the equity curves directly comparable, BTC is fed in EVERY config
(prepended so it's processed before the alt each bar, no lookahead) but is
never traded — only the alt symbol gets a strategy. This means even the V1
baseline column sees the same 2-symbol equity sampling as the V2 columns, so
Sharpe / max-DD are apples-to-apples across all four.

Usage:
    python scripts/ab_v2.py --symbol ETH/USDT --source ccxt \
        --interval 30m --start 2024-01-01 --end 2025-01-01

Run `python scripts/ab_v2.py --help` for all options.
"""

import argparse
import copy
import logging
import os
import sys

import pandas as pd

# Allow running as `python scripts/ab_v2.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli
from analytics.metrics import compute_all


# ---------------------------------------------------------------------------
# Configuration variants
# ---------------------------------------------------------------------------

def _variant_overrides() -> dict:
    """Map of column label → the V2 flag overrides that define it."""
    return {
        "(a) V1 baseline": dict(
            ep_btc_gate_enabled=False, ep_btc_gate_mode="off",
            ep_rs_filter_enabled=False, ep_volume_filter_enabled=False,
            ep_pullback_memory_bars=0, ep_btc_flatten_on_break=False,
        ),
        "(b) +BTC gate": dict(
            ep_btc_gate_enabled=True,
            ep_rs_filter_enabled=False, ep_volume_filter_enabled=False,
            ep_pullback_memory_bars=0, ep_btc_flatten_on_break=False,
        ),
        "(c) +RS": dict(
            ep_btc_gate_enabled=True, ep_rs_filter_enabled=True,
            ep_volume_filter_enabled=False,
            ep_pullback_memory_bars=0, ep_btc_flatten_on_break=False,
        ),
        "(d) full V2": dict(
            ep_btc_gate_enabled=True, ep_rs_filter_enabled=True,
            ep_volume_filter_enabled=True,
        ),  # pullback_memory / flatten left at base cfg values
    }


def build_base_cfg(args) -> dict:
    """Start from the CLI CONFIG and apply the shared run settings."""
    cfg = copy.deepcopy(cli.CONFIG)
    cfg["strategy"]        = "ema_pullback"
    cfg["data_source"]     = args.source
    cfg["symbols"]         = [args.symbol]
    cfg["interval"]        = args.interval
    cfg["start"]           = args.start
    cfg["end"]             = args.end
    cfg["ep_direction"]    = args.direction
    cfg["ep_btc_symbol"]   = args.btc_symbol
    cfg["periods_per_year"] = args.periods_per_year
    cfg["refresh_cache"]   = args.refresh_cache
    return cfg


# ---------------------------------------------------------------------------
# Lean runner (no result folders / charts / Monte Carlo)
# ---------------------------------------------------------------------------

def run_one(cfg: dict):
    """
    Run a single backtest. Returns (portfolio, strategies, metrics).

    BTC is forced into the feed for every variant (see module docstring) by
    building the feed from [btc_symbol, *symbols] while strategies are built
    from `symbols` only — so BTC is consumed for regime state but never traded.
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

    eq     = portfolio.equity_series()
    trades = portfolio.trade_dataframe()
    metrics = compute_all(
        eq, trades,
        risk_free_rate=cfg.get("risk_free_rate", 0.0),
        periods_per_year=cfg["periods_per_year"],
    )
    return portfolio, strategies, metrics


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def _avg_r(trades: pd.DataFrame) -> float:
    """
    Mean R-multiple across trades that carry a stop price.
    R = realized pnl / planned dollar risk, where risk = qty × |entry − stop|.
    """
    if trades is None or trades.empty or "stop_price" not in trades.columns:
        return float("nan")
    rs = []
    for _, t in trades.iterrows():
        stop = t.get("stop_price")
        if stop is None or pd.isna(stop):
            continue
        risk = abs(t["entry_price"] - stop) * t["quantity"]
        if risk > 0:
            rs.append(t["pnl"] / risk)
    return float(sum(rs) / len(rs)) if rs else float("nan")


def _pnl_by_reason(trades: pd.DataFrame) -> dict:
    """exit_reason → (count, total_pnl)."""
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        return {}
    out = {}
    for reason, grp in trades.groupby("exit_reason"):
        out[reason] = (len(grp), float(grp["pnl"].sum()))
    return out


def analyze(portfolio, strategies, metrics, initial_capital: float) -> dict:
    trades = portfolio.trade_dataframe()
    eq = portfolio.equity_series()
    final_equity = float(eq.iloc[-1]) if len(eq) else initial_capital
    # Merge filter funnels across strategies (one per traded symbol).
    setups = entries = 0
    rejections: dict = {}
    for s in strategies:
        if hasattr(s, "filter_stats"):
            fs = s.filter_stats()
            setups += fs["setups"]
            entries += fs["entries"]
            for k, v in fs["rejections"].items():
                rejections[k] = rejections.get(k, 0) + v
    return {
        "trades":       metrics["total_trades"],
        "win_rate":     metrics["win_rate"],
        "avg_r":        _avg_r(trades),
        "expectancy":   metrics["expectancy"],
        "profit_factor": metrics["profit_factor"],
        "max_dd":       metrics["max_drawdown_pct"],
        "sharpe":       metrics["sharpe_ratio"],
        "final_equity": final_equity,
        "pnl_by_reason": _pnl_by_reason(trades),
        "setups":       setups,
        "entries":      entries,
        "rejections":   rejections,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v, kind="f2"):
    if isinstance(v, float) and (v != v):  # NaN
        return "—"
    if kind == "pct":
        return f"{v * 100:5.1f}%"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "money":
        return f"{v:,.0f}"
    return str(v)


def print_report(results: dict, base_cfg: dict, min_trades_flag: int):
    labels = list(results.keys())
    col_w = max(14, max(len(l) for l in labels) + 1)
    LABEL_W = 22  # wide enough for the longest row label ("  rej:pullback_memory")

    def header(title):
        print("\n  " + title.ljust(LABEL_W) + "".join(l.rjust(col_w) for l in labels))
        print("  " + "-" * (LABEL_W + col_w * len(labels)))

    def row(name, getter, kind="f2"):
        cells = "".join(_fmt(getter(results[l]), kind).rjust(col_w) for l in labels)
        print(f"  {name:<{LABEL_W}}{cells}")

    print("\n" + "=" * 78)
    print(f"  EMA Pullback V2 — A/B comparison")
    print(f"  {base_cfg['symbols'][0]} vs {base_cfg['ep_btc_symbol']} | "
          f"{base_cfg['data_source']} | {base_cfg['interval']} | "
          f"{base_cfg['start']} → {base_cfg['end']} | dir={base_cfg['ep_direction']}")
    print("=" * 78)

    header("PERFORMANCE")
    row("Trades",        lambda r: r["trades"],        "raw")
    row("Win rate",      lambda r: r["win_rate"],      "pct")
    row("Avg R",         lambda r: r["avg_r"],         "f2")
    row("Expectancy $",  lambda r: r["expectancy"],    "f2")
    row("Profit factor", lambda r: r["profit_factor"], "f2")
    row("Max DD",        lambda r: r["max_dd"],        "pct")
    row("Sharpe",        lambda r: r["sharpe"],        "f2")
    row("Final equity",  lambda r: r["final_equity"],  "money")

    # P&L by exit reason — union of reasons seen across all columns.
    reasons = sorted({rsn for r in results.values() for rsn in r["pnl_by_reason"]})
    if reasons:
        header("P&L BY EXIT")
        for rsn in reasons:
            def cell(r):
                if rsn in r["pnl_by_reason"]:
                    n, pnl = r["pnl_by_reason"][rsn]
                    return f"{pnl:,.0f}({n})"
                return "—"
            cells = "".join(cell(results[l]).rjust(col_w) for l in labels)
            print(f"  {rsn:<{LABEL_W}}{cells}")

    # Filter funnel.
    header("FILTER FUNNEL")
    row("Setups",  lambda r: r["setups"],  "raw")
    row("Entries", lambda r: r["entries"], "raw")
    rej_keys = ["regime", "adx", "supertrend", "warmup",
                "btc_gate", "rs", "volume", "pullback_memory"]
    for k in rej_keys:
        row(f"  rej:{k}", lambda r, k=k: r["rejections"].get(k, 0), "raw")

    # Watch-point from the V2 plan.
    print("\n" + "=" * 78)
    full = results.get("(d) full V2")
    if full is not None and full["trades"] < min_trades_flag:
        print(f"  ⚠  Full V2 produced {full['trades']} trades (< {min_trades_flag}). "
              f"Filters may be too strict — inspect the funnel above before tuning.")
    else:
        print("  Deltas between columns show what each filter is worth.")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="A/B harness for EMA Pullback V2 filters.")
    p.add_argument("--symbol", default="ETH/USDT", help="alt symbol to trade")
    p.add_argument("--btc-symbol", dest="btc_symbol", default="BTC/USDT",
                   help="symbol driving the BTC gate / RS filter")
    p.add_argument("--source", default="ccxt",
                   choices=["yfinance", "alpaca", "ccxt", "forex"])
    p.add_argument("--interval", default="30m")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--direction", default="long", choices=["long", "short", "both"])
    p.add_argument("--periods-per-year", dest="periods_per_year", type=int, default=17520,
                   help="for Sharpe/CAGR annualization (30m crypto ≈ 365*48 = 17520)")
    p.add_argument("--min-trades", dest="min_trades", type=int, default=30,
                   help="flag full-V2 runs below this trade count as too strict")
    p.add_argument("--refresh-cache", dest="refresh_cache", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show engine INFO logs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    base_cfg = build_base_cfg(args)
    results = {}
    for label, overrides in _variant_overrides().items():
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        print(f"Running {label} ...", flush=True)
        portfolio, strategies, metrics = run_one(cfg)
        results[label] = analyze(portfolio, strategies, metrics, cfg["initial_capital"])

    print_report(results, base_cfg, args.min_trades)


if __name__ == "__main__":
    main()
