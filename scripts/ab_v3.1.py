"""
scripts/ab_v31.py — A/B harness for the V3.1 entry refinements.

Unlike ab_v2.py (which pools one baseline run and buckets trades post-hoc), the
V3.1 filters change the *set* of trades that occur:

  * extension filter — vetoes longs stretched too far above EMA200 (longs only).
  * reset gate       — one entry per trend leg; re-arms on an EMA50 cross. This
                       is PATH-DEPENDENT (whether an entry is "first of the leg"
                       depends on which earlier entries fired), so it cannot be
                       recovered from an all-off baseline. It must be run live.

So this is a plain config-grid A/B: four configurations, each a full backtest,
per symbol, with results pooled across symbols into one row per cell.

    (a) V3 baseline : ext off, reset off
    (b) +ext        : ext on,  reset off
    (c) +reset      : ext off, reset on
    (d) +both       : ext on,  reset on

Everything else stays at the validated V3 config (fresh_touch on, RS shorts-only,
no BTC gate, no volume). The deliverable is the per-cell table: trade count, win
rate, avg R (per round-trip), profit factor, max drawdown, final equity — pooled
across symbols. The result you want from a good refinement is FEWER trades with
PF flat-or-up; and the SHORT-side numbers must be identical between (a)&(b) and
between (c)&(d) for ext (long-only) — a built-in wiring check.

The 2x2 grid is also swept across a stop/TP1 plane (--stop-mults × --tp1-rs,
default {1.5, 2.0}×ATR stop and {2.5, 3.0}R TP1), printing one report block per
(stop, tp1) combination so the refinement effect can be read at each setting.

Usage:
    python scripts/ab_v31.py --symbols ETH/USDT SOL/USDT NEAR/USDT FET/USDT \
        --source ccxt --interval 30m --start 2021-06-12 --end 2026-06-12 \
        --ext-max-pct 10.0 --stop-mults 1.5 2.0 --tp1-rs 2.5 3.0
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli


# ---------------------------------------------------------------------------
# The 2x2 grid. Only the two V3.1 flags vary; everything else is validated V3.
# ---------------------------------------------------------------------------

def _grid(ext_max_pct: float) -> dict:
    return {
        "(a) V3 baseline": dict(ep_ext_filter_enabled=False, ep_reset_gate_enabled=False),
        "(b) +ext":        dict(ep_ext_filter_enabled=True,  ep_reset_gate_enabled=False),
        "(c) +reset":      dict(ep_ext_filter_enabled=False, ep_reset_gate_enabled=True),
        "(d) +both":       dict(ep_ext_filter_enabled=True,  ep_reset_gate_enabled=True),
    }


def _v3_config() -> dict:
    """The validated V3 entry config — shared by every cell."""
    return dict(
        ep_btc_gate_enabled=False,
        ep_rs_filter_sides="short",
        ep_volume_filter_enabled=False,
        ep_pullback_memory_bars=0,
        ep_fresh_touch_required=True,
        ep_daily_supertrend_filter=True,
    )


def build_base_cfg(args) -> dict:
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
    cfg["ep_ext_max_pct"]   = args.ext_max_pct
    cfg.update(_v3_config())
    return cfg


# ---------------------------------------------------------------------------
# Runner (BTC forced into feed for RS state, never traded)
# ---------------------------------------------------------------------------

def run_one(cfg: dict):
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
# Round-trip reconstruction (same logic as ab_v2.py — one R per entry)
# ---------------------------------------------------------------------------

def round_trips(trades: pd.DataFrame, symbol: str) -> list:
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
            "symbol": symbol, "side": side, "entry_time": entry_time,
            "pnl": pnl, "R": (pnl / risk) if (risk and risk > 0) else np.nan,
            "win": pnl > 0,
        })
    return out


def cell_metrics(rows: list, initial_capital: float) -> dict:
    """Pooled stats for one grid cell. Equity-based metrics are pooled across
    symbols by chaining each symbol's trades in time order (approximate, but
    consistent across cells, which is what the comparison needs)."""
    if not rows:
        return dict(n=0, n_long=0, n_short=0, win=float("nan"), avg_r=float("nan"),
                    pf=float("nan"), max_dd=float("nan"), final_eq=initial_capital,
                    short_n=0, short_pnl=0.0)
    df = pd.DataFrame(rows).sort_values("entry_time")
    gross_w = df.loc[df.pnl > 0, "pnl"].sum()
    gross_l = -df.loc[df.pnl < 0, "pnl"].sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    # Pooled equity curve (chained PnL) for a rough max-DD comparison.
    eq = initial_capital + df["pnl"].cumsum()
    running_max = eq.cummax()
    dd = ((eq - running_max) / running_max).min()

    shorts = df[df.side.str.lower() == "short"]
    return dict(
        n=len(df),
        n_long=int((df.side.str.lower() == "long").sum()),
        n_short=int((df.side.str.lower() == "short").sum()),
        win=float(df["win"].mean()),
        avg_r=float(df["R"].mean(skipna=True)),
        pf=float(pf),
        max_dd=float(dd),
        final_eq=float(eq.iloc[-1]),
        short_n=len(shorts),
        short_pnl=float(shorts["pnl"].sum()),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict, base_cfg: dict):
    labels = list(results.keys())
    W = 16
    LW = 16

    def row(name, getter, kind="f2"):
        cells = ""
        for l in labels:
            v = getter(results[l])
            if isinstance(v, float) and v != v:
                s = "—"
            elif kind == "pct":
                s = f"{v*100:.1f}%"
            elif kind == "money":
                s = f"{v:,.0f}"
            elif kind == "raw":
                s = f"{v}"
            else:
                s = f"{v:.2f}"
            cells += s.rjust(W)
        print(f"  {name:<{LW}}{cells}")

    print("\n" + "=" * (LW + 2 + W * len(labels)))
    print(f"  EMA Pullback V3.1 — refinement A/B  "
          f"[stop={base_cfg['ep_atr_stop_mult']}×ATR, tp1={base_cfg['ep_tp1_r']}R]")
    print(f"  {', '.join(base_cfg['symbols'])} | {base_cfg['interval']} | "
          f"{base_cfg['start']} → {base_cfg['end']} | dir={base_cfg['ep_direction']} | "
          f"ext_max={base_cfg['ep_ext_max_pct']}%")
    print("=" * (LW + 2 + W * len(labels)))
    print("  " + " " * LW + "".join(l.rjust(W) for l in labels))
    print("  " + "-" * (LW + W * len(labels)))
    row("Trades",       lambda r: r["n"],        "raw")
    row("  long",       lambda r: r["n_long"],   "raw")
    row("  short",      lambda r: r["n_short"],  "raw")
    row("Win rate",     lambda r: r["win"],      "pct")
    row("Avg R",        lambda r: r["avg_r"],    "f2")
    row("Profit factor",lambda r: r["pf"],       "f2")
    row("Max DD",       lambda r: r["max_dd"],   "pct")
    row("Final equity", lambda r: r["final_eq"], "money")

    # Wiring check: ext is long-only, so short PnL must be identical a==b and c==d.
    print("  " + "-" * (LW + W * len(labels)))
    a, b = results["(a) V3 baseline"], results["(b) +ext"]
    c, d = results["(c) +reset"], results["(d) +both"]
    def close(x, y): return abs(x - y) < 1e-6
    ok_ab = a["short_n"] == b["short_n"] and close(a["short_pnl"], b["short_pnl"])
    ok_cd = c["short_n"] == d["short_n"] and close(c["short_pnl"], d["short_pnl"])
    print(f"\n  WIRING CHECK (ext is long-only → short side must match):")
    print(f"    (a)==(b) shorts: {'OK' if ok_ab else 'MISMATCH'} "
          f"[a: {a['short_n']}tr ${a['short_pnl']:+.0f} | b: {b['short_n']}tr ${b['short_pnl']:+.0f}]")
    print(f"    (c)==(d) shorts: {'OK' if ok_cd else 'MISMATCH'} "
          f"[c: {c['short_n']}tr ${c['short_pnl']:+.0f} | d: {d['short_n']}tr ${d['short_pnl']:+.0f}]")
    if not (ok_ab and ok_cd):
        print("    ⚠  Extension filter is touching shorts — check the side guard in _ext_ok().")

    print("\n  READ: a good refinement shows FEWER trades with PF flat-or-up and")
    print("        DD flat-or-down. If PF falls, later-leg entries were adding edge.")
    print("=" * (LW + 2 + W * len(labels)) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="A/B harness for EMA Pullback V3.1 refinements.")
    p.add_argument("--symbols", nargs="+",
                   default=["ETH/USDT", "SOL/USDT", "NEAR/USDT", "FET/USDT"])
    p.add_argument("--btc-symbol", dest="btc_symbol", default="BTC/USDT")
    p.add_argument("--source", default="ccxt",
                   choices=["yfinance", "alpaca", "ccxt", "forex"])
    p.add_argument("--interval", default="30m")
    p.add_argument("--start", default="2021-06-12")
    p.add_argument("--end", default="2026-06-12")
    p.add_argument("--direction", default="both", choices=["long", "short", "both"])
    p.add_argument("--ext-max-pct", dest="ext_max_pct", type=float, default=10.0,
                   help="max %% above EMA200 to allow a long (extension filter)")
    p.add_argument("--stop-mults", dest="stop_mults", nargs="+", type=float,
                   default=[1.5, 2.0],
                   help="ATR stop multipliers to sweep (ep_atr_stop_mult)")
    p.add_argument("--tp1-rs", dest="tp1_rs", nargs="+", type=float,
                   default=[2.0,2.5, 3.0],
                   help="TP1 R-multiples to sweep (ep_tp1_r)")
    p.add_argument("--periods-per-year", dest="periods_per_year", type=int, default=17520)
    p.add_argument("--refresh-cache", dest="refresh_cache", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    base_cfg = build_base_cfg(args)
    grid = _grid(args.ext_max_pct)

    # Sweep the stop/TP1 plane; run the full ext/reset A/B at each (stop, tp1).
    for stop_mult in args.stop_mults:
        for tp1_r in args.tp1_rs:
            results = {}
            for label, overrides in grid.items():
                pooled_rows = []
                for sym in args.symbols:
                    cfg = copy.deepcopy(base_cfg)
                    cfg.update(overrides)
                    cfg["ep_atr_stop_mult"] = stop_mult
                    cfg["ep_tp1_r"]         = tp1_r
                    cfg["symbols"] = [sym]
                    print(f"Running stop={stop_mult}×ATR tp1={tp1_r}R | {label} — {sym} ...",
                          flush=True)
                    portfolio, strategies = run_one(cfg)
                    trades = portfolio.trade_dataframe()
                    for strat in strategies:
                        pooled_rows.extend(round_trips(trades, strat.symbol))
                results[label] = cell_metrics(pooled_rows, base_cfg["initial_capital"])

            report_cfg = copy.deepcopy(base_cfg)
            report_cfg["ep_atr_stop_mult"] = stop_mult
            report_cfg["ep_tp1_r"]         = tp1_r
            print_report(results, report_cfg)


if __name__ == "__main__":
    main()