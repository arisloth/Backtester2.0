"""
pooled_filter_analysis.py — pool per-trade CSVs (with filter flags) across
symbols and measure what each filter is actually worth on true per-trade data.

Expects the trade-list format exported by scripts/ab_v2.py:
  symbol,side,entry_time,exit_time,entry_price,stop_price,quantity,pnl,R,
  hold_bars,win,pass_btc_gate,pass_rs,pass_volume,pullback_consec,
  pass_pullback_memory

Saves a JSON report to scripts/analysis/<symbols>.json alongside the console
output. The JSON mirrors every section of the console report and is loadable
by any downstream analysis script.

Usage:
    python scripts/pooled_filtered_analysis.py scripts/results/*.csv
    python scripts/pooled_filtered_analysis.py scripts/results/Sol.csv scripts/results/Eth.csv
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

FILTERS = ["pass_btc_gate", "pass_rs", "pass_volume", "pass_pullback_memory"]
BOOL_COLS = ["win"] + FILTERS

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        for c in BOOL_COLS:
            d[c] = d[c].astype(str).str.lower() == "true"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["fresh_touch"] = df["pullback_consec"] == 0  # inverted-memory candidate
    return df


# ---------------------------------------------------------------------------
# Stats helpers — dict form (used for JSON) and string form (used for console)
# ---------------------------------------------------------------------------

def stats_dict(d) -> dict:
    if len(d) == 0:
        return {"n": 0, "win_rate": None, "avg_r": None, "exp_per_trade": None,
                "profit_factor": None, "total_pnl": 0.0}
    gross_w = float(d.loc[d.pnl > 0, "pnl"].sum())
    gross_l = float(-d.loc[d.pnl < 0, "pnl"].sum())
    pf = gross_w / gross_l if gross_l > 0 else None  # None = infinite (all winners)
    r_vals = d["R"].dropna()
    return {
        "n":             int(len(d)),
        "win_rate":      round(float(d["win"].mean()), 4),
        "avg_r":         round(float(r_vals.mean()), 4) if len(r_vals) else None,
        "exp_per_trade": round(float(d["pnl"].mean()), 4),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "total_pnl":     round(float(d["pnl"].sum()), 2),
    }


def stats_str(d, label: str) -> str:
    if len(d) == 0:
        return f"{label:38s} n=  0"
    s = stats_dict(d)
    pf_str = f"{s['profit_factor']:5.2f}" if s["profit_factor"] is not None else "  inf"
    return (f"{label:38s} n={s['n']:4d}  WR={s['win_rate']*100:5.1f}%  "
            f"avgR={s['avg_r']:+5.2f}  exp$={s['exp_per_trade']:+7.2f}  "
            f"PF={pf_str}  totPnL={s['total_pnl']:+8.0f}")


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_filter_table(df, scope_label):
    print(f"\n--- {scope_label} ---")
    print(stats_str(df, "ALL"))
    for f in FILTERS:
        print(stats_str(df[df[f]], f"  {f}=PASS"))
        print(stats_str(df[~df[f]], f"  {f}=FAIL"))
    print(stats_str(df[df.fresh_touch], "  fresh_touch (consec==0)"))
    print(stats_str(df[~df.fresh_touch], "  not fresh (consec>0)"))


def print_consistency(df, mask_name, mask):
    print(f"\nPer-symbol consistency for [{mask_name}]:")
    for sym, d in df.groupby("symbol"):
        a, b = d[mask(d)], d[~mask(d)]
        def pf(x):
            gl = -x.loc[x.pnl < 0, "pnl"].sum()
            return x.loc[x.pnl > 0, "pnl"].sum() / gl if gl > 0 else float("inf")
        print(f"  {sym:12s}  IN: n={len(a):3d} PF={pf(a):5.2f} pnl={a.pnl.sum():+7.0f}   "
              f"OUT: n={len(b):3d} PF={pf(b):5.2f} pnl={b.pnl.sum():+7.0f}")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_pnl_ci(d, n_boot=5000, seed=42):
    """95% CI on total PnL via trade resampling."""
    if len(d) < 10:
        return None
    rng = np.random.default_rng(seed)
    pnl = d["pnl"].to_numpy()
    sums = rng.choice(pnl, size=(n_boot, len(pnl)), replace=True).sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "positive_at_95": bool(lo > 0)}


# ---------------------------------------------------------------------------
# Build JSON report (mirrors every console section)
# ---------------------------------------------------------------------------

def _filter_bucket(df) -> dict:
    """Stats for one scope (all/LONG/SHORT): all trades + per-filter pass/fail."""
    out = {"all": stats_dict(df)}
    for f in FILTERS:
        out[f] = {"pass": stats_dict(df[df[f]]), "fail": stats_dict(df[~df[f]])}
    out["fresh_touch"] = {
        "pass": stats_dict(df[df.fresh_touch]),
        "fail": stats_dict(df[~df.fresh_touch]),
    }
    return out


def _consistency_dict(df, mask) -> dict:
    out = {}
    for sym, d in df.groupby("symbol"):
        a, b = d[mask(d)], d[~mask(d)]
        out[sym] = {"in": stats_dict(a), "out": stats_dict(b)}
    return out


def build_report(df) -> dict:
    symbols = sorted(df.symbol.unique().tolist())
    v3 = df[df.pass_rs & df.fresh_touch]
    ci = bootstrap_pnl_ci(v3)

    return {
        "meta": {
            "symbols":      symbols,
            "total_trades": int(len(df)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "performance": {
            "all":   _filter_bucket(df),
            "LONG":  _filter_bucket(df[df.side == "LONG"]),
            "SHORT": _filter_bucket(df[df.side == "SHORT"]),
        },
        "consistency": {
            "pass_rs":       _consistency_dict(df, lambda d: d.pass_rs),
            "fresh_touch":   _consistency_dict(df, lambda d: d.fresh_touch),
            "pass_btc_gate": _consistency_dict(df, lambda d: d.pass_btc_gate),
        },
        "v3_candidate": {
            "definition": "pass_rs AND fresh_touch (no BTC gate, no volume filter)",
            "all":    stats_dict(v3),
            "LONG":   stats_dict(v3[v3.side == "LONG"]),
            "SHORT":  stats_dict(v3[v3.side == "SHORT"]),
            "bootstrap_95_ci": ci,
            "pct_of_baseline": round(len(v3) / max(len(df), 1) * 100, 1),
        },
        "sanity": {
            "losses_past_stop":       int((df["R"] < -1.05).sum()),
            "duplicate_entry_times":  int(df.duplicated(subset=["symbol", "entry_time"]).sum()),
            "win_true_negative_r":    int(len(df[(df.win) & (df["R"] < 0)])),
        },
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_json(report: dict) -> str:
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    coins = [s.split("/")[0].capitalize() for s in report["meta"]["symbols"]]
    fname = "_".join(coins) + ".json"
    path  = os.path.join(ANALYSIS_DIR, fname)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(paths):
    df = load(paths)
    symbols = ", ".join(sorted(df.symbol.unique()))

    section(f"POOLED PER-TRADE FILTER ANALYSIS  |  {symbols}  |  {len(df)} trades")

    print_filter_table(df, "POOLED (both directions)")
    for s in ["LONG", "SHORT"]:
        print_filter_table(df[df.side == s], s)

    section("CONSISTENCY ACROSS SYMBOLS (edge must hold broadly, not on one coin)")
    print_consistency(df, "pass_rs",       lambda d: d.pass_rs)
    print_consistency(df, "fresh_touch",   lambda d: d.fresh_touch)
    print_consistency(df, "pass_btc_gate", lambda d: d.pass_btc_gate)

    section("CANDIDATE V3: pass_rs AND fresh_touch (no BTC gate, no volume filter)")
    v3 = df[df.pass_rs & df.fresh_touch]
    print(stats_str(df, "V1 (all trades)"))
    print(stats_str(v3, "V3 candidate"))
    for s in ["LONG", "SHORT"]:
        print(stats_str(v3[v3.side == s], f"V3 {s}"))
    ci = bootstrap_pnl_ci(v3)
    if ci is not None:
        verdict = "edge > noise at 95%" if ci["positive_at_95"] else "NOT distinguishable from noise"
        print(f"\nBootstrap 95% CI on V3 total PnL: [{ci['lo']:+.0f}, {ci['hi']:+.0f}]  → {verdict}")
    print(f"V3 trade frequency: {len(v3)} of {len(df)} "
          f"({len(v3)/max(len(df),1)*100:.0f}% of baseline setups)")

    section("SANITY")
    print(f"Losses worse than -1.05R (slippage past stop): {(df['R'] < -1.05).sum()}")
    print(f"Duplicated entry_times within a symbol: "
          f"{df.duplicated(subset=['symbol','entry_time']).sum()}")
    bad = df[(df.win) & (df["R"] < 0)]
    print(f"Rows flagged win=True with negative R: {len(bad)}")

    report   = build_report(df)
    json_path = save_json(report)
    print(f"\nReport saved → {json_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
