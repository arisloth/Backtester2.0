"""
results.py — Save backtest outputs to a per-run folder under results/.

Folder structure:
    results/
        run_1_20260407_141247/
            trades.csv
            equity.csv
            metrics.json
            report.txt
"""

import json
import os
from datetime import datetime


def next_run_number() -> int:
    """Return the next run number by counting existing run_ folders under results/."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return 1
    existing = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ]
    return len(existing) + 1


def save_results(cfg: dict, metrics: dict, eq, trades) -> str:
    """
    Save trade log, equity curve, metrics JSON, and a human-readable report
    into a new numbered run folder.  Returns the path to the folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num   = next_run_number()
    run_dir   = os.path.join("results", f"run_{run_num}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Trade log CSV
    trades_path = os.path.join(run_dir, "trades.csv")
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        import pandas as pd
        pd.DataFrame().to_csv(trades_path, index=False)

    # Equity curve CSV
    eq.to_frame(name="equity").to_csv(os.path.join(run_dir, "equity.csv"))

    # Metrics JSON
    serializable = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    serializable["_config"] = {
        k: v for k, v in cfg.items()
        if isinstance(v, (int, float, str, bool, list, type(None)))
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Human-readable report
    from analytics.report import backtest_report, save_report
    mc = metrics.get("monte_carlo")
    report_text = backtest_report(cfg, metrics, eq, trades, mc=mc)
    save_report(report_text, run_dir)
    print(report_text)

    print(f"\n  Results saved to: {run_dir}/")
    print(f"    trades.csv  |  equity.csv  |  metrics.json  |  report.txt")

    return run_dir
