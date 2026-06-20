"""
db/backfill.py — load existing results/run_*/ folders into the database.

The CLI has been writing per-run folders long before the DB existed. This
one-off (re-runnable) script reconstructs Run rows from the files on disk so
historical runs show up in the dashboard. It is idempotent by results_dir:
folders already imported are skipped.

Usage:
    python -m db.backfill                 # import all of results/
    python -m db.backfill --results DIR   # custom results directory
    python -m db.backfill --reset         # drop + recreate tables first
"""

import argparse
import json
import os

import pandas as pd

from db.engine import session_scope, init_db, engine
from db.models import Base, Run, Trade, EquityPoint, OptimizerResult, OptimizerTrial, WfWindow
from db.ingest import (
    _build_metricset, build_trade_dicts, build_equity_dicts, _bulk_insert_children,
    _serialize_config, parse_run_dir, _json_safe, _clean_float,
)

# Columns in walk-forward summary.csv that are NOT strategy params.
_WF_KNOWN_COLS = {
    "is_start", "is_end", "oos_start", "oos_end",
    "is_sharpe_ratio", "oos_sharpe_ratio", "is_trades", "oos_trades",
    "oos_win_rate", "oos_profit_factor", "oos_max_drawdown",
}


def _read_trades(path: str) -> pd.DataFrame:
    """Read a trades.csv, tolerating the empty-file case."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def _read_equity(path: str) -> pd.Series:
    """Read an equity.csv (index = timestamp, column = equity)."""
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError:
        return pd.Series(dtype=float)
    if "equity" in df.columns:
        return df["equity"]
    return df.iloc[:, 0] if df.shape[1] else pd.Series(dtype=float)


def _backfill_backtest(s, run_dir: str, metrics: dict):
    cfg = metrics.pop("_config", {}) or {}
    run_num, created_at = parse_run_dir(run_dir)
    run = Run(
        run_num=run_num, run_type="backtest", status="done",
        data_source=cfg.get("data_source"), symbols=_json_safe(cfg.get("symbols")),
        start=cfg.get("start"), end=cfg.get("end"),
        interval=cfg.get("interval"), strategy=cfg.get("strategy"),
        config=_serialize_config(cfg), results_dir=run_dir,
    )
    if created_at is not None:
        run.created_at = created_at
    run.metrics = _build_metricset(metrics)
    s.add(run)
    s.flush()
    _bulk_insert_children(
        s, run.id,
        build_trade_dicts(_read_trades(os.path.join(run_dir, "trades.csv"))),
        build_equity_dicts(_read_equity(os.path.join(run_dir, "equity.csv"))),
    )


def _backfill_walkforward(s, run_dir: str, summary: dict):
    cfg = summary.get("_config", {}) or {}
    run_num, created_at = parse_run_dir(run_dir)
    run = Run(
        run_num=run_num, run_type="walkforward", status="done",
        data_source=cfg.get("data_source"), symbols=_json_safe(cfg.get("symbols")),
        start=summary.get("start") or cfg.get("start"),
        end=summary.get("end") or cfg.get("end"),
        interval=cfg.get("interval"), strategy=cfg.get("strategy"),
        config=_serialize_config(cfg), results_dir=run_dir,
    )
    if created_at is not None:
        run.created_at = created_at
    run.metrics = _build_metricset({
        "sharpe_ratio": summary.get("oos_sharpe"),
        "win_rate": summary.get("oos_win_rate"),
        "total_trades": summary.get("oos_total_trades"),
    })

    # Per-window rows from summary.csv.
    csv_path = os.path.join(run_dir, "summary.csv")
    if os.path.exists(csv_path):
        try:
            wf = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            wf = pd.DataFrame()
        for i, row in enumerate(wf.to_dict("records")):
            params = {k: _json_safe(v) for k, v in row.items() if k not in _WF_KNOWN_COLS}
            run.wf_windows.append(WfWindow(
                window_num=i + 1,
                is_start=row.get("is_start"), is_end=row.get("is_end"),
                oos_start=row.get("oos_start"), oos_end=row.get("oos_end"),
                best_params=params,
                is_sharpe=_clean_float(row.get("is_sharpe_ratio")),
                oos_sharpe=_clean_float(row.get("oos_sharpe_ratio")),
                oos_win_rate=_clean_float(row.get("oos_win_rate")),
                oos_profit_factor=_clean_float(row.get("oos_profit_factor")),
                oos_max_drawdown=_clean_float(row.get("oos_max_drawdown")),
                oos_trades=int(row["oos_trades"]) if row.get("oos_trades") is not None and not pd.isna(row.get("oos_trades")) else None,
            ))

    s.add(run)
    s.flush()
    # Combined OOS trades, if present.
    _bulk_insert_children(
        s, run.id,
        build_trade_dicts(_read_trades(os.path.join(run_dir, "trades.csv"))),
        [],
    )


def _backfill_optimize(s, run_dir: str, summary: dict):
    cfg = summary.get("_config", {}) or {}
    run_num, created_at = parse_run_dir(run_dir)
    run = Run(
        run_num=run_num, run_type="optimize", status="done",
        data_source=cfg.get("data_source"), symbols=_json_safe(cfg.get("symbols")),
        start=cfg.get("start"), end=cfg.get("end"),
        interval=cfg.get("interval"), strategy=cfg.get("strategy"),
        config=_serialize_config(cfg), results_dir=run_dir,
    )
    if created_at is not None:
        run.created_at = created_at
    oos_metrics = summary.get("oos_metrics", {}) or {}
    run.metrics = _build_metricset(oos_metrics)
    opt = OptimizerResult(
        mode=summary.get("mode", "simple"),
        metric=summary.get("metric", "sharpe_ratio"),
        best_params=_json_safe(summary.get("best_params", {})),
        is_metrics=_json_safe(summary.get("is_metrics", {})),
        oos_metrics=_json_safe(oos_metrics),
        overfit_diagnostics=_json_safe(summary.get("overfit_diagnostics", {})),
        is_start=summary.get("is_start"), is_end=summary.get("is_end"),
        oos_start=summary.get("oos_start"), oos_end=summary.get("oos_end"),
    )
    # IS leaderboard from all_runs.csv.
    runs_csv = os.path.join(run_dir, "all_runs.csv")
    if os.path.exists(runs_csv):
        try:
            ar = pd.read_csv(runs_csv)
        except pd.errors.EmptyDataError:
            ar = pd.DataFrame()
        for i, row in enumerate(ar.to_dict("records")):
            opt.trials.append(OptimizerTrial(rank=i + 1, row=_json_safe(row)))
    run.optimizer_result = opt
    s.add(run)
    s.flush()
    _bulk_insert_children(
        s, run.id,
        build_trade_dicts(_read_trades(os.path.join(run_dir, "trades.csv"))),
        [],
    )


def backfill(results_dir: str = "results") -> dict:
    """Import all run_* folders under results_dir. Returns a count summary."""
    init_db()
    counts = {"backtest": 0, "optimize": 0, "walkforward": 0, "skipped": 0, "error": 0}

    if not os.path.isdir(results_dir):
        print(f"No results directory at '{results_dir}'.")
        return counts

    with session_scope() as s:
        existing = {r for (r,) in s.query(Run.results_dir).all() if r}

        for name in sorted(os.listdir(results_dir)):
            run_dir = os.path.join(results_dir, name)
            if not (os.path.isdir(run_dir) and name.startswith("run_")):
                continue
            if run_dir in existing:
                counts["skipped"] += 1
                continue

            metrics_path = os.path.join(run_dir, "metrics.json")
            summary_path = os.path.join(run_dir, "summary.json")
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        _backfill_backtest(s, run_dir, json.load(f))
                    counts["backtest"] += 1
                elif os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    if summary.get("mode") == "walkforward":
                        _backfill_walkforward(s, run_dir, summary)
                        counts["walkforward"] += 1
                    else:
                        _backfill_optimize(s, run_dir, summary)
                        counts["optimize"] += 1
                else:
                    counts["skipped"] += 1
            except Exception as e:  # one bad folder shouldn't abort the import
                counts["error"] += 1
                print(f"  ! {name}: {type(e).__name__}: {e}")

    return counts


def main():
    ap = argparse.ArgumentParser(description="Backfill backtester results into the DB.")
    ap.add_argument("--results", default="results", help="results directory (default: results)")
    ap.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    args = ap.parse_args()

    if args.reset:
        print("Dropping and recreating all tables...")
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

    counts = backfill(args.results)
    print("\nBackfill complete:")
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
