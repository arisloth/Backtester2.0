"""
api/runner.py — headless entry points for launching runs.

These wrap the existing engine/optimizer code so a caller (the FastAPI job
worker, a script, a test) can launch a run without the interactive CLI:
execute → persist to the DB → return the new run id plus a JSON-safe summary.

Every config is layered over main.CONFIG so partial configs from the dashboard
still carry all the keys the engine expects. Nothing here renders charts or
writes a results/ folder — the database is the system of record for these runs.
"""

from typing import Optional

from db import init_db, session_scope
from db.ingest import (
    ingest_backtest, ingest_optimize, ingest_walkforward,
    METRIC_KEYS, _mc_to_dict, _json_safe, _clean_float,
)


def _full_cfg(cfg: Optional[dict]) -> dict:
    """Layer a partial config over the CONFIG defaults so all keys are present."""
    from main import CONFIG
    full = dict(CONFIG)
    full.update(cfg or {})
    return full


def _scalar_metrics(metrics: dict) -> dict:
    """JSON-safe headline metrics (scalars + monte carlo summary)."""
    out = {k: _clean_float(metrics.get(k)) for k in METRIC_KEYS}
    mc = _mc_to_dict(metrics.get("monte_carlo"))
    if mc is not None:
        out["monte_carlo"] = mc
    return out


def run_backtest(cfg: Optional[dict] = None, *, persist: bool = True) -> dict:
    """Run a single backtest. Returns {run_id, metrics}."""
    from main import execute_backtest
    full = _full_cfg(cfg)
    eq, trades, metrics = execute_backtest(full, run_mc=True, verbose=False)

    run_id = None
    if persist:
        init_db()
        with session_scope() as s:
            run_id = ingest_backtest(full, metrics, eq, trades, session=s)

    return {"run_id": run_id, "metrics": _scalar_metrics(metrics)}


def run_optimize(
    cfg: Optional[dict],
    param_grid: dict,
    *,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
    persist: bool = True,
) -> dict:
    """Run an IS/OOS grid search. Returns {run_id, best_params, is/oos metrics, ...}."""
    from main import _apply_periods_per_year
    from analytics.optimizer import optimize

    full = _full_cfg(cfg)
    _apply_periods_per_year(full)
    result = optimize(
        base_cfg=full, param_grid=param_grid,
        is_start=is_start, is_end=is_end, oos_start=oos_start, oos_end=oos_end,
        metric=metric, min_trades=min_trades,
    )

    run_id = None
    if persist:
        init_db()
        with session_scope() as s:
            run_id = ingest_optimize(
                result, full, metric=metric,
                is_start=is_start, is_end=is_end, oos_start=oos_start, oos_end=oos_end,
                session=s,
            )

    return {
        "run_id": run_id,
        "best_params": _json_safe(result.best_params),
        "is_metrics": _json_safe(result.best_is_metrics),
        "oos_metrics": _json_safe(result.oos_metrics),
        "overfit_diagnostics": _json_safe(result.overfit_diagnostics),
    }


def run_walkforward(
    cfg: Optional[dict],
    param_grid: dict,
    *,
    start: str,
    end: str,
    train_months: int = 36,
    test_months: int = 6,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
    n_jobs: int = 1,
    persist: bool = True,
) -> dict:
    """Run a month-based walk-forward. Returns {run_id, oos_sharpe, ...}."""
    from main import _apply_periods_per_year
    from analytics.optimizer import walk_forward_months

    full = _full_cfg(cfg)
    _apply_periods_per_year(full)
    result = walk_forward_months(
        base_cfg=full, param_grid=param_grid,
        start=start, end=end,
        train_months=train_months, test_months=test_months,
        metric=metric, min_trades=min_trades, n_jobs=n_jobs,
    )

    run_id = None
    if persist:
        init_db()
        with session_scope() as s:
            run_id = ingest_walkforward(
                result, full,
                train_months=train_months, test_months=test_months,
                start=start, end=end, metric=metric, session=s,
            )

    return {
        "run_id": run_id,
        "oos_sharpe": _clean_float(result.oos_sharpe),
        "oos_win_rate": _clean_float(result.oos_win_rate),
        "oos_total_trades": int(result.oos_total_trades) if result.oos_total_trades is not None else None,
        "n_windows": len(result.windows),
    }
