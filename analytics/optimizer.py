"""
analytics/optimizer.py — Parameter optimization and walk-forward validation.

Two modes:

  1. Simple IS/OOS split
     Optimize params on in-sample data, validate on out-of-sample once.
     Use this to find robust params without overfitting.

  2. Walk-forward
     Roll IS/OOS windows forward across the full date range.
     Each window picks the best params on IS, tests on OOS.
     Stitch OOS results together for a realistic view of real-world performance.

Quick usage
-----------
    from analytics.optimizer import optimize, walk_forward

    param_grid = {
        "fvg_atr_stop_mult": [0.5, 0.75, 1.0],
        "fvg_tp_atr_mult":   [2.0, 3.0, 4.0],
    }

    # Simple IS/OOS
    result = optimize(
        base_cfg=cfg,
        param_grid=param_grid,
        is_start="2018-01-01",
        is_end="2021-12-31",
        oos_start="2022-01-01",
        oos_end="2024-12-31",
        metric="sharpe_ratio",    # metric to maximize on IS
    )
    print(result.best_params)
    print(result.oos_metrics)
    print(result.all_results)     # full ranked table of all IS runs

    # Walk-forward
    wf = walk_forward(
        base_cfg=cfg,
        param_grid=param_grid,
        start="2015-01-01",
        end="2024-12-31",
        train_years=3,
        test_years=1,
        metric="sharpe_ratio",
    )
    print(wf.summary)             # per-window best params + IS/OOS metrics
    print(wf.oos_sharpe)          # combined OOS Sharpe
"""

import copy
import itertools
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """Result of a single IS/OOS optimization run."""
    best_params: dict
    best_is_metrics: dict
    oos_metrics: dict
    all_results: pd.DataFrame   # all IS param combinations, ranked by metric


@dataclass
class WalkForwardWindow:
    """One IS/OOS window within a walk-forward run."""
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict
    is_metrics: dict
    oos_metrics: dict


@dataclass
class WalkForwardResult:
    """Aggregated result of a walk-forward run."""
    windows: List[WalkForwardWindow]
    summary: pd.DataFrame       # one row per window — params + IS/OOS key metrics
    oos_sharpe: float           # trade-count-weighted OOS Sharpe across all windows
    oos_win_rate: float         # trade-count-weighted OOS win rate
    oos_total_trades: int       # total OOS trades across all windows


# ---------------------------------------------------------------------------
# Core: silent single backtest run
# ---------------------------------------------------------------------------

def _run_single(cfg: dict) -> dict:
    """
    Run one backtest silently (no prints, no charts, no file saves).
    Returns the metrics dict from analytics.metrics.compute_all.
    """
    # Suppress all logging below WARNING for clean optimizer output
    import logging as _logging
    _logging.disable(_logging.INFO)
    try:
        from main import build_data_handler, build_strategy, build_fill_model, build_cost_model
        from core.portfolio import Portfolio
        from core.broker import Broker
        from core.engine import Engine
        from analytics.metrics import compute_all

        feed       = build_data_handler(cfg)
        strategies = [build_strategy(cfg, s) for s in cfg["symbols"]]
        portfolio  = Portfolio(
            initial_capital=cfg["initial_capital"],
            position_size_pct=cfg["position_size_pct"],
        )
        broker = Broker(
            fill_model=build_fill_model(cfg),
            cost_model=build_cost_model(cfg),
            fill_ratio=cfg.get("fill_ratio", 1.0),
        )
        engine = Engine(
            data_handler=feed,
            strategies=strategies,
            portfolio=portfolio,
            broker=broker,
        )
        engine.run()

        eq     = portfolio.equity_series()
        trades = portfolio.trade_dataframe()
        return compute_all(
            eq, trades,
            risk_free_rate=cfg.get("risk_free_rate", 0.0),
            periods_per_year=cfg.get("periods_per_year", 252),
        )
    finally:
        _logging.disable(_logging.NOTSET)


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

def _param_combinations(param_grid: Dict[str, list]) -> List[dict]:
    """Return a list of all param dicts from the Cartesian product of param_grid."""
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_grid(base_cfg: dict, param_grid: Dict[str, list], metric: str) -> pd.DataFrame:
    """
    Run all parameter combinations from param_grid on the date range in base_cfg.
    Returns a DataFrame with one row per combination, sorted by metric descending.
    """
    combos  = _param_combinations(param_grid)
    total   = len(combos)
    records = []

    for i, params in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(params)
        logger.info(f"  [{i}/{total}] {params}")

        try:
            metrics = _run_single(cfg)
        except Exception as e:
            logger.warning(f"  Run failed for {params}: {e}")
            metrics = {}

        row = dict(params)
        row.update({
            "sharpe_ratio":  metrics.get("sharpe_ratio",  float("nan")),
            "sortino_ratio": metrics.get("sortino_ratio", float("nan")),
            "cagr":          metrics.get("cagr",          float("nan")),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", float("nan")),
            "win_rate":      metrics.get("win_rate",      float("nan")),
            "profit_factor": metrics.get("profit_factor", float("nan")),
            "expectancy":    metrics.get("expectancy",    float("nan")),
            "total_trades":  metrics.get("total_trades",  0),
            "_metrics":      metrics,   # full dict stored for later
        })
        records.append(row)

    df = pd.DataFrame(records)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API: simple IS/OOS optimize
# ---------------------------------------------------------------------------

def optimize(
    base_cfg: dict,
    param_grid: Dict[str, list],
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
) -> OptimizeResult:
    """
    Grid search over param_grid on [is_start, is_end], then validate the
    best params on [oos_start, oos_end].

    Parameters
    ----------
    base_cfg : dict
        Base config dict (from main.py CONFIG). Date fields will be overridden.
    param_grid : dict
        Mapping of config key → list of values to try.
        e.g. {"fvg_atr_stop_mult": [0.5, 0.75, 1.0], "fvg_tp_atr_mult": [2.0, 3.0]}
    is_start / is_end : str
        In-sample date range ("YYYY-MM-DD").
    oos_start / oos_end : str
        Out-of-sample date range ("YYYY-MM-DD").
    metric : str
        Metric to maximize when selecting best IS params.
        Must be a key returned by analytics.metrics.compute_all.
    min_trades : int
        Minimum number of trades required for a result to be eligible.
        Prevents selecting low-activity param sets that look good by luck.

    Returns
    -------
    OptimizeResult
    """
    combos = _param_combinations(param_grid)
    print(f"\n  Optimizer: {len(combos)} combinations × IS {is_start}→{is_end}")
    print(f"  Metric: {metric}  |  OOS: {oos_start}→{oos_end}\n")

    # --- In-sample grid search ---
    is_cfg = copy.deepcopy(base_cfg)
    is_cfg["start"] = is_start
    is_cfg["end"]   = is_end

    results = _run_grid(is_cfg, param_grid, metric)

    # Filter by min_trades
    eligible = results[results["total_trades"] >= min_trades]
    if eligible.empty:
        logger.warning(
            f"No param combination produced >= {min_trades} trades in IS. "
            "Using best overall result."
        )
        eligible = results

    best_row    = eligible.iloc[0]
    best_params = {k: best_row[k] for k in param_grid.keys()}
    best_is_metrics = best_row["_metrics"]

    print(f"  Best IS params: {best_params}")
    print(f"  IS {metric}: {best_row.get(metric, 'n/a'):.4f}  |  trades: {int(best_row['total_trades'])}")

    # --- Out-of-sample validation ---
    print(f"\n  Running OOS validation...")
    oos_cfg = copy.deepcopy(base_cfg)
    oos_cfg.update(best_params)
    oos_cfg["start"] = oos_start
    oos_cfg["end"]   = oos_end
    oos_metrics = _run_single(oos_cfg)

    print(f"  OOS {metric}: {oos_metrics.get(metric, 'n/a'):.4f}  |  trades: {oos_metrics.get('total_trades', 0)}")

    # Drop internal _metrics column before returning
    display_results = results.drop(columns=["_metrics"], errors="ignore")

    return OptimizeResult(
        best_params=best_params,
        best_is_metrics=best_is_metrics,
        oos_metrics=oos_metrics,
        all_results=display_results,
    )


# ---------------------------------------------------------------------------
# Public API: walk-forward
# ---------------------------------------------------------------------------

def _date_add_years(d: date, years: int) -> date:
    """Add N years to a date, clamping Feb 29 if needed."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def _date_add_months(d: date, months: int) -> date:
    """Add N months to a date, clamping to end of month if needed."""
    month = d.month - 1 + months
    year  = d.year + month // 12
    month = month % 12 + 1
    import calendar
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def _build_windows_months(start: str, end: str, train_months: int, test_months: int) -> List[dict]:
    """
    Build walk-forward windows using month-based train/test sizes.
    """
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)
    windows = []

    is_start_d = start_d
    while True:
        is_end_d    = _date_add_months(is_start_d, train_months) - timedelta(days=1)
        oos_start_d = is_end_d + timedelta(days=1)
        oos_end_d   = _date_add_months(oos_start_d, test_months) - timedelta(days=1)

        if oos_start_d >= end_d:
            break
        oos_end_d = min(oos_end_d, end_d)

        windows.append({
            "is_start":  is_start_d.isoformat(),
            "is_end":    is_end_d.isoformat(),
            "oos_start": oos_start_d.isoformat(),
            "oos_end":   oos_end_d.isoformat(),
        })

        is_start_d = _date_add_months(is_start_d, test_months)

    return windows


def _build_windows(start: str, end: str, train_years: int, test_years: int) -> List[dict]:
    """
    Build a list of {is_start, is_end, oos_start, oos_end} dicts by rolling
    a fixed-size IS window forward by test_years each step.
    """
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)
    windows = []

    is_start_d = start_d
    while True:
        is_end_d   = _date_add_years(is_start_d, train_years) - timedelta(days=1)
        oos_start_d = is_end_d + timedelta(days=1)
        oos_end_d   = _date_add_years(oos_start_d, test_years) - timedelta(days=1)

        if oos_start_d >= end_d:
            break
        oos_end_d = min(oos_end_d, end_d)

        windows.append({
            "is_start":  is_start_d.isoformat(),
            "is_end":    is_end_d.isoformat(),
            "oos_start": oos_start_d.isoformat(),
            "oos_end":   oos_end_d.isoformat(),
        })

        is_start_d = _date_add_years(is_start_d, test_years)

    return windows


def walk_forward(
    base_cfg: dict,
    param_grid: Dict[str, list],
    start: str,
    end: str,
    train_years: int = 3,
    test_years: int = 1,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
) -> WalkForwardResult:
    """
    Rolling walk-forward optimization.

    For each window, runs a grid search on the IS period to find the best
    params, then tests those params on the OOS period. Rolls forward by
    test_years and repeats.

    Parameters
    ----------
    base_cfg : dict
        Base config dict (from main.py CONFIG). Date fields will be overridden.
    param_grid : dict
        Mapping of config key → list of values to try.
    start / end : str
        Full date range for the walk-forward ("YYYY-MM-DD").
    train_years : int
        Length of each IS window in years (default 3).
    test_years : int
        Length of each OOS window in years, also the step size (default 1).
    metric : str
        Metric to maximize on IS (default "sharpe_ratio").
    min_trades : int
        Min trades required in IS for a result to be eligible.

    Returns
    -------
    WalkForwardResult
    """
    windows_spec = _build_windows(start, end, train_years, test_years)
    combos = _param_combinations(param_grid)

    print(f"\n  Walk-Forward: {len(windows_spec)} windows × {len(combos)} combinations")
    print(f"  Train {train_years}y / Test {test_years}y  |  Metric: {metric}\n")

    wf_windows: List[WalkForwardWindow] = []
    summary_rows = []

    for idx, w in enumerate(windows_spec, 1):
        print(f"  ── Window {idx}/{len(windows_spec)}: "
              f"IS {w['is_start']}→{w['is_end']}  |  OOS {w['oos_start']}→{w['oos_end']}")

        # IS grid search
        is_cfg = copy.deepcopy(base_cfg)
        is_cfg["start"] = w["is_start"]
        is_cfg["end"]   = w["is_end"]
        results = _run_grid(is_cfg, param_grid, metric)

        eligible = results[results["total_trades"] >= min_trades]
        if eligible.empty:
            eligible = results

        best_row    = eligible.iloc[0]
        best_params = {k: best_row[k] for k in param_grid.keys()}
        is_metrics  = best_row["_metrics"]

        print(f"     Best: {best_params}  |  IS {metric}={best_row.get(metric, float('nan')):.3f}")

        # OOS test
        oos_cfg = copy.deepcopy(base_cfg)
        oos_cfg.update(best_params)
        oos_cfg["start"] = w["oos_start"]
        oos_cfg["end"]   = w["oos_end"]

        try:
            oos_metrics = _run_single(oos_cfg)
        except Exception as e:
            logger.warning(f"OOS run failed: {e}")
            oos_metrics = {}

        oos_sharpe = oos_metrics.get(metric, float("nan"))
        oos_trades = oos_metrics.get("total_trades", 0)
        print(f"     OOS {metric}={oos_sharpe:.3f}  |  trades={oos_trades}\n")

        wf_windows.append(WalkForwardWindow(
            is_start=w["is_start"],
            is_end=w["is_end"],
            oos_start=w["oos_start"],
            oos_end=w["oos_end"],
            best_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
        ))

        row = dict(best_params)
        row.update({
            "is_start":  w["is_start"],
            "is_end":    w["is_end"],
            "oos_start": w["oos_start"],
            "oos_end":   w["oos_end"],
            f"is_{metric}":  best_row.get(metric, float("nan")),
            f"oos_{metric}": oos_sharpe,
            "is_trades":  int(best_row.get("total_trades", 0)),
            "oos_trades": oos_trades,
            "oos_win_rate":      oos_metrics.get("win_rate", float("nan")),
            "oos_profit_factor": oos_metrics.get("profit_factor", float("nan")),
            "oos_max_drawdown":  oos_metrics.get("max_drawdown_pct", float("nan")),
        })
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Aggregate: trade-count-weighted OOS Sharpe
    total_oos_trades = sum(w.oos_metrics.get("total_trades", 0) for w in wf_windows)
    if total_oos_trades > 0:
        weighted_sharpe = sum(
            w.oos_metrics.get("sharpe_ratio", 0.0) * w.oos_metrics.get("total_trades", 0)
            for w in wf_windows
        ) / total_oos_trades
        weighted_win_rate = sum(
            w.oos_metrics.get("win_rate", 0.0) * w.oos_metrics.get("total_trades", 0)
            for w in wf_windows
        ) / total_oos_trades
    else:
        weighted_sharpe   = float("nan")
        weighted_win_rate = float("nan")

    print(f"\n  Walk-Forward Complete")
    print(f"  OOS Sharpe (weighted): {weighted_sharpe:.3f}")
    print(f"  OOS Win Rate (weighted): {weighted_win_rate:.1%}")
    print(f"  Total OOS trades: {total_oos_trades}")

    return WalkForwardResult(
        windows=wf_windows,
        summary=summary,
        oos_sharpe=weighted_sharpe,
        oos_win_rate=weighted_win_rate,
        oos_total_trades=total_oos_trades,
    )


def walk_forward_months(
    base_cfg: dict,
    param_grid: Dict[str, list],
    start: str,
    end: str,
    train_months: int = 36,
    test_months: int = 6,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
) -> WalkForwardResult:
    """
    Walk-forward using month-based train/test windows instead of years.
    Identical to walk_forward() but uses _build_windows_months internally.
    """
    windows_spec = _build_windows_months(start, end, train_months, test_months)
    combos = _param_combinations(param_grid)

    print(f"\n  Walk-Forward: {len(windows_spec)} windows × {len(combos)} combinations")
    print(f"  Train {train_months}m / Test {test_months}m  |  Metric: {metric}\n")

    # Reuse the same core loop as walk_forward by injecting windows_spec
    wf_windows: List[WalkForwardWindow] = []
    summary_rows = []

    for idx, w in enumerate(windows_spec, 1):
        print(f"  ── Window {idx}/{len(windows_spec)}: "
              f"IS {w['is_start']}→{w['is_end']}  |  OOS {w['oos_start']}→{w['oos_end']}")

        is_cfg = copy.deepcopy(base_cfg)
        is_cfg["start"] = w["is_start"]
        is_cfg["end"]   = w["is_end"]
        results = _run_grid(is_cfg, param_grid, metric)

        eligible = results[results["total_trades"] >= min_trades]
        if eligible.empty:
            eligible = results

        best_row    = eligible.iloc[0]
        best_params = {k: best_row[k] for k in param_grid.keys()}
        is_metrics  = best_row["_metrics"]

        print(f"     Best: {best_params}  |  IS {metric}={best_row.get(metric, float('nan')):.3f}")

        oos_cfg = copy.deepcopy(base_cfg)
        oos_cfg.update(best_params)
        oos_cfg["start"] = w["oos_start"]
        oos_cfg["end"]   = w["oos_end"]

        try:
            oos_metrics = _run_single(oos_cfg)
        except Exception as e:
            logger.warning(f"OOS run failed: {e}")
            oos_metrics = {}

        oos_sharpe = oos_metrics.get(metric, float("nan"))
        oos_trades = oos_metrics.get("total_trades", 0)
        print(f"     OOS {metric}={oos_sharpe:.3f}  |  trades={oos_trades}\n")

        wf_windows.append(WalkForwardWindow(
            is_start=w["is_start"], is_end=w["is_end"],
            oos_start=w["oos_start"], oos_end=w["oos_end"],
            best_params=best_params, is_metrics=is_metrics, oos_metrics=oos_metrics,
        ))

        row = dict(best_params)
        row.update({
            "is_start": w["is_start"], "is_end": w["is_end"],
            "oos_start": w["oos_start"], "oos_end": w["oos_end"],
            f"is_{metric}": best_row.get(metric, float("nan")),
            f"oos_{metric}": oos_sharpe,
            "is_trades": int(best_row.get("total_trades", 0)),
            "oos_trades": oos_trades,
            "oos_win_rate":      oos_metrics.get("win_rate", float("nan")),
            "oos_profit_factor": oos_metrics.get("profit_factor", float("nan")),
            "oos_max_drawdown":  oos_metrics.get("max_drawdown_pct", float("nan")),
        })
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    total_oos_trades = sum(w.oos_metrics.get("total_trades", 0) for w in wf_windows)
    if total_oos_trades > 0:
        weighted_sharpe = sum(
            w.oos_metrics.get("sharpe_ratio", 0.0) * w.oos_metrics.get("total_trades", 0)
            for w in wf_windows
        ) / total_oos_trades
        weighted_win_rate = sum(
            w.oos_metrics.get("win_rate", 0.0) * w.oos_metrics.get("total_trades", 0)
            for w in wf_windows
        ) / total_oos_trades
    else:
        weighted_sharpe   = float("nan")
        weighted_win_rate = float("nan")

    print(f"\n  Walk-Forward Complete")
    print(f"  OOS Sharpe (weighted): {weighted_sharpe:.3f}")
    print(f"  OOS Win Rate (weighted): {weighted_win_rate:.1%}")
    print(f"  Total OOS trades: {total_oos_trades}")

    return WalkForwardResult(
        windows=wf_windows,
        summary=summary,
        oos_sharpe=weighted_sharpe,
        oos_win_rate=weighted_win_rate,
        oos_total_trades=total_oos_trades,
    )
