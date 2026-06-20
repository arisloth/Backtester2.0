"""
db/ingest.py — persist completed runs into the database.

Maps the backtester's in-memory result objects to ORM rows:
  - ingest_backtest    ← cfg, metrics dict, equity Series, trades DataFrame
  - ingest_optimize    ← analytics.optimizer.OptimizeResult
  - ingest_walkforward ← analytics.optimizer.WalkForwardResult

Each function accepts an optional `session`. If given, the caller owns the
transaction (the row is flushed so its id is available, but not committed). If
omitted, a private session_scope() is opened and committed. All functions
return the new run's integer id.
"""

import math
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import insert

from db.engine import session_scope
from db.models import (
    Run, MetricSet, Trade, EquityPoint,
    OptimizerResult, OptimizerTrial, WfWindow,
)

# Scalar metric keys produced by analytics.metrics.compute_all().
METRIC_KEYS = [
    "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "max_drawdown_bars",
    "cagr", "total_trades", "long_trades", "short_trades", "win_rate",
    "profit_factor", "avg_win", "avg_loss", "expectancy",
]

# The subset of METRIC_KEYS that are integer counts (kept as ints in the DB and
# in API payloads); everything else in METRIC_KEYS is a float.
INT_METRIC_KEYS = ("max_drawdown_bars", "total_trades", "long_trades", "short_trades")

# Equity curves can run to hundreds of thousands of points (per-event, not
# per-bar). Storing every point bloats the DB and can't be plotted raw, so we
# downsample to at most this many points (uniform stride, endpoints kept) when
# persisting. The exact curve always remains on disk in equity.csv.
MAX_EQUITY_POINTS = 20_000

_RUN_DIR_RE = re.compile(r"run_(\d+)_(\d{8})_(\d{6})")


# ------------------------------------------------------------------
# Coercion helpers
# ------------------------------------------------------------------
def _clean_float(v):
    """Return a JSON/DB-safe float: inf and nan become None; numpy → python."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isinf(f) or math.isnan(f):
        return None
    return f


def _clean_int(v):
    """Return a JSON/DB-safe int: None, NaN, and non-finite become None."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _iso(v):
    """Timestamp-like → ISO string; passes through plain strings; None safe."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        return v
    try:
        return pd.Timestamp(v).isoformat()
    except Exception:
        return str(v)


def _json_safe(obj):
    """Recursively coerce dicts/lists of numpy/pandas scalars into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(x) for k, x in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return _clean_float(obj)
    # numpy scalars expose .item(); fall back to str for anything exotic.
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    f = _clean_float(obj)
    return f if f is not None else str(obj)


def parse_run_dir(run_dir: Optional[str]):
    """Extract (run_num, created_at) from a 'results/run_N_YYYYMMDD_HHMMSS' path."""
    if not run_dir:
        return None, None
    m = _RUN_DIR_RE.search(run_dir)
    if not m:
        return None, None
    run_num = int(m.group(1))
    try:
        created_at = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
    except ValueError:
        created_at = None
    return run_num, created_at


def _serialize_config(cfg: dict) -> dict:
    """JSON-safe copy of a config dict (matches main.py's _save_results filter)."""
    if not cfg:
        return {}
    return {
        k: v for k, v in cfg.items()
        if isinstance(v, (int, float, str, bool, list, type(None)))
    }


def _mc_to_dict(mc) -> Optional[dict]:
    """Serialize a MonteCarloResults dataclass (or dict) to a JSON-safe summary."""
    if mc is None:
        return None
    fields = [
        "n_iterations", "initial_capital", "p_profit", "p_dd_exceed",
        "dd_threshold", "median_equity", "pct5_equity", "pct95_equity",
        "assumption_note",
    ]
    out = {}
    for f in fields:
        val = mc.get(f) if isinstance(mc, dict) else getattr(mc, f, None)
        if isinstance(val, str) or val is None:
            out[f] = val
        else:
            out[f] = _clean_float(val)
    return out


def _build_metricset(metrics: dict) -> MetricSet:
    """Create a MetricSet row from a compute_all()-style metrics dict."""
    ms = MetricSet()
    for k in METRIC_KEYS:
        val = metrics.get(k)
        ms_val = _clean_int(val) if k in INT_METRIC_KEYS else _clean_float(val)
        setattr(ms, k, ms_val)
    ms.monte_carlo = _mc_to_dict(metrics.get("monte_carlo"))
    return ms


_TRADE_COLS = [
    "symbol", "side", "entry_time", "exit_time", "entry_price",
    "exit_price", "quantity", "pnl", "pnl_pct", "commission",
    "slippage", "stop_price", "tp_price", "exit_reason", "hold_bars",
]


def build_trade_dicts(trades: pd.DataFrame):
    """Coerce a trade-log DataFrame into a list of plain dicts for bulk insert
    (no run_id). May be empty."""
    if trades is None or trades.empty:
        return []
    str_fields = {"symbol", "side", "exit_reason"}
    time_fields = {"entry_time", "exit_time"}
    int_fields = {"hold_bars"}
    out = []
    for rec in trades.to_dict("records"):
        row = {}
        for col in _TRADE_COLS:
            val = rec.get(col)
            if col in time_fields:
                row[col] = _iso(val)
            elif col in str_fields:
                row[col] = None if val is None else str(val)
            elif col in int_fields:
                cf = _clean_float(val)
                row[col] = int(cf) if cf is not None else None
            else:
                row[col] = _clean_float(val)
        out.append(row)
    return out


def build_equity_dicts(eq: pd.Series, max_points: int = MAX_EQUITY_POINTS):
    """Coerce an equity Series into a list of {timestamp, equity} dicts,
    downsampled to at most max_points (uniform stride, endpoints kept)."""
    if eq is None or len(eq) == 0:
        return []
    n = len(eq)
    if max_points and n > max_points:
        # Uniform stride keeping first and last point.
        import numpy as np
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        eq = eq.iloc[idx]
    out = []
    for ts, val in eq.items():
        v = _clean_float(val)
        if v is None:
            continue
        out.append({"timestamp": _iso(ts), "equity": v})
    return out


def stitch_equity_curves(curves):
    """Chain per-window OOS equity curves into one continuous curve.

    Each walk-forward window is an independent backtest that starts fresh at
    initial_capital, so concatenating the raw dollar levels would reset to the
    starting capital at every window boundary — a sawtooth, not a tradable
    equity curve. Instead we compound each window's returns onto the running
    equity: the first window is kept as-is, and every later window is rescaled
    so its opening point continues from the previous window's closing equity.

    Returns a single chronological Series, or None if there's nothing to chain.
    """
    cleaned = [c.dropna() for c in curves if isinstance(c, pd.Series)]
    cleaned = [c for c in cleaned if not c.empty]
    if not cleaned:
        return None
    stitched = [cleaned[0]]
    running = float(cleaned[0].iloc[-1])
    for c in cleaned[1:]:
        base = float(c.iloc[0])
        # Rescale by growth factor (multiplicative compounding); fall back to an
        # additive offset only if the window opens at zero (no growth factor).
        scaled = (c / base * running) if base != 0 else (c - base + running)
        stitched.append(scaled)
        running = float(scaled.iloc[-1])
    return pd.concat(stitched)


def _bulk_insert_children(session, run_id: int, trade_dicts, equity_dicts):
    """Fast executemany insert of a run's trades and equity points."""
    if trade_dicts:
        session.execute(insert(Trade), [{**d, "run_id": run_id} for d in trade_dicts])
    if equity_dicts:
        session.execute(insert(EquityPoint), [{**d, "run_id": run_id} for d in equity_dicts])


# ------------------------------------------------------------------
# Public ingest functions
# ------------------------------------------------------------------
def _run_in_session(fn, session):
    """Run fn(session) either in the caller's session or a private scope."""
    if session is not None:
        run_id = fn(session)
        session.flush()
        return run_id
    with session_scope() as s:
        return fn(s)


def ingest_backtest(
    cfg: dict,
    metrics: dict,
    eq: pd.Series,
    trades: pd.DataFrame,
    *,
    run_dir: Optional[str] = None,
    status: str = "done",
    session=None,
) -> int:
    """Persist a single backtest run. Returns the new run id."""
    def _do(s):
        run_num, created_at = parse_run_dir(run_dir)
        run = Run(
            run_num=run_num,
            run_type="backtest",
            status=status,
            data_source=(cfg or {}).get("data_source"),
            symbols=_json_safe((cfg or {}).get("symbols")),
            start=(cfg or {}).get("start"),
            end=(cfg or {}).get("end"),
            interval=(cfg or {}).get("interval"),
            strategy=(cfg or {}).get("strategy"),
            config=_serialize_config(cfg),
            results_dir=run_dir,
        )
        if created_at is not None:
            run.created_at = created_at
        run.metrics = _build_metricset(metrics or {})
        s.add(run)
        s.flush()
        _bulk_insert_children(s, run.id, build_trade_dicts(trades), build_equity_dicts(eq))
        return run.id

    return _run_in_session(_do, session)


def ingest_optimize(
    result,
    cfg: dict,
    *,
    mode: str = "simple",
    metric: str = "sharpe_ratio",
    is_start: Optional[str] = None,
    is_end: Optional[str] = None,
    oos_start: Optional[str] = None,
    oos_end: Optional[str] = None,
    run_dir: Optional[str] = None,
    status: str = "done",
    session=None,
) -> int:
    """Persist an IS/OOS OptimizeResult. Returns the new run id."""
    def _do(s):
        run_num, created_at = parse_run_dir(run_dir)
        run = Run(
            run_num=run_num,
            run_type="optimize",
            status=status,
            data_source=(cfg or {}).get("data_source"),
            symbols=_json_safe((cfg or {}).get("symbols")),
            start=(cfg or {}).get("start"),
            end=(cfg or {}).get("end"),
            interval=(cfg or {}).get("interval"),
            strategy=(cfg or {}).get("strategy"),
            config=_serialize_config(cfg),
            results_dir=run_dir,
        )
        if created_at is not None:
            run.created_at = created_at

        # OOS metrics double as the run's headline MetricSet for list views.
        oos_metrics = getattr(result, "oos_metrics", {}) or {}
        run.metrics = _build_metricset(oos_metrics)

        opt = OptimizerResult(
            mode=mode,
            metric=metric,
            best_params=_json_safe(getattr(result, "best_params", {})),
            is_metrics=_json_safe(getattr(result, "best_is_metrics", {})),
            oos_metrics=_json_safe(oos_metrics),
            overfit_diagnostics=_json_safe(getattr(result, "overfit_diagnostics", {})),
            is_start=is_start, is_end=is_end, oos_start=oos_start, oos_end=oos_end,
        )
        all_results = getattr(result, "all_results", None)
        if isinstance(all_results, pd.DataFrame) and not all_results.empty:
            for i, row in enumerate(all_results.to_dict("records")):
                opt.trials.append(OptimizerTrial(rank=i + 1, row=_json_safe(row)))
        run.optimizer_result = opt

        s.add(run)
        s.flush()
        return run.id

    return _run_in_session(_do, session)


def ingest_walkforward(
    result,
    cfg: dict,
    *,
    train_months: Optional[int] = None,
    test_months: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    metric: str = "sharpe_ratio",
    run_dir: Optional[str] = None,
    status: str = "done",
    session=None,
) -> int:
    """Persist a WalkForwardResult, stitching per-window OOS trades/equity.
    Returns the new run id."""
    def _do(s):
        run_num, created_at = parse_run_dir(run_dir)
        # Fold the walk-forward run params into the config snapshot — there's no
        # dedicated WF metadata table, and these describe how the run was set up.
        wf_config = _serialize_config(cfg)
        for k, v in (("train_months", train_months), ("test_months", test_months),
                     ("metric", metric)):
            if v is not None:
                wf_config[k] = v
        run = Run(
            run_num=run_num,
            run_type="walkforward",
            status=status,
            data_source=(cfg or {}).get("data_source"),
            symbols=_json_safe((cfg or {}).get("symbols")),
            start=start or (cfg or {}).get("start"),
            end=end or (cfg or {}).get("end"),
            interval=(cfg or {}).get("interval"),
            strategy=(cfg or {}).get("strategy"),
            config=wf_config,
            results_dir=run_dir,
        )
        if created_at is not None:
            run.created_at = created_at

        # Headline metrics from the aggregate OOS figures.
        run.metrics = _build_metricset({
            "sharpe_ratio": getattr(result, "oos_sharpe", None),
            "win_rate": getattr(result, "oos_win_rate", None),
            "total_trades": getattr(result, "oos_total_trades", None),
        })

        windows = getattr(result, "windows", []) or []
        stitched_trades = []
        stitched_equity = []
        for i, w in enumerate(windows):
            is_m = getattr(w, "is_metrics", {}) or {}
            oos_m = getattr(w, "oos_metrics", {}) or {}
            run.wf_windows.append(WfWindow(
                window_num=i + 1,
                is_start=getattr(w, "is_start", None),
                is_end=getattr(w, "is_end", None),
                oos_start=getattr(w, "oos_start", None),
                oos_end=getattr(w, "oos_end", None),
                best_params=_json_safe(getattr(w, "best_params", {})),
                is_sharpe=_clean_float(is_m.get("sharpe_ratio")),
                oos_sharpe=_clean_float(oos_m.get("sharpe_ratio")),
                oos_win_rate=_clean_float(oos_m.get("win_rate")),
                oos_profit_factor=_clean_float(oos_m.get("profit_factor")),
                oos_max_drawdown=_clean_float(oos_m.get("max_drawdown_pct")),
                oos_trades=int(oos_m["total_trades"]) if oos_m.get("total_trades") is not None else None,
            ))
            wt = getattr(w, "oos_trades", None)
            if isinstance(wt, pd.DataFrame) and not wt.empty:
                stitched_trades.append(wt)
            we = getattr(w, "oos_equity", None)
            if isinstance(we, pd.Series) and len(we) > 0:
                stitched_equity.append(we)

        s.add(run)
        s.flush()
        trade_df = pd.concat(stitched_trades, ignore_index=True) if stitched_trades else None
        equity_s = stitch_equity_curves(stitched_equity)
        _bulk_insert_children(s, run.id, build_trade_dicts(trade_df), build_equity_dicts(equity_s))
        return run.id

    return _run_in_session(_do, session)
