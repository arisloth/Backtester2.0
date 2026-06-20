"""
tests/test_db.py — Unit tests for the db/ persistence layer.

Each test runs against a throwaway on-disk SQLite file (FK enforcement on) so
ingest + read round-trips are exercised exactly as in production, without
touching the real backtester.db.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from db.models import (
    Base, Run, MetricSet, Trade, EquityPoint,
    OptimizerResult, OptimizerTrial, WfWindow,
)
from db.ingest import (
    ingest_backtest, ingest_optimize, ingest_walkforward,
    build_equity_dicts, parse_run_dir, MAX_EQUITY_POINTS,
)


def _equity(values, start="2020-01-01", freq="D") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="equity")


def _trades(pnls):
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "symbol": "BTC/USDT", "side": "LONG" if pnl >= 0 else "SHORT",
            "entry_time": ts, "exit_time": ts + pd.Timedelta(hours=4),
            "entry_price": 100.0, "exit_price": 100.0 + pnl,
            "quantity": 1.5, "pnl": pnl, "pnl_pct": pnl / 100.0,
            "commission": 0.1, "slippage": 0.05,
            "stop_price": 95.0, "tp_price": 110.0,
            "exit_reason": "tp" if pnl >= 0 else "stop", "hold_bars": 7,
        }
        for pnl in pnls
    ]
    return pd.DataFrame(rows)


_METRICS = {
    "sharpe_ratio": 1.23, "sortino_ratio": 1.55, "max_drawdown_pct": -0.12,
    "max_drawdown_bars": 30, "cagr": 0.18, "total_trades": 3,
    "long_trades": 2, "short_trades": 1, "win_rate": 0.667,
    "profit_factor": 2.1, "avg_win": 12.0, "avg_loss": -8.0, "expectancy": 4.0,
}

_CFG = {
    "data_source": "ccxt", "symbols": ["BTC/USDT"], "start": "2020-01-01",
    "end": "2024-12-31", "interval": "4h", "strategy": "ema_pullback",
    "initial_capital": 1000.0, "ep_adx_min": 25.0,
}


class _DbTestCase(unittest.TestCase):
    """Base providing a fresh temp-file SQLite engine + session per test."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)

        @event.listens_for(self.engine, "connect")
        def _fk(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def tearDown(self):
        self.engine.dispose()
        try:
            os.remove(self.path)
        except OSError:
            pass


class TestIngestBacktest(_DbTestCase):

    def test_round_trip(self):
        eq = _equity([1000, 1010, 1005, 1020])
        trades = _trades([12.0, -8.0, 12.0])
        with self.Session() as s:
            run_id = ingest_backtest(_CFG, _METRICS, eq, trades,
                                     run_dir="results/run_7_20260101_120000", session=s)
            s.commit()

        with self.Session() as s:
            run = s.get(Run, run_id)
            self.assertEqual(run.run_type, "backtest")
            self.assertEqual(run.run_num, 7)
            self.assertEqual(run.strategy, "ema_pullback")
            self.assertEqual(run.symbols, ["BTC/USDT"])
            # config snapshot kept
            self.assertEqual(run.config["ep_adx_min"], 25.0)
            # metrics mapped
            self.assertAlmostEqual(run.metrics.sharpe_ratio, 1.23)
            self.assertEqual(run.metrics.total_trades, 3)
            # children
            self.assertEqual(len(run.trades), 3)
            self.assertEqual(len(run.equity_points), 4)
            # trade fields coerced correctly
            t = sorted(run.trades, key=lambda x: x.pnl)[0]
            self.assertEqual(t.side, "SHORT")
            self.assertEqual(t.exit_reason, "stop")
            self.assertEqual(t.hold_bars, 7)
            self.assertIsInstance(t.entry_time, str)  # ISO string

    def test_empty_trades_and_equity(self):
        with self.Session() as s:
            run_id = ingest_backtest(_CFG, _METRICS, pd.Series(dtype=float),
                                     pd.DataFrame(), session=s)
            s.commit()
        with self.Session() as s:
            run = s.get(Run, run_id)
            self.assertEqual(len(run.trades), 0)
            self.assertEqual(len(run.equity_points), 0)
            self.assertIsNotNone(run.metrics)

    def test_inf_profit_factor_becomes_none(self):
        m = dict(_METRICS, profit_factor=float("inf"))
        with self.Session() as s:
            run_id = ingest_backtest(_CFG, m, _equity([1000, 1001]), _trades([5.0]), session=s)
            s.commit()
        with self.Session() as s:
            self.assertIsNone(s.get(Run, run_id).metrics.profit_factor)

    def test_delete_cascades_to_children(self):
        with self.Session() as s:
            run_id = ingest_backtest(_CFG, _METRICS, _equity([1, 2, 3]), _trades([1.0]), session=s)
            s.commit()
        with self.Session() as s:
            s.delete(s.get(Run, run_id))
            s.commit()
        with self.Session() as s:
            self.assertEqual(s.query(MetricSet).count(), 0)
            self.assertEqual(s.query(Trade).count(), 0)
            self.assertEqual(s.query(EquityPoint).count(), 0)


class TestEquityDownsample(unittest.TestCase):

    def test_caps_to_max_points_keeping_endpoints(self):
        n = MAX_EQUITY_POINTS * 3
        eq = _equity(list(range(n)))
        dicts = build_equity_dicts(eq)
        self.assertLessEqual(len(dicts), MAX_EQUITY_POINTS)
        self.assertEqual(dicts[0]["equity"], 0.0)
        self.assertEqual(dicts[-1]["equity"], float(n - 1))

    def test_small_series_unchanged(self):
        eq = _equity([10, 20, 30])
        self.assertEqual(len(build_equity_dicts(eq)), 3)


class TestParseRunDir(unittest.TestCase):

    def test_parses_num_and_timestamp(self):
        num, created = parse_run_dir("results/run_42_20260612_154942")
        self.assertEqual(num, 42)
        self.assertEqual(created.year, 2026)
        self.assertEqual(created.hour, 15)

    def test_none_for_unparseable(self):
        self.assertEqual(parse_run_dir(None), (None, None))
        self.assertEqual(parse_run_dir("results/weird"), (None, None))


class _FakeOptimizeResult:
    def __init__(self):
        self.best_params = {"ep_adx_min": 20.0}
        self.best_is_metrics = dict(_METRICS, sharpe_ratio=1.8)
        self.oos_metrics = dict(_METRICS, sharpe_ratio=0.9)
        self.overfit_diagnostics = {"available": True, "deflated_sharpe_prob": 0.7}
        self.all_results = pd.DataFrame([
            {"ep_adx_min": 20.0, "sharpe_ratio": 1.8, "total_trades": 50},
            {"ep_adx_min": 25.0, "sharpe_ratio": 1.2, "total_trades": 40},
        ])


class TestIngestOptimize(_DbTestCase):

    def test_round_trip(self):
        with self.Session() as s:
            run_id = ingest_optimize(
                _FakeOptimizeResult(), _CFG, metric="sharpe_ratio",
                is_start="2020-01-01", is_end="2022-12-31",
                oos_start="2023-01-01", oos_end="2024-12-31", session=s)
            s.commit()
        with self.Session() as s:
            run = s.get(Run, run_id)
            self.assertEqual(run.run_type, "optimize")
            # OOS metrics become the headline metricset
            self.assertAlmostEqual(run.metrics.sharpe_ratio, 0.9)
            opt = run.optimizer_result
            self.assertEqual(opt.best_params, {"ep_adx_min": 20.0})
            self.assertEqual(opt.overfit_diagnostics["available"], True)
            self.assertEqual(len(opt.trials), 2)
            self.assertEqual(opt.trials[0].rank, 1)


class _FakeWindow:
    def __init__(self, i):
        self.is_start, self.is_end = "2020-01-01", "2022-12-31"
        self.oos_start, self.oos_end = "2023-01-01", "2023-06-30"
        self.best_params = {"ep_adx_min": 20.0 + i}
        self.is_metrics = {"sharpe_ratio": 1.5}
        self.oos_metrics = {"sharpe_ratio": 0.8, "win_rate": 0.6,
                            "profit_factor": 1.4, "max_drawdown_pct": -0.1,
                            "total_trades": 10}
        self.oos_equity = _equity([1000 + i, 1010 + i])
        self.oos_trades = _trades([3.0, -2.0])


class _FakeWfResult:
    def __init__(self):
        self.windows = [_FakeWindow(0), _FakeWindow(1)]
        self.summary = pd.DataFrame()
        self.oos_sharpe = 0.85
        self.oos_win_rate = 0.6
        self.oos_total_trades = 20


class TestIngestWalkforward(_DbTestCase):

    def test_round_trip(self):
        with self.Session() as s:
            run_id = ingest_walkforward(
                _FakeWfResult(), _CFG, train_months=24, test_months=6,
                start="2020-01-01", end="2024-12-31", session=s)
            s.commit()
        with self.Session() as s:
            run = s.get(Run, run_id)
            self.assertEqual(run.run_type, "walkforward")
            self.assertAlmostEqual(run.metrics.sharpe_ratio, 0.85)
            self.assertEqual(len(run.wf_windows), 2)
            self.assertEqual(run.wf_windows[0].window_num, 1)
            self.assertAlmostEqual(run.wf_windows[0].oos_win_rate, 0.6)
            # stitched OOS trades + equity from both windows
            self.assertEqual(len(run.trades), 4)
            self.assertEqual(len(run.equity_points), 4)


if __name__ == "__main__":
    unittest.main()
