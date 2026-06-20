"""
tests/test_api.py — Unit tests for api/config_schema.py and api/runner.py.

The runner tests stub out the engine/optimizer (no network, no DB writes) so
they assert the runner's orchestration and JSON-safe return shapes
deterministically. The DB persistence path itself is covered by test_db.py.
"""

import json
import unittest
from unittest import mock

import pandas as pd

from api.config_schema import build_schema
from api import runner


# ------------------------------------------------------------------
# config schema
# ------------------------------------------------------------------
class TestConfigSchema(unittest.TestCase):

    def setUp(self):
        self.schema = build_schema()

    def test_json_serializable(self):
        json.dumps(self.schema)  # raises if any non-serializable value slipped in

    def test_has_run_types_and_sections(self):
        self.assertEqual(self.schema["run_types"], ["backtest", "optimize", "walkforward"])
        keys = {s["key"] for s in self.schema["sections"]}
        self.assertTrue({"data", "capital", "slippage", "commission", "analytics"} <= keys)

    def test_strategy_sections_tagged(self):
        by_key = {s["key"]: s for s in self.schema["sections"]}
        self.assertEqual(by_key["strategy_sma"]["strategy"], "sma_cross")
        self.assertEqual(by_key["strategy_fvg"]["strategy"], "fvg")
        self.assertEqual(by_key["strategy_ep"]["strategy"], "ema_pullback")
        # data section is not strategy-specific
        self.assertNotIn("strategy", by_key["data"])

    def test_ep_section_is_dynamic_and_nonempty(self):
        ep = {s["key"]: s for s in self.schema["sections"]}["strategy_ep"]
        self.assertGreater(len(ep["fields"]), 10)
        self.assertTrue(all(f["key"].startswith("ep_") for f in ep["fields"]))

    def test_enum_field_carries_options(self):
        ds = next(f for f in self.schema["sections"][0]["fields"] if f["key"] == "data_source")
        self.assertEqual(ds["type"], "select")
        self.assertIn("ccxt", ds["options"])

    def test_options_include_intervals_per_source(self):
        intervals = self.schema["options"]["intervals"]
        self.assertIn("ccxt", intervals)
        self.assertIn("4h", intervals["ccxt"])

    def test_defaults_match_config(self):
        from main import CONFIG
        self.assertEqual(self.schema["defaults"]["initial_capital"], CONFIG["initial_capital"])


# ------------------------------------------------------------------
# runner orchestration (stubbed engine/optimizer, persist=False)
# ------------------------------------------------------------------
def _fake_metrics():
    return {
        "sharpe_ratio": 1.5, "sortino_ratio": 1.8, "max_drawdown_pct": -0.1,
        "max_drawdown_bars": 12, "cagr": 0.2, "total_trades": 5,
        "long_trades": 3, "short_trades": 2, "win_rate": 0.6,
        "profit_factor": float("inf"),  # exercises inf coercion
        "avg_win": 10.0, "avg_loss": -5.0, "expectancy": 4.0,
    }


class TestRunBacktest(unittest.TestCase):

    def test_returns_scalar_metrics_no_persist(self):
        eq = pd.Series([1000, 1010], name="equity")
        trades = pd.DataFrame([{"pnl": 5.0}])
        with mock.patch("main.execute_backtest", return_value=(eq, trades, _fake_metrics())) as m:
            out = runner.run_backtest({"symbols": ["SPY"]}, persist=False)
        m.assert_called_once()
        self.assertIsNone(out["run_id"])
        self.assertAlmostEqual(out["metrics"]["sharpe_ratio"], 1.5)
        self.assertEqual(out["metrics"]["total_trades"], 5)
        # count fields stay ints (not floats); float metrics stay floats
        for k in ("total_trades", "long_trades", "short_trades", "max_drawdown_bars"):
            self.assertIsInstance(out["metrics"][k], int)
        self.assertIsInstance(out["metrics"]["sharpe_ratio"], float)
        # inf profit factor coerced to None, and the whole payload is JSON-safe
        self.assertIsNone(out["metrics"]["profit_factor"])
        json.dumps(out)


class _FakeOpt:
    best_params = {"fast": 10}
    best_is_metrics = {"sharpe_ratio": 1.9}
    oos_metrics = {"sharpe_ratio": 1.1, "total_trades": 20}
    overfit_diagnostics = {"available": True, "deflated_sharpe_prob": 0.8}


class TestRunOptimize(unittest.TestCase):

    def test_returns_best_params_no_persist(self):
        with mock.patch("analytics.optimizer.optimize", return_value=_FakeOpt()) as m:
            out = runner.run_optimize(
                {"symbols": ["SPY"]}, {"fast": [10, 20]},
                is_start="2020-01-01", is_end="2021-12-31",
                oos_start="2022-01-01", oos_end="2022-12-31", persist=False)
        m.assert_called_once()
        self.assertIsNone(out["run_id"])
        self.assertEqual(out["best_params"], {"fast": 10})
        self.assertAlmostEqual(out["oos_metrics"]["sharpe_ratio"], 1.1)
        json.dumps(out)


class _FakeWf:
    windows = [object(), object()]
    oos_sharpe = 0.9
    oos_win_rate = 0.55
    oos_total_trades = 42


class TestRunWalkforward(unittest.TestCase):

    def test_returns_oos_aggregates_no_persist(self):
        with mock.patch("analytics.optimizer.walk_forward_months", return_value=_FakeWf()) as m:
            out = runner.run_walkforward(
                {"symbols": ["SPY"]}, {"fast": [10, 20]},
                start="2020-01-01", end="2024-12-31",
                train_months=24, test_months=6, persist=False)
        m.assert_called_once()
        self.assertIsNone(out["run_id"])
        self.assertAlmostEqual(out["oos_sharpe"], 0.9)
        self.assertEqual(out["oos_total_trades"], 42)
        self.assertEqual(out["n_windows"], 2)
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
