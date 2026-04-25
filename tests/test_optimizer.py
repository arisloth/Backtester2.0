"""
tests/test_optimizer.py -- Optimizer regression tests.
"""

import os
import queue
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from analytics.optimizer import _run_single, optimize
from analytics.overfit import deflated_sharpe_ratio
from core.event import (
    FillEvent, MarketEvent, OrderSide, SignalDirection, SignalEvent,
)
from data.base import DataHandler
from execution.cost_model import ZeroCommission
from execution.fill_model import FixedSlippage
from strategy.base import Strategy


def _make_bar(symbol: str, ts: str, o: float, c: float) -> dict:
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "open": o,
        "high": max(o, c) + 1,
        "low": min(o, c) - 1,
        "close": c,
        "volume": 1_000_000.0,
        "symbol": symbol,
        "asset_class": "stock",
    }


class SyntheticFeed(DataHandler):
    """Deterministic in-memory feed used by optimizer tests."""

    def __init__(self, bars):
        self._bars = bars
        self._index = 0
        self._current = {}

    def has_more(self) -> bool:
        return self._index < len(self._bars)

    def update_bars(self, events: queue.Queue) -> None:
        bar = self._bars[self._index]
        self._current[bar["symbol"]] = bar
        events.put(MarketEvent(
            symbol=bar["symbol"],
            asset_class=bar["asset_class"],
            timestamp=bar["timestamp"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        ))
        self._index += 1

    def current_bars(self):
        return self._current


class OneTradeStrategy(Strategy):
    """Stateful strategy: enter once, exit once."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bar_count = 0
        self._in_position = False

    def on_bar(self, market_event: MarketEvent):
        if market_event.symbol != self.symbol:
            return None

        self._bar_count += 1
        if self._bar_count == 1 and not self._in_position:
            return SignalEvent(
                symbol=self.symbol,
                asset_class="stock",
                timestamp=market_event.timestamp,
                direction=SignalDirection.LONG,
            )
        if self._bar_count == 3 and self._in_position:
            return SignalEvent(
                symbol=self.symbol,
                asset_class="stock",
                timestamp=market_event.timestamp,
                direction=SignalDirection.EXIT,
            )
        return None

    def on_fill(self, fill_event: FillEvent) -> None:
        if fill_event.symbol == self.symbol:
            self._in_position = fill_event.side == OrderSide.BUY


class TestOptimizerRunIsolation(unittest.TestCase):

    def test_run_single_repeated_config_is_identical(self):
        bars = [
            _make_bar("SPY", "2024-01-01", o=100, c=100),
            _make_bar("SPY", "2024-01-02", o=102, c=102),
            _make_bar("SPY", "2024-01-03", o=106, c=106),
            _make_bar("SPY", "2024-01-04", o=108, c=108),
        ]
        cfg = {
            "symbols": ["SPY"],
            "initial_capital": 10_000.0,
            "position_size_pct": 0.5,
            "risk_pct": 0.02,
            "short_borrow_rate": 0.0,
            "fill_ratio": 1.0,
            "risk_free_rate": 0.0,
            "periods_per_year": 252,
        }

        feeds = []
        strategies = []

        def build_feed(_cfg):
            feed = SyntheticFeed(list(bars))
            feeds.append(feed)
            return feed

        def build_strategy(_cfg, symbol):
            strategy = OneTradeStrategy(symbol)
            strategies.append(strategy)
            return strategy

        with patch("main.build_data_handler", side_effect=build_feed) as data_builder, \
             patch("main.build_strategy", side_effect=build_strategy) as strategy_builder, \
             patch("main.build_fill_model", side_effect=lambda _cfg: FixedSlippage(pct=0.0)) as fill_builder, \
             patch("main.build_cost_model", side_effect=lambda _cfg: ZeroCommission()) as cost_builder:
            first_metrics, first_equity, first_trades = _run_single(
                cfg, return_equity=True, return_trades=True
            )
            second_metrics, second_equity, second_trades = _run_single(
                cfg, return_equity=True, return_trades=True
            )

        self.assertEqual(first_metrics, second_metrics)
        assert_series_equal(first_equity, second_equity)
        assert_frame_equal(first_trades, second_trades)

        self.assertEqual(data_builder.call_count, 2)
        self.assertEqual(strategy_builder.call_count, 2)
        self.assertEqual(fill_builder.call_count, 2)
        self.assertEqual(cost_builder.call_count, 2)
        self.assertEqual(len({id(feed) for feed in feeds}), 2)
        self.assertEqual(len({id(strategy) for strategy in strategies}), 2)

    def test_simple_optimize_summary_json_is_byte_identical_across_runs(self):
        import main

        bars = [
            _make_bar("SPY", "2024-01-01", o=100, c=100),
            _make_bar("SPY", "2024-01-02", o=102, c=102),
            _make_bar("SPY", "2024-01-03", o=106, c=106),
            _make_bar("SPY", "2024-01-04", o=108, c=108),
        ]
        base_cfg = {
            "symbols": ["SPY"],
            "initial_capital": 10_000.0,
            "position_size_pct": 0.5,
            "risk_pct": 0.02,
            "short_borrow_rate": 0.0,
            "fill_ratio": 1.0,
            "risk_free_rate": 0.0,
            "periods_per_year": 252,
        }
        opt_cfg = {
            "mode": "simple",
            "base_cfg": base_cfg,
            "param_grid": {"fast": [5, 10]},
            "metric": "sharpe_ratio",
            "is_start": "2024-01-01",
            "is_end": "2024-01-04",
            "oos_start": "2024-01-01",
            "oos_end": "2024-01-04",
        }

        def build_feed(_cfg):
            return SyntheticFeed(list(bars))

        def build_strategy(_cfg, symbol):
            return OneTradeStrategy(symbol)

        def run_and_save(path):
            result = optimize(
                base_cfg=base_cfg,
                param_grid=opt_cfg["param_grid"],
                is_start=opt_cfg["is_start"],
                is_end=opt_cfg["is_end"],
                oos_start=opt_cfg["oos_start"],
                oos_end=opt_cfg["oos_end"],
                metric=opt_cfg["metric"],
                min_trades=1,
            )
            payload = main._simple_optimize_summary_payload(
                result, opt_cfg, opt_cfg["metric"]
            )
            main._write_summary_json(path, payload)

        with tempfile.TemporaryDirectory() as tmp, \
             patch("main.build_data_handler", side_effect=build_feed), \
             patch("main.build_strategy", side_effect=build_strategy), \
             patch("main.build_fill_model", side_effect=lambda _cfg: FixedSlippage(pct=0.0)), \
             patch("main.build_cost_model", side_effect=lambda _cfg: ZeroCommission()):
            first_path = os.path.join(tmp, "first_summary.json")
            second_path = os.path.join(tmp, "second_summary.json")
            run_and_save(first_path)
            run_and_save(second_path)

            with open(first_path, "rb") as f:
                first_bytes = f.read()
            with open(second_path, "rb") as f:
                second_bytes = f.read()

        self.assertEqual(first_bytes, second_bytes)


class TestOverfitDiagnostics(unittest.TestCase):

    def test_dsr_unavailable_for_too_few_trials(self):
        result = deflated_sharpe_ratio([1.0], pd.Series([0.01, -0.01, 0.02]))

        self.assertFalse(result["available"])
        self.assertIn("at least 2", result["reason"])

    def test_dsr_high_when_best_sharpe_exceeds_trial_distribution(self):
        result = deflated_sharpe_ratio(
            [0.1, 0.2, 0.3, 2.0],
            pd.Series([0.01, 0.012, 0.009, 0.011] * 30),
        )

        self.assertTrue(result["available"])
        self.assertGreater(result["deflated_sharpe_prob"], 0.95)
        self.assertFalse(result["warning"])

    def test_dsr_warns_for_many_similar_sharpes(self):
        result = deflated_sharpe_ratio(
            [0.98, 0.99, 1.0, 1.0, 0.99],
            pd.Series([0.01, -0.005] * 30),
        )

        self.assertTrue(result["available"])
        self.assertLess(result["deflated_sharpe_prob"], 0.95)
        self.assertTrue(result["warning"])

    def test_dsr_unavailable_for_short_return_series(self):
        result = deflated_sharpe_ratio([0.5, 1.0], pd.Series([0.01, -0.01]))

        self.assertFalse(result["available"])
        self.assertIn("return observations", result["reason"])


class TestOptimizerOverfitIntegration(unittest.TestCase):

    def test_optimize_result_includes_overfit_diagnostics(self):
        grid = pd.DataFrame([
            {"fast": 5, "sharpe_ratio": 0.2, "total_trades": 10, "_metrics": {"sharpe_ratio": 0.2}},
            {"fast": 10, "sharpe_ratio": 1.5, "total_trades": 10, "_metrics": {"sharpe_ratio": 1.5}},
        ]).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

        def run_single(_cfg, return_equity=False, return_trades=False):
            if return_equity:
                equity = pd.Series(
                    [10_000, 10_100, 10_250, 10_200, 10_400],
                    index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
                    name="equity",
                )
                return {"sharpe_ratio": 1.5}, equity
            return {"sharpe_ratio": 1.2, "total_trades": 3}

        with patch("analytics.optimizer._run_grid", return_value=grid), \
             patch("analytics.optimizer._run_single", side_effect=run_single):
            result = optimize(
                base_cfg={"initial_capital": 10_000},
                param_grid={"fast": [5, 10]},
                is_start="2024-01-01",
                is_end="2024-02-01",
                oos_start="2024-02-02",
                oos_end="2024-03-01",
                min_trades=1,
            )

        self.assertIn("available", result.overfit_diagnostics)
        self.assertIn("deflated_sharpe_prob", result.overfit_diagnostics)


if __name__ == "__main__":
    unittest.main()
