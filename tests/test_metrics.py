"""
tests/test_metrics.py — Unit tests for analytics/metrics.py and monte_carlo.py.
"""

import unittest

import numpy as np
import pandas as pd

from analytics.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_duration,
    cagr, trade_metrics, compute_all,
)
from analytics.monte_carlo import run_monte_carlo


def _equity(values, start="2020-01-01", freq="B") -> pd.Series:
    """Build a pd.Series equity curve from a list of values."""
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="equity")


def _trades(pnls) -> pd.DataFrame:
    """Build a minimal trade DataFrame from a list of P&L values."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        {
            "symbol": "SPY", "side": "LONG",
            "entry_time": ts, "exit_time": ts,
            "entry_price": 100.0, "exit_price": 100.0 + pnl,
            "quantity": 1.0, "pnl": pnl, "commission": 0.0,
        }
        for pnl in pnls
    ]
    return pd.DataFrame(rows)


class TestSharpeRatio(unittest.TestCase):

    def test_flat_equity_returns_zero(self):
        eq = _equity([100] * 252)
        self.assertAlmostEqual(sharpe_ratio(eq), 0.0)

    def test_positive_trending_equity_positive_sharpe(self):
        values = [100 + i * 0.1 for i in range(252)]
        eq = _equity(values)
        self.assertGreater(sharpe_ratio(eq), 0.0)

    def test_declining_equity_negative_sharpe(self):
        values = [100 - i * 0.1 for i in range(252)]
        eq = _equity(values)
        self.assertLess(sharpe_ratio(eq), 0.0)

    def test_single_point_returns_zero(self):
        eq = _equity([100])
        self.assertAlmostEqual(sharpe_ratio(eq), 0.0)


class TestSortinoRatio(unittest.TestCase):

    def test_no_downside_returns_zero(self):
        # Equity only goes up — no downside deviation
        values = [100 + i for i in range(252)]
        eq = _equity(values)
        # Sortino is 0 because downside_std is 0 when no negative excess returns
        self.assertAlmostEqual(sortino_ratio(eq), 0.0)

    def test_mixed_returns_nonzero(self):
        np.random.seed(1)
        returns = np.random.randn(252) * 0.01 + 0.0005
        values = [100.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        eq = _equity(values)
        result = sortino_ratio(eq)
        self.assertIsInstance(result, float)


class TestMaxDrawdown(unittest.TestCase):

    def test_no_drawdown_returns_zero(self):
        eq = _equity([100, 101, 102, 103])
        self.assertAlmostEqual(max_drawdown(eq), 0.0)

    def test_drawdown_is_negative(self):
        eq = _equity([100, 110, 90, 95])
        dd = max_drawdown(eq)
        self.assertLess(dd, 0.0)

    def test_drawdown_magnitude(self):
        # Peak = 110, trough = 90 → DD = (90-110)/110 ≈ -0.1818
        eq = _equity([100, 110, 90, 95])
        dd = max_drawdown(eq)
        self.assertAlmostEqual(dd, (90 - 110) / 110, places=5)

    def test_empty_series_returns_zero(self):
        self.assertAlmostEqual(max_drawdown(pd.Series(dtype=float)), 0.0)


class TestMaxDrawdownDuration(unittest.TestCase):

    def test_no_drawdown_duration_zero(self):
        eq = _equity([100, 101, 102, 103])
        self.assertEqual(max_drawdown_duration(eq), 0)

    def test_drawdown_duration(self):
        # Below peak from bar 1 to bar 3 → duration = 3
        eq = _equity([100, 90, 85, 95, 101])
        dur = max_drawdown_duration(eq)
        self.assertGreater(dur, 0)


class TestCAGR(unittest.TestCase):

    def test_flat_equity_zero_cagr(self):
        eq = _equity([100] * 252)
        self.assertAlmostEqual(cagr(eq), 0.0, places=4)

    def test_doubling_equity_positive_cagr(self):
        # 252 bars (1 year), 100 → 200 → CAGR ≈ 100%
        eq = _equity([100 + i * (100 / 251) for i in range(252)])
        result = cagr(eq)
        self.assertGreater(result, 0.5)

    def test_short_series_returns_zero(self):
        self.assertAlmostEqual(cagr(_equity([100])), 0.0)


class TestTradeMetrics(unittest.TestCase):

    def test_empty_trades_returns_zeros(self):
        result = trade_metrics(pd.DataFrame())
        self.assertEqual(result["total_trades"], 0)
        self.assertAlmostEqual(result["win_rate"], 0.0)

    def test_all_winners(self):
        result = trade_metrics(_trades([100, 200, 50]))
        self.assertAlmostEqual(result["win_rate"], 1.0)
        self.assertEqual(result["total_trades"], 3)
        self.assertEqual(result["profit_factor"], 9999.0)  # sentinel: all wins, no losses

    def test_all_losers(self):
        result = trade_metrics(_trades([-100, -50, -200]))
        self.assertAlmostEqual(result["win_rate"], 0.0)
        self.assertAlmostEqual(result["profit_factor"], 0.0)

    def test_mixed_trades(self):
        result = trade_metrics(_trades([100, -50, 200, -100]))
        self.assertAlmostEqual(result["win_rate"], 0.5)
        # gross profit = 300, gross loss = 150 → PF = 2.0
        self.assertAlmostEqual(result["profit_factor"], 2.0)

    def test_expectancy(self):
        result = trade_metrics(_trades([100, -50]))
        # win_rate=0.5, avg_win=100, avg_loss=-50
        # expectancy = 0.5*100 + 0.5*(-50) = 25
        self.assertAlmostEqual(result["expectancy"], 25.0)

    def test_long_short_counts(self):
        df = pd.DataFrame([
            {"symbol": "SPY", "side": "LONG",  "pnl": 100, "entry_time": pd.Timestamp("2024-01-01"),
             "exit_time": pd.Timestamp("2024-01-02"), "entry_price": 100, "exit_price": 101,
             "quantity": 1, "commission": 0},
            {"symbol": "SPY", "side": "SHORT", "pnl": -50, "entry_time": pd.Timestamp("2024-01-01"),
             "exit_time": pd.Timestamp("2024-01-02"), "entry_price": 100, "exit_price": 100.5,
             "quantity": 1, "commission": 0},
        ])
        result = trade_metrics(df)
        self.assertEqual(result["long_trades"],  1)
        self.assertEqual(result["short_trades"], 1)


class TestComputeAll(unittest.TestCase):

    def test_returns_all_keys(self):
        eq = _equity([100 + i * 0.05 for i in range(252)])
        result = compute_all(eq, pd.DataFrame())
        expected_keys = [
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "max_drawdown_bars",
            "cagr", "total_trades", "long_trades", "short_trades",
            "win_rate", "profit_factor", "avg_win", "avg_loss", "expectancy",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")


class TestMonteCarlo(unittest.TestCase):

    def test_all_winning_trades_high_p_profit(self):
        trades = _trades([100] * 20)
        results = run_monte_carlo(trades, initial_capital=10_000, n=200, seed=42)
        self.assertGreater(results.p_profit, 0.95)

    def test_all_losing_trades_low_p_profit(self):
        trades = _trades([-100] * 20)
        results = run_monte_carlo(trades, initial_capital=10_000, n=200, seed=42)
        self.assertLess(results.p_profit, 0.05)

    def test_output_shape(self):
        trades = _trades([50, -30, 80, -20, 100])
        results = run_monte_carlo(trades, initial_capital=10_000, n=100, seed=0)
        self.assertEqual(len(results.terminal_equities), 100)
        self.assertEqual(len(results.max_drawdowns), 100)

    def test_paths_stored_when_requested(self):
        trades = _trades([50, -30, 80])
        results = run_monte_carlo(trades, initial_capital=10_000, n=50,
                                  return_paths=True, seed=0)
        self.assertIsNotNone(results.equity_paths)
        self.assertEqual(results.equity_paths.shape, (50, len(trades) + 1))

    def test_paths_not_stored_by_default(self):
        trades = _trades([50, -30, 80])
        results = run_monte_carlo(trades, initial_capital=10_000, n=50, seed=0)
        self.assertIsNone(results.equity_paths)

    def test_empty_trades_raises(self):
        with self.assertRaises(ValueError):
            run_monte_carlo(pd.DataFrame(), initial_capital=10_000)

    def test_percentiles_ordered(self):
        trades = _trades([50, -30, 80, -20, 100, -10, 60])
        results = run_monte_carlo(trades, initial_capital=10_000, n=500, seed=7)
        self.assertLessEqual(results.pct5_equity, results.median_equity)
        self.assertLessEqual(results.median_equity, results.pct95_equity)


if __name__ == "__main__":
    unittest.main()
