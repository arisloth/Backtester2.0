"""
tests/test_report.py -- Report formatting regressions.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from analytics.optimizer import OptimizeResult
from analytics.optimizer import WalkForwardResult, WalkForwardWindow
from analytics.report import optimize_report, walkforward_report


class TestWalkForwardReport(unittest.TestCase):

    def test_fold_oos_return_uses_actual_return_not_cagr(self):
        initial = 10_000.0
        equity = pd.Series(
            [initial, 11_000.0],
            index=pd.DatetimeIndex([
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-06-30", tz="UTC"),
            ]),
            name="equity",
        )
        window = WalkForwardWindow(
            is_start="2023-01-01",
            is_end="2023-12-31",
            oos_start="2024-01-01",
            oos_end="2024-06-30",
            best_params={"fvg_atr_stop_mult": 1.0},
            is_metrics={},
            oos_metrics={
                "cagr": 0.999,
                "sharpe_ratio": 1.23,
                "max_drawdown_pct": -0.05,
                "total_trades": 3,
                "expectancy": 100.0,
                "win_rate": 0.67,
                "profit_factor": 2.0,
            },
            oos_equity=equity,
            oos_trades=pd.DataFrame(),
        )
        result = WalkForwardResult(
            windows=[window],
            summary=pd.DataFrame(),
            oos_sharpe=1.23,
            oos_win_rate=0.67,
            oos_total_trades=3,
        )
        cfg = {
            "symbols": ["SPY"],
            "strategy": "fvg",
            "fvg_direction": "long",
            "initial_capital": initial,
            "data_source": "yfinance",
            "interval": "1d",
            "slippage_model": "fixed",
            "commission_model": "zero",
        }
        opt_cfg = {
            "wf_start": "2023-01-01",
            "wf_end": "2024-06-30",
            "train_months": 12,
            "test_months": 6,
            "metric": "sharpe_ratio",
        }

        with patch("analytics.report._fetch_buyhold", return_value=0.02):
            report = walkforward_report(cfg, result, opt_cfg)

        self.assertIn("+10.00%", report)
        self.assertNotIn("+99.90%", report)


class TestOptimizeReport(unittest.TestCase):

    def test_optimize_report_renders_overfit_diagnostic(self):
        result = OptimizeResult(
            best_params={"fast": 10},
            best_is_metrics={"sharpe_ratio": 1.5, "cagr": 0.2, "total_trades": 10},
            oos_metrics={"sharpe_ratio": 0.5, "cagr": 0.05, "total_trades": 3},
            all_results=pd.DataFrame([{"fast": 10, "sharpe_ratio": 1.5, "total_trades": 10}]),
            overfit_diagnostics={
                "available": True,
                "n_trials": 4,
                "best_is_sharpe": 1.5,
                "expected_max_sharpe": 1.2,
                "deflated_sharpe_prob": 0.91,
                "threshold": 0.95,
                "warning": True,
            },
        )
        cfg = {
            "symbols": ["SPY"],
            "strategy": "sma_cross",
            "initial_capital": 10_000,
            "data_source": "yfinance",
            "interval": "1d",
        }
        opt_cfg = {
            "metric": "sharpe_ratio",
            "is_start": "2024-01-01",
            "is_end": "2024-02-01",
            "oos_start": "2024-02-02",
            "oos_end": "2024-03-01",
        }

        with patch("analytics.report._fetch_buyhold", return_value=0.01):
            report = optimize_report(cfg, result, opt_cfg)

        self.assertIn("OVERFITTING DIAGNOSTIC", report)
        self.assertIn("Deflated Sharpe Ratio", report)
        self.assertIn("WARNING", report)


if __name__ == "__main__":
    unittest.main()
