"""
tests/test_data_alignment.py -- Multi-symbol data alignment diagnostics.
"""

import logging
import unittest

import pandas as pd

from data.alignment import align_symbol_data


def _df(days):
    timestamps = [pd.Timestamp(day, tz="UTC") for day in days]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0 + i for i in range(len(timestamps))],
        "high": [101.0 + i for i in range(len(timestamps))],
        "low": [99.0 + i for i in range(len(timestamps))],
        "close": [100.5 + i for i in range(len(timestamps))],
        "volume": [1_000_000.0 for _ in timestamps],
    })


class TestSymbolAlignment(unittest.TestCase):

    def test_full_overlap_logs_no_warning_level_data_loss(self):
        data = {
            "SPY": _df(["2024-01-01", "2024-01-02"]),
            "QQQ": _df(["2024-01-01", "2024-01-02"]),
        }
        logger = logging.getLogger("tests.alignment.full")

        with self.assertLogs("tests.alignment.full", level="INFO") as captured:
            aligned = align_symbol_data(data, logger, context="Test symbols")

        self.assertEqual(len(aligned["SPY"]), 2)
        self.assertEqual(len(aligned["QQQ"]), 2)
        self.assertFalse(any(record.startswith("WARNING:") for record in captured.output))

    def test_partial_overlap_aligns_to_common_timestamps(self):
        data = {
            "SPY": _df(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "QQQ": _df(["2024-01-02", "2024-01-03", "2024-01-04"]),
        }
        logger = logging.getLogger("tests.alignment.partial")

        aligned = align_symbol_data(data, logger, context="Test symbols")

        expected = [
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
        ]
        self.assertEqual(list(aligned["SPY"]["timestamp"]), expected)
        self.assertEqual(list(aligned["QQQ"]["timestamp"]), expected)

    def test_partial_overlap_logs_dropped_and_missing_counts(self):
        data = {
            "SPY": _df(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "QQQ": _df(["2024-01-02", "2024-01-03", "2024-01-04"]),
        }
        logger_name = "tests.alignment.logging"
        logger = logging.getLogger(logger_name)

        with self.assertLogs(logger_name, level="WARNING") as captured:
            align_symbol_data(data, logger, context="Test symbols")

        logs = "\n".join(captured.output)
        self.assertIn("2/4 union timestamps removed", logs)
        self.assertIn("SPY: original=3 aligned=2 dropped=1 missing_vs_union=1", logs)
        self.assertIn("QQQ: original=3 aligned=2 dropped=1 missing_vs_union=1", logs)

    def test_no_overlap_raises_value_error(self):
        data = {
            "SPY": _df(["2024-01-01"]),
            "QQQ": _df(["2024-01-02"]),
        }
        logger = logging.getLogger("tests.alignment.none")

        with self.assertRaises(ValueError):
            align_symbol_data(data, logger, context="Test symbols")


if __name__ == "__main__":
    unittest.main()
