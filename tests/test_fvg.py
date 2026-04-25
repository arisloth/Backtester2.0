"""
tests/test_fvg.py -- FVG strategy exit-policy tests.
"""

import unittest

import pandas as pd

from core.event import MarketEvent
from strategy.examples.fvg import FVGStrategy


def _event(o=100.0, h=110.0, l=90.0, c=100.0):
    return MarketEvent(
        symbol="SPY",
        asset_class="stock",
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1_000_000.0,
    )


class TestFVGExitPolicy(unittest.TestCase):

    def test_long_stop_wins_when_stop_and_tp_hit_same_bar(self):
        strategy = FVGStrategy("SPY")
        strategy._position_side = "long"
        strategy._stop_price = 95.0
        strategy._tp2_price = 105.0

        signal = strategy._check_exit(_event(h=106.0, l=94.0))

        self.assertIsNotNone(signal)
        self.assertEqual(signal.exit_reason, "stop")
        self.assertEqual(signal.stop_price, 95.0)
        self.assertIsNone(signal.tp_price)

    def test_short_stop_wins_when_stop_and_tp_hit_same_bar(self):
        strategy = FVGStrategy("SPY")
        strategy._position_side = "short"
        strategy._stop_price = 105.0
        strategy._tp2_price = 95.0

        signal = strategy._check_exit(_event(h=106.0, l=94.0))

        self.assertIsNotNone(signal)
        self.assertEqual(signal.exit_reason, "stop")
        self.assertEqual(signal.stop_price, 105.0)
        self.assertIsNone(signal.tp_price)


if __name__ == "__main__":
    unittest.main()
