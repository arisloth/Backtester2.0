"""
tests/test_ema_pullback.py — EMA Pullback strategy unit tests.

Mirrors the style of tests/test_fvg.py: direct strategy instantiation,
hand-crafted bar streams or hand-set state, assertions on emitted signals
and `_check_exit` behavior.
"""

import unittest
from typing import Optional

import pandas as pd

from core.event import (
    FillEvent,
    MarketEvent,
    OrderSide,
    SignalDirection,
)
from strategy.examples.ema_pullback import EMAPullbackStrategy


def _event(o=100.0, h=110.0, l=90.0, c=100.0, t="2024-01-01"):
    return MarketEvent(
        symbol="ETH/USDT",
        asset_class="crypto",
        timestamp=pd.Timestamp(t, tz="UTC"),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1_000_000.0,
    )


def _fill(side: OrderSide, price: float, qty: float = 1.0):
    return FillEvent(
        symbol="ETH/USDT",
        asset_class="crypto",
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        side=side,
        quantity=qty,
        fill_price=price,
        commission=0.0,
        slippage=0.0,
    )


def _make_strategy(**overrides) -> EMAPullbackStrategy:
    defaults = dict(symbol="ETH/USDT", asset_class="crypto", direction="long")
    defaults.update(overrides)
    return EMAPullbackStrategy(**defaults)


def _drive(strategy: EMAPullbackStrategy, bars):
    """Feed bars in order, collecting any emitted signals."""
    signals = []
    for b in bars:
        ev = _event(**b)
        sig = strategy.on_bar(ev)
        if sig is not None:
            signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Exit policy
# ---------------------------------------------------------------------------

class TestEMAPullbackExitPolicy(unittest.TestCase):
    """Direct _check_exit tests — mirror tests/test_fvg.py:28-52."""

    def test_long_stop_wins_when_stop_and_tp1_hit_same_bar(self):
        s = _make_strategy()
        s._in_position    = True
        s._position_side  = "long"
        s._stop_price     = 95.0
        s._tp1_price      = 105.0
        s._tp2_price      = 110.0
        s._entry_bar      = 0
        s._bar_count      = 1
        s._atr            = 1.0
        s._entry_price    = 100.0

        sig = s._check_exit(_event(h=106.0, l=94.0))

        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "stop")
        self.assertEqual(sig.stop_price, 95.0)
        self.assertIsNone(sig.tp_price)
        self.assertEqual(sig.strength, 1.0)

    def test_short_stop_wins_when_stop_and_tp1_hit_same_bar(self):
        s = _make_strategy(direction="short")
        s._in_position    = True
        s._position_side  = "short"
        s._stop_price     = 105.0
        s._tp1_price      = 95.0
        s._tp2_price      = 90.0
        s._entry_bar      = 0
        s._bar_count      = 1
        s._atr            = 1.0
        s._entry_price    = 100.0

        sig = s._check_exit(_event(h=106.0, l=94.0))

        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "stop")
        self.assertEqual(sig.stop_price, 105.0)
        self.assertIsNone(sig.tp_price)
        self.assertEqual(sig.strength, 1.0)

    def test_long_tp1_partial_strength(self):
        s = _make_strategy(tp1_ratio=0.5)
        s._in_position    = True
        s._position_side  = "long"
        s._stop_price     = 95.0
        s._tp1_price      = 105.0
        s._tp2_price      = 110.0
        s._entry_bar      = 0
        s._bar_count      = 1
        s._atr            = 1.0
        s._entry_price    = 100.0

        sig = s._check_exit(_event(h=106.0, l=99.0))

        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "tp1")
        self.assertAlmostEqual(sig.strength, 0.5)
        self.assertEqual(sig.tp_price, 105.0)

    def test_timeout_fires_at_max_hold_bars(self):
        s = _make_strategy(max_hold_bars=10)
        s._in_position    = True
        s._position_side  = "long"
        s._stop_price     = 95.0
        s._tp1_price      = 105.0
        s._tp2_price      = 110.0
        s._entry_bar      = 0
        s._bar_count      = 10  # exactly max_hold_bars later
        s._atr            = 1.0
        s._entry_price    = 100.0

        sig = s._check_exit(_event(h=104.0, l=96.0))

        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "timeout")


# ---------------------------------------------------------------------------
# TP1 → BE move (via on_fill)
# ---------------------------------------------------------------------------

class TestTP1BreakEvenMove(unittest.TestCase):

    def test_long_tp1_partial_fill_moves_stop_to_be(self):
        s = _make_strategy()
        s._in_position        = True
        s._position_side      = "long"
        s._entry_price        = 100.0
        s._stop_price         = 95.0
        s._tp1_price          = 105.0
        s._tp2_price          = 110.0
        s._tp1_pending_fill   = True
        s._exit_pending       = True
        s._stop_distance      = 5.0

        # TP1 partial sell fill arrives
        s.on_fill(_fill(OrderSide.SELL, price=105.0, qty=0.5))

        self.assertTrue(s._tp1_hit)
        self.assertFalse(s._tp1_pending_fill)
        self.assertFalse(s._exit_pending)
        self.assertEqual(s._stop_price, 100.0)  # BE
        # Still in position with runner
        self.assertTrue(s._in_position)

    def test_short_tp1_partial_fill_moves_stop_to_be(self):
        s = _make_strategy(direction="short")
        s._in_position        = True
        s._position_side      = "short"
        s._entry_price        = 100.0
        s._stop_price         = 105.0
        s._tp1_price          = 95.0
        s._tp2_price          = 90.0
        s._tp1_pending_fill   = True
        s._exit_pending       = True
        s._stop_distance      = 5.0

        # TP1 partial buy-to-cover fill arrives
        s.on_fill(_fill(OrderSide.BUY, price=95.0, qty=0.5))

        self.assertTrue(s._tp1_hit)
        self.assertFalse(s._tp1_pending_fill)
        self.assertEqual(s._stop_price, 100.0)  # BE
        self.assertTrue(s._in_position)


# ---------------------------------------------------------------------------
# Runner modes
# ---------------------------------------------------------------------------

class TestRunnerModes(unittest.TestCase):
    """After TP1 is hit, runner exits according to runner_mode."""

    def _setup_long_after_tp1(self, runner_mode: str, **kw) -> EMAPullbackStrategy:
        s = _make_strategy(runner_mode=runner_mode, **kw)
        s._in_position    = True
        s._position_side  = "long"
        s._entry_price    = 100.0
        s._stop_price     = 100.0  # BE after TP1
        s._tp1_price      = 105.0
        s._tp1_hit        = True
        s._stop_distance  = 5.0
        s._entry_bar      = 0
        s._bar_count      = 5
        s._atr            = 2.0
        return s

    def test_fixed_r_tp2_exits_at_target(self):
        s = self._setup_long_after_tp1("fixed_r", runner_fixed_r=4.0)
        # TP2 = 100 + 4 * 5 = 120
        s._tp2_price = 120.0

        # Bar prints to 121 — should hit TP2
        sig = s._check_exit(_event(h=121.0, l=119.0))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "tp2")
        self.assertEqual(sig.tp_price, 120.0)

        # Bar that doesn't reach 120 → no exit
        s2 = self._setup_long_after_tp1("fixed_r", runner_fixed_r=4.0)
        s2._tp2_price = 120.0
        self.assertIsNone(s2._check_exit(_event(h=119.0, l=110.0)))

    def test_structure_tp2_exits_at_overhead(self):
        s = self._setup_long_after_tp1("structure")
        s._tp2_price = 115.0  # overhead resistance

        sig = s._check_exit(_event(h=116.0, l=110.0))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "tp2")
        self.assertEqual(sig.tp_price, 115.0)

    def test_atr_trail_exits_when_low_pierces_trail(self):
        s = self._setup_long_after_tp1("atr_trail", atr_trail_mult=2.0)
        s._tp2_price = None
        s._trail_anchor = 120.0  # highest since entry
        # trail = 120 - 2*2 = 116

        sig = s._check_exit(_event(h=119.0, l=115.0))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "trail")
        self.assertAlmostEqual(sig.stop_price, 116.0)

    def test_atr_trail_does_not_exit_above_trail(self):
        s = self._setup_long_after_tp1("atr_trail", atr_trail_mult=2.0)
        s._tp2_price = None
        s._trail_anchor = 120.0
        # trail = 116; bar low 117 → no exit
        self.assertIsNone(s._check_exit(_event(h=121.0, l=117.0)))

    def test_short_atr_trail(self):
        s = _make_strategy(direction="short", runner_mode="atr_trail", atr_trail_mult=2.0)
        s._in_position    = True
        s._position_side  = "short"
        s._entry_price    = 100.0
        s._stop_price     = 100.0  # BE
        s._tp1_hit        = True
        s._entry_bar      = 0
        s._bar_count      = 5
        s._atr            = 2.0
        s._trail_anchor   = 80.0  # lowest since entry
        # trail = 80 + 2*2 = 84

        sig = s._check_exit(_event(h=85.0, l=82.0))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.exit_reason, "trail")
        self.assertAlmostEqual(sig.stop_price, 84.0)


# ---------------------------------------------------------------------------
# Entry gating (regime + ADX + pullback)
# ---------------------------------------------------------------------------

class TestEntryGates(unittest.TestCase):
    """Test _check_entry directly with hand-set indicators."""

    def _stage(self, **kw):
        """Build a strategy with indicators pre-seeded so _check_entry can fire."""
        # Disable Supertrend filter so we test the EMA/ADX/pullback gates in isolation.
        kw.setdefault("daily_supertrend_filter", False)
        s = _make_strategy(swing_lookback=3, **kw)
        # Pre-populate bars so swing_low scan has data
        s._bars.append({"open": 99.0, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 1.0})
        s._bars.append({"open": 100.0, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1.0})
        s._bar_count = 2
        s._atr        = 1.0
        s._adx        = 30.0
        s._ema_fast   = 99.0
        s._ema_slow   = 95.0
        s._ema_trend  = 90.0
        return s

    def test_long_entry_fires_on_touch_and_hold(self):
        s = self._stage()
        # Bar low pierces EMA_fast (99.0), close holds above it
        # Append the would-be current bar so the swing scan includes it
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1

        ev = _event(o=100.0, h=100.5, l=98.9, c=99.5)
        sig = s._check_entry(ev)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, SignalDirection.LONG)
        self.assertIsNotNone(sig.stop_price)
        self.assertLess(sig.stop_price, ev.close)

    def test_skip_when_regime_tangled(self):
        # Tangled stack: ema_fast < ema_slow (not the trend stack)
        s = self._stage()
        s._ema_fast = 90.0
        s._ema_slow = 95.0
        s._ema_trend = 99.0
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1

        sig = s._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNone(sig)

    def test_skip_when_adx_below_min(self):
        s = self._stage()
        s._adx = 15.0  # below default 25
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1

        sig = s._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNone(sig)

    def test_skip_when_no_pullback_touch(self):
        # Price hovering above EMA_fast without touching it
        s = self._stage()
        bar = {"open": 100.0, "high": 101.0, "low": 100.5, "close": 100.8, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1

        sig = s._check_entry(_event(o=100.0, h=101.0, l=100.5, c=100.8))
        self.assertIsNone(sig)

    def test_skip_when_close_breaks_below_ema(self):
        # Wicked through EMA and CLOSED below — invalidation, not a pullback
        s = self._stage()
        bar = {"open": 99.5, "high": 99.5, "low": 98.0, "close": 98.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1

        # close 98.5 is BELOW ema_fast 99 → not "held"
        sig = s._check_entry(_event(o=99.5, h=99.5, l=98.0, c=98.5))
        self.assertIsNone(sig)


# ---------------------------------------------------------------------------
# Indicator sanity (ATR + ADX warm-up behavior)
# ---------------------------------------------------------------------------

class TestIndicatorWarmup(unittest.TestCase):

    def test_atr_and_adx_eventually_non_none(self):
        s = _make_strategy(atr_period=5, adx_period=5, ema_fast=3, ema_slow=5, ema_trend=10)
        # Feed 50 bars of trending data
        bars = []
        price = 100.0
        for i in range(50):
            price += 1.0
            bars.append({
                "o": price - 0.5,
                "h": price + 0.5,
                "l": price - 1.0,
                "c": price,
            })
        for b in bars:
            s.on_bar(_event(o=b["o"], h=b["h"], l=b["l"], c=b["c"]))

        self.assertIsNotNone(s._atr)
        self.assertGreater(s._atr, 0)
        self.assertIsNotNone(s._adx)
        # Strongly trending up → ADX should be elevated
        self.assertGreater(s._adx, 25.0)
        self.assertIsNotNone(s._ema_fast)
        self.assertIsNotNone(s._ema_slow)
        self.assertIsNotNone(s._ema_trend)

    def test_adx_low_on_flat_data(self):
        s = _make_strategy(atr_period=5, adx_period=5, ema_fast=3, ema_slow=5, ema_trend=10)
        # Feed flat oscillating data — should NOT register as a trend
        bars = []
        for i in range(50):
            mid = 100.0 + (0.1 if i % 2 == 0 else -0.1)
            bars.append({
                "o": mid,
                "h": mid + 0.05,
                "l": mid - 0.05,
                "c": mid,
            })
        for b in bars:
            s.on_bar(_event(o=b["o"], h=b["h"], l=b["l"], c=b["c"]))

        self.assertIsNotNone(s._adx)
        # No directional movement → ADX should stay well below 25
        self.assertLess(s._adx, 25.0)


# ---------------------------------------------------------------------------
# End-to-end: regime + ADX + pullback → entry signal in a synthetic stream
# ---------------------------------------------------------------------------

class TestEndToEndEntry(unittest.TestCase):

    def test_entry_signal_emitted_on_synthetic_trend_pullback(self):
        """
        Build a deterministic up-trend that establishes the EMA stack and
        elevated ADX, then inject a pullback bar that touches EMA_fast and
        closes back above. The strategy should emit a LONG entry.
        """
        s = _make_strategy(
            ema_fast=10, ema_slow=20, ema_trend=30,
            adx_period=10, atr_period=10, adx_min=20.0,
            swing_lookback=3, touch_tol_atr=0.5,
            daily_supertrend_filter=False,  # tested separately
        )

        emitted = []

        # 80 bars of strong up-trend
        price = 100.0
        for i in range(80):
            price += 1.0
            ev = _event(
                o=price - 0.3,
                h=price + 0.3,
                l=price - 0.5,
                c=price,
                t=f"2024-01-{(i % 28) + 1:02d}",
            )
            sig = s.on_bar(ev)
            if sig is not None:
                emitted.append(sig)

        # By now: ema_fast > ema_slow > ema_trend (up-trend), ADX elevated
        self.assertIsNotNone(s._ema_fast)
        self.assertIsNotNone(s._ema_slow)
        self.assertIsNotNone(s._ema_trend)
        self.assertGreater(s._ema_fast, s._ema_slow)
        self.assertGreater(s._ema_slow, s._ema_trend)
        self.assertGreater(s._adx, 20.0)

        # Inject a pullback bar that grazes EMA_fast and closes back above
        ema_fast_val = s._ema_fast
        atr = s._atr
        # Low pierces just below ema_fast; close finishes above it
        pull_bar = _event(
            o=ema_fast_val + 0.1,
            h=ema_fast_val + 0.2,
            l=ema_fast_val - 0.05,
            c=ema_fast_val + 0.15,
            t="2024-02-01",
        )
        sig = s.on_bar(pull_bar)
        self.assertIsNotNone(sig, "Expected a LONG entry signal on the pullback bar")
        self.assertEqual(sig.direction, SignalDirection.LONG)
        self.assertIsNotNone(sig.stop_price)
        self.assertLess(sig.stop_price, pull_bar.close)


# ---------------------------------------------------------------------------
# 1d Supertrend regime filter
# ---------------------------------------------------------------------------

class TestSupertrendFilter(unittest.TestCase):
    """Direct tests of the Supertrend day-aggregator and the entry gate."""

    def _bar(self, day: int, hour: int, o=100.0, h=101.0, l=99.0, c=100.0):
        return _event(o=o, h=h, l=l, c=c, t=f"2024-01-{day:02d} {hour:02d}:00:00")

    def test_day_rollover_finalizes_daily_bar(self):
        s = _make_strategy(st_atr_period=2, st_multiplier=3.0)
        # Two 4h bars on day 1
        s._update_daily_supertrend(self._bar(day=1, hour=0, h=105.0, l=95.0, c=100.0))
        s._update_daily_supertrend(self._bar(day=1, hour=4, h=110.0, l=99.0, c=108.0))
        # Still day 1 — nothing finalized yet
        self.assertIsNone(s._st_prev_close)
        self.assertEqual(s._current_day_ohlc["high"], 110.0)
        self.assertEqual(s._current_day_ohlc["low"],  95.0)
        self.assertEqual(s._current_day_ohlc["close"], 108.0)

        # Day rolls over — day 1's bar should be finalized into Supertrend state
        s._update_daily_supertrend(self._bar(day=2, hour=0, c=109.0))
        self.assertEqual(s._st_prev_close, 108.0)  # day 1's close
        # Day 2 in-progress
        self.assertEqual(s._current_day_ohlc["close"], 109.0)

    def test_supertrend_goes_green_on_uptrend(self):
        s = _make_strategy(st_atr_period=3, st_multiplier=2.0)
        # Feed 10 daily bars in a clean up-trend; one bar per day
        for day in range(1, 11):
            close = 100.0 + day * 2.0
            s._update_daily_supertrend(self._bar(day=day, hour=0, h=close + 1, l=close - 1, c=close))
        # Force a final rollover so day 10 gets finalized too
        s._update_daily_supertrend(self._bar(day=11, hour=0, c=125.0))
        self.assertEqual(s._st_direction, 1, "Expected Supertrend green on strong up-trend")

    def test_supertrend_goes_red_on_downtrend(self):
        s = _make_strategy(st_atr_period=3, st_multiplier=2.0)
        for day in range(1, 11):
            close = 100.0 - day * 2.0
            s._update_daily_supertrend(self._bar(day=day, hour=0, h=close + 1, l=close - 1, c=close))
        s._update_daily_supertrend(self._bar(day=11, hour=0, c=75.0))
        self.assertEqual(s._st_direction, -1, "Expected Supertrend red on strong down-trend")

    def test_filter_blocks_entries_during_warmup(self):
        s = _make_strategy(daily_supertrend_filter=True)
        # Pre-seed every OTHER indicator so only the Supertrend gate could fail
        s._bars.append({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 1.0})
        s._bar_count = 1
        s._atr        = 1.0
        s._adx        = 30.0
        s._ema_fast   = 99.0
        s._ema_slow   = 95.0
        s._ema_trend  = 90.0
        # _st_direction is None (warm-up) → must block
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1
        sig = s._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNone(sig)

    def test_green_supertrend_allows_long_blocks_short(self):
        s = _make_strategy(direction="both", daily_supertrend_filter=True, swing_lookback=3)
        s._bars.append({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 1.0})
        s._bar_count = 1
        s._atr        = 1.0
        s._adx        = 30.0
        s._st_direction = 1  # green

        # Long-friendly stack
        s._ema_fast, s._ema_slow, s._ema_trend = 99.0, 95.0, 90.0
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1
        sig = s._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, SignalDirection.LONG)

        # Reset, now try a short-friendly stack with ST still green → should block
        s2 = _make_strategy(direction="both", daily_supertrend_filter=True, swing_lookback=3)
        s2._bars.append({"open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0, "volume": 1.0})
        s2._bar_count = 1
        s2._atr, s2._adx = 1.0, 30.0
        s2._st_direction = 1  # green — should NOT allow shorts
        s2._ema_fast, s2._ema_slow, s2._ema_trend = 101.0, 105.0, 110.0  # down-stack
        bar2 = {"open": 100.0, "high": 101.1, "low": 100.5, "close": 100.5, "volume": 1.0}
        s2._bars.append(bar2)
        s2._bar_count += 1
        sig2 = s2._check_entry(_event(o=100.0, h=101.1, l=100.5, c=100.5))
        self.assertIsNone(sig2)

    def test_red_supertrend_allows_short_blocks_long(self):
        s = _make_strategy(direction="both", daily_supertrend_filter=True, swing_lookback=3)
        s._bars.append({"open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0, "volume": 1.0})
        s._bar_count = 1
        s._atr        = 1.0
        s._adx        = 30.0
        s._st_direction = -1  # red

        # Short-friendly stack: close < ema_fast < ema_slow < ema_trend
        s._ema_fast, s._ema_slow, s._ema_trend = 101.0, 105.0, 110.0
        bar = {"open": 100.0, "high": 101.1, "low": 100.5, "close": 100.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1
        sig = s._check_entry(_event(o=100.0, h=101.1, l=100.5, c=100.5))
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, SignalDirection.SHORT)

        # Long-friendly stack with ST red → should block
        s2 = _make_strategy(direction="both", daily_supertrend_filter=True, swing_lookback=3)
        s2._bars.append({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 1.0})
        s2._bar_count = 1
        s2._atr, s2._adx = 1.0, 30.0
        s2._st_direction = -1
        s2._ema_fast, s2._ema_slow, s2._ema_trend = 99.0, 95.0, 90.0
        bar2 = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s2._bars.append(bar2)
        s2._bar_count += 1
        sig2 = s2._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNone(sig2)

    def test_filter_off_ignores_supertrend(self):
        # Same as warmup-block test but with filter disabled → entry fires.
        s = _make_strategy(daily_supertrend_filter=False, swing_lookback=3)
        s._bars.append({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 1.0})
        s._bar_count = 1
        s._atr, s._adx = 1.0, 30.0
        s._ema_fast, s._ema_slow, s._ema_trend = 99.0, 95.0, 90.0
        # _st_direction stays None — should NOT block
        bar = {"open": 100.0, "high": 100.5, "low": 98.9, "close": 99.5, "volume": 1.0}
        s._bars.append(bar)
        s._bar_count += 1
        sig = s._check_entry(_event(o=100.0, h=100.5, l=98.9, c=99.5))
        self.assertIsNotNone(sig)


class TestTP1FullCloseBug(unittest.TestCase):
    """
    Regression: when tp1_ratio=1.0 the TP1 signal closes 100% of the position
    in the portfolio. The strategy used to enter the "partial fill" branch
    in on_fill, leaving _in_position=True forever and blocking new entries.
    """

    def test_long_tp1_ratio_1_resets_position(self):
        s = _make_strategy(tp1_ratio=1.0)
        s._in_position        = True
        s._position_side      = "long"
        s._entry_price        = 100.0
        s._stop_price         = 95.0
        s._tp1_price          = 105.0
        s._tp2_price          = 110.0
        s._tp1_pending_fill   = True
        s._exit_pending       = True
        s._stop_distance      = 5.0
        s._entry_bar          = 10

        # Full-close TP1 fill arrives (because ratio=1.0)
        s.on_fill(_fill(OrderSide.SELL, price=105.0, qty=1.0))

        # Strategy must NOT think it's still holding a runner.
        self.assertFalse(s._in_position)
        self.assertIsNone(s._position_side)
        self.assertIsNone(s._stop_price)
        self.assertFalse(s._tp1_hit)
        self.assertFalse(s._tp1_pending_fill)
        self.assertFalse(s._exit_pending)
        self.assertIsNone(s._entry_bar)

    def test_short_tp1_ratio_1_resets_position(self):
        s = _make_strategy(direction="short", tp1_ratio=1.0)
        s._in_position        = True
        s._position_side      = "short"
        s._entry_price        = 100.0
        s._stop_price         = 105.0
        s._tp1_price          = 95.0
        s._tp2_price          = 90.0
        s._tp1_pending_fill   = True
        s._exit_pending       = True
        s._stop_distance      = 5.0
        s._entry_bar          = 10

        s.on_fill(_fill(OrderSide.BUY, price=95.0, qty=1.0))

        self.assertFalse(s._in_position)
        self.assertIsNone(s._position_side)
        self.assertIsNone(s._stop_price)

    def test_long_tp1_ratio_below_1_still_keeps_runner(self):
        # Sanity: the partial-then-runner path still works for tp1_ratio < 1.0.
        s = _make_strategy(tp1_ratio=0.5)
        s._in_position        = True
        s._position_side      = "long"
        s._entry_price        = 100.0
        s._stop_price         = 95.0
        s._tp1_price          = 105.0
        s._tp2_price          = 110.0
        s._tp1_pending_fill   = True
        s._exit_pending       = True
        s._stop_distance      = 5.0

        s.on_fill(_fill(OrderSide.SELL, price=105.0, qty=0.5))

        # Still in position, BE move applied.
        self.assertTrue(s._in_position)
        self.assertTrue(s._tp1_hit)
        self.assertEqual(s._stop_price, 100.0)


if __name__ == "__main__":
    unittest.main()
