"""
strategy/examples/fvg.py — Fair Value Gap (FVG) strategy.

Detection (3-bar imbalance pattern):
  Bullish FVG: high[i-2] < low[i]  → gap zone = [high[i-2], low[i]]
  Bearish FVG: low[i-2]  > high[i] → gap zone = [high[i],   low[i-2]]

Entry: price retraces into the gap and closes inside the zone (above gap_low for longs).
Stop:  gap_low - atr_stop_mult * ATR  (longs) / gap_high + atr_stop_mult * ATR (shorts).
TP:    fill_price + tp_atr_mult * ATR (longs) / fill_price - tp_atr_mult * ATR (shorts).
If a bar ranges through both stop and TP, the strategy exits at the stop first.
This is a conservative OHLC-only assumption because intrabar ordering is unknown.

Filters (all optional, on by default for longs-only equity trading):
  ema200_filter      — close must be above EMA200 for longs, below for shorts
  order_block_filter — bar[i-2] must be an opposing candle (ICT order block)
  min_gap_atr        — gap must be at least N × ATR wide

Note on fill timing: signals are emitted at bar i's close. The broker fills
at bar i's open (current architecture simplification). For daily bars this
introduces a small fill-price approximation. Stop/TP prices are computed from
the actual fill price in on_fill() to compensate.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from core.event import FillEvent, MarketEvent, OrderSide, SignalDirection, SignalEvent
from strategy.base import Strategy


# ── Hardcoded strategy constants ─────────────────────────────────────────────
# Change these to adjust behaviour. NOT constructor params. NOT optimizer grid.

MAX_HOLD_BARS: int = 50     # Force-exit after N bars if neither stop nor TP is hit.
                             # 4h bars: 50 ≈ 12 trading days. 1h bars: 50 ≈ 7 days.

TP1_ENABLED: bool = True    # Two-stage take profit. Set False to revert to single-TP.
TP1_RATIO: float  = 1.0     # TP1 distance = ratio × stop_distance. 1.0 = 1:1 RR.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _PendingGap:
    """A detected FVG zone waiting for a retracement entry."""
    side: str        # "long" | "short"
    gap_low: float
    gap_high: float
    stop_price: float
    atr: float       # ATR at detection time, used for TP after fill
    created_bar: int


class FVGStrategy(Strategy):
    """
    Fair Value Gap strategy for the event-driven backtester.

    Parameters
    ----------
    symbol : str
        Ticker to trade. Must match the data feed symbol.
    asset_class : str
        "stock" | "crypto" | "forex".
    direction : str
        "long" (default), "short", or "both".
    atr_period : int
        ATR lookback period in bars (default 14).
    atr_stop_mult : float
        Stop placed at gap_low - N×ATR for longs (default 0.75).
    tp_atr_mult : float
        Take profit at fill_price + N×ATR for longs (default 3.0).
    ema200_filter : bool
        Require close > EMA200 for longs, < EMA200 for shorts (default True).
    order_block_filter : bool
        Require bar[i-2] to be an opposing candle — bearish for bull gaps,
        bullish for bear gaps (default True).
    min_gap_atr : float
        Minimum gap width as a multiple of ATR. 0 = off (default 0.25).
    max_gap_age : int
        Bars before a pending gap is discarded (default 10).
    """

    def __init__(
        self,
        symbol: str,
        asset_class: str = "stock",
        direction: str = "long",
        atr_period: int = 14,
        atr_stop_mult: float = 0.75,
        tp_atr_mult: float = 3.0,
        ema200_filter: bool = True,
        order_block_filter: bool = True,
        min_gap_atr: float = 0.25,
        max_gap_age: int = 10,
    ):
        if direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got '{direction}'")

        self.symbol            = symbol
        self.asset_class       = asset_class
        self.direction         = direction
        self.atr_period        = atr_period
        self.atr_stop_mult     = atr_stop_mult
        self.tp_atr_mult       = tp_atr_mult
        self.ema200_filter     = ema200_filter
        self.order_block_filter = order_block_filter
        self.min_gap_atr       = min_gap_atr
        self.max_gap_age       = max_gap_age

        # Bar history — need enough for EMA200 + ATR + FVG lookback
        self._bars: deque = deque(maxlen=max(203, atr_period + 3))
        self._bar_count: int = 0

        # Pending gaps (detected but not yet entered)
        self._pending_gaps: List[_PendingGap] = []

        # Active position state
        self._in_position:    bool            = False
        self._position_side:  Optional[str]   = None  # "long" | "short"
        self._stop_price:     Optional[float] = None
        self._tp_price:       Optional[float] = None
        self._exit_pending:   bool            = False  # avoid double-exit signals

        # max_hold_bars: bar count at position entry
        self._entry_bar: Optional[int] = None

        # Two-stage TP state
        self._entry_price:     Optional[float] = None
        self._stop_distance:   Optional[float] = None
        self._tp1_price:       Optional[float] = None
        self._tp2_price:       Optional[float] = None
        self._tp1_hit:         bool            = False
        self._tp1_pending_fill: bool           = False  # set when TP1 signal emitted

        # Stash the gap's ATR/stop between signal and fill (fill price not known yet)
        self._pending_atr:    Optional[float] = None
        self._pending_stop:   Optional[float] = None
        self._pending_side:   Optional[str]   = None

        # EMA200 — incremental update
        self._ema200:         Optional[float] = None
        self._ema200_alpha:   float           = 2.0 / 201.0

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        if event.symbol != self.symbol:
            return None

        self._bars.append({
            "open":  event.open,
            "high":  event.high,
            "low":   event.low,
            "close": event.close,
            "volume": event.volume,
        })
        self._bar_count += 1
        self._update_ema200(event.close)

        # Need enough bars for ATR (+ 1 for prev close) and FVG detection (3)
        if len(self._bars) < max(self.atr_period + 2, 3):
            return None

        atr = self._compute_atr()
        if atr is None or atr <= 0:
            return None

        # --- In position: check stop and TP ---
        if self._in_position and not self._exit_pending:
            signal = self._check_exit(event)
            if signal:
                self._exit_pending = True
                return signal

        # --- Not in position: detect new gaps, check pending entries ---
        if not self._in_position and not self._exit_pending:
            self._detect_fvg(atr)
            self._expire_and_invalidate_gaps(event)
            return self._check_entry(event, atr)

        return None

    def on_fill(self, fill: FillEvent) -> None:
        if fill.symbol != self.symbol:
            return

        if fill.side == OrderSide.BUY:
            if self._in_position and self._position_side == "short" and self._tp1_pending_fill:
                # TP1 partial cover of short — stay in position with half size
                self._tp1_hit          = True
                self._tp1_pending_fill = False
                self._exit_pending     = False
                if self._entry_price is not None:
                    self._stop_price = self._entry_price  # move stop to breakeven
            elif self._in_position and self._position_side == "short":
                # Full exit of short (stop, tp2, or timeout)
                self._reset_position()
            else:
                # Opening a long
                self._in_position   = True
                self._position_side = "long"
                self._exit_pending  = False
                self._entry_bar     = self._bar_count
                if self._pending_atr is not None:
                    self._entry_price   = fill.fill_price
                    self._stop_distance = abs(fill.fill_price - self._pending_stop)
                    full_tp = fill.fill_price + self.tp_atr_mult * self._pending_atr
                    if TP1_ENABLED and self._stop_distance > 0:
                        self._tp1_price = fill.fill_price + TP1_RATIO * self._stop_distance
                        self._tp2_price = full_tp
                    else:
                        self._tp1_price = None
                        self._tp2_price = None
                    self._tp_price  = full_tp  # keep in sync for None-guard
                    self._stop_price = self._pending_stop
                    self._tp1_hit   = False
                    self._tp1_pending_fill = False
                    self._pending_atr  = None
                    self._pending_stop = None
                    self._pending_side = None

        elif fill.side == OrderSide.SELL:
            if self._in_position and self._position_side == "long":
                if self._tp1_pending_fill:
                    # TP1 partial close — stay in position with half size
                    self._tp1_hit          = True
                    self._tp1_pending_fill = False
                    self._exit_pending     = False
                    if self._entry_price is not None:
                        self._stop_price = self._entry_price  # move stop to breakeven
                else:
                    # Full exit (stop, tp2, or timeout)
                    self._reset_position()
            elif not self._in_position and self._pending_side == "short":
                # Opening a short
                self._in_position   = True
                self._position_side = "short"
                self._exit_pending  = False
                self._entry_bar     = self._bar_count
                if self._pending_atr is not None:
                    self._entry_price   = fill.fill_price
                    self._stop_distance = abs(fill.fill_price - self._pending_stop)
                    full_tp = fill.fill_price - self.tp_atr_mult * self._pending_atr
                    if TP1_ENABLED and self._stop_distance > 0:
                        self._tp1_price = fill.fill_price - TP1_RATIO * self._stop_distance
                        self._tp2_price = full_tp
                    else:
                        self._tp1_price = None
                        self._tp2_price = None
                    self._tp_price  = full_tp  # keep in sync for None-guard
                    self._stop_price = self._pending_stop
                    self._tp1_hit   = False
                    self._tp1_pending_fill = False
                    self._pending_atr  = None
                    self._pending_stop = None
                    self._pending_side = None
            else:
                # Closing a short (full exit)
                self._reset_position()

    # ------------------------------------------------------------------
    # FVG detection
    # ------------------------------------------------------------------

    def _detect_fvg(self, atr: float) -> None:
        """Check if the last 3 bars form a valid FVG and queue it."""
        bars = list(self._bars)
        if len(bars) < 3:
            return

        b0 = bars[-3]   # bar i-2: the order block / pre-impulse bar
        # b1 = bars[-2]  # bar i-1: the impulse bar (strong directional move)
        b2 = bars[-1]   # bar i:   the current bar

        # --- Bullish FVG ---
        if self.direction in ("long", "both") and b0["high"] < b2["low"]:
            gap_low  = b0["high"]
            gap_high = b2["low"]

            # Order block: b0 must be bearish (close < open)
            if self.order_block_filter and b0["close"] >= b0["open"]:
                pass
            # Min gap size
            elif self.min_gap_atr > 0 and (gap_high - gap_low) < self.min_gap_atr * atr:
                pass
            else:
                self._pending_gaps.append(_PendingGap(
                    side="long",
                    gap_low=gap_low,
                    gap_high=gap_high,
                    stop_price=gap_low - self.atr_stop_mult * atr,
                    atr=atr,
                    created_bar=self._bar_count,
                ))

        # --- Bearish FVG ---
        if self.direction in ("short", "both") and b0["low"] > b2["high"]:
            gap_low  = b2["high"]
            gap_high = b0["low"]

            # Order block: b0 must be bullish (close > open)
            if self.order_block_filter and b0["close"] <= b0["open"]:
                pass
            elif self.min_gap_atr > 0 and (gap_high - gap_low) < self.min_gap_atr * atr:
                pass
            else:
                self._pending_gaps.append(_PendingGap(
                    side="short",
                    gap_low=gap_low,
                    gap_high=gap_high,
                    stop_price=gap_high + self.atr_stop_mult * atr,
                    atr=atr,
                    created_bar=self._bar_count,
                ))

    def _expire_and_invalidate_gaps(self, event: MarketEvent) -> None:
        """Remove gaps that aged out or were invalidated by price action."""
        valid = []
        for gap in self._pending_gaps:
            # Age out
            if (self._bar_count - gap.created_bar) > self.max_gap_age:
                continue
            # Invalidation: price closes through the gap entirely
            if gap.side == "long" and event.close < gap.gap_low:
                continue
            if gap.side == "short" and event.close > gap.gap_high:
                continue
            valid.append(gap)
        self._pending_gaps = valid

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _check_entry(self, event: MarketEvent, atr: float) -> Optional[SignalEvent]:
        """Return a signal if price has retraced into a valid pending gap."""
        for gap in list(self._pending_gaps):
            if gap.side == "long":
                # Price must touch or enter the gap zone from above
                # and close inside (confirming the gap is still valid)
                in_zone = event.low <= gap.gap_high and event.close >= gap.gap_low
                if not in_zone:
                    continue

                # EMA200 filter
                if self.ema200_filter and self._ema200 is not None:
                    if event.close <= self._ema200:
                        continue

                self._pending_atr  = gap.atr
                self._pending_stop = gap.stop_price
                self._pending_side = "long"
                self._pending_gaps.remove(gap)

                return SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=event.timestamp,
                    direction=SignalDirection.LONG,
                    strategy_id="fvg",
                    stop_price=gap.stop_price,
                    # tp_price unknown until fill price is known — set in on_fill
                )

            elif gap.side == "short":
                in_zone = event.high >= gap.gap_low and event.close <= gap.gap_high
                if not in_zone:
                    continue

                if self.ema200_filter and self._ema200 is not None:
                    if event.close >= self._ema200:
                        continue

                self._pending_atr  = gap.atr
                self._pending_stop = gap.stop_price
                self._pending_side = "short"
                self._pending_gaps.remove(gap)

                return SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=event.timestamp,
                    direction=SignalDirection.SHORT,
                    strategy_id="fvg",
                    stop_price=gap.stop_price,
                    # tp_price unknown until fill price is known — set in on_fill
                )

        return None

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit(self, event: MarketEvent) -> Optional[SignalEvent]:
        """Return an EXIT signal if stop, TP, or max hold bars was hit this bar."""
        if self._stop_price is None:
            return None

        # 1. Timeout — force-exit after MAX_HOLD_BARS bars
        if self._entry_bar is not None:
            if (self._bar_count - self._entry_bar) >= MAX_HOLD_BARS:
                return SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=event.timestamp,
                    direction=SignalDirection.EXIT,
                    strategy_id="fvg",
                    exit_reason="timeout",
                )

        # 2. Stop / TP1 / TP2. Stop is checked first as a conservative
        # OHLC-only rule when a bar ranges through both stop and target.
        exit_reason: str   = ""
        strength:    float = 1.0
        sig_stop:    Optional[float] = None
        sig_tp:      Optional[float] = None

        if self._position_side == "long":
            if event.low <= self._stop_price:
                exit_reason = "stop"
                sig_stop    = self._stop_price
            elif TP1_ENABLED and not self._tp1_hit and self._tp1_price is not None:
                if event.high >= self._tp1_price:
                    exit_reason = "tp1"
                    strength    = 0.5
                    sig_tp      = self._tp1_price
            elif self._tp2_price is not None and event.high >= self._tp2_price:
                exit_reason = "tp"
                sig_tp      = self._tp2_price
            elif not TP1_ENABLED and self._tp_price is not None and event.high >= self._tp_price:
                exit_reason = "tp"
                sig_tp      = self._tp_price

        elif self._position_side == "short":
            if event.high >= self._stop_price:
                exit_reason = "stop"
                sig_stop    = self._stop_price
            elif TP1_ENABLED and not self._tp1_hit and self._tp1_price is not None:
                if event.low <= self._tp1_price:
                    exit_reason = "tp1"
                    strength    = 0.5
                    sig_tp      = self._tp1_price
            elif self._tp2_price is not None and event.low <= self._tp2_price:
                exit_reason = "tp"
                sig_tp      = self._tp2_price
            elif not TP1_ENABLED and self._tp_price is not None and event.low <= self._tp_price:
                exit_reason = "tp"
                sig_tp      = self._tp_price

        if not exit_reason:
            return None

        if exit_reason == "tp1":
            self._tp1_pending_fill = True  # on_fill will detect partial close

        return SignalEvent(
            symbol=self.symbol,
            asset_class=self.asset_class,
            timestamp=event.timestamp,
            direction=SignalDirection.EXIT,
            strategy_id="fvg",
            strength=strength,
            exit_reason=exit_reason,
            stop_price=sig_stop,
            tp_price=sig_tp,
        )

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _update_ema200(self, close: float) -> None:
        """Incrementally update the 200-bar EMA."""
        if self._ema200 is None:
            # Bootstrap: initialize as SMA of first 200 closes
            if len(self._bars) >= 200:
                closes = [b["close"] for b in list(self._bars)[-200:]]
                self._ema200 = sum(closes) / 200.0
        else:
            self._ema200 = self._ema200_alpha * close + (1 - self._ema200_alpha) * self._ema200

    def _compute_atr(self) -> Optional[float]:
        """Simple ATR: mean true range over the last atr_period bars."""
        bars = list(self._bars)
        if len(bars) < self.atr_period + 1:
            return None
        true_ranges = []
        for i in range(1, self.atr_period + 1):
            b    = bars[-i]
            prev = bars[-(i + 1)]
            tr = max(
                b["high"] - b["low"],
                abs(b["high"] - prev["close"]),
                abs(b["low"]  - prev["close"]),
            )
            true_ranges.append(tr)
        return sum(true_ranges) / len(true_ranges)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_position(self) -> None:
        self._in_position      = False
        self._position_side    = None
        self._stop_price       = None
        self._tp_price         = None
        self._exit_pending     = False
        # Clear any pending gaps when position closes so we start fresh
        self._pending_gaps.clear()
        # max_hold_bars
        self._entry_bar        = None
        # two-stage TP
        self._entry_price      = None
        self._stop_distance    = None
        self._tp1_price        = None
        self._tp2_price        = None
        self._tp1_hit          = False
        self._tp1_pending_fill = False
