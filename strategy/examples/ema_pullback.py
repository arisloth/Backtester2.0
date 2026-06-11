"""
strategy/examples/ema_pullback.py — EMA Pullback Strategy.

Idea: in a confirmed trend, price pulls back to an EMA support and you enter
*there* — buying the dip into the trend, not chasing the extension. The EMA
gives a specific level to bid into with a logical stop, which fixes the
"entered mid-range / at resistance" problem.

Entry pipeline (longs; mirror for shorts):
  1. Regime gate    : close > EMA_fast > EMA_slow > EMA_trend
  2. Trend strength : ADX > adx_min  (Wilder ADX, default period 14)
  3. Pullback       : bar.low <= chosen_ema + touch_tol_atr * ATR
                      AND bar.close > chosen_ema  (touched the EMA but held)

Stop placement (longs):
  stop = min(swing_low - 0.1*ATR, close - atr_stop_mult*ATR)
  (the safer / further-below candidate)

Exits (in priority order, mirrors fvg.py):
  1. max_hold_bars timeout
  2. Stop (conservative: stop wins if same bar ranges through both stop and TP)
  3. TP1 partial exit at entry + tp1_r * stop_distance; stop moves to BE
  4. Runner exit, mode-dependent:
       structure  : TP2 = recent overhead swing high (computed at entry)
       atr_trail  : trail stop = trail_anchor - atr_trail_mult * ATR
                    (trail_anchor = highest high since entry for longs)
       fixed_r    : TP2 = entry + runner_fixed_r * stop_distance

Note on fill timing: signals emit at bar i's close; broker fills at bar i's
open (current architecture simplification). Stop/TP prices are computed from
the actual fill price in on_fill() to compensate.

Scope decisions for v1 (see CLAUDE.md plan): no BTC regime gate, no OBV,
no confirmation-candle filter, no orderbook. Coin's own EMA stack + ADX
are the only regime filters.
"""

from collections import deque
from typing import List, Optional

import pandas as pd

from core.event import FillEvent, MarketEvent, OrderSide, SignalDirection, SignalEvent
from strategy.base import Strategy


class EMAPullbackStrategy(Strategy):
    """
    Trend-pullback strategy on EMA support.

    Parameters
    ----------
    symbol : str
    asset_class : str
        "stock" | "crypto" | "forex".
    direction : str
        "long" (default), "short", or "both".
    ema_fast, ema_slow, ema_trend : int
        EMA stack periods. Defaults 20 / 50 / 200.
    pullback_ema : str
        Which EMA price must touch: "ema_fast" (default) or "ema_slow".
    touch_tol_atr : float
        How close to the EMA the bar must come, in ATR units. 0 = exact touch.
    adx_period : int
        Wilder ADX period (default 14).
    adx_min : float
        Minimum ADX for a real trend (default 25).
    atr_period : int
        ATR period (default 14). Wilder smoothing (RMA) like the canonical ATR.
    atr_stop_mult : float
        Stop = close - N * ATR for longs (mirror for shorts). Default 1.5.
    swing_lookback : int
        Bars to scan for the pullback swing low (default 5).
    tp1_r : float
        TP1 distance in R multiples of the initial stop (default 2.0).
    tp1_ratio : float
        Fraction of position closed at TP1 (default 0.5).
    runner_mode : str
        How TP2 / runner exit is handled:
          "structure" : TP2 at the recent overhead swing high (default).
          "atr_trail" : trail stop on the runner.
          "fixed_r"   : TP2 at entry + runner_fixed_r * stop_distance.
    runner_fixed_r : float
        TP2 distance in R for `fixed_r` mode (default 4.0).
    atr_trail_mult : float
        Trail distance in ATR units for `atr_trail` mode (default 2.5).
    structure_lookback : int
        Bars to scan for overhead swing high in `structure` mode (default 50).
    max_hold_bars : int
        Force-exit after N bars (default 100).
    daily_supertrend_filter : bool
        If True, only take longs when the 1d Supertrend is green (uptrend) and
        only take shorts when it's red (downtrend). Daily bars are aggregated
        from the incoming intraday bars on UTC day-rollover, so there's no
        lookahead — the filter uses the most recently CLOSED daily bar.
        Default True.
    st_atr_period : int
        Supertrend ATR period on the daily bars (default 10).
    st_multiplier : float
        Supertrend ATR multiplier (default 3.0).
    """

    def __init__(
        self,
        symbol: str,
        asset_class: str = "crypto",
        direction: str = "long",
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_trend: int = 200,
        pullback_ema: str = "ema_fast",
        touch_tol_atr: float = 0.1,
        adx_period: int = 14,
        adx_min: float = 25.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        swing_lookback: int = 5,
        tp1_r: float = 2.0,
        tp1_ratio: float = 0.5,
        runner_mode: str = "structure",
        runner_fixed_r: float = 4.0,
        atr_trail_mult: float = 2.5,
        structure_lookback: int = 50,
        max_hold_bars: int = 100,
        daily_supertrend_filter: bool = True,
        st_atr_period: int = 10,
        st_multiplier: float = 3.0,
    ):
        if direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got '{direction}'")
        if pullback_ema not in ("ema_fast", "ema_slow"):
            raise ValueError(f"pullback_ema must be 'ema_fast' or 'ema_slow', got '{pullback_ema}'")
        if runner_mode not in ("structure", "atr_trail", "fixed_r"):
            raise ValueError(f"runner_mode must be 'structure'|'atr_trail'|'fixed_r', got '{runner_mode}'")
        if not (0.0 < tp1_ratio <= 1.0):
            raise ValueError(f"tp1_ratio must be in (0, 1], got {tp1_ratio}")

        self.symbol             = symbol
        self.asset_class        = asset_class
        self.direction          = direction
        self.ema_fast_n         = ema_fast
        self.ema_slow_n         = ema_slow
        self.ema_trend_n        = ema_trend
        self.pullback_ema_name  = pullback_ema
        self.touch_tol_atr      = touch_tol_atr
        self.adx_period         = adx_period
        self.adx_min            = adx_min
        self.atr_period         = atr_period
        self.atr_stop_mult      = atr_stop_mult
        self.swing_lookback     = swing_lookback
        self.tp1_r              = tp1_r
        self.tp1_ratio          = tp1_ratio
        self.runner_mode        = runner_mode
        self.runner_fixed_r     = runner_fixed_r
        self.atr_trail_mult     = atr_trail_mult
        self.structure_lookback = structure_lookback
        self.max_hold_bars      = max_hold_bars
        self.daily_supertrend_filter = daily_supertrend_filter
        self.st_atr_period      = st_atr_period
        self.st_multiplier      = st_multiplier

        # Bar history — needs to cover EMA_trend warm-up, ADX warm-up (~2*N),
        # structure_lookback, and swing_lookback.
        history_len = max(ema_trend + 5, adx_period * 3, structure_lookback + 5, swing_lookback + 5)
        self._bars: deque = deque(maxlen=history_len)
        self._bar_count: int = 0

        # EMA state (incremental — same pattern as fvg.py:476-484)
        self._ema_fast:  Optional[float] = None
        self._ema_slow:  Optional[float] = None
        self._ema_trend: Optional[float] = None
        self._alpha_fast  = 2.0 / (ema_fast + 1.0)
        self._alpha_slow  = 2.0 / (ema_slow + 1.0)
        self._alpha_trend = 2.0 / (ema_trend + 1.0)

        # ATR state (Wilder/RMA smoothing)
        self._atr: Optional[float] = None
        self._atr_seed_sum: float = 0.0
        self._atr_seed_count: int = 0

        # ADX state (Wilder smoothing for +DM, -DM, TR; then DX → ADX)
        self._adx:           Optional[float] = None
        self._smooth_pdm:    Optional[float] = None
        self._smooth_ndm:    Optional[float] = None
        self._smooth_tr_adx: Optional[float] = None
        self._adx_dx_buffer: List[float] = []  # collects DX values before ADX is first seeded
        self._adx_seed_pdm_sum: float = 0.0
        self._adx_seed_ndm_sum: float = 0.0
        self._adx_seed_tr_sum:  float = 0.0
        self._adx_seed_count:   int   = 0

        # Active position state
        self._in_position:    bool            = False
        self._position_side:  Optional[str]   = None  # "long" | "short"
        self._stop_price:     Optional[float] = None
        self._tp1_price:      Optional[float] = None
        self._tp2_price:      Optional[float] = None
        self._exit_pending:   bool            = False  # avoid double-exit signals
        self._entry_bar:      Optional[int]   = None

        # Two-stage TP state
        self._entry_price:     Optional[float] = None
        self._stop_distance:   Optional[float] = None
        self._tp1_hit:         bool            = False
        self._tp1_pending_fill: bool           = False

        # Trailing-stop state (atr_trail mode only)
        self._trail_anchor: Optional[float] = None

        # Stash between signal and fill (fill price not known when signal fires)
        self._pending_stop: Optional[float] = None
        self._pending_side: Optional[str]   = None
        self._pending_atr:  Optional[float] = None

        # --- Daily Supertrend (regime filter) ---
        # We aggregate intraday bars into 1d OHLC on UTC day-rollover and
        # update Supertrend on each newly CLOSED daily bar. The filter then
        # uses the latest direction value — which always reflects the most
        # recently closed daily bar, never the in-progress one.
        self._current_day: Optional[pd.Timestamp] = None
        self._current_day_ohlc: Optional[dict]    = None  # {open, high, low, close}

        # Supertrend state (carried across closed daily bars)
        self._st_direction:    Optional[int]   = None  # +1 = green/up, -1 = red/down
        self._st_atr:          Optional[float] = None
        self._st_atr_seed_sum: float           = 0.0
        self._st_atr_seed_cnt: int             = 0
        self._st_prev_close:   Optional[float] = None
        self._st_prev_upper:   Optional[float] = None  # prior bar's final upper band
        self._st_prev_lower:   Optional[float] = None  # prior bar's final lower band

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        if event.symbol != self.symbol:
            return None

        # Aggregate daily bars + update Supertrend on day-rollover.
        # Done FIRST so the regime filter reflects the most recent CLOSED day.
        self._update_daily_supertrend(event)

        # Update indicators BEFORE storing bar, because ATR/ADX need previous bar's close.
        self._update_atr_and_adx(event)
        self._update_emas(event.close)

        self._bars.append({
            "open":  event.open,
            "high":  event.high,
            "low":   event.low,
            "close": event.close,
            "volume": event.volume,
        })
        self._bar_count += 1

        # In position → check exits first
        if self._in_position and not self._exit_pending:
            sig = self._check_exit(event)
            if sig is not None:
                self._exit_pending = True
                return sig

        # Update trailing anchor (atr_trail mode) every bar after TP1 hit
        if self._in_position and self._tp1_hit and self.runner_mode == "atr_trail":
            if self._position_side == "long":
                if self._trail_anchor is None or event.high > self._trail_anchor:
                    self._trail_anchor = event.high
            else:  # short
                if self._trail_anchor is None or event.low < self._trail_anchor:
                    self._trail_anchor = event.low

        # Not in position → check entry
        if not self._in_position and not self._exit_pending:
            return self._check_entry(event)

        return None

    def on_fill(self, fill: FillEvent) -> None:
        if fill.symbol != self.symbol:
            return

        if fill.side == OrderSide.BUY:
            if self._in_position and self._position_side == "short" and self._tp1_pending_fill:
                # TP1 fill on a short. If tp1_ratio >= 1.0 the portfolio just
                # covered the FULL position — treat as a full exit, not a partial.
                # Otherwise it's a true partial: mark TP1 hit, move stop to BE,
                # and keep the runner alive.
                if self.tp1_ratio >= 1.0:
                    self._reset_position()
                else:
                    self._tp1_hit          = True
                    self._tp1_pending_fill = False
                    self._exit_pending     = False
                    if self._entry_price is not None:
                        self._stop_price = self._entry_price  # BE
                    if self.runner_mode == "atr_trail":
                        self._trail_anchor = fill.fill_price  # init at fill
            elif self._in_position and self._position_side == "short":
                # Full cover of short (stop / tp2 / trail / timeout)
                self._reset_position()
            else:
                # Opening a long
                self._open_position("long", fill.fill_price)

        elif fill.side == OrderSide.SELL:
            if self._in_position and self._position_side == "long" and self._tp1_pending_fill:
                # TP1 fill on a long. If tp1_ratio >= 1.0 the portfolio just
                # sold the FULL position — treat as a full exit, not a partial.
                if self.tp1_ratio >= 1.0:
                    self._reset_position()
                else:
                    self._tp1_hit          = True
                    self._tp1_pending_fill = False
                    self._exit_pending     = False
                    if self._entry_price is not None:
                        self._stop_price = self._entry_price  # BE
                    if self.runner_mode == "atr_trail":
                        self._trail_anchor = fill.fill_price  # init at fill
            elif self._in_position and self._position_side == "long":
                # Full exit of long (stop / tp2 / trail / timeout)
                self._reset_position()
            elif not self._in_position and self._pending_side == "short":
                # Opening a short
                self._open_position("short", fill.fill_price)
            else:
                # Closing-leg of short — handled in BUY branch; this is defensive.
                self._reset_position()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _check_entry(self, event: MarketEvent) -> Optional[SignalEvent]:
        # Don't re-emit while a prior entry signal is awaiting its fill.
        if self._pending_side is not None:
            return None
        if self._atr is None or self._atr <= 0:
            return None
        if self._adx is None or self._adx < self.adx_min:
            return None
        if self._ema_fast is None or self._ema_slow is None or self._ema_trend is None:
            return None

        # 1d Supertrend regime filter — strict: block during warm-up too.
        if self.daily_supertrend_filter and self._st_direction is None:
            return None

        chosen_ema = self._ema_fast if self.pullback_ema_name == "ema_fast" else self._ema_slow
        tol = self.touch_tol_atr * self._atr

        # --- LONG ---
        if self.direction in ("long", "both"):
            # 1d Supertrend gate: only longs when daily ST is green (+1)
            if self.daily_supertrend_filter and self._st_direction != 1:
                pass  # fall through, try short branch
            # Regime gate: close > ema_fast > ema_slow > ema_trend
            elif (event.close > self._ema_fast > self._ema_slow > self._ema_trend):
                # Pullback gate: low pierced (or grazed) the EMA, close held above it
                if event.low <= chosen_ema + tol and event.close > chosen_ema:
                    stop = self._compute_long_stop(event.close)
                    if stop is not None and stop < event.close:
                        self._pending_stop = stop
                        self._pending_side = "long"
                        self._pending_atr  = self._atr
                        return SignalEvent(
                            symbol=self.symbol,
                            asset_class=self.asset_class,
                            timestamp=event.timestamp,
                            direction=SignalDirection.LONG,
                            strategy_id="ema_pullback",
                            stop_price=stop,
                            # tp_price set in on_fill once fill price is known
                        )

        # --- SHORT ---
        if self.direction in ("short", "both"):
            # 1d Supertrend gate: only shorts when daily ST is red (-1)
            if self.daily_supertrend_filter and self._st_direction != -1:
                return None
            # Regime gate: close < ema_fast < ema_slow < ema_trend
            if (event.close < self._ema_fast < self._ema_slow < self._ema_trend):
                # Pullback gate: high pierced the EMA from below, close held below it
                if event.high >= chosen_ema - tol and event.close < chosen_ema:
                    stop = self._compute_short_stop(event.close)
                    if stop is not None and stop > event.close:
                        self._pending_stop = stop
                        self._pending_side = "short"
                        self._pending_atr  = self._atr
                        return SignalEvent(
                            symbol=self.symbol,
                            asset_class=self.asset_class,
                            timestamp=event.timestamp,
                            direction=SignalDirection.SHORT,
                            strategy_id="ema_pullback",
                            stop_price=stop,
                        )

        return None

    def _compute_long_stop(self, close: float) -> Optional[float]:
        """Safer of: swing_low - small ATR buffer, OR close - atr_stop_mult * ATR."""
        if self._atr is None or self._atr <= 0:
            return None
        # The current bar is already in self._bars
        recent = list(self._bars)[-self.swing_lookback:]
        if not recent:
            return None
        swing_low = min(b["low"] for b in recent)
        cand1 = swing_low - 0.1 * self._atr
        cand2 = close - self.atr_stop_mult * self._atr
        return min(cand1, cand2)  # lower = further below entry = safer

    def _compute_short_stop(self, close: float) -> Optional[float]:
        if self._atr is None or self._atr <= 0:
            return None
        recent = list(self._bars)[-self.swing_lookback:]
        if not recent:
            return None
        swing_high = max(b["high"] for b in recent)
        cand1 = swing_high + 0.1 * self._atr
        cand2 = close + self.atr_stop_mult * self._atr
        return max(cand1, cand2)  # higher = further above entry = safer

    def _open_position(self, side: str, fill_price: float) -> None:
        """Initialize position state and TP2 (after fill price is known)."""
        if self._pending_stop is None or self._pending_atr is None:
            # Defensive — shouldn't happen, but don't crash if state is inconsistent
            return

        self._in_position     = True
        self._position_side   = side
        self._exit_pending    = False
        self._entry_bar       = self._bar_count
        self._entry_price     = fill_price
        self._stop_price      = self._pending_stop
        self._stop_distance   = abs(fill_price - self._pending_stop)
        self._tp1_hit         = False
        self._tp1_pending_fill = False
        self._trail_anchor    = None

        if self._stop_distance <= 0:
            # Degenerate; treat as no TPs and let max_hold_bars exit it
            self._tp1_price = None
            self._tp2_price = None
        else:
            if side == "long":
                self._tp1_price = fill_price + self.tp1_r * self._stop_distance
                self._tp2_price = self._compute_long_tp2(fill_price)
            else:  # short
                self._tp1_price = fill_price - self.tp1_r * self._stop_distance
                self._tp2_price = self._compute_short_tp2(fill_price)

        self._pending_stop = None
        self._pending_side = None
        self._pending_atr  = None

    def _compute_long_tp2(self, fill_price: float) -> Optional[float]:
        if self._stop_distance is None or self._stop_distance <= 0:
            return None
        if self.runner_mode == "fixed_r":
            return fill_price + self.runner_fixed_r * self._stop_distance
        if self.runner_mode == "atr_trail":
            return None  # exit via trailing stop, not a fixed price
        # structure: highest high over last structure_lookback bars (excluding current)
        # that sits above fill_price; fall back to fixed_r if none.
        bars = list(self._bars)
        # Exclude the current (entry) bar
        scan = bars[-(self.structure_lookback + 1):-1] if len(bars) > 1 else []
        overhead = [b["high"] for b in scan if b["high"] > fill_price]
        if overhead:
            return max(overhead)
        return fill_price + self.runner_fixed_r * self._stop_distance

    def _compute_short_tp2(self, fill_price: float) -> Optional[float]:
        if self._stop_distance is None or self._stop_distance <= 0:
            return None
        if self.runner_mode == "fixed_r":
            return fill_price - self.runner_fixed_r * self._stop_distance
        if self.runner_mode == "atr_trail":
            return None
        bars = list(self._bars)
        scan = bars[-(self.structure_lookback + 1):-1] if len(bars) > 1 else []
        underhead = [b["low"] for b in scan if b["low"] < fill_price]
        if underhead:
            return min(underhead)
        return fill_price - self.runner_fixed_r * self._stop_distance

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _check_exit(self, event: MarketEvent) -> Optional[SignalEvent]:
        if self._stop_price is None:
            return None

        # 1. Timeout
        if self._entry_bar is not None:
            if (self._bar_count - self._entry_bar) >= self.max_hold_bars:
                return SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=event.timestamp,
                    direction=SignalDirection.EXIT,
                    strategy_id="ema_pullback",
                    exit_reason="timeout",
                )

        # 2. Stop / TP1 / Runner. Stop wins if both hit same bar (conservative).
        exit_reason: str = ""
        strength: float = 1.0
        sig_stop: Optional[float] = None
        sig_tp:   Optional[float] = None

        if self._position_side == "long":
            if event.low <= self._stop_price:
                exit_reason = "stop"
                sig_stop    = self._stop_price
            elif (not self._tp1_hit) and self._tp1_price is not None and event.high >= self._tp1_price:
                exit_reason = "tp1"
                strength    = self.tp1_ratio
                sig_tp      = self._tp1_price
            elif self._tp1_hit:
                # Runner exit
                if self.runner_mode == "atr_trail":
                    trail = self._current_long_trail()
                    if trail is not None and event.low <= trail:
                        exit_reason = "trail"
                        sig_stop    = trail
                elif self._tp2_price is not None and event.high >= self._tp2_price:
                    exit_reason = "tp2"
                    sig_tp      = self._tp2_price

        elif self._position_side == "short":
            if event.high >= self._stop_price:
                exit_reason = "stop"
                sig_stop    = self._stop_price
            elif (not self._tp1_hit) and self._tp1_price is not None and event.low <= self._tp1_price:
                exit_reason = "tp1"
                strength    = self.tp1_ratio
                sig_tp      = self._tp1_price
            elif self._tp1_hit:
                if self.runner_mode == "atr_trail":
                    trail = self._current_short_trail()
                    if trail is not None and event.high >= trail:
                        exit_reason = "trail"
                        sig_stop    = trail
                elif self._tp2_price is not None and event.low <= self._tp2_price:
                    exit_reason = "tp2"
                    sig_tp      = self._tp2_price

        if not exit_reason:
            return None

        if exit_reason == "tp1":
            self._tp1_pending_fill = True  # on_fill detects partial close

        return SignalEvent(
            symbol=self.symbol,
            asset_class=self.asset_class,
            timestamp=event.timestamp,
            direction=SignalDirection.EXIT,
            strategy_id="ema_pullback",
            strength=strength,
            exit_reason=exit_reason,
            stop_price=sig_stop,
            tp_price=sig_tp,
        )

    def _current_long_trail(self) -> Optional[float]:
        if self._trail_anchor is None or self._atr is None:
            return None
        return self._trail_anchor - self.atr_trail_mult * self._atr

    def _current_short_trail(self) -> Optional[float]:
        if self._trail_anchor is None or self._atr is None:
            return None
        return self._trail_anchor + self.atr_trail_mult * self._atr

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _update_emas(self, close: float) -> None:
        """Seed each EMA as SMA over its first N closes, then update incrementally."""
        # We update AFTER computing other indicators but BEFORE appending the new bar.
        # The new bar will be appended right after this in on_bar, so for SMA seeding
        # we look at self._bars + the incoming close.
        # Use future bar_count = self._bar_count + 1 (this bar is about to be added).
        n_so_far = self._bar_count  # bars already stored, before append
        closes_after = [b["close"] for b in self._bars] + [close]

        # ema_fast
        if self._ema_fast is None:
            if len(closes_after) >= self.ema_fast_n:
                self._ema_fast = sum(closes_after[-self.ema_fast_n:]) / self.ema_fast_n
        else:
            self._ema_fast = self._alpha_fast * close + (1 - self._alpha_fast) * self._ema_fast

        # ema_slow
        if self._ema_slow is None:
            if len(closes_after) >= self.ema_slow_n:
                self._ema_slow = sum(closes_after[-self.ema_slow_n:]) / self.ema_slow_n
        else:
            self._ema_slow = self._alpha_slow * close + (1 - self._alpha_slow) * self._ema_slow

        # ema_trend
        if self._ema_trend is None:
            if len(closes_after) >= self.ema_trend_n:
                self._ema_trend = sum(closes_after[-self.ema_trend_n:]) / self.ema_trend_n
        else:
            self._ema_trend = self._alpha_trend * close + (1 - self._alpha_trend) * self._ema_trend

    def _update_atr_and_adx(self, event: MarketEvent) -> None:
        """
        Incremental Wilder ATR + ADX. Needs the previous bar's close (and high/low
        for +DM/-DM), so we look at self._bars[-1] BEFORE appending the new bar.
        """
        if not self._bars:
            return  # first bar — no prev to diff against

        prev = self._bars[-1]
        high, low, close = event.high, event.low, event.close
        prev_high, prev_low, prev_close = prev["high"], prev["low"], prev["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        up_move = high - prev_high
        dn_move = prev_low - low
        pdm = up_move if (up_move > dn_move and up_move > 0) else 0.0
        ndm = dn_move if (dn_move > up_move and dn_move > 0) else 0.0

        # --- ATR (Wilder / RMA) ---
        if self._atr is None:
            # Seed by averaging the first atr_period TRs.
            self._atr_seed_sum += tr
            self._atr_seed_count += 1
            if self._atr_seed_count >= self.atr_period:
                self._atr = self._atr_seed_sum / self.atr_period
                self._atr_seed_sum = 0.0
        else:
            self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

        # --- ADX (Wilder smoothing of +DM, -DM, TR; then DX → ADX) ---
        n = self.adx_period
        if self._smooth_pdm is None:
            # Seed smoothed +DM, -DM, TR by summing first N values.
            self._adx_seed_pdm_sum += pdm
            self._adx_seed_ndm_sum += ndm
            self._adx_seed_tr_sum  += tr
            self._adx_seed_count   += 1
            if self._adx_seed_count >= n:
                self._smooth_pdm    = self._adx_seed_pdm_sum
                self._smooth_ndm    = self._adx_seed_ndm_sum
                self._smooth_tr_adx = self._adx_seed_tr_sum
                self._compute_dx_for_adx()
        else:
            # Wilder smoothing: prev * (N-1)/N + current
            self._smooth_pdm    = self._smooth_pdm    - (self._smooth_pdm    / n) + pdm
            self._smooth_ndm    = self._smooth_ndm    - (self._smooth_ndm    / n) + ndm
            self._smooth_tr_adx = self._smooth_tr_adx - (self._smooth_tr_adx / n) + tr
            self._compute_dx_for_adx()

    def _update_daily_supertrend(self, event: MarketEvent) -> None:
        """
        Aggregate the incoming intraday bar into a daily OHLC. When the UTC
        day rolls over, finalize the prior day's bar and update Supertrend.

        We use the UTC date of the bar's timestamp as the day key. Crypto bars
        from CCXT/Binance are UTC-aligned so this is unambiguous. For the
        rare case where a strategy is fed in non-UTC bars, results are still
        deterministic, just bucketed by UTC date.
        """
        # Day key — date portion of the UTC timestamp.
        ts = event.timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        day_key = ts.normalize()

        if self._current_day is None:
            # First bar ever — start a fresh in-progress day.
            self._current_day = day_key
            self._current_day_ohlc = {
                "open": event.open, "high": event.high,
                "low":  event.low,  "close": event.close,
            }
            return

        if day_key == self._current_day:
            # Same day — extend in-progress OHLC.
            d = self._current_day_ohlc
            if event.high > d["high"]:
                d["high"] = event.high
            if event.low < d["low"]:
                d["low"] = event.low
            d["close"] = event.close
            return

        # --- Day rolled over: finalize prior day, update Supertrend ---
        finalized = self._current_day_ohlc
        self._supertrend_step(finalized["high"], finalized["low"], finalized["close"])

        # Start a fresh day from the current bar.
        self._current_day = day_key
        self._current_day_ohlc = {
            "open": event.open, "high": event.high,
            "low":  event.low,  "close": event.close,
        }

    def _supertrend_step(self, high: float, low: float, close: float) -> None:
        """
        Incremental Supertrend update for a single closed daily bar.

        Sets `self._st_direction` to +1 (green/up) or -1 (red/down), or leaves
        it None until enough daily bars exist to seed ATR.
        """
        # First bar — no prev_close yet, can't compute TR.
        if self._st_prev_close is None:
            self._st_prev_close = close
            return

        tr = max(
            high - low,
            abs(high - self._st_prev_close),
            abs(low  - self._st_prev_close),
        )

        # ATR (Wilder/RMA smoothing)
        n = self.st_atr_period
        if self._st_atr is None:
            self._st_atr_seed_sum += tr
            self._st_atr_seed_cnt += 1
            if self._st_atr_seed_cnt >= n:
                self._st_atr = self._st_atr_seed_sum / n
        else:
            self._st_atr = (self._st_atr * (n - 1) + tr) / n

        # Need ATR to compute bands. Bump prev_close and bail until ATR is ready.
        if self._st_atr is None:
            self._st_prev_close = close
            return

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.st_multiplier * self._st_atr
        basic_lower = hl2 - self.st_multiplier * self._st_atr

        # Carry-forward logic. Uses the PRIOR bar's close (still in _st_prev_close).
        prev_close = self._st_prev_close
        if self._st_prev_upper is None or basic_upper < self._st_prev_upper or prev_close > self._st_prev_upper:
            final_upper = basic_upper
        else:
            final_upper = self._st_prev_upper

        if self._st_prev_lower is None or basic_lower > self._st_prev_lower or prev_close < self._st_prev_lower:
            final_lower = basic_lower
        else:
            final_lower = self._st_prev_lower

        # Direction. Seed on first valid bar by close vs. mid-bands.
        if self._st_direction is None:
            self._st_direction = 1 if close > final_upper else -1
        else:
            if self._st_direction == 1 and close < final_lower:
                self._st_direction = -1
            elif self._st_direction == -1 and close > final_upper:
                self._st_direction = 1
            # else: maintain direction

        # Roll state forward for the next daily bar.
        self._st_prev_close = close
        self._st_prev_upper = final_upper
        self._st_prev_lower = final_lower

    def _compute_dx_for_adx(self) -> None:
        """Compute the current DX, then either buffer it (seeding) or smooth into ADX."""
        if self._smooth_tr_adx is None or self._smooth_tr_adx <= 0:
            return
        pdi = 100.0 * self._smooth_pdm / self._smooth_tr_adx
        ndi = 100.0 * self._smooth_ndm / self._smooth_tr_adx
        denom = pdi + ndi
        if denom == 0:
            dx = 0.0
        else:
            dx = 100.0 * abs(pdi - ndi) / denom

        n = self.adx_period
        if self._adx is None:
            self._adx_dx_buffer.append(dx)
            if len(self._adx_dx_buffer) >= n:
                self._adx = sum(self._adx_dx_buffer) / n
                self._adx_dx_buffer = []
        else:
            self._adx = (self._adx * (n - 1) + dx) / n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_position(self) -> None:
        self._in_position      = False
        self._position_side    = None
        self._stop_price       = None
        self._tp1_price        = None
        self._tp2_price        = None
        self._exit_pending     = False
        self._entry_bar        = None
        self._entry_price      = None
        self._stop_distance    = None
        self._tp1_hit          = False
        self._tp1_pending_fill = False
        self._trail_anchor     = None
