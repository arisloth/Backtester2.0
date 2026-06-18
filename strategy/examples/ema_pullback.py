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

V2 additions (market-wide regime awareness + entry-quality filters):
  1. BTC regime gate   : block alt longs when BTC is bearish, alt shorts when
                         BTC is strongly bullish (BTC EMA20/50 on the trading
                         timeframe). Optional flatten-on-break exit override.
  2. Relative strength : only long alts outperforming BTC over rs_lookback
                         bars; only short alts underperforming.
  3. Volume confirm    : entry bar volume must exceed vol_mult × the trailing
                         average over vol_lookback bars.
  4. Pullback memory   : require pullback_memory_bars consecutive closes on the
                         correct side of the chosen EMA *before* the touch bar,
                         so a bar chopping through the EMA in a flat market does
                         not count as a genuine pullback.
  5. Instrumentation   : per-filter rejection counters (see filter_stats()).

Feed ordering requirement (no-lookahead): when the BTC gate or RS filter is
enabled, the backtest runner MUST feed the BTC bar for time T *before* the
alt's bar for time T. With the existing multi-symbol feeds this is satisfied
by listing btc_symbol first in the feed's `symbols` list — the engine drains
MarketEvents in symbol order, so BTC state is current when the alt bar runs.
All V2 filters use only data available at the bar's close (no lookahead).
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

    --- V2 parameters ---
    btc_symbol : str
        Symbol whose bars drive the BTC regime gate / RS filter. Default
        "BTC/USDT". Bars for this symbol are consumed to update BTC state and
        never traded. The runner must feed this symbol's bar for time T before
        the alt's bar for time T (see module docstring).
    btc_gate_enabled : bool
        Master switch for the BTC regime gate (default True).
    btc_gate_mode : str
        "ema_stack"     : longs need btc_close > btc_ema20 > btc_ema50; shorts
                          need btc_close < btc_ema20 < btc_ema50. (default)
        "ema20_reclaim" : looser — longs need btc_close > btc_ema20; shorts
                          need btc_close < btc_ema20.
        "off"           : no gate (A/B comparison).
    btc_ema_fast, btc_ema_slow : int
        BTC EMA periods on the trading timeframe (default 20 / 50).
    btc_flatten_on_break : bool
        If True, emit an EXIT (exit_reason="btc_break") when BTC closes below
        its EMA50 while long (mirror above EMA50 while short). Default False so
        V1-vs-V2 comparisons isolate the entry gate first.
    rs_filter_sides : str
        Which side(s) the relative-strength-vs-BTC veto applies to:
        "short" (default, V3 — never short an alt outperforming BTC), "both"
        (V2 behavior — also require longs to outperform), or "off". A legacy
        `rs_filter_enabled` bool is still accepted and maps True→"both",
        False→"off".
    rs_lookback : int
        Lookback in bars for the RS return comparison (default 48 = 24h on 30m).
    rs_min_spread : float
        Minimum return spread the alt must beat BTC by. Long needs
        alt_ret > btc_ret + rs_min_spread; short needs alt_ret < btc_ret -
        rs_min_spread. Default 0.0 (just "outperforming").
    volume_filter_enabled : bool
        Master switch for the entry-bar volume confirmation (default False — V3
        drops it from the entry path; machinery kept for A/B).
    vol_lookback : int
        Bars in the trailing average-volume window (default 20).
    vol_mult : float
        Entry bar volume must exceed vol_mult × the trailing average (default 1.5).
    pullback_memory_bars : int
        Require this many consecutive closes on the correct side of the chosen
        EMA *before* the touch bar. Default 0 (disabled). Superseded by
        fresh_touch_required in V3 but kept for A/B comparison.
    fresh_touch_required : bool
        V3 entry gate (default True): take the trade only on the FIRST pullback
        after price reclaims the chosen EMA (longs) / loses it (shorts) — i.e.
        the consecutive-close count on the trade's side is 0 at the touch bar.
        The inverse of pullback_memory.
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
        touch_tol_atr: float = 0.3,
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
        # --- V2 / V3 ---
        btc_symbol: str = "BTC/USDT",
        btc_gate_enabled: bool = False,       # V3: BTC demoted from entry veto to (future) sizing input
        btc_gate_mode: str = "ema_stack",
        btc_ema_fast: int = 20,
        btc_ema_slow: int = 50,
        btc_flatten_on_break: bool = False,
        rs_filter_sides: str = "short",       # V3: "short" | "both" | "off"; RS only vetoes shorts
        rs_lookback: int = 48,
        rs_min_spread: float = 0.0,
        volume_filter_enabled: bool = False,  # V3: volume filter deleted from the entry path
        vol_lookback: int = 20,
        vol_mult: float = 1.5,
        pullback_memory_bars: int = 0,        # V3: superseded by fresh_touch_required (kept for A/B)
        fresh_touch_required: bool = True,    # V3: enter only on the first reclaim/retest of the EMA
        # --- V3.1 candidate refinements (default off; A/B before trusting) 
        ext_filter_enabled: bool = False,   # block longs stretched too far above EMA200
        ext_max_pct: float = 10.0,          # max % above EMA200 to allow a long
        reset_gate_enabled: bool = False,   # one entry per leg; re-arm on EMA50 cross
        rs_filter_enabled: Optional[bool] = None,  # legacy shim: True→"both", False→"off"
    ):
        if direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got '{direction}'")
        if pullback_ema not in ("ema_fast", "ema_slow"):
            raise ValueError(f"pullback_ema must be 'ema_fast' or 'ema_slow', got '{pullback_ema}'")
        if runner_mode not in ("structure", "atr_trail", "fixed_r"):
            raise ValueError(f"runner_mode must be 'structure'|'atr_trail'|'fixed_r', got '{runner_mode}'")
        if not (0.0 < tp1_ratio <= 1.0):
            raise ValueError(f"tp1_ratio must be in (0, 1], got {tp1_ratio}")
        if btc_gate_mode not in ("ema_stack", "ema20_reclaim", "off"):
            raise ValueError(
                f"btc_gate_mode must be 'ema_stack'|'ema20_reclaim'|'off', got '{btc_gate_mode}'"
            )
        # Legacy shim: an explicit rs_filter_enabled (V2 API) maps onto the new
        # rs_filter_sides switch so older configs/tests keep working.
        if rs_filter_enabled is not None:
            rs_filter_sides = "both" if rs_filter_enabled else "off"
        if rs_filter_sides not in ("short", "both", "off"):
            raise ValueError(
                f"rs_filter_sides must be 'short'|'both'|'off', got '{rs_filter_sides}'"
            )

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

        # --- V2 params ---
        self.btc_symbol            = btc_symbol
        self.btc_gate_enabled      = btc_gate_enabled
        self.btc_gate_mode         = btc_gate_mode
        self.btc_ema_fast_n        = btc_ema_fast
        self.btc_ema_slow_n        = btc_ema_slow
        self.btc_flatten_on_break  = btc_flatten_on_break
        self.rs_filter_sides       = rs_filter_sides
        self.rs_lookback           = rs_lookback
        self.rs_min_spread         = rs_min_spread
        self.volume_filter_enabled = volume_filter_enabled
        self.vol_lookback          = vol_lookback
        self.vol_mult              = vol_mult
        self.pullback_memory_bars  = pullback_memory_bars
        self.fresh_touch_required  = fresh_touch_required
        self.ext_filter_enabled    = ext_filter_enabled
        self.ext_max_pct           = ext_max_pct
        self.reset_gate_enabled    = reset_gate_enabled

        # Bar history — needs to cover EMA_trend warm-up, ADX warm-up (~2*N),
        # structure_lookback, swing_lookback, RS lookback, and the volume window.
        history_len = max(
            ema_trend + 5, adx_period * 3, structure_lookback + 5, swing_lookback + 5,
            rs_lookback + 5, vol_lookback + 5,
        )
        self._bars: deque = deque(maxlen=history_len)
        self._bar_count: int = 0

        # --- BTC regime state (incremental EMAs on the trading timeframe) ---
        # _btc_closes feeds both EMA seeding and the RS-vs-BTC return lookback.
        self._btc_close:    Optional[float] = None
        self._btc_ema_fast: Optional[float] = None
        self._btc_ema_slow: Optional[float] = None
        self._btc_alpha_fast = 2.0 / (btc_ema_fast + 1.0)
        self._btc_alpha_slow = 2.0 / (btc_ema_slow + 1.0)
        self._btc_closes: deque = deque(maxlen=max(btc_ema_slow, rs_lookback) + 5)

        # --- Volume confirmation: trailing window of PRIOR bars' volumes ---
        # Updated at end of on_bar, so during a bar's entry check the window
        # holds only bars strictly before the current (entry) bar.
        self._vol_window: deque = deque(maxlen=vol_lookback)
        self._vol_sum: float = 0.0

        # --- Pullback memory: consecutive closes above/below the chosen EMA ---
        # Snapshotted before update: during a bar's entry check these counters
        # reflect bars strictly before the current one.
        self._consec_above: int = 0
        self._consec_below: int = 0

        # --- Instrumentation: per-filter rejection counters ---
        self._reject_counts = {
            "regime": 0, "adx": 0, "supertrend": 0, "warmup": 0,
            "btc_gate": 0, "rs": 0, "volume": 0, "pullback_memory": 0,
            "fresh_touch": 0, "extension": 0, "reset_gate": 0,
        }
        self._setup_count: int = 0   # valid regime + pullback-touch setups seen
        self._entry_count: int = 0   # entry signals actually emitted
        # Per-emitted-entry record of the RAW V2-filter verdicts at that bar
        # (evaluated regardless of whether each filter is enabled). Lets external
        # A/B tooling pool every trade and bucket its realized outcome by filter
        # state — see scripts/ab_v2.py and entry_filter_log().
        self._entry_filter_log: list = []

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
        # V3.1 reset-gate arming (one entry per trend leg)
        self._long_armed:  bool = True
        self._short_armed: bool = True

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
        # BTC bars drive the regime gate / RS filter — update BTC state and
        # never trade them. (When btc_symbol == self.symbol the user is trading
        # BTC itself, so fall through and process it normally.)
        if event.symbol == self.btc_symbol and self.btc_symbol != self.symbol:
            self._update_btc_state(event)
            return None

        if event.symbol != self.symbol:
            return None

        # If we're trading BTC itself, keep its regime state current from the
        # same bar (filters are typically disabled in this case).
        if self.btc_symbol == self.symbol:
            self._update_btc_state(event)

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

        signal: Optional[SignalEvent] = None

        # In position → check exits first
        if self._in_position and not self._exit_pending:
            sig = self._check_exit(event)
            if sig is not None:
                self._exit_pending = True
                signal = sig

        if signal is None:
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
                signal = self._check_entry(event)

        # End-of-bar rolling updates. Run on EVERY bar (regardless of position
        # or early exit) so the volume window and pullback-memory counters stay
        # correct, and AFTER the entry check so they reflect bars strictly
        # before the current one when that check reads them.
        self._update_entry_quality_state(event)

        return signal

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
        if self._ema_fast is None or self._ema_slow is None or self._ema_trend is None:
            return None

        # ADX trend-strength gate.
        if self._adx is None or self._adx < self.adx_min:
            self._reject_counts["adx"] += 1
            return None

        # 1d Supertrend regime filter — strict: block during warm-up too.
        if self.daily_supertrend_filter and self._st_direction is None:
            self._reject_counts["supertrend"] += 1
            return None

        # V2 filter warm-up — strict, like the Supertrend convention: if a gate
        # is enabled but its state isn't seeded yet, block entries.
        if self.btc_gate_enabled and self.btc_gate_mode != "off" and not self._btc_gate_ready():
            self._reject_counts["warmup"] += 1
            return None
        if self.rs_filter_sides != "off" and not self._rs_ready():
            self._reject_counts["warmup"] += 1
            return None
        if self.volume_filter_enabled and not self._vol_ready():
            self._reject_counts["warmup"] += 1
            return None

        chosen_ema = self._ema_fast if self.pullback_ema_name == "ema_fast" else self._ema_slow
        tol = self.touch_tol_atr * self._atr

        # --- LONG ---
        if self.direction in ("long", "both"):
            sig = self._try_long(event, chosen_ema, tol)
            if sig is not None:
                return sig

        # --- SHORT ---
        if self.direction in ("short", "both"):
            sig = self._try_short(event, chosen_ema, tol)
            if sig is not None:
                return sig

        return None

    def _try_long(self, event: MarketEvent, chosen_ema: float, tol: float) -> Optional[SignalEvent]:
        # 1d Supertrend gate: only longs when daily ST is green (+1)
        if self.daily_supertrend_filter and self._st_direction != 1:
            self._reject_counts["supertrend"] += 1
            return None
        # Regime gate: close > ema_fast > ema_slow > ema_trend
        if not (event.close > self._ema_fast > self._ema_slow > self._ema_trend):
            self._reject_counts["regime"] += 1
            return None
        # Pullback trigger: low pierced (or grazed) the EMA, close held above it.
        # Absence of a touch is "no setup", not a filter rejection.
        if not (event.low <= chosen_ema + tol and event.close > chosen_ema):
            return None

        # A genuine regime + pullback setup exists — now apply V2 quality filters.
        self._setup_count += 1
        if not self._btc_gate_allows("long"):
            self._reject_counts["btc_gate"] += 1
            return None
        if not self._rs_allows("long"):
            self._reject_counts["rs"] += 1
            return None
        if not self._volume_confirms(event):
            self._reject_counts["volume"] += 1
            return None
        if not self._pullback_memory_ok("long"):
            self._reject_counts["pullback_memory"] += 1
            return None
        if not self._fresh_touch_ok("long"):
            self._reject_counts["fresh_touch"] += 1
            return None
        if not self._ext_ok("long"):
            self._reject_counts["extension"] += 1
            return None
        if not self._reset_gate_ok("long"):
            self._reject_counts["reset_gate"] += 1
            return None

        stop = self._compute_long_stop(event.close)
        if stop is None or stop >= event.close:
            return None
        self._pending_stop = stop
        self._pending_side = "long"
        self._pending_atr  = self._atr
        self._entry_count += 1
        self._long_armed = False
        self._log_entry_verdicts("long", event)
        return SignalEvent(
            symbol=self.symbol,
            asset_class=self.asset_class,
            timestamp=event.timestamp,
            direction=SignalDirection.LONG,
            strategy_id="ema_pullback",
            stop_price=stop,
            # tp_price set in on_fill once fill price is known
        )

    def _try_short(self, event: MarketEvent, chosen_ema: float, tol: float) -> Optional[SignalEvent]:
        # 1d Supertrend gate: only shorts when daily ST is red (-1)
        if self.daily_supertrend_filter and self._st_direction != -1:
            self._reject_counts["supertrend"] += 1
            return None
        # Regime gate: close < ema_fast < ema_slow < ema_trend
        if not (event.close < self._ema_fast < self._ema_slow < self._ema_trend):
            self._reject_counts["regime"] += 1
            return None
        # Pullback trigger: high pierced the EMA from below, close held below it.
        if not (event.high >= chosen_ema - tol and event.close < chosen_ema):
            return None

        self._setup_count += 1
        if not self._btc_gate_allows("short"):
            self._reject_counts["btc_gate"] += 1
            return None
        if not self._rs_allows("short"):
            self._reject_counts["rs"] += 1
            return None
        if not self._volume_confirms(event):
            self._reject_counts["volume"] += 1
            return None
        if not self._pullback_memory_ok("short"):
            self._reject_counts["pullback_memory"] += 1
            return None
        if not self._fresh_touch_ok("short"):
            self._reject_counts["fresh_touch"] += 1
            return None
        if not self._ext_ok("short"):
            self._reject_counts["extension"] += 1
            return None
        if not self._reset_gate_ok("short"):
            self._reject_counts["reset_gate"] += 1
            return None

        stop = self._compute_short_stop(event.close)
        if stop is None or stop <= event.close:
            return None
        self._pending_stop = stop
        self._pending_side = "short"
        self._pending_atr  = self._atr
        self._entry_count += 1
        self._short_armed = False
        self._log_entry_verdicts("short", event)
        return SignalEvent(
            symbol=self.symbol,
            asset_class=self.asset_class,
            timestamp=event.timestamp,
            direction=SignalDirection.SHORT,
            strategy_id="ema_pullback",
            stop_price=stop,
        )

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

        # BTC regime-break flatten. Only fires when neither stop nor a TP target
        # triggered this bar, so a stop stays conservative and wins. Flattens the
        # full position (including a runner) at the next bar's open (market exit).
        if not exit_reason and self.btc_flatten_on_break and self._btc_break_triggered():
            exit_reason = "btc_break"
            strength    = 1.0

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
    # V2 — BTC regime gate
    # ------------------------------------------------------------------

    def _update_btc_state(self, event: MarketEvent) -> None:
        """
        Update BTC close, EMAs, and the RS close history from a BTC bar.

        Same incremental pattern as the alt EMAs: each EMA is seeded as an SMA
        over its first N closes, then updated recursively. Must be called before
        the alt's bar for the same timestamp (see module docstring on ordering).
        """
        close = event.close
        self._btc_closes.append(close)
        self._btc_close = close

        # Seed each EMA as the SMA over its first N closes, then go incremental.
        if self._btc_ema_fast is None:
            if len(self._btc_closes) >= self.btc_ema_fast_n:
                self._btc_ema_fast = (
                    sum(list(self._btc_closes)[-self.btc_ema_fast_n:]) / self.btc_ema_fast_n
                )
        else:
            self._btc_ema_fast = (
                self._btc_alpha_fast * close + (1 - self._btc_alpha_fast) * self._btc_ema_fast
            )

        if self._btc_ema_slow is None:
            if len(self._btc_closes) >= self.btc_ema_slow_n:
                self._btc_ema_slow = (
                    sum(list(self._btc_closes)[-self.btc_ema_slow_n:]) / self.btc_ema_slow_n
                )
        else:
            self._btc_ema_slow = (
                self._btc_alpha_slow * close + (1 - self._btc_alpha_slow) * self._btc_ema_slow
            )

    def _btc_gate_ready(self) -> bool:
        """True once the BTC EMAs the active gate mode needs are seeded."""
        if self.btc_gate_mode == "off":
            return True
        if self.btc_gate_mode == "ema20_reclaim":
            return self._btc_ema_fast is not None
        return self._btc_ema_fast is not None and self._btc_ema_slow is not None

    def _btc_gate_allows(self, side: str) -> bool:
        """Whether the BTC regime permits an entry on `side` ('long'|'short')."""
        if not self.btc_gate_enabled or self.btc_gate_mode == "off":
            return True
        return self._btc_gate_raw(side)

    def _btc_gate_raw(self, side: str) -> bool:
        """The gate's verdict ignoring the enabled flag (for A/B instrumentation).
        Uses btc_gate_mode; mode 'off' has nothing to test → True. Warm-up → False."""
        if self.btc_gate_mode == "off":
            return True
        c, f = self._btc_close, self._btc_ema_fast
        if c is None or f is None:
            return False  # warm-up — defensive; _check_entry blocks this earlier
        if self.btc_gate_mode == "ema20_reclaim":
            return c > f if side == "long" else c < f
        # ema_stack
        s = self._btc_ema_slow
        if s is None:
            return False
        return (c > f > s) if side == "long" else (c < f < s)

    def _btc_break_triggered(self) -> bool:
        """For the flatten-on-break exit: BTC closed through its EMA50 against us."""
        if self._btc_close is None or self._btc_ema_slow is None:
            return False
        if self._position_side == "long":
            return self._btc_close < self._btc_ema_slow
        if self._position_side == "short":
            return self._btc_close > self._btc_ema_slow
        return False

    # ------------------------------------------------------------------
    # V2 — Relative strength vs. BTC
    # ------------------------------------------------------------------

    def _rs_ready(self) -> bool:
        """Need rs_lookback+1 closes for both the alt and BTC to form a return."""
        return (
            len(self._bars) >= self.rs_lookback + 1
            and len(self._btc_closes) >= self.rs_lookback + 1
        )

    def _rs_allows(self, side: str) -> bool:
        """Long requires the alt to out-return BTC over rs_lookback (mirror short).

        V3: the RS veto applies only to the side(s) named by rs_filter_sides —
        default "short" never vetoes a long (you can be long an alt that's lagging
        BTC, but you should not short one that's outrunning it).
        """
        if self.rs_filter_sides == "off":
            return True
        if self.rs_filter_sides == "short" and side != "short":
            return True
        return self._rs_raw(side)

    def _rs_raw(self, side: str) -> bool:
        """RS verdict ignoring the enabled flag (for A/B instrumentation)."""
        if not self._rs_ready():
            return False  # can't evaluate yet — treat as "did not confirm"
        alt_now  = self._bars[-1]["close"]
        alt_then = self._bars[-1 - self.rs_lookback]["close"]
        btc_now  = self._btc_closes[-1]
        btc_then = self._btc_closes[-1 - self.rs_lookback]
        if alt_then <= 0 or btc_then <= 0:
            return True  # degenerate baseline — don't block on bad data
        alt_ret = alt_now / alt_then - 1.0
        btc_ret = btc_now / btc_then - 1.0
        if side == "long":
            return alt_ret > btc_ret + self.rs_min_spread
        return alt_ret < btc_ret - self.rs_min_spread

    # ------------------------------------------------------------------
    # V2 — Volume confirmation
    # ------------------------------------------------------------------

    def _vol_ready(self) -> bool:
        return len(self._vol_window) >= self.vol_lookback

    def _volume_confirms(self, event: MarketEvent) -> bool:
        """Entry bar volume must exceed vol_mult × the trailing average."""
        if not self.volume_filter_enabled:
            return True
        return self._volume_raw(event)

    def _volume_raw(self, event: MarketEvent) -> bool:
        """Volume verdict ignoring the enabled flag (for A/B instrumentation)."""
        if not self._vol_window:
            return False
        avg = self._vol_sum / len(self._vol_window)
        if avg <= 0:
            return True  # no volume info — don't block
        return event.volume > self.vol_mult * avg

    # ------------------------------------------------------------------
    # V2 — Pullback memory
    # ------------------------------------------------------------------

    def _pullback_memory_ok(self, side: str) -> bool:
        """
        Require pullback_memory_bars consecutive closes on the correct side of
        the chosen EMA *before* the current bar. The counters are snapshotted
        before being updated with the current close (see _update_entry_quality_state),
        so when read here they reflect only prior bars.
        """
        if self.pullback_memory_bars <= 0:
            return True
        if side == "long":
            return self._consec_above >= self.pullback_memory_bars
        return self._consec_below >= self.pullback_memory_bars

    def _fresh_touch_ok(self, side: str) -> bool:
        """
        V3 entry gate (the inverse of pullback memory): enter only on the FIRST
        pullback after price reclaims the chosen EMA. The consecutive-close
        counters are snapshotted before the current bar's update, so == 0 means
        the prior bar did NOT close on the trade's side of the EMA — i.e. the
        current touch-and-hold is a fresh reclaim/retest, not the Nth bar of an
        already-extended leg.
        """
        if not self.fresh_touch_required:
            return True
        if side == "long":
            return self._consec_above == 0
        return self._consec_below == 0

    def _ext_ok(self, side: str) -> bool:
        """V3.1: block LONGS stretched too far above EMA200. Shorts exempt —
        short entries live in downtrends where price is naturally below the
        mean, so a distance cap there blocks them when they work best."""
        if not self.ext_filter_enabled or side != "long":
            return True
        if self._ema_trend is None or self._ema_trend <= 0:
            return True
        ext_pct = (self._bars[-1]["close"] - self._ema_trend) / self._ema_trend * 100.0
        return ext_pct <= self.ext_max_pct

    def _reset_gate_ok(self, side: str) -> bool:
        """V3.1: at most one entry per trend leg; re-arms on EMA50 cross
        (see _update_entry_quality_state)."""
        if not self.reset_gate_enabled:
            return True
        return self._long_armed if side == "long" else self._short_armed

    # ------------------------------------------------------------------
    # V2 — End-of-bar rolling state + instrumentation
    # ------------------------------------------------------------------

    def _update_entry_quality_state(self, event: MarketEvent) -> None:
        """
        Roll the volume window and pullback-memory counters forward with the
        current bar. Called at the END of on_bar so these structures reflect
        bars strictly before the current one when the entry check reads them.
        """
        # Volume window (subtract the evicted leftmost before appending).
        if len(self._vol_window) == self.vol_lookback and self._vol_window:
            self._vol_sum -= self._vol_window[0]
        self._vol_window.append(event.volume)
        self._vol_sum += event.volume

        # Pullback memory: consecutive closes vs. the chosen EMA.
        chosen = self._ema_fast if self.pullback_ema_name == "ema_fast" else self._ema_slow
        if chosen is not None:
            if event.close > chosen:
                self._consec_above += 1
                self._consec_below = 0
            elif event.close < chosen:
                self._consec_below += 1
                self._consec_above = 0
            else:  # exactly on the EMA — neither side
                self._consec_above = 0
                self._consec_below = 0

        # V3.1 reset gate: re-arm a side once price closes back through EMA50.
        if self._ema_slow is not None:
            if event.close < self._ema_slow:
                self._long_armed = True
            if event.close > self._ema_slow:
                self._short_armed = True

    def _log_entry_verdicts(self, side: str, event: MarketEvent) -> None:
        """Record the raw V2-filter verdicts for an entry that is being emitted.

        Evaluated independently of each filter's enabled flag, so a baseline run
        (all V2 filters OFF) still captures what each filter *would* have decided.
        Appended in emission order; aligns 1:1 (in order) with the resulting
        round-trips. `pullback_consec` is the raw consecutive-close count so the
        consumer can threshold it freely.
        """
        consec = self._consec_above if side == "long" else self._consec_below
        self._entry_filter_log.append({
            "time":            event.timestamp,
            "side":            side,
            "btc_gate":        self._btc_gate_raw(side),
            "rs":              self._rs_raw(side),
            "volume":          self._volume_raw(event),
            "pullback_consec": consec,
        })

    def entry_filter_log(self) -> list:
        """Per-emitted-entry raw V2-filter verdicts, in emission order.

        See _log_entry_verdicts. Consumed by scripts/ab_v2.py to bucket realized
        trade outcomes by filter state with full sample size.
        """
        return list(self._entry_filter_log)

    def filter_stats(self) -> dict:
        """
        Snapshot of entry-funnel instrumentation for end-of-run diagnostics.

        `setups` is the number of genuine regime + pullback-touch setups seen;
        the per-filter counts under `rejections` say how many of those (or, for
        adx/supertrend/warmup, how many candidate bars) each filter killed.
        `entries` is the number of entry signals actually emitted.
        """
        return {
            "setups": self._setup_count,
            "entries": self._entry_count,
            "rejections": dict(self._reject_counts),
        }

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
