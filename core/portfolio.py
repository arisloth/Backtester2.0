"""
core/portfolio.py — Position tracking, cash management, and equity curve.

Responsibilities:
- Convert SignalEvents into sized OrderEvents (position sizing lives here).
- Track open positions and cash after each FillEvent.
- Mark positions to market on each bar (unrealized P&L).
- Record the equity curve and a full trade log for analytics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from core.event import (
    FillEvent, MarketEvent, OrderEvent, OrderSide, OrderType,
    SignalDirection, SignalEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracks a single open position for one symbol."""
    symbol: str
    side: OrderSide           # BUY (long) or SELL (short)
    quantity: float           # units held (always positive)
    avg_price: float          # average entry price (cost basis)
    realized_pnl: float = 0.0

    def market_value(self, current_price: float) -> float:
        """Current mark-to-market value of the position."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == OrderSide.BUY:
            return self.quantity * (current_price - self.avg_price)
        else:
            return self.quantity * (self.avg_price - current_price)


@dataclass
class TradeRecord:
    """Immutable record of a completed round-trip trade."""
    symbol: str
    side: str                        # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float                       # net P&L after commissions
    pnl_pct: float                   # P&L as % of entry value
    commission: float
    slippage: float                  # total slippage cost (entry + exit)
    stop_price: Optional[float]      # planned stop loss at entry
    tp_price: Optional[float]        # planned take profit at entry
    exit_reason: str                 # "stop" | "tp" | "signal" | ""
    hold_bars: int                   # number of bars position was held


class Portfolio:
    """
    Manages cash, positions, and the equity curve.

    Parameters
    ----------
    initial_capital : float
        Starting cash in base currency (USD).
    position_size_pct : float
        Fraction of current equity to allocate per signal (0.0–1.0).
        Default 0.1 = 10% per trade.
    short_initial_margin : float
        Fraction of short notional required as initial margin. Default 0.50
        matches Reg-T-style stock margin.
    """

    def __init__(self, initial_capital: float = 1_000.0, position_size_pct: float = 0.10,
                 risk_pct: float = 0.02, short_borrow_rate: float = 0.0,
                 short_initial_margin: float = 0.50):
        if short_initial_margin <= 0:
            raise ValueError(f"short_initial_margin must be > 0, got {short_initial_margin}")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size_pct = position_size_pct
        self.risk_pct = risk_pct
        self.short_borrow_rate = short_borrow_rate  # annualized; deducted daily from cash
        self.short_initial_margin = short_initial_margin

        # symbol → Position
        self.positions: Dict[str, Position] = {}

        # symbol → latest bar price (updated on every MarketEvent)
        self.current_prices: Dict[str, float] = {}

        # Equity curve: list of (timestamp, equity) tuples
        self.equity_curve: List[tuple] = []

        # Completed round-trip trades
        self.trade_log: List[TradeRecord] = []

        # Per-position metadata for trade log
        self._entry_times: Dict[str, pd.Timestamp] = {}
        self._entry_commissions: Dict[str, float] = {}
        self._entry_slippage: Dict[str, float] = {}
        self._entry_bar: Dict[str, int] = {}
        self._stop_prices: Dict[str, Optional[float]] = {}
        self._tp_prices: Dict[str, Optional[float]] = {}
        self._exit_reasons: Dict[str, str] = {}
        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # Called by Engine
    # ------------------------------------------------------------------

    def update_market(self, event: MarketEvent) -> None:
        """
        Update the latest price for this symbol and record an equity snapshot.
        Called once per MarketEvent, before strategies run.
        """
        self.current_prices[event.symbol] = event.close
        self._bar_count += 1

        # Deduct daily short borrow cost for any open short positions on this symbol
        if self.short_borrow_rate > 0:
            pos = self.positions.get(event.symbol)
            if pos is not None and pos.side == OrderSide.SELL:
                daily_cost = pos.quantity * event.close * (self.short_borrow_rate / 252)
                self.cash -= daily_cost

        # One equity snapshot PER TIMESTAMP, not per MarketEvent. The feed can
        # emit several symbols at the same timestamp (a multi-symbol basket, or
        # a non-traded reference feed like BTC for the RS filter). Appending per
        # event would record N points per bar, inflating the series length and
        # corrupting any length-based metric (CAGR divides equity by series
        # length → years). Overwrite the same-timestamp point so we keep the
        # last, fully-marked snapshot for that bar.
        equity = self._total_equity()
        if self.equity_curve and self.equity_curve[-1][0] == event.timestamp:
            self.equity_curve[-1] = (event.timestamp, equity)
        else:
            self.equity_curve.append((event.timestamp, equity))

    def generate_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """
        Convert a SignalEvent into a sized OrderEvent.

        Sizing rule: allocate `position_size_pct` of current equity.
        EXIT signals generate a sell/cover order for the full open position.
        We do not open a position if one already exists in the same direction.
        """
        symbol = signal.symbol
        direction = signal.direction
        equity = self._total_equity()

        # --- EXIT ---
        if direction == SignalDirection.EXIT:
            if symbol not in self.positions:
                logger.debug(f"EXIT signal for {symbol} but no open position — ignored.")
                return None
            pos = self.positions[symbol]
            side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            # Store exit reason from signal metadata
            self._exit_reasons[symbol] = signal.exit_reason or "signal"
            # Pass intended fill price so broker can apply gap protection
            fill_override = signal.stop_price if signal.exit_reason == "stop" else (
                            signal.tp_price   if signal.exit_reason in ("tp", "tp1") else None)
            # Honour strength for partial exits (e.g. TP1 half-close); 1.0 = full close
            exit_qty = pos.quantity * min(max(signal.strength, 0.0), 1.0)
            if exit_qty <= 0:
                return None
            return OrderEvent(
                symbol=symbol,
                asset_class=signal.asset_class,
                timestamp=signal.timestamp,
                order_type=OrderType.MARKET,
                side=side,
                quantity=exit_qty,
                fill_price_override=fill_override,
                exit_reason=signal.exit_reason or "",
            )

        # --- LONG ---
        if direction == SignalDirection.LONG:
            if symbol in self.positions and self.positions[symbol].side == OrderSide.BUY:
                logger.debug(f"Already long {symbol} — signal ignored.")
                return None
            price = self.current_prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(f"No price available for {symbol} — cannot size order.")
                return None
            quantity = self._size_position(
                equity, signal.strength, price, signal.stop_price, side=OrderSide.BUY
            )
            if quantity <= 0:
                return None
            # Stash signal metadata for trade log
            self._stop_prices[symbol] = signal.stop_price
            self._tp_prices[symbol]   = signal.tp_price
            return OrderEvent(
                symbol=symbol,
                asset_class=signal.asset_class,
                timestamp=signal.timestamp,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=quantity,
            )

        # --- SHORT ---
        if direction == SignalDirection.SHORT:
            if symbol in self.positions and self.positions[symbol].side == OrderSide.SELL:
                logger.debug(f"Already short {symbol} — signal ignored.")
                return None
            price = self.current_prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(f"No price available for {symbol} — cannot size order.")
                return None
            quantity = self._size_position(
                equity, signal.strength, price, signal.stop_price, side=OrderSide.SELL
            )
            if quantity <= 0:
                return None
            self._stop_prices[symbol] = signal.stop_price
            self._tp_prices[symbol]   = signal.tp_price
            return OrderEvent(
                symbol=symbol,
                asset_class=signal.asset_class,
                timestamp=signal.timestamp,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=quantity,
            )

        return None

    def update_fill(self, fill: FillEvent) -> None:
        """
        Update cash and positions after a fill. Records completed trades.

        Dispatch is based on the *existing position* relative to the fill side,
        not the fill side alone:
          - no position           → open a new position on fill side
          - same-side position    → scale into existing position
          - opposite-side position → close it (records a TradeRecord). If the
                                     fill quantity exceeds the open position,
                                     the remainder opens a new opposite-side
                                     position (a flip).
        """
        if fill.quantity <= 0:
            return

        # Adjust cash. net_cost is signed: + for buys (outflow), − for sells (inflow).
        self.cash -= fill.net_cost

        pos = self.positions.get(fill.symbol)

        if pos is None:
            self._open_position(fill)
        elif pos.side == fill.side:
            self._scale_in(fill, pos)
        else:
            self._close_position(fill, pos)

    def finalize(self) -> None:
        """
        Called by the engine at the end of the backtest. Closes any remaining
        open positions at the last known price (mark-to-market close).
        """
        for symbol, pos in list(self.positions.items()):
            price = self.current_prices.get(symbol, pos.avg_price)
            pnl = pos.unrealized_pnl(price)
            logger.info(
                f"Unclosed position {symbol}: {pos.quantity} units, "
                f"unrealized P&L = {pnl:.2f}"
            )
        # Equity curve final point is already appended by the last update_market call.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_equity(self) -> float:
        """Cash + mark-to-market value of all open positions."""
        return self.cash + sum(
            pos.quantity * self.current_prices.get(sym, pos.avg_price)
            if pos.side == OrderSide.BUY
            else -pos.quantity * self.current_prices.get(sym, pos.avg_price)
            for sym, pos in self.positions.items()
        )

    def _size_position(self, equity: float, strength: float, price: float,
                       stop_price: Optional[float] = None,
                       side: OrderSide = OrderSide.BUY) -> float:
        """
        Return the number of units to trade.

        If a stop price is provided, size so that hitting the stop loses exactly
        risk_pct × current equity (fixed-risk sizing).

        If no stop price is available, fall back to allocating position_size_pct
        of current equity (used by strategies that don't set stops, e.g. SMA cross).

        Shorts are capped by aggregate initial margin, so short-sale proceeds
        cannot be recycled into unlimited additional short exposure.
        """
        stop_distance = abs(price - stop_price) if stop_price is not None else None

        if stop_distance and stop_distance > 0:
            risk_amount = equity * self.risk_pct * strength
            quantity    = risk_amount / stop_distance
        else:
            alloc    = equity * self.position_size_pct * strength
            quantity = alloc / price if price > 0 else 0.0

        if price <= 0:
            return 0.0

        if side == OrderSide.SELL:
            max_quantity = self._available_short_margin(equity) / (price * self.short_initial_margin)
        else:
            # Longs cannot spend more cash than we have.
            max_quantity = self.cash / price

        return min(quantity, max_quantity)

    def _available_short_margin(self, equity: float) -> float:
        """Equity remaining after Reg-T-style initial margin on open shorts."""
        used_margin = sum(
            pos.quantity * self.current_prices.get(sym, pos.avg_price) * self.short_initial_margin
            for sym, pos in self.positions.items()
            if pos.side == OrderSide.SELL
        )
        return max(0.0, equity - used_margin)

    def _open_position(self, fill: FillEvent) -> None:
        """Open a brand-new position on `fill.side` for this symbol."""
        symbol = fill.symbol
        self.positions[symbol] = Position(
            symbol=symbol,
            side=fill.side,
            quantity=fill.quantity,
            avg_price=fill.fill_price,
        )
        self._entry_times[symbol]       = fill.timestamp
        self._entry_commissions[symbol] = fill.commission
        self._entry_slippage[symbol]    = fill.slippage
        self._entry_bar[symbol]         = self._bar_count

    def _scale_in(self, fill: FillEvent, pos: Position) -> None:
        """Add to an existing same-side position; weighted-average the cost basis."""
        total_cost      = pos.avg_price * pos.quantity + fill.fill_price * fill.quantity
        pos.quantity   += fill.quantity
        pos.avg_price   = total_cost / pos.quantity
        # Carry incremental entry cost into the position's running entry totals so that
        # a future close prorates correctly.
        symbol = fill.symbol
        self._entry_commissions[symbol] = self._entry_commissions.get(symbol, 0.0) + fill.commission
        self._entry_slippage[symbol]    = self._entry_slippage.get(symbol, 0.0)    + fill.slippage

    def _close_position(self, fill: FillEvent, pos: Position) -> None:
        """
        Close (or partially close) an opposite-side position. Records a TradeRecord.

        If `fill.quantity` exceeds the open position size, the remainder opens a
        new position on `fill.side` at `fill.fill_price` (a flip). Commissions
        and slippage on the fill are prorated by quantity between the two legs.
        """
        symbol      = fill.symbol
        closed_qty  = min(fill.quantity, pos.quantity)
        # Prorate the fill's commission/slippage over the closed portion only;
        # the remainder (if any) belongs to the newly opened opposite-side leg.
        close_frac      = closed_qty / fill.quantity if fill.quantity else 1.0
        close_comm_fill = fill.commission * close_frac
        close_slip_fill = fill.slippage   * close_frac

        # Prorate the *entry* commission for partial closes of a scaled-in position.
        entry_comm  = self._entry_commissions.get(symbol, 0.0) * (closed_qty / pos.quantity)
        entry_slip  = self._entry_slippage.get(symbol, 0.0)    * (closed_qty / pos.quantity)

        if pos.side == OrderSide.BUY:
            # Closing a long: profit if exit > entry.
            pnl_gross = closed_qty * (fill.fill_price - pos.avg_price)
            side_str  = "LONG"
        else:
            # Closing a short: profit if exit < entry.
            pnl_gross = closed_qty * (pos.avg_price - fill.fill_price)
            side_str  = "SHORT"

        pnl         = pnl_gross - close_comm_fill - entry_comm
        entry_value = closed_qty * pos.avg_price
        pnl_pct     = pnl / entry_value if entry_value else 0.0
        hold_bars   = self._bar_count - self._entry_bar.get(symbol, self._bar_count)

        self.trade_log.append(TradeRecord(
            symbol=symbol,
            side=side_str,
            entry_time=self._entry_times.get(symbol, fill.timestamp),
            exit_time=fill.timestamp,
            entry_price=pos.avg_price,
            exit_price=fill.fill_price,
            quantity=closed_qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=close_comm_fill + entry_comm,
            slippage=entry_slip + close_slip_fill,
            stop_price=self._stop_prices.get(symbol),
            tp_price=self._tp_prices.get(symbol),
            exit_reason=self._exit_reasons.get(symbol, "signal"),
            hold_bars=hold_bars,
        ))

        # Decrement and clean up if fully closed.
        pos.quantity -= closed_qty
        if pos.quantity <= 1e-9:
            del self.positions[symbol]
            for d in (self._entry_times, self._entry_commissions, self._entry_slippage,
                      self._entry_bar, self._stop_prices, self._tp_prices, self._exit_reasons):
                d.pop(symbol, None)
        else:
            # Partial close of a scaled-in position: leave residual entry costs.
            self._entry_commissions[symbol] -= entry_comm
            self._entry_slippage[symbol]    -= entry_slip

        # Flip: if the fill was larger than the open position, the leftover
        # quantity opens a new position in the fill's direction at fill_price.
        leftover = fill.quantity - closed_qty
        if leftover > 1e-9:
            logger.warning(
                f"{symbol}: fill of {fill.quantity} flipped through a "
                f"{pos.side.name} position of {closed_qty} — opening "
                f"{fill.side.name} {leftover} at {fill.fill_price}."
            )
            self.positions[symbol] = Position(
                symbol=symbol,
                side=fill.side,
                quantity=leftover,
                avg_price=fill.fill_price,
            )
            self._entry_times[symbol]       = fill.timestamp
            self._entry_commissions[symbol] = fill.commission - close_comm_fill
            self._entry_slippage[symbol]    = fill.slippage   - close_slip_fill
            self._entry_bar[symbol]         = self._bar_count
            # Stale stop/tp/exit_reason metadata from the closed leg should not
            # carry into the new position; they're cleared by the cleanup above
            # when the close was full. On a flip, the close was full by definition,
            # so they were popped — nothing to do here.

    # ------------------------------------------------------------------
    # Convenience accessors for analytics
    # ------------------------------------------------------------------

    def equity_series(self) -> pd.Series:
        """Return the equity curve as a time-indexed pandas Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        times, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

    def trade_dataframe(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trade_log])
