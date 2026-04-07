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
    side: str                  # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float                 # net P&L after commissions
    commission: float


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
    """

    def __init__(self, initial_capital: float = 100_000.0, position_size_pct: float = 0.1):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size_pct = position_size_pct

        # symbol → Position
        self.positions: Dict[str, Position] = {}

        # symbol → latest bar price (updated on every MarketEvent)
        self.current_prices: Dict[str, float] = {}

        # Equity curve: list of (timestamp, equity) tuples
        self.equity_curve: List[tuple] = []

        # Completed round-trip trades
        self.trade_log: List[TradeRecord] = []

        # Track entry timestamps for trade log
        self._entry_times: Dict[str, pd.Timestamp] = {}
        self._entry_commissions: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Called by Engine
    # ------------------------------------------------------------------

    def update_market(self, event: MarketEvent) -> None:
        """
        Update the latest price for this symbol and record an equity snapshot.
        Called once per MarketEvent, before strategies run.
        """
        self.current_prices[event.symbol] = event.close
        equity = self._total_equity()
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
            return OrderEvent(
                symbol=symbol,
                asset_class=signal.asset_class,
                timestamp=signal.timestamp,
                order_type=OrderType.MARKET,
                side=side,
                quantity=pos.quantity,
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
            quantity = self._size_position(equity, signal.strength, price)
            if quantity <= 0:
                return None
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
            quantity = self._size_position(equity, signal.strength, price)
            if quantity <= 0:
                return None
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
        """
        symbol = fill.symbol

        # Adjust cash
        self.cash -= fill.net_cost

        if fill.side == OrderSide.BUY:
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)

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
        unrealized = sum(
            pos.unrealized_pnl(self.current_prices.get(sym, pos.avg_price))
            for sym, pos in self.positions.items()
        )
        return self.cash + sum(
            pos.quantity * self.current_prices.get(sym, pos.avg_price)
            if pos.side == OrderSide.BUY
            else pos.quantity * pos.avg_price  # short: cash already received at entry
            for sym, pos in self.positions.items()
        )

    def _size_position(self, equity: float, strength: float, price: float) -> float:
        """Return the number of units to trade given current equity and signal strength."""
        alloc = equity * self.position_size_pct * strength
        # Ensure we don't allocate more cash than we have (long side only check)
        alloc = min(alloc, self.cash)
        return alloc / price if price > 0 else 0.0

    def _apply_buy(self, fill: FillEvent) -> None:
        symbol = fill.symbol
        if symbol in self.positions and self.positions[symbol].side == OrderSide.BUY:
            # Scale into an existing long — update average price
            pos = self.positions[symbol]
            total_cost = pos.avg_price * pos.quantity + fill.fill_price * fill.quantity
            pos.quantity += fill.quantity
            pos.avg_price = total_cost / pos.quantity
        else:
            # New long position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=fill.quantity,
                avg_price=fill.fill_price,
            )
            self._entry_times[symbol] = fill.timestamp
            self._entry_commissions[symbol] = fill.commission

    def _apply_sell(self, fill: FillEvent) -> None:
        symbol = fill.symbol

        if symbol not in self.positions:
            # Opening a short
            self.positions[symbol] = Position(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=fill.quantity,
                avg_price=fill.fill_price,
            )
            self._entry_times[symbol] = fill.timestamp
            self._entry_commissions[symbol] = fill.commission
            return

        pos = self.positions[symbol]

        if pos.side == OrderSide.BUY:
            # Closing (or partially closing) a long
            closed_qty = min(fill.quantity, pos.quantity)
            pnl = closed_qty * (fill.fill_price - pos.avg_price) - fill.commission
            pnl -= self._entry_commissions.get(symbol, 0.0) * (closed_qty / pos.quantity)

            self.trade_log.append(TradeRecord(
                symbol=symbol,
                side="LONG",
                entry_time=self._entry_times.get(symbol, fill.timestamp),
                exit_time=fill.timestamp,
                entry_price=pos.avg_price,
                exit_price=fill.fill_price,
                quantity=closed_qty,
                pnl=pnl,
                commission=fill.commission,
            ))

            pos.quantity -= closed_qty
            if pos.quantity <= 1e-9:
                del self.positions[symbol]
                self._entry_times.pop(symbol, None)
                self._entry_commissions.pop(symbol, None)

        else:
            # Covering a short
            closed_qty = min(fill.quantity, pos.quantity)
            pnl = closed_qty * (pos.avg_price - fill.fill_price) - fill.commission
            pnl -= self._entry_commissions.get(symbol, 0.0) * (closed_qty / pos.quantity)

            self.trade_log.append(TradeRecord(
                symbol=symbol,
                side="SHORT",
                entry_time=self._entry_times.get(symbol, fill.timestamp),
                exit_time=fill.timestamp,
                entry_price=pos.avg_price,
                exit_price=fill.fill_price,
                quantity=closed_qty,
                pnl=pnl,
                commission=fill.commission,
            ))

            pos.quantity -= closed_qty
            if pos.quantity <= 1e-9:
                del self.positions[symbol]
                self._entry_times.pop(symbol, None)
                self._entry_commissions.pop(symbol, None)

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
