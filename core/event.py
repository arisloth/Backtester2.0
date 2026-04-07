"""
core/event.py — Event types for the backtester event loop.

Flow: MarketEvent → Strategy → SignalEvent → Portfolio → OrderEvent → Broker → FillEvent → Portfolio
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MarketEvent:
    """
    Fired once per bar for each symbol. Carries the normalized OHLCV bar
    that the strategy is allowed to act on. Strategy must never see any
    bar beyond the one in this event (no lookahead).
    """
    type: EventType = field(default=EventType.MARKET, init=False)

    symbol: str
    asset_class: str          # "stock" | "crypto" | "forex"
    timestamp: pd.Timestamp   # UTC bar open time
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SignalEvent:
    """
    Emitted by a strategy after processing a MarketEvent. Expresses intent
    to go long, short, or exit — not a concrete order yet. Portfolio converts
    this into an OrderEvent with proper sizing.
    """
    type: EventType = field(default=EventType.SIGNAL, init=False)

    symbol: str
    asset_class: str
    timestamp: pd.Timestamp
    direction: SignalDirection
    strength: float = 1.0     # 0.0–1.0, used for position sizing if desired
    strategy_id: str = ""     # tag for multi-strategy runs
    # Optional metadata passed through to the trade log
    stop_price:  float = None   # planned stop loss price
    tp_price:    float = None   # planned take profit price
    exit_reason: str   = ""     # e.g. "stop", "tp", "signal"


@dataclass
class OrderEvent:
    """
    Concrete order produced by the portfolio after sizing a SignalEvent.
    Sent to the broker for simulated execution.
    """
    type: EventType = field(default=EventType.ORDER, init=False)

    symbol: str
    asset_class: str
    timestamp: pd.Timestamp
    order_type: OrderType     # MARKET or LIMIT
    side: OrderSide           # BUY or SELL
    quantity: float           # number of shares/units (always positive)
    limit_price: Optional[float] = None        # required if order_type == LIMIT
    fill_price_override: Optional[float] = None  # stop/TP exit price; broker applies gap protection
    exit_reason: str = ""                        # "stop" | "tp" | "" — used by broker for gap logic


@dataclass
class FillEvent:
    """
    Returned by the broker after simulating order execution. Contains the
    actual fill price (after slippage) and total cost (commission included).
    Portfolio uses this to update positions and the equity curve.
    """
    type: EventType = field(default=EventType.FILL, init=False)

    symbol: str
    asset_class: str
    timestamp: pd.Timestamp   # bar at which the fill occurred
    side: OrderSide
    quantity: float           # units filled (may be < order quantity for partial fills)
    fill_price: float         # price after slippage
    commission: float         # total commission cost in base currency
    slippage: float           # slippage cost in base currency (informational)

    @property
    def trade_value(self) -> float:
        """Gross value of the fill, before commission."""
        return self.quantity * self.fill_price

    @property
    def net_cost(self) -> float:
        """
        Net cash impact of the fill.
        Positive = cash outflow (buy), negative = cash inflow (sell).
        """
        if self.side == OrderSide.BUY:
            return self.trade_value + self.commission
        else:
            return -(self.trade_value - self.commission)
