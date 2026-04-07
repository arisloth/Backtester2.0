"""
strategy/base.py — Abstract Strategy interface.

All strategies inherit from Strategy. The engine calls:
  - on_bar(market_event)  → SignalEvent | None  (once per bar)
  - on_fill(fill_event)   → None                (once per fill)

Strategies must be self-contained. They communicate only via events —
no direct imports from core/, no access to portfolio state, no future data.
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.event import FillEvent, MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses should maintain their own internal state (e.g. price history,
    indicator values, position flags) and update it inside on_bar / on_fill.
    """

    @abstractmethod
    def on_bar(self, market_event: MarketEvent) -> Optional[SignalEvent]:
        """
        Called once per bar for each symbol. Receives the current bar's
        OHLCV data and returns a SignalEvent if the strategy wants to act,
        or None to do nothing.

        Parameters
        ----------
        market_event : MarketEvent
            The current bar. This is the only data the strategy is allowed
            to act on — no future bars, no external lookups.

        Returns
        -------
        SignalEvent | None
        """

    @abstractmethod
    def on_fill(self, fill_event: FillEvent) -> None:
        """
        Called when one of this strategy's orders is filled. Use this to
        update internal state (e.g. flip an in-position flag, record entry
        price for stop-loss calculation).

        Parameters
        ----------
        fill_event : FillEvent
            The completed fill from the broker.
        """
