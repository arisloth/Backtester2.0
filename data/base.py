"""
data/base.py — Abstract DataHandler interface.

All data feeds (yfinance, Alpaca, CCXT, forex) implement this interface.
The engine only ever talks to DataHandler — never directly to any data source.

Required methods:
    has_more()       → bool
    update_bars(q)   → None   (pushes MarketEvents onto the event queue)
    current_bars()   → dict   (returns latest bar dict per symbol, for broker fill logic)
"""

from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict


class DataHandler(ABC):
    """
    Abstract base class for all data feeds.

    Subclasses must load or stream data and expose it one bar at a time
    so the engine can process events sequentially without lookahead.
    """

    @abstractmethod
    def has_more(self) -> bool:
        """
        Return True if there are more bars to process, False when exhausted.
        The engine calls this at the top of every iteration.
        """

    @abstractmethod
    def update_bars(self, events: Queue) -> None:
        """
        Advance by one bar and push a MarketEvent onto the event queue for
        each symbol. Called once per engine loop iteration.

        Parameters
        ----------
        events : Queue
            The engine's central event queue. Push MarketEvent instances here.
        """

    @abstractmethod
    def current_bars(self) -> Dict[str, dict]:
        """
        Return the most recently emitted bar for each symbol as a plain dict.

        Used by the broker to fill orders against the current bar's OHLCV.
        Dict format per symbol:
            {
                "timestamp": pd.Timestamp,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "symbol": str,
                "asset_class": str,
            }
        """
