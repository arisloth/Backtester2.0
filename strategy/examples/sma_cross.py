"""
strategy/examples/sma_cross.py — Simple moving average crossover strategy.

Entry rules:
  - BUY  when fast SMA crosses above slow SMA (golden cross)
  - EXIT when fast SMA crosses below slow SMA (death cross)

This is the smoke-test strategy. Run it on SPY 2020–2024 to verify the
engine, portfolio, broker, and data feed all wire together correctly.

Usage:
    from strategy.examples.sma_cross import SMACrossStrategy
    strategy = SMACrossStrategy(symbol="SPY", fast=50, slow=200)
"""

from collections import deque
from typing import Optional

from core.event import FillEvent, MarketEvent, SignalEvent, SignalDirection
from strategy.base import Strategy


class SMACrossStrategy(Strategy):
    """
    SMA crossover on a single symbol.

    Parameters
    ----------
    symbol : str
        The ticker to trade (must match the data feed symbol).
    fast : int
        Fast SMA period in bars (default 50).
    slow : int
        Slow SMA period in bars (default 200).
    asset_class : str
        Passed through to SignalEvent (default "stock").
    """

    def __init__(
        self,
        symbol: str,
        fast: int = 50,
        slow: int = 200,
        asset_class: str = "stock",
    ):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")

        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.asset_class = asset_class

        # Rolling close price buffers — only keep as many as needed
        self._closes: deque = deque(maxlen=slow)

        # Track whether we're currently in a position to avoid repeat signals
        self._in_position: bool = False

        # Previous bar's SMA values to detect crossovers
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None

    def on_bar(self, market_event: MarketEvent) -> Optional[SignalEvent]:
        # Ignore bars for other symbols (multi-asset engine support)
        if market_event.symbol != self.symbol:
            return None

        self._closes.append(market_event.close)

        # Not enough data yet to compute both SMAs
        if len(self._closes) < self.slow:
            return None

        fast_sma = sum(list(self._closes)[-self.fast:]) / self.fast
        slow_sma = sum(self._closes) / self.slow

        signal = None

        if self._prev_fast is not None and self._prev_slow is not None:
            was_above = self._prev_fast > self._prev_slow
            is_above  = fast_sma > slow_sma

            # Golden cross: fast crosses above slow → go long
            if not was_above and is_above and not self._in_position:
                signal = SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=market_event.timestamp,
                    direction=SignalDirection.LONG,
                    strategy_id="sma_cross",
                )

            # Death cross: fast crosses below slow → exit
            elif was_above and not is_above and self._in_position:
                signal = SignalEvent(
                    symbol=self.symbol,
                    asset_class=self.asset_class,
                    timestamp=market_event.timestamp,
                    direction=SignalDirection.EXIT,
                    strategy_id="sma_cross",
                )

        self._prev_fast = fast_sma
        self._prev_slow = slow_sma

        return signal

    def on_fill(self, fill_event: FillEvent) -> None:
        if fill_event.symbol != self.symbol:
            return
        # Toggle position flag based on what just filled
        from core.event import OrderSide
        if fill_event.side == OrderSide.BUY:
            self._in_position = True
        else:
            self._in_position = False
