"""
core/engine.py — Main event loop for the backtester.

Drives the simulation bar-by-bar. On each bar:
  1. DataHandler yields MarketEvents (one per symbol).
  2. Each MarketEvent is passed to every Strategy.
  3. SignalEvents are queued and passed to the Portfolio for sizing.
  4. OrderEvents are queued and passed to the Broker for execution.
  5. FillEvents are passed back to Portfolio and the originating Strategy.

Nothing in this file knows about yfinance, Alpaca, specific strategies, or
commission models — it only talks to the abstract interfaces.
"""

import queue
import logging
from typing import List

from core.event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger(__name__)


class Engine:
    """
    Event-driven backtest engine.

    Parameters
    ----------
    data_handler : DataHandler
        Implements data/base.py DataHandler. Yields bars one at a time.
    strategies : list[Strategy]
        One or more strategy instances (strategy/base.py Strategy).
    portfolio : Portfolio
        Tracks positions, cash, and the equity curve (core/portfolio.py).
    broker : Broker
        Simulates order execution with slippage + commissions (core/broker.py).
    """

    def __init__(self, data_handler, strategies, portfolio, broker):
        self.data_handler = data_handler
        self.strategies = strategies if isinstance(strategies, list) else [strategies]
        self.portfolio = portfolio
        self.broker = broker

        # Central event queue — all components communicate through this.
        self.events: queue.Queue = queue.Queue()

        self.running = False
        self.bar_count = 0

    def run(self) -> None:
        """
        Execute the full backtest. Loops until the data handler is exhausted.
        """
        logger.info("Backtest started.")
        self.running = True

        while self.running:
            # Advance the data handler by one bar. If exhausted, stop.
            if not self.data_handler.has_more():
                logger.info(f"Data exhausted after {self.bar_count} bars. Backtest complete.")
                self.running = False
                break

            # Step 1: generate MarketEvents for this bar (one per symbol).
            self.data_handler.update_bars(self.events)
            self.bar_count += 1

            # Step 2: drain the event queue.
            self._process_events()

        # Let the portfolio finalize the equity curve / close open positions.
        self.portfolio.finalize()
        logger.info("Backtest finished.")

    def _process_events(self) -> None:
        """Drain the event queue, routing each event to the right handler."""
        while not self.events.empty():
            event = self.events.get(block=False)

            if event.type == EventType.MARKET:
                self._handle_market(event)

            elif event.type == EventType.SIGNAL:
                self._handle_signal(event)

            elif event.type == EventType.ORDER:
                self._handle_order(event)

            elif event.type == EventType.FILL:
                self._handle_fill(event)

            else:
                logger.warning(f"Unknown event type: {event.type}")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_market(self, event: MarketEvent) -> None:
        """
        Pass the bar to every strategy. Each strategy may emit a SignalEvent,
        which gets pushed onto the queue for the next iteration of the drain loop.
        """
        # Update portfolio's view of current prices (for mark-to-market).
        self.portfolio.update_market(event)

        for strategy in self.strategies:
            signal = strategy.on_bar(event)
            if signal is not None:
                logger.debug(f"Signal: {signal.symbol} {signal.direction} @ {event.timestamp}")
                self.events.put(signal)

    def _handle_signal(self, event: SignalEvent) -> None:
        """
        Portfolio converts the signal into a sized OrderEvent and pushes it
        onto the queue.
        """
        order = self.portfolio.generate_order(event)
        if order is not None:
            logger.debug(f"Order: {order.symbol} {order.side} {order.quantity} @ {event.timestamp}")
            self.events.put(order)

    def _handle_order(self, event: OrderEvent) -> None:
        """
        Broker simulates execution. If filled, a FillEvent is pushed onto
        the queue. If the order cannot be filled this bar (e.g. limit not
        reached), nothing is pushed.
        """
        fill = self.broker.execute_order(event, self.data_handler.current_bars())
        if fill is not None:
            logger.debug(
                f"Fill: {fill.symbol} {fill.side} {fill.quantity} "
                f"@ {fill.fill_price:.4f} (slip={fill.slippage:.4f}, comm={fill.commission:.4f})"
            )
            self.events.put(fill)

    def _handle_fill(self, event: FillEvent) -> None:
        """
        Portfolio updates positions and equity curve. Each strategy's on_fill
        hook is called so it can track its own state (e.g. mark as in-position).
        """
        self.portfolio.update_fill(event)

        for strategy in self.strategies:
            strategy.on_fill(event)
