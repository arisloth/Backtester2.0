"""
tests/test_engine.py — Unit tests for the event loop and portfolio.

Tests use a synthetic in-memory data feed so no network calls are made.
"""

import queue
import unittest
from typing import Dict, List, Optional

import pandas as pd

from core.engine import Engine
from core.event import (
    FillEvent, MarketEvent, OrderSide, OrderType,
    SignalDirection, SignalEvent,
)
from core.portfolio import Portfolio
from core.broker import Broker
from data.base import DataHandler
from execution.fill_model import FixedSlippage
from execution.cost_model import ZeroCommission
from strategy.base import Strategy


# ------------------------------------------------------------------
# Helpers — synthetic data feed and deterministic strategy
# ------------------------------------------------------------------

def _make_bar(symbol: str, ts: str, o=100.0, h=102.0, l=99.0, c=101.0, v=1_000_000.0) -> dict:
    return {
        "timestamp":   pd.Timestamp(ts, tz="UTC"),
        "open":  o, "high": h, "low": l, "close": c, "volume": v,
        "symbol": symbol, "asset_class": "stock",
    }


class SyntheticFeed(DataHandler):
    """Replays a pre-built list of bar dicts, one per call to update_bars."""

    def __init__(self, bars: List[dict]):
        self._bars = bars
        self._index = 0
        self._current: Dict[str, dict] = {}

    def has_more(self) -> bool:
        return self._index < len(self._bars)

    def update_bars(self, events: queue.Queue) -> None:
        bar = self._bars[self._index]
        self._current[bar["symbol"]] = bar
        events.put(MarketEvent(
            symbol=bar["symbol"], asset_class=bar["asset_class"],
            timestamp=bar["timestamp"],
            open=bar["open"], high=bar["high"],
            low=bar["low"],  close=bar["close"],
            volume=bar["volume"],
        ))
        self._index += 1

    def current_bars(self) -> Dict[str, dict]:
        return self._current


class BuyOnBarOneStrategy(Strategy):
    """Emits a single LONG signal on the first bar, then EXIT on bar 3."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bar_count = 0
        self._in_position = False

    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        if event.symbol != self.symbol:
            return None
        self._bar_count += 1
        if self._bar_count == 1 and not self._in_position:
            return SignalEvent(symbol=self.symbol, asset_class="stock",
                               timestamp=event.timestamp, direction=SignalDirection.LONG)
        if self._bar_count == 3 and self._in_position:
            return SignalEvent(symbol=self.symbol, asset_class="stock",
                               timestamp=event.timestamp, direction=SignalDirection.EXIT)
        return None

    def on_fill(self, fill: FillEvent) -> None:
        if fill.symbol != self.symbol:
            return
        self._in_position = fill.side == OrderSide.BUY


class NeverSignalStrategy(Strategy):
    def on_bar(self, event): return None
    def on_fill(self, fill): pass


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestEngine(unittest.TestCase):

    def _build(self, bars, strategy=None, capital=10_000.0):
        feed = SyntheticFeed(bars)
        strat = strategy or NeverSignalStrategy()
        portfolio = Portfolio(initial_capital=capital, position_size_pct=0.9)
        broker = Broker(fill_model=FixedSlippage(pct=0.0), cost_model=ZeroCommission())
        engine = Engine(data_handler=feed, strategies=strat, portfolio=portfolio, broker=broker)
        return engine, portfolio

    def test_bar_count(self):
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(5)]
        engine, _ = self._build(bars)
        engine.run()
        self.assertEqual(engine.bar_count, 5)

    def test_no_signal_no_position(self):
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(5)]
        engine, portfolio = self._build(bars)
        engine.run()
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(len(portfolio.trade_log), 0)

    def test_equity_curve_length(self):
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(5)]
        engine, portfolio = self._build(bars)
        engine.run()
        # One equity snapshot per bar
        self.assertEqual(len(portfolio.equity_curve), 5)

    def test_initial_equity_equals_capital(self):
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(3)]
        engine, portfolio = self._build(bars, capital=50_000.0)
        engine.run()
        self.assertAlmostEqual(portfolio.equity_curve[0][1], 50_000.0)

    def test_buy_signal_opens_position(self):
        # 5 bars; strategy buys bar 1, exits bar 3
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(5)]
        strategy = BuyOnBarOneStrategy("SPY")
        engine, portfolio = self._build(bars, strategy=strategy)
        engine.run()
        # After exit on bar 3, trade log should have 1 completed trade
        self.assertEqual(len(portfolio.trade_log), 1)

    def test_completed_trade_pnl_sign(self):
        # Prices rise bar-by-bar so the long trade should be profitable
        bars = [
            _make_bar("SPY", "2024-01-01", o=100, h=101, l=99,  c=100),
            _make_bar("SPY", "2024-01-02", o=101, h=102, l=100, c=101),  # fill BUY here
            _make_bar("SPY", "2024-01-03", o=105, h=106, l=104, c=105),
            _make_bar("SPY", "2024-01-04", o=108, h=109, l=107, c=108),  # fill SELL here
            _make_bar("SPY", "2024-01-05", o=110, h=111, l=109, c=110),
        ]
        strategy = BuyOnBarOneStrategy("SPY")
        engine, portfolio = self._build(bars, strategy=strategy, capital=10_000.0)
        engine.run()
        self.assertEqual(len(portfolio.trade_log), 1)
        self.assertGreater(portfolio.trade_log[0].pnl, 0)

    def test_multi_strategy(self):
        """Engine accepts a list of strategies and calls each."""
        bars = [_make_bar("SPY", f"2024-01-0{i+1}") for i in range(3)]
        s1 = NeverSignalStrategy()
        s2 = NeverSignalStrategy()
        feed = SyntheticFeed(bars)
        portfolio = Portfolio(initial_capital=10_000)
        broker = Broker(fill_model=FixedSlippage(pct=0.0), cost_model=ZeroCommission())
        engine = Engine(data_handler=feed, strategies=[s1, s2], portfolio=portfolio, broker=broker)
        engine.run()
        self.assertEqual(engine.bar_count, 3)


class TestPortfolio(unittest.TestCase):

    def test_cash_decreases_on_buy(self):
        portfolio = Portfolio(initial_capital=10_000, position_size_pct=1.0)
        # Manually fire a market event to set the price
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        fill = FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.BUY, quantity=10.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        )
        portfolio.update_fill(fill)
        self.assertLess(portfolio.cash, 10_000.0)

    def test_cash_increases_on_sell(self):
        portfolio = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        # Open a long first
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.BUY, quantity=10.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        cash_after_buy = portfolio.cash
        # Close it
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.SELL, quantity=10.0,
            fill_price=105.0, commission=0.0, slippage=0.0,
        ))
        self.assertGreater(portfolio.cash, cash_after_buy)

    def test_short_equity_marks_to_market(self):
        portfolio = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.SELL, quantity=10.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        self.assertAlmostEqual(portfolio.cash, 11_000.0)

        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            open=120, high=121, low=119, close=120, volume=1_000_000,
        ))
        self.assertAlmostEqual(portfolio.equity_curve[-1][1], 9_800.0)

        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-03", tz="UTC"),
            open=80, high=81, low=79, close=80, volume=1_000_000,
        ))
        self.assertAlmostEqual(portfolio.equity_curve[-1][1], 10_200.0)

    def test_short_position_size_respects_initial_margin(self):
        portfolio = Portfolio(
            initial_capital=10_000,
            position_size_pct=10.0,
            short_initial_margin=0.50,
        )
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))

        order = portfolio.generate_order(SignalEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            direction=SignalDirection.SHORT,
        ))

        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertAlmostEqual(order.quantity, 200.0)

    def test_short_position_size_uses_aggregate_margin(self):
        portfolio = Portfolio(
            initial_capital=10_000,
            position_size_pct=10.0,
            short_initial_margin=0.50,
        )
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.SELL, quantity=200.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        portfolio.update_market(MarketEvent(
            symbol="QQQ", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))

        order = portfolio.generate_order(SignalEvent(
            symbol="QQQ", asset_class="stock", timestamp=ts,
            direction=SignalDirection.SHORT,
        ))

        self.assertIsNone(order)

    def test_position_removed_after_full_close(self):
        portfolio = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.BUY, quantity=5.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.SELL, quantity=5.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        self.assertNotIn("SPY", portfolio.positions)

    def test_equity_series_returns_series(self):
        portfolio = Portfolio(initial_capital=10_000)
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        eq = portfolio.equity_series()
        self.assertIsInstance(eq, pd.Series)
        self.assertEqual(len(eq), 1)


class TestFillDispatch(unittest.TestCase):
    """
    Regression tests for the fill-dispatch logic in Portfolio.update_fill.

    Earlier code dispatched on `fill.side` alone: a BUY fill would unconditionally
    call `_apply_buy`, which on an existing SHORT silently overwrote it with a
    phantom LONG of the same quantity (no TradeRecord, equity inflated by
    +qty*price). These tests pin down the correct behavior.
    """

    def _open_short(self, portfolio, qty=10.0, price=100.0, ts=None):
        ts = ts or pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=price, high=price + 1, low=price - 1, close=price, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.SELL, quantity=qty,
            fill_price=price, commission=0.0, slippage=0.0,
        ))

    def test_close_short_records_short_trade(self):
        """Closing a short with a BUY fill should produce a SHORT TradeRecord, not silently flip to a phantom long."""
        portfolio = Portfolio(initial_capital=10_000)
        self._open_short(portfolio, qty=10.0, price=100.0)
        # Buy back at $90 → +$100 PnL on the short.
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            open=90, high=91, low=89, close=90, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.BUY, quantity=10.0,
            fill_price=90.0, commission=0.0, slippage=0.0,
        ))

        self.assertEqual(len(portfolio.trade_log), 1, "Expected exactly one completed trade")
        rec = portfolio.trade_log[0]
        self.assertEqual(rec.side, "SHORT")
        self.assertAlmostEqual(rec.entry_price, 100.0)
        self.assertAlmostEqual(rec.exit_price, 90.0)
        self.assertAlmostEqual(rec.pnl, 100.0)
        # Position must be fully gone — no phantom long left over.
        self.assertNotIn("SPY", portfolio.positions)

    def test_close_short_does_not_inflate_equity(self):
        """After a winning short closes, equity = initial + realized PnL — not double."""
        portfolio = Portfolio(initial_capital=10_000)
        self._open_short(portfolio, qty=10.0, price=100.0)
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            open=90, high=91, low=89, close=90, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.BUY, quantity=10.0,
            fill_price=90.0, commission=0.0, slippage=0.0,
        ))
        # Tick another bar to trigger an equity snapshot post-close.
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-03", tz="UTC"),
            open=90, high=91, low=89, close=90, volume=1_000_000,
        ))
        self.assertAlmostEqual(portfolio.cash, 10_100.0)
        self.assertAlmostEqual(portfolio.equity_curve[-1][1], 10_100.0)

    def test_close_long_with_excess_qty_flips_to_short(self):
        """A SELL fill larger than the open long should close it and open a residual short."""
        portfolio = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.BUY, quantity=10.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        # Sell 15 at $100: closes 10 long (flat PnL), opens 5 short.
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            side=OrderSide.SELL, quantity=15.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        self.assertEqual(len(portfolio.trade_log), 1)
        self.assertEqual(portfolio.trade_log[0].side, "LONG")
        self.assertAlmostEqual(portfolio.trade_log[0].quantity, 10.0)
        self.assertIn("SPY", portfolio.positions)
        self.assertEqual(portfolio.positions["SPY"].side, OrderSide.SELL)
        self.assertAlmostEqual(portfolio.positions["SPY"].quantity, 5.0)

    def test_scale_into_long_averages_price(self):
        """Two BUY fills on the same long should weighted-average the cost basis."""
        portfolio = Portfolio(initial_capital=100_000)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        portfolio.update_market(MarketEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            open=100, high=101, low=99, close=100, volume=1_000_000,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.BUY, quantity=10.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock", timestamp=ts,
            side=OrderSide.BUY, quantity=10.0,
            fill_price=110.0, commission=0.0, slippage=0.0,
        ))
        pos = portfolio.positions["SPY"]
        self.assertAlmostEqual(pos.quantity, 20.0)
        self.assertAlmostEqual(pos.avg_price, 105.0)
        self.assertEqual(len(portfolio.trade_log), 0, "Scaling in must not create a TradeRecord")

    def test_zero_qty_fill_is_noop(self):
        portfolio = Portfolio(initial_capital=10_000)
        cash_before = portfolio.cash
        portfolio.update_fill(FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            side=OrderSide.BUY, quantity=0.0,
            fill_price=100.0, commission=0.0, slippage=0.0,
        ))
        self.assertEqual(portfolio.cash, cash_before)
        self.assertNotIn("SPY", portfolio.positions)


if __name__ == "__main__":
    unittest.main()
