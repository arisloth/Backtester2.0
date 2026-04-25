"""
tests/test_broker.py — Unit tests for Broker, FillModel, and CostModel.
"""

import unittest

import pandas as pd

from core.broker import Broker
from core.event import OrderEvent, OrderSide, OrderType
from core.portfolio import Portfolio
from execution.fill_model import FixedSlippage, VolatilitySlippage, VolumeImpactSlippage
from execution.cost_model import (
    ZeroCommission, PerShareCommission, PercentCommission, SpreadCommission
)


def _bar(o=100.0, h=105.0, l=95.0, c=101.0, v=1_000_000.0, ts="2024-01-02"):
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "symbol": "SPY", "asset_class": "stock",
    }


def _order(side=OrderSide.BUY, qty=10.0, order_type=OrderType.MARKET, limit=None,
           fill_price_override=None, exit_reason=""):
    return OrderEvent(
        symbol="SPY", asset_class="stock",
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        order_type=order_type, side=side, quantity=qty, limit_price=limit,
        fill_price_override=fill_price_override, exit_reason=exit_reason,
    )


class TestBrokerMarketOrders(unittest.TestCase):

    def test_market_buy_fills_at_open(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(_order(OrderSide.BUY), {"SPY": _bar(o=100.0)})
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 100.0)

    def test_market_sell_fills_at_open(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(_order(OrderSide.SELL), {"SPY": _bar(o=100.0)})
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 100.0)

    def test_no_bar_returns_none(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(_order(), {})  # empty bars dict
        self.assertIsNone(fill)

    def test_fill_ratio_reduces_quantity(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission(), fill_ratio=0.5)
        fill = broker.execute_order(_order(qty=10.0), {"SPY": _bar()})
        self.assertAlmostEqual(fill.quantity, 5.0)

    def test_invalid_fill_ratio_raises(self):
        with self.assertRaises(ValueError):
            Broker(FixedSlippage(pct=0.0), ZeroCommission(), fill_ratio=0.0)
        with self.assertRaises(ValueError):
            Broker(FixedSlippage(pct=0.0), ZeroCommission(), fill_ratio=1.5)

    def test_long_stop_exit_requires_bar_touch(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.SELL,
            fill_price_override=95.0,
            exit_reason="stop",
        )

        with self.assertRaises(ValueError):
            broker.execute_order(order, {"SPY": _bar(o=100.0, h=104.0, l=96.0)})

    def test_long_stop_exit_fills_when_bar_touches_stop(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.SELL,
            fill_price_override=95.0,
            exit_reason="stop",
        )
        fill = broker.execute_order(order, {"SPY": _bar(o=100.0, h=104.0, l=95.0)})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 95.0)

    def test_short_stop_exit_requires_bar_touch(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.BUY,
            fill_price_override=105.0,
            exit_reason="stop",
        )

        with self.assertRaises(ValueError):
            broker.execute_order(order, {"SPY": _bar(o=100.0, h=104.0, l=96.0)})

    def test_short_stop_exit_fills_when_bar_touches_stop(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.BUY,
            fill_price_override=105.0,
            exit_reason="stop",
        )
        fill = broker.execute_order(order, {"SPY": _bar(o=100.0, h=105.0, l=96.0)})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 105.0)

    def test_long_stop_gap_down_fills_at_open(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.SELL,
            fill_price_override=95.0,
            exit_reason="stop",
        )
        fill = broker.execute_order(order, {"SPY": _bar(o=90.0, h=94.0, l=89.0)})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 90.0)

    def test_short_stop_gap_up_fills_at_open(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(
            OrderSide.BUY,
            fill_price_override=105.0,
            exit_reason="stop",
        )
        fill = broker.execute_order(order, {"SPY": _bar(o=110.0, h=111.0, l=106.0)})

        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 110.0)

    def test_stop_exit_requires_stop_price(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        order = _order(OrderSide.SELL, exit_reason="stop")

        with self.assertRaises(ValueError):
            broker.execute_order(order, {"SPY": _bar()})

    def test_zero_volume_market_order_does_not_fill(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(_order(OrderSide.BUY), {"SPY": _bar(v=0.0)})

        self.assertIsNone(fill)

    def test_zero_volume_limit_order_does_not_fill(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(
            _order(OrderSide.BUY, order_type=OrderType.LIMIT, limit=97.0),
            {"SPY": _bar(l=95.0, v=0.0)},
        )

        self.assertIsNone(fill)

    def test_below_min_volume_order_does_not_fill(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission(), min_fill_volume=1_000.0)
        fill = broker.execute_order(_order(OrderSide.BUY), {"SPY": _bar(v=999.0)})

        self.assertIsNone(fill)

    def test_invalid_min_fill_volume_raises(self):
        with self.assertRaises(ValueError):
            Broker(FixedSlippage(pct=0.0), ZeroCommission(), min_fill_volume=-1.0)

    def test_consecutive_partial_fills_sum_to_order_quantity_and_cash(self):
        broker = Broker(
            FixedSlippage(pct=0.0),
            PerShareCommission(rate=0.10, minimum=0.0),
            fill_ratio=0.5,
        )
        portfolio = Portfolio(initial_capital=10_000.0)
        order = _order(OrderSide.BUY, qty=10.0)
        bars = {"SPY": _bar(o=100.0)}

        first_fill = broker.execute_order(order, bars)
        second_fill = broker.execute_order(order, bars)
        portfolio.update_fill(first_fill)
        portfolio.update_fill(second_fill)

        pos = portfolio.positions["SPY"]
        self.assertAlmostEqual(first_fill.quantity, 5.0)
        self.assertAlmostEqual(second_fill.quantity, 5.0)
        self.assertAlmostEqual(pos.quantity, 10.0)
        self.assertAlmostEqual(pos.avg_price, 100.0)
        self.assertAlmostEqual(portfolio.cash, 8_999.0)


class TestBrokerLimitOrders(unittest.TestCase):

    def test_buy_limit_triggers_when_low_below_limit(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        # bar low=95, limit=97 → low trades through limit → should fill
        fill = broker.execute_order(
            _order(OrderSide.BUY, order_type=OrderType.LIMIT, limit=97.0),
            {"SPY": _bar(l=95.0)}
        )
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 97.0)

    def test_buy_limit_does_not_trigger_when_low_above_limit(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        # bar low=98, limit=97 → price never reaches limit → no fill
        fill = broker.execute_order(
            _order(OrderSide.BUY, order_type=OrderType.LIMIT, limit=97.0),
            {"SPY": _bar(l=98.0)}
        )
        self.assertIsNone(fill)

    def test_sell_limit_triggers_when_high_above_limit(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        # bar high=105, limit=103 → high trades through limit → should fill
        fill = broker.execute_order(
            _order(OrderSide.SELL, order_type=OrderType.LIMIT, limit=103.0),
            {"SPY": _bar(h=105.0)}
        )
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 103.0)

    def test_sell_limit_does_not_trigger_when_high_below_limit(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        # bar high=102, limit=103 → price never reaches limit → no fill
        fill = broker.execute_order(
            _order(OrderSide.SELL, order_type=OrderType.LIMIT, limit=103.0),
            {"SPY": _bar(h=102.0)}
        )
        self.assertIsNone(fill)

    def test_limit_order_missing_price_returns_none(self):
        broker = Broker(FixedSlippage(pct=0.0), ZeroCommission())
        fill = broker.execute_order(
            _order(OrderSide.BUY, order_type=OrderType.LIMIT, limit=None),
            {"SPY": _bar()}
        )
        self.assertIsNone(fill)


class TestSlippageModels(unittest.TestCase):

    def test_fixed_slippage_buy_increases_price(self):
        model = FixedSlippage(pct=0.01)
        adj = model.calculate(base_price=100.0, side=OrderSide.BUY, quantity=1, bar=_bar())
        self.assertAlmostEqual(adj, 1.0)  # 1% of 100

    def test_fixed_slippage_sell_decreases_price(self):
        model = FixedSlippage(pct=0.01)
        adj = model.calculate(base_price=100.0, side=OrderSide.SELL, quantity=1, bar=_bar())
        self.assertAlmostEqual(adj, -1.0)

    def test_fixed_slippage_zero(self):
        model = FixedSlippage(pct=0.0)
        adj = model.calculate(base_price=100.0, side=OrderSide.BUY, quantity=1, bar=_bar())
        self.assertAlmostEqual(adj, 0.0)

    def test_volatility_slippage_scales_with_range(self):
        model = VolatilitySlippage(atr_multiplier=0.5)
        bar_wide  = _bar(h=110.0, l=90.0)   # range = 20
        bar_narrow = _bar(h=101.0, l=99.0)  # range = 2
        adj_wide   = model.calculate(100.0, OrderSide.BUY, 1, bar_wide)
        adj_narrow = model.calculate(100.0, OrderSide.BUY, 1, bar_narrow)
        self.assertGreater(adj_wide, adj_narrow)

    def test_volatility_slippage_flat_bar_is_zero(self):
        model = VolatilitySlippage(atr_multiplier=0.5)
        bar_flat = _bar(h=100.0, l=100.0)
        adj = model.calculate(100.0, OrderSide.BUY, 1, bar_flat)
        self.assertAlmostEqual(adj, 0.0)

    def test_volume_impact_increases_with_order_size(self):
        model = VolumeImpactSlippage(base_pct=0.0005, impact_factor=0.1)
        bar = _bar(v=1_000_000.0)
        small_adj = model.calculate(100.0, OrderSide.BUY, 100,     bar)
        large_adj = model.calculate(100.0, OrderSide.BUY, 100_000, bar)
        self.assertGreater(large_adj, small_adj)

    def test_volume_impact_no_volume_falls_back_to_base(self):
        model = VolumeImpactSlippage(base_pct=0.01, impact_factor=0.5)
        bar_no_vol = _bar(v=0.0)
        adj = model.calculate(100.0, OrderSide.BUY, 100, bar_no_vol)
        self.assertAlmostEqual(adj, 1.0)  # base_pct=0.01 * 100


class TestCostModels(unittest.TestCase):

    def test_zero_commission(self):
        model = ZeroCommission()
        self.assertAlmostEqual(model.calculate(100.0, 10, "stock"), 0.0)

    def test_per_share_minimum_applies(self):
        model = PerShareCommission(rate=0.005, minimum=1.0)
        # 10 shares * $0.005 = $0.05 → below minimum → should return $1.00
        self.assertAlmostEqual(model.calculate(100.0, 10, "stock"), 1.0)

    def test_per_share_above_minimum(self):
        model = PerShareCommission(rate=0.005, minimum=1.0)
        # 500 shares * $0.005 = $2.50 → above minimum
        self.assertAlmostEqual(model.calculate(100.0, 500, "stock"), 2.50)

    def test_percent_commission(self):
        model = PercentCommission(default_pct=0.001)
        # 0.1% of $100 * 10 = $1.00
        self.assertAlmostEqual(model.calculate(100.0, 10, "stock"), 1.0)

    def test_percent_commission_per_asset_class(self):
        model = PercentCommission(default_pct=0.001, rates={"crypto": 0.002, "stock": 0.0})
        self.assertAlmostEqual(model.calculate(100.0, 10, "stock"),  0.0)
        self.assertAlmostEqual(model.calculate(100.0, 10, "crypto"), 2.0)

    def test_spread_commission(self):
        model = SpreadCommission(spread_pips=2.0, pip_value=0.0001)
        # half_spread = 1 pip = 0.0001, * 10000 units = 1.0
        self.assertAlmostEqual(model.calculate(1.10, 10_000, "forex"), 1.0)


class TestFillEventProperties(unittest.TestCase):

    def test_net_cost_buy(self):
        from core.event import FillEvent
        fill = FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            side=OrderSide.BUY, quantity=10.0,
            fill_price=100.0, commission=5.0, slippage=0.0,
        )
        # Buy: trade_value + commission = 1000 + 5 = 1005
        self.assertAlmostEqual(fill.net_cost, 1005.0)

    def test_net_cost_sell(self):
        from core.event import FillEvent
        fill = FillEvent(
            symbol="SPY", asset_class="stock",
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            side=OrderSide.SELL, quantity=10.0,
            fill_price=100.0, commission=5.0, slippage=0.0,
        )
        # Sell: -(trade_value - commission) = -(1000 - 5) = -995
        self.assertAlmostEqual(fill.net_cost, -995.0)


if __name__ == "__main__":
    unittest.main()
