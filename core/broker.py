"""
core/broker.py — Simulated order execution with realistic fill logic.

Responsibilities:
- Apply slippage via a pluggable FillModel.
- Apply commissions via a pluggable CostModel.
- Enforce fill rules:
    - Market orders fill at next bar's open (caller passes current_bars which
      is the bar *after* the signal bar).
    - Limit orders fill only if price trades *through* the limit (not just touches).
    - Partial fills supported via fill_ratio (0.0–1.0).
"""

import logging
from typing import Dict, Optional

from core.event import FillEvent, OrderEvent, OrderSide, OrderType

logger = logging.getLogger(__name__)


class Broker:
    """
    Simulated broker. Wires together a FillModel (slippage) and a CostModel
    (commissions) to produce realistic FillEvents.

    Parameters
    ----------
    fill_model : FillModel
        Instance from execution/fill_model.py. Computes slippage.
    cost_model : CostModel
        Instance from execution/cost_model.py. Computes commission.
    fill_ratio : float
        Fraction of requested quantity that gets filled (1.0 = full fill).
        Set below 1.0 to simulate illiquid assets.
    """

    def __init__(self, fill_model, cost_model, fill_ratio: float = 1.0):
        self.fill_model = fill_model
        self.cost_model = cost_model

        if not (0.0 < fill_ratio <= 1.0):
            raise ValueError(f"fill_ratio must be in (0, 1], got {fill_ratio}")
        self.fill_ratio = fill_ratio

    def execute_order(
        self,
        order: OrderEvent,
        current_bars: Dict[str, dict],
    ) -> Optional[FillEvent]:
        """
        Attempt to fill an order against the current bar.

        Parameters
        ----------
        order : OrderEvent
            The order to fill.
        current_bars : dict[symbol -> bar dict]
            The bar being processed (next bar's OHLCV for market orders).
            Bar dict must have keys: open, high, low, close, volume, timestamp.

        Returns
        -------
        FillEvent if filled, None if the order cannot be filled this bar
        (e.g. limit not triggered).
        """
        bar = current_bars.get(order.symbol)
        if bar is None:
            logger.warning(f"No bar data for {order.symbol} — order not filled.")
            return None

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar)
        else:
            logger.error(f"Unknown order type: {order.order_type}")
            return None

    # ------------------------------------------------------------------
    # Fill logic
    # ------------------------------------------------------------------

    def _fill_market(self, order: OrderEvent, bar: dict) -> FillEvent:
        """
        Market orders fill at the bar's open price (next bar after signal).
        Slippage is applied on top of the open.
        """
        base_price = bar["open"]
        filled_qty = order.quantity * self.fill_ratio

        slippage_amount = self.fill_model.calculate(
            base_price=base_price,
            side=order.side,
            quantity=filled_qty,
            bar=bar,
        )
        fill_price = base_price + slippage_amount

        commission = self.cost_model.calculate(
            fill_price=fill_price,
            quantity=filled_qty,
            asset_class=order.asset_class,
        )

        return FillEvent(
            symbol=order.symbol,
            asset_class=order.asset_class,
            timestamp=bar["timestamp"],
            side=order.side,
            quantity=filled_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(slippage_amount) * filled_qty,
        )

    def _fill_limit(self, order: OrderEvent, bar: dict) -> Optional[FillEvent]:
        """
        Limit orders fill only if the bar trades *through* the limit price
        (i.e. low < limit for buys, high > limit for sells).
        Fills at the limit price (no additional slippage — the limit is the
        worst acceptable price, and we assume we get it if triggered).
        """
        limit = order.limit_price
        if limit is None:
            logger.error(f"LIMIT order for {order.symbol} has no limit_price — rejected.")
            return None

        bar_low = bar["low"]
        bar_high = bar["high"]

        if order.side == OrderSide.BUY:
            # Buy limit triggers only if price trades *below* the limit
            if bar_low >= limit:
                logger.debug(f"Buy limit {limit} not triggered (low={bar_low}).")
                return None
        else:
            # Sell limit triggers only if price trades *above* the limit
            if bar_high <= limit:
                logger.debug(f"Sell limit {limit} not triggered (high={bar_high}).")
                return None

        filled_qty = order.quantity * self.fill_ratio
        fill_price = limit  # filled at the limit, no extra slippage

        commission = self.cost_model.calculate(
            fill_price=fill_price,
            quantity=filled_qty,
            asset_class=order.asset_class,
        )

        return FillEvent(
            symbol=order.symbol,
            asset_class=order.asset_class,
            timestamp=bar["timestamp"],
            side=order.side,
            quantity=filled_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=0.0,  # limit order: no slippage beyond the limit itself
        )
