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
    min_fill_volume : float
        Minimum bar volume required to fill any order. A bar with volume <= 0
        is always treated as halted/unfillable.
    """

    def __init__(self, fill_model, cost_model, fill_ratio: float = 1.0,
                 min_fill_volume: float = 0.0):
        self.fill_model = fill_model
        self.cost_model = cost_model

        if not (0.0 < fill_ratio <= 1.0):
            raise ValueError(f"fill_ratio must be in (0, 1], got {fill_ratio}")
        if min_fill_volume < 0:
            raise ValueError(f"min_fill_volume must be >= 0, got {min_fill_volume}")
        self.fill_ratio = fill_ratio
        self.min_fill_volume = min_fill_volume

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
        if not self._bar_can_fill(order, bar):
            return None

        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar)
        else:
            logger.error(f"Unknown order type: {order.order_type}")
            return None

    def _bar_can_fill(self, order: OrderEvent, bar: dict) -> bool:
        volume = float(bar.get("volume", 0.0))
        if volume <= 0:
            logger.warning(f"{order.symbol} bar volume is zero — order not filled.")
            return False
        if volume < self.min_fill_volume:
            logger.warning(
                f"{order.symbol} bar volume {volume} below min_fill_volume "
                f"{self.min_fill_volume} — order not filled."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Fill logic
    # ------------------------------------------------------------------

    def _fill_market(self, order: OrderEvent, bar: dict) -> FillEvent:
        """
        Market orders fill at the bar's open price (next bar after signal).
        Slippage is applied on top of the open.

        For stop/TP exits the intended fill price is passed via fill_price_override.
        Gap protection is applied so adverse overnight gaps worsen the fill:
          - Stop exits:  fill = worst of (stop_price, bar_open)  — gap through = bad
          - TP exits:    fill = best  of (tp_price,   bar_open)  — gap through = good
        """
        bar_open = bar["open"]

        if order.exit_reason == "stop":
            self._validate_stop_trigger(order, bar)

        if order.fill_price_override is not None:
            override = order.fill_price_override
            if order.exit_reason == "stop":
                # Gap against you → you get the open (worse than stop)
                if order.side == OrderSide.SELL:
                    base_price = min(override, bar_open)   # long stop: gap down = bad
                else:
                    base_price = max(override, bar_open)   # short stop: gap up = bad
            else:  # "tp" or other override
                # Gap in your favour → you get the open (better than TP)
                if order.side == OrderSide.SELL:
                    base_price = max(override, bar_open)   # long TP: gap up = good
                else:
                    base_price = min(override, bar_open)   # short TP: gap down = good
        else:
            base_price = bar_open

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

    def _validate_stop_trigger(self, order: OrderEvent, bar: dict) -> None:
        """Fail loudly if a strategy emits a stop exit the bar did not touch."""
        stop_price = order.fill_price_override
        if stop_price is None:
            raise ValueError(f"Stop exit for {order.symbol} requires a stop price.")

        if order.side == OrderSide.SELL:
            if bar["low"] > stop_price:
                raise ValueError(
                    f"Long stop exit for {order.symbol} at {stop_price} was not touched "
                    f"(bar low={bar['low']})."
                )
        else:
            if bar["high"] < stop_price:
                raise ValueError(
                    f"Short stop exit for {order.symbol} at {stop_price} was not touched "
                    f"(bar high={bar['high']})."
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
