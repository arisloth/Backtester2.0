"""
execution/fill_model.py — Slippage models for simulated order execution.

Three models, selectable per backtest run:
  1. FixedSlippage     — flat % per side (simple baseline)
  2. VolatilitySlippage — proportional to ATR (realistic in trending/choppy conditions)
  3. VolumeImpactSlippage — scales with order size relative to average daily volume

All models implement the same interface:
    calculate(base_price, side, quantity, bar) -> float

The return value is a signed price adjustment (+ = worse fill for buyer, - = worse for seller).
Broker adds this directly to the base fill price.
"""

from abc import ABC, abstractmethod

from core.event import OrderSide


class FillModel(ABC):
    """Abstract base for all slippage models."""

    @abstractmethod
    def calculate(
        self,
        base_price: float,
        side: OrderSide,
        quantity: float,
        bar: dict,
    ) -> float:
        """
        Compute the signed slippage adjustment to apply to base_price.

        Parameters
        ----------
        base_price : float
            The reference price (e.g. bar open for market orders).
        side : OrderSide
            BUY or SELL. Slippage always moves the fill price against the trader.
        quantity : float
            Units being traded (used by volume-impact model).
        bar : dict
            Full OHLCV bar dict. Models may use high, low, volume, etc.

        Returns
        -------
        float
            Signed price delta. Add to base_price to get the fill price.
            Positive for buys (you pay more), negative for sells (you receive less).
        """


class FixedSlippage(FillModel):
    """
    Flat percentage slippage per side.

    Parameters
    ----------
    pct : float
        Slippage as a fraction of price (e.g. 0.0005 = 0.05%).
    """

    def __init__(self, pct: float = 0.0005):
        self.pct = pct

    def calculate(self, base_price: float, side: OrderSide, quantity: float, bar: dict) -> float:
        adjustment = base_price * self.pct
        return adjustment if side == OrderSide.BUY else -adjustment


class VolatilitySlippage(FillModel):
    """
    Slippage scaled by the bar's high-low range as a proxy for ATR.
    Higher volatility bars → larger slippage.

    Parameters
    ----------
    atr_multiplier : float
        Fraction of the bar's H-L range to use as slippage.
        E.g. 0.1 = 10% of the bar range.
    """

    def __init__(self, atr_multiplier: float = 0.1):
        self.atr_multiplier = atr_multiplier

    def calculate(self, base_price: float, side: OrderSide, quantity: float, bar: dict) -> float:
        bar_range = bar.get("high", base_price) - bar.get("low", base_price)
        adjustment = bar_range * self.atr_multiplier
        # Floor at zero in case high == low (flat bar)
        adjustment = max(adjustment, 0.0)
        return adjustment if side == OrderSide.BUY else -adjustment


class VolumeImpactSlippage(FillModel):
    """
    Slippage that grows with order size relative to bar volume.
    Relevant for small/mid-cap stocks where large orders move the market.

    Model: slippage = base_pct + impact_factor * (quantity / bar_volume)

    Parameters
    ----------
    base_pct : float
        Minimum slippage regardless of order size (e.g. 0.0005 = 0.05%).
    impact_factor : float
        How aggressively order size drives slippage.
        E.g. 0.1 means a 10%-of-volume order adds 1% extra slippage.
    """

    def __init__(self, base_pct: float = 0.0005, impact_factor: float = 0.1):
        self.base_pct = base_pct
        self.impact_factor = impact_factor

    def calculate(self, base_price: float, side: OrderSide, quantity: float, bar: dict) -> float:
        bar_volume = bar.get("volume", 0.0)

        if bar_volume > 0:
            participation = quantity / bar_volume
            slippage_pct = self.base_pct + self.impact_factor * participation
        else:
            # No volume data — fall back to base slippage
            slippage_pct = self.base_pct

        adjustment = base_price * slippage_pct
        return adjustment if side == OrderSide.BUY else -adjustment
