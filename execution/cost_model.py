"""
execution/cost_model.py — Commission models for simulated order execution.

Three models, configurable per asset class:
  ZeroCommission    — $0 (Alpaca stocks default)
  PerShareCommission — fixed cost per share (IB-style)
  PercentCommission  — % of trade value (crypto exchanges)
  SpreadCommission   — fixed pip/point spread (forex)

All models implement the same interface:
    calculate(fill_price, quantity, asset_class) -> float

Returns the total commission cost in base currency (always positive).
"""

from abc import ABC, abstractmethod


class CostModel(ABC):
    """Abstract base for all commission models."""

    @abstractmethod
    def calculate(self, fill_price: float, quantity: float, asset_class: str) -> float:
        """
        Compute the total commission for a fill.

        Parameters
        ----------
        fill_price : float
            The price at which the order was filled (post-slippage).
        quantity : float
            Number of units filled (always positive).
        asset_class : str
            "stock" | "crypto" | "forex" — models may apply different
            rates per asset class.

        Returns
        -------
        float
            Total commission in base currency (always >= 0).
        """


class ZeroCommission(CostModel):
    """
    No commission. Default for Alpaca stock trading.
    """

    def calculate(self, fill_price: float, quantity: float, asset_class: str) -> float:
        return 0.0


class PerShareCommission(CostModel):
    """
    Fixed cost per share/unit. IB-style pricing for stocks.

    Parameters
    ----------
    rate : float
        Cost per share in base currency (e.g. 0.005 = $0.005/share).
    minimum : float
        Minimum commission per order (e.g. $1.00).
    """

    def __init__(self, rate: float = 0.005, minimum: float = 1.0):
        self.rate = rate
        self.minimum = minimum

    def calculate(self, fill_price: float, quantity: float, asset_class: str) -> float:
        return max(self.rate * quantity, self.minimum)


class PercentCommission(CostModel):
    """
    Percentage of trade value. Standard for crypto exchanges.

    Per-asset-class rates can be supplied; falls back to `default_pct`
    if the asset class isn't in the rates dict.

    Parameters
    ----------
    default_pct : float
        Default rate as a fraction (e.g. 0.001 = 0.1% Binance taker fee).
    rates : dict[str, float]
        Optional per-asset-class overrides, e.g. {"crypto": 0.001, "stock": 0.0}.
    """

    def __init__(self, default_pct: float = 0.001, rates: dict = None):
        self.default_pct = default_pct
        self.rates = rates or {}

    def calculate(self, fill_price: float, quantity: float, asset_class: str) -> float:
        pct = self.rates.get(asset_class, self.default_pct)
        return fill_price * quantity * pct


class SpreadCommission(CostModel):
    """
    Spread-based commission for forex. Models the bid-ask spread as a
    round-trip cost applied at entry (half-spread per side).

    Parameters
    ----------
    spread_pips : float
        Full bid-ask spread in pips (e.g. 2.0 for a 2-pip spread on EUR/USD).
    pip_value : float
        Value of one pip in base currency per unit (e.g. 0.0001 for most
        USD pairs). Defaults to 0.0001.
    """

    def __init__(self, spread_pips: float = 2.0, pip_value: float = 0.0001):
        self.spread_pips = spread_pips
        self.pip_value = pip_value

    def calculate(self, fill_price: float, quantity: float, asset_class: str) -> float:
        # Half-spread per side = full spread cost per round-trip entry
        half_spread = (self.spread_pips / 2) * self.pip_value
        return half_spread * quantity
