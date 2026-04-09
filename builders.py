"""
builders.py — Factory functions that wire cfg dicts into concrete objects.

Imported by main.py, optimizer.py, and any other entry point that needs
to construct a full backtest from a configuration dictionary.
"""


def build_data_handler(cfg: dict):
    source = cfg["data_source"]

    if source == "yfinance":
        from data.yfinance_feed import YFinanceFeed
        return YFinanceFeed(
            symbols=cfg["symbols"],
            start=cfg["start"],
            end=cfg["end"],
            interval=cfg["interval"],
        )

    elif source == "alpaca":
        from data.alpaca_feed import AlpacaFeed
        return AlpacaFeed(
            symbols=cfg["symbols"],
            start=cfg["start"],
            end=cfg["end"],
            timeframe=cfg["interval"],
        )

    elif source == "ccxt":
        from data.ccxt_feed import CCXTFeed
        return CCXTFeed(
            symbols=cfg["symbols"],
            start=cfg["start"],
            end=cfg["end"],
            timeframe=cfg["interval"],
        )

    elif source == "forex":
        from data.forex_feed import ForexFeed
        return ForexFeed(
            symbols=cfg["symbols"],
            start=cfg["start"],
            end=cfg["end"],
            interval=cfg["interval"],
        )

    else:
        raise ValueError(f"Unknown data_source: '{source}'")


def build_strategy(cfg: dict, symbol: str):
    name = cfg["strategy"]

    asset_class = {
        "yfinance": "stock", "alpaca": "stock",
        "ccxt": "crypto",    "forex": "forex",
    }.get(cfg["data_source"], "stock")

    if name == "sma_cross":
        from strategy.examples.sma_cross import SMACrossStrategy
        return SMACrossStrategy(
            symbol=symbol,
            fast=cfg["fast"],
            slow=cfg["slow"],
            asset_class=asset_class,
        )

    elif name == "fvg":
        from strategy.examples.fvg import FVGStrategy
        return FVGStrategy(
            symbol=symbol,
            asset_class=asset_class,
            direction=cfg["fvg_direction"],
            atr_period=cfg["fvg_atr_period"],
            atr_stop_mult=cfg["fvg_atr_stop_mult"],
            tp_atr_mult=cfg["fvg_tp_atr_mult"],
            ema200_filter=cfg["fvg_ema200_filter"],
            order_block_filter=cfg["fvg_order_block_filter"],
            min_gap_atr=cfg["fvg_min_gap_atr"],
            max_gap_age=cfg["fvg_max_gap_age"],
            max_hold_bars=cfg["fvg_max_hold_bars"],
            tp1_enabled=cfg["fvg_tp1_enabled"],
            tp1_ratio=cfg["fvg_tp1_ratio"],
        )

    else:
        raise ValueError(f"Unknown strategy: '{name}'")


def build_fill_model(cfg: dict):
    model = cfg["slippage_model"]

    if model == "fixed":
        from execution.fill_model import FixedSlippage
        return FixedSlippage(pct=cfg["slippage_pct"])

    elif model == "volatility":
        from execution.fill_model import VolatilitySlippage
        return VolatilitySlippage(atr_multiplier=cfg["atr_multiplier"])

    elif model == "volume_impact":
        from execution.fill_model import VolumeImpactSlippage
        return VolumeImpactSlippage(
            base_pct=cfg["slippage_pct"],
            impact_factor=cfg["impact_factor"],
        )

    else:
        raise ValueError(f"Unknown slippage_model: '{model}'")


def build_cost_model(cfg: dict):
    model = cfg["commission_model"]

    if model == "zero":
        from execution.cost_model import ZeroCommission
        return ZeroCommission()

    elif model == "per_share":
        from execution.cost_model import PerShareCommission
        return PerShareCommission(
            rate=cfg["commission_rate"],
            minimum=cfg["commission_minimum"],
        )

    elif model == "percent":
        from execution.cost_model import PercentCommission
        return PercentCommission(default_pct=cfg["commission_rate"])

    elif model == "spread":
        from execution.cost_model import SpreadCommission
        return SpreadCommission(spread_pips=cfg["spread_pips"])

    else:
        raise ValueError(f"Unknown commission_model: '{model}'")
