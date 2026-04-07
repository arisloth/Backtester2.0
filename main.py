"""
main.py — Entry point for running a backtest.

Configure your backtest here and run:
    python main.py

Edit the CONFIG section below to change symbols, dates, strategy,
slippage/commission models, and analytics options.
"""

import logging
import os

# ------------------------------------------------------------------
# Logging — set to INFO for progress, DEBUG for full event trace
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================================
# CONFIG — edit this section to configure your backtest
# ==================================================================

CONFIG = {
    # --- Data ---
    "data_source": "yfinance",       # "yfinance" | "alpaca" | "ccxt" | "forex"
    "symbols":     ["SPY"],
    "start":       "2020-01-01",
    "end":         "2024-12-31",
    "interval":    "1d",             # yfinance: "1d","1h" etc. | alpaca: "1Day","1Hour"

    # --- Strategy ---
    "strategy":    "sma_cross",      # "sma_cross" (add more as you port them)
    "fast":        50,               # SMA fast period
    "slow":        200,              # SMA slow period

    # --- Capital & sizing ---
    "initial_capital":    100_000.0,
    "position_size_pct":  0.95,      # fraction of equity per trade

    # --- Slippage model ---
    # "fixed" | "volatility" | "volume_impact"
    "slippage_model": "fixed",
    "slippage_pct":   0.0005,        # used by fixed model (0.05%)
    "atr_multiplier": 0.1,           # used by volatility model
    "impact_factor":  0.1,           # used by volume_impact model

    # --- Commission model ---
    # "zero" | "per_share" | "percent" | "spread"
    "commission_model":   "zero",
    "commission_rate":    0.005,     # per_share: $/share | percent: fraction
    "commission_minimum": 1.0,       # per_share minimum
    "spread_pips":        2.0,       # forex spread model

    # --- Partial fills ---
    "fill_ratio": 1.0,               # 1.0 = full fill, 0.5 = 50% partial

    # --- Analytics ---
    "risk_free_rate":    0.0,
    "periods_per_year":  252,
    "monte_carlo_n":     1000,
    "monte_carlo_dd_threshold": 0.20,

    # --- Output ---
    # Set to a directory path to save charts as PNGs, or None to display interactively
    "chart_output_dir": None,
}

# ==================================================================
# End of CONFIG
# ==================================================================


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

    if name == "sma_cross":
        from strategy.examples.sma_cross import SMACrossStrategy
        asset_class = {
            "yfinance": "stock", "alpaca": "stock",
            "ccxt": "crypto",    "forex": "forex",
        }.get(cfg["data_source"], "stock")
        return SMACrossStrategy(
            symbol=symbol,
            fast=cfg["fast"],
            slow=cfg["slow"],
            asset_class=asset_class,
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


def run(cfg: dict = None) -> dict:
    """
    Run a full backtest from CONFIG and return the metrics dict.
    Pass a custom cfg dict to override CONFIG programmatically.
    """
    if cfg is None:
        cfg = CONFIG

    # --- Wire up components ---
    feed        = build_data_handler(cfg)
    strategies  = [build_strategy(cfg, s) for s in cfg["symbols"]]
    fill_model  = build_fill_model(cfg)
    cost_model  = build_cost_model(cfg)

    from core.portfolio import Portfolio
    from core.broker import Broker
    from core.engine import Engine

    portfolio = Portfolio(
        initial_capital=cfg["initial_capital"],
        position_size_pct=cfg["position_size_pct"],
    )
    broker = Broker(
        fill_model=fill_model,
        cost_model=cost_model,
        fill_ratio=cfg["fill_ratio"],
    )
    engine = Engine(
        data_handler=feed,
        strategies=strategies,
        portfolio=portfolio,
        broker=broker,
    )

    # --- Run ---
    logger.info(
        f"Starting backtest: {cfg['symbols']} | {cfg['data_source']} | "
        f"{cfg['start']} → {cfg['end']} | strategy={cfg['strategy']}"
    )
    engine.run()

    # --- Metrics ---
    from analytics.metrics import compute_all, print_summary
    eq     = portfolio.equity_series()
    trades = portfolio.trade_dataframe()
    metrics = compute_all(
        eq, trades,
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
    )
    print_summary(metrics, initial_capital=cfg["initial_capital"])

    # --- Monte Carlo (only if there are completed trades) ---
    if not trades.empty:
        from analytics.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            trades,
            initial_capital=cfg["initial_capital"],
            n=cfg["monte_carlo_n"],
            dd_threshold=cfg["monte_carlo_dd_threshold"],
            return_paths=True,
            seed=42,
        )
        print(mc.summary())
        metrics["monte_carlo"] = mc
    else:
        logger.info("No completed trades — skipping Monte Carlo.")

    # --- Charts ---
    from analytics.visualizer import plot_all
    plot_all(eq, trades, save_dir=cfg["chart_output_dir"])

    return metrics


if __name__ == "__main__":
    run()
