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
    "cache_ttl_days": 7,             # refresh cached provider data after this many days
    "refresh_cache": False,          # bypass cache for this run; CLI: --refresh-cache

    # --- Strategy ---
    "strategy":    "sma_cross",      # "sma_cross" | "fvg" | "ema_pullback"
    "fast":        50,               # SMA fast period
    "slow":        200,              # SMA slow period

    # --- FVG strategy params (used when strategy = "fvg") ---
    "fvg_direction":         "long",   # "long" | "short" | "both"
    "fvg_atr_period":        14,
    "fvg_atr_stop_mult":     0.75,     # stop = gap_low - N * ATR
    "fvg_tp_atr_mult":       3.0,      # TP   = fill_price + N * ATR
    "fvg_ema200_filter":     True,
    "fvg_order_block_filter":True,
    "fvg_min_gap_atr":       0.25,
    "fvg_max_gap_age":       10,

    # --- EMA Pullback strategy params (used when strategy = "ema_pullback") ---
    "ep_direction":           "long",       # "long" | "short" | "both"
    "ep_ema_fast":            20,
    "ep_ema_slow":            50,
    "ep_ema_trend":           200,
    "ep_pullback_ema":        "ema_fast",   # "ema_fast" | "ema_slow"
    "ep_touch_tol_atr":       0.3,          # how close low must come to EMA (ATR units)
    "ep_adx_period":          14,
    "ep_adx_min":             25.0,         # Wilder ADX gate
    "ep_atr_period":          14,
    "ep_atr_stop_mult":       1.5,          # stop = close - N * ATR (also clamped by swing)
    "ep_swing_lookback":      5,            # bars to scan for pullback swing low/high
    "ep_tp1_r":               2.0,          # TP1 = entry + N * stop_distance
    "ep_tp1_ratio":           0.5,          # fraction of size taken off at TP1
    "ep_runner_mode":         "structure",  # "structure" | "atr_trail" | "fixed_r"
    "ep_runner_fixed_r":      4.0,          # TP2 R-multiple for fixed_r mode
    "ep_atr_trail_mult":      2.5,          # trail distance in ATR units
    "ep_structure_lookback":  50,           # bars to scan for overhead swing high
    "ep_max_hold_bars":       100,          # timeout exit
    "ep_supertrend_filter":   True,         # 1d Supertrend regime gate (green→longs, red→shorts)
    "ep_st_atr_period":       10,           # Supertrend ATR period on daily bars
    "ep_st_multiplier":       3.0,          # Supertrend ATR multiplier

    # --- EMA Pullback V2: market-wide regime + entry-quality filters ---
    # When the BTC gate or RS filter is enabled, BTC bars are auto-added to the
    # data feed (prepended so they're processed before the alt at each bar).
    "ep_btc_symbol":          "BTC/USDT",   # symbol driving the BTC regime gate / RS filter
    "ep_btc_gate_enabled":    False,        # V3: BTC demoted from entry veto to (future) sizing input
    "ep_btc_gate_mode":       "ema_stack",  # "ema_stack" | "ema20_reclaim" | "off"
    "ep_btc_ema_fast":        20,
    "ep_btc_ema_slow":        50,
    "ep_btc_flatten_on_break":False,        # also EXIT when BTC breaks its EMA50 against us
    "ep_rs_filter_sides":     "short",      # V3: "short" | "both" | "off" — RS only vetoes shorts
    "ep_rs_lookback":         48,           # bars for the RS return comparison (24h on 30m)
    "ep_rs_min_spread":       0.0,          # required out/under-performance margin vs BTC
    "ep_volume_filter_enabled": False,      # V3: volume filter dropped from the entry path
    "ep_vol_lookback":        20,
    "ep_vol_mult":            1.5,
    "ep_pullback_memory_bars":0,            # V3: superseded by fresh-touch (kept for A/B)
    "ep_fresh_touch_required":True,         # V3: enter only on the first EMA reclaim/retest

    # --- Capital & sizing ---
    "initial_capital":    1_000.0,
    "risk_pct":           0.02,      # fraction of current equity to risk per trade (stop-based sizing)
    "position_size_pct":  0.10,      # fallback fraction used when signal has no stop price
    "short_borrow_rate":  0.0,       # annualized borrow cost for short positions (e.g. 0.03 = 3%)
    "short_initial_margin": 0.50,    # initial margin for shorts (Reg-T stock default = 50%)

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
    "min_fill_volume": 0.0,          # bars with volume <= 0 never fill; >0 sets liquidity threshold

    # --- Analytics ---
    "risk_free_rate":    0.0,
    "periods_per_year":  252,
    "monte_carlo_n":     1000,
    "monte_carlo_dd_threshold": 0.20,
    "monte_carlo_method": "iid",     # "iid" or "block"
    "monte_carlo_block_size": None,  # None = sqrt(number of trades) for block bootstrap

    # --- Output ---
    # Set to False to skip chart generation entirely
    "plot_charts": True,
    # Set to a directory path to save charts as PNGs, or None to display interactively
    "chart_output_dir": None,
}

# ==================================================================
# End of CONFIG
# ==================================================================


# ------------------------------------------------------------------
# Annualization factor (periods_per_year)
# ------------------------------------------------------------------
# Sharpe/Sortino/CAGR are annualized by sqrt(periods_per_year) / n_years,
# where periods_per_year = number of bars in one year. This depends on BOTH
# the bar size and the market calendar:
#   crypto   — trades 24 / 7 / 365
#   equities — ~252 sessions of 6.5h
#   forex    — ~24h across 252 trading days
# The old static CONFIG default of 252 is only correct for *daily equity*
# bars. On 1h crypto it under-annualizes by ~35x (252 vs 8760), which craters
# every annualized metric in the report. We derive the right value per run.

# Provider-native interval strings → canonical keys used in the table below.
_INTERVAL_ALIASES = {
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1hour": "1h", "4hour": "4h", "1day": "1d", "1week": "1w", "1wk": "1w",
}

# Bars per year by market calendar and canonical interval.
_BARS_PER_YEAR = {
    "crypto": {  # 24 / 7 / 365
        "1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520,
        "1h": 8760, "4h": 2190, "12h": 730, "1d": 365, "3d": 122, "1w": 52,
    },
    "equity": {  # 252 sessions × 6.5h
        "1m": 98280, "5m": 19656, "15m": 6552, "30m": 3276,
        "1h": 1638, "4h": 410, "1d": 252, "1w": 52,
    },
    "forex": {  # 24h × 252 trading days
        "1m": 362880, "5m": 72576, "15m": 24192, "30m": 12096,
        "1h": 6048, "4h": 1512, "12h": 756, "1d": 252, "1w": 52,
    },
}


def _calendar_for(data_source: str, symbols: list) -> str:
    """Classify the market calendar (crypto / forex / equity) for a run."""
    if data_source == "ccxt":
        return "crypto"
    if data_source == "forex":
        return "forex"
    # yfinance can also serve crypto (BTC-USD) and forex (EURUSD=X).
    syms = [str(s).upper() for s in symbols]
    if any(s.endswith("=X") for s in syms):
        return "forex"
    if any(("/" in s) or s.endswith(("-USD", "-USDT")) for s in syms):
        return "crypto"
    return "equity"


def _apply_periods_per_year(cfg: dict) -> dict:
    """Set cfg['periods_per_year'] from the interval + market calendar so that
    annualized metrics are correct. Overrides the static CONFIG default."""
    cal = _calendar_for(cfg.get("data_source", ""), cfg.get("symbols", []))
    raw = str(cfg.get("interval", "1d")).strip().lower()
    key = _INTERVAL_ALIASES.get(raw, raw)
    ppy = _BARS_PER_YEAR.get(cal, {}).get(key)
    if ppy is None:
        logger.warning(
            "Could not derive periods_per_year for interval=%r source=%r; "
            "falling back to %d. Annualized metrics may be off.",
            cfg.get("interval"), cfg.get("data_source"), cfg.get("periods_per_year", 252),
        )
        return cfg
    cfg["periods_per_year"] = ppy
    return cfg


def feed_symbols(cfg: dict) -> list:
    """
    The symbol list the data feed should load.

    For the EMA Pullback V2 strategy, the BTC regime gate / RS filter need BTC
    bars even though BTC itself isn't traded. We prepend btc_symbol so the feed
    emits it FIRST at each timestamp — the engine drains MarketEvents in symbol
    order, so each alt strategy's BTC state is current when its own bar runs
    (no lookahead). BTC is added to the feed only, never to the strategy list,
    so it is never traded unless the user explicitly trades it.
    """
    syms = list(cfg["symbols"])
    if cfg.get("strategy") != "ema_pullback":
        return syms
    gate_on = (cfg.get("ep_btc_gate_enabled", False)
               and cfg.get("ep_btc_gate_mode", "ema_stack") != "off")
    rs_on = cfg.get("ep_rs_filter_sides", "short") != "off" \
        or cfg.get("ep_rs_filter_enabled", False)  # legacy key still honored
    needs_btc = gate_on or rs_on or cfg.get("ep_btc_flatten_on_break", False)
    btc = cfg.get("ep_btc_symbol", "BTC/USDT")
    if needs_btc and btc not in syms:
        syms = [btc] + syms
    return syms


def build_data_handler(cfg: dict):
    source = cfg["data_source"]
    symbols = feed_symbols(cfg)

    if source == "yfinance":
        from data.yfinance_feed import YFinanceFeed
        return YFinanceFeed(
            symbols=symbols,
            start=cfg["start"],
            end=cfg["end"],
            interval=cfg["interval"],
            cache_ttl_days=cfg.get("cache_ttl_days", 7),
            refresh_cache=cfg.get("refresh_cache", False),
        )

    elif source == "alpaca":
        from data.alpaca_feed import AlpacaFeed
        return AlpacaFeed(
            symbols=symbols,
            start=cfg["start"],
            end=cfg["end"],
            timeframe=cfg["interval"],
            cache_ttl_days=cfg.get("cache_ttl_days", 7),
            refresh_cache=cfg.get("refresh_cache", False),
        )

    elif source == "ccxt":
        from data.ccxt_feed import CCXTFeed
        return CCXTFeed(
            symbols=symbols,
            start=cfg["start"],
            end=cfg["end"],
            timeframe=cfg["interval"],
            cache_ttl_days=cfg.get("cache_ttl_days", 7),
            refresh_cache=cfg.get("refresh_cache", False),
        )

    elif source == "forex":
        from data.forex_feed import ForexFeed
        return ForexFeed(
            symbols=symbols,
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
        )

    elif name == "ema_pullback":
        from strategy.examples.ema_pullback import EMAPullbackStrategy
        return EMAPullbackStrategy(
            symbol=symbol,
            asset_class=asset_class,
            direction=cfg["ep_direction"],
            ema_fast=cfg["ep_ema_fast"],
            ema_slow=cfg["ep_ema_slow"],
            ema_trend=cfg["ep_ema_trend"],
            pullback_ema=cfg["ep_pullback_ema"],
            touch_tol_atr=cfg["ep_touch_tol_atr"],
            adx_period=cfg["ep_adx_period"],
            adx_min=cfg["ep_adx_min"],
            atr_period=cfg["ep_atr_period"],
            atr_stop_mult=cfg["ep_atr_stop_mult"],
            swing_lookback=cfg["ep_swing_lookback"],
            tp1_r=cfg["ep_tp1_r"],
            tp1_ratio=cfg["ep_tp1_ratio"],
            runner_mode=cfg["ep_runner_mode"],
            runner_fixed_r=cfg["ep_runner_fixed_r"],
            atr_trail_mult=cfg["ep_atr_trail_mult"],
            structure_lookback=cfg["ep_structure_lookback"],
            max_hold_bars=cfg["ep_max_hold_bars"],
            daily_supertrend_filter=cfg["ep_supertrend_filter"],
            st_atr_period=cfg["ep_st_atr_period"],
            st_multiplier=cfg["ep_st_multiplier"],
            # --- V2 (cfg.get so older configs without these keys still work) ---
            btc_symbol=cfg.get("ep_btc_symbol", "BTC/USDT"),
            btc_gate_enabled=cfg.get("ep_btc_gate_enabled", True),
            btc_gate_mode=cfg.get("ep_btc_gate_mode", "ema_stack"),
            btc_ema_fast=cfg.get("ep_btc_ema_fast", 20),
            btc_ema_slow=cfg.get("ep_btc_ema_slow", 50),
            btc_flatten_on_break=cfg.get("ep_btc_flatten_on_break", False),
            rs_filter_sides=cfg.get("ep_rs_filter_sides", "short"),
            rs_filter_enabled=cfg.get("ep_rs_filter_enabled", None),  # legacy shim
            rs_lookback=cfg.get("ep_rs_lookback", 48),
            rs_min_spread=cfg.get("ep_rs_min_spread", 0.0),
            volume_filter_enabled=cfg.get("ep_volume_filter_enabled", False),
            vol_lookback=cfg.get("ep_vol_lookback", 20),
            vol_mult=cfg.get("ep_vol_mult", 1.5),
            pullback_memory_bars=cfg.get("ep_pullback_memory_bars", 0),
            fresh_touch_required=cfg.get("ep_fresh_touch_required", True),
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

    # Annualization factor must match the bar size + market calendar, not the
    # static CONFIG default (which only suits daily equity bars).
    _apply_periods_per_year(cfg)

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
        risk_pct=cfg.get("risk_pct", 0.02),
        short_borrow_rate=cfg.get("short_borrow_rate", 0.0),
        short_initial_margin=cfg.get("short_initial_margin", 0.50),
    )
    broker = Broker(
        fill_model=fill_model,
        cost_model=cost_model,
        fill_ratio=cfg["fill_ratio"],
        min_fill_volume=cfg.get("min_fill_volume", 0.0),
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
            method=cfg.get("monte_carlo_method", "iid"),
            block_size=cfg.get("monte_carlo_block_size"),
        )
        print(mc.summary())
        metrics["monte_carlo"] = mc
    else:
        logger.info("No completed trades — skipping Monte Carlo.")

    # --- Charts ---
    if cfg.get("plot_charts", True):
        from analytics.visualizer import plot_all
        plot_all(eq, trades, save_dir=cfg["chart_output_dir"])

    # --- Auto-save results ---
    _save_results(cfg, metrics, eq, trades)

    return metrics


def _next_run_number() -> int:
    """Return the next run number by counting existing run folders under results/."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return 1
    existing = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ]
    return len(existing) + 1


def _save_results(cfg: dict, metrics: dict, eq, trades) -> None:
    """
    Save trade log, equity curve, and metrics into a per-run folder.

    Folder structure:
        results/
            run_1_20260407_141247/
                trades.csv
                equity.csv
                metrics.json
    """
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num   = _next_run_number()
    run_dir   = os.path.join("results", f"run_{run_num}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Trade log CSV
    trades_path = os.path.join(run_dir, "trades.csv")
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        import pandas as pd
        pd.DataFrame().to_csv(trades_path, index=False)

    # Equity curve CSV
    equity_path = os.path.join(run_dir, "equity.csv")
    eq.to_frame(name="equity").to_csv(equity_path)

    # Metrics JSON
    metrics_path = os.path.join(run_dir, "metrics.json")
    serializable = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    serializable["_config"] = {
        k: v for k, v in cfg.items()
        if isinstance(v, (int, float, str, bool, list, type(None)))
    }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Human-readable report
    from analytics.report import backtest_report, save_report
    mc = metrics.get("monte_carlo")
    report_text = backtest_report(cfg, metrics, eq, trades, mc=mc)
    report_path = save_report(report_text, run_dir)
    print(report_text)

    # Mirror into the SQLite results DB (additive; never breaks the CLI run).
    _persist_to_db("backtest", cfg=cfg, metrics=metrics, eq=eq, trades=trades, run_dir=run_dir)

    print(f"\n  Results saved to: {run_dir}/")
    print(f"    trades.csv  |  equity.csv  |  metrics.json  |  report.txt")


def _persist_to_db(kind: str, *, run_dir: str, **kw) -> None:
    """
    Best-effort mirror of a finished run into the SQLite results DB.

    Wrapped so a DB/import error can never break an existing CLI run — the
    per-run folder remains the source of truth on disk regardless.
    """
    try:
        from db import init_db
        from db import ingest
        init_db()
        if kind == "backtest":
            ingest.ingest_backtest(
                kw["cfg"], kw["metrics"], kw["eq"], kw["trades"], run_dir=run_dir,
            )
        elif kind == "optimize":
            ingest.ingest_optimize(
                kw["result"], kw["cfg"], metric=kw.get("metric", "sharpe_ratio"),
                is_start=kw.get("is_start"), is_end=kw.get("is_end"),
                oos_start=kw.get("oos_start"), oos_end=kw.get("oos_end"),
                run_dir=run_dir,
            )
        elif kind == "walkforward":
            ingest.ingest_walkforward(
                kw["result"], kw["cfg"],
                train_months=kw.get("train_months"), test_months=kw.get("test_months"),
                start=kw.get("start"), end=kw.get("end"),
                metric=kw.get("metric", "sharpe_ratio"), run_dir=run_dir,
            )
    except Exception as e:
        logger.warning(f"DB persist skipped ({kind}): {type(e).__name__}: {e}")


def _choose(label: str, options: list, default_index: int = 0) -> str:
    """
    Print a numbered menu horizontally and return the chosen value.
    Options is a list of (display_label, value) tuples.
    """
    parts = []
    for i, (display, _) in enumerate(options, 1):
        marker = " *" if i == default_index + 1 else ""
        parts.append(f"{i}) {display}{marker}")
    print(f"\n  {label}:  " + "   |   ".join(parts))
    while True:
        raw = input(f"  Enter number [default {default_index + 1}]: ").strip()
        if raw == "":
            return options[default_index][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"    Please enter a number between 1 and {len(options)}.")


def _prompt(label: str, default, cast=str):
    """Free-text prompt with a default value."""
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"    Invalid value '{raw}'. Using default: {default}")
        return default


def _prompt_bool(label: str, default: bool) -> bool:
    """Prompt for a yes/no value."""
    raw = input(f"  {label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1", "true")


def _resolve_lookback(period: str) -> tuple:
    """Convert a lookback string like '3y' to (start, end) date strings."""
    from datetime import date, timedelta
    years = int(period.replace("y", ""))
    end   = date.today()
    start = date(end.year - years, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def prompt_config() -> dict:
    """Interactively build a config dict from terminal prompts."""
    print("\n" + "=" * 50)
    print("  Backtester — Backtest Setup")
    print("  ( * = default )")
    print("=" * 50)

    cfg = _prompt_base_config()

    # --- Look-back period ---
    valid_periods = ["1y","2y","3y","4y","5y","10y"]
    print(f"\n  Backtest period  [{' | '.join(valid_periods)}]")
    while True:
        raw = input("  Enter period [default 5y]: ").strip().lower()
        if raw == "":
            raw = "5y"
        if raw in valid_periods:
            period = raw
            break
        print(f"    Invalid. Choose from: {' | '.join(valid_periods)}")
    cfg["start"], cfg["end"] = _resolve_lookback(period)
    print(f"  → {cfg['start']} to {cfg['end']}")

    # --- Strategy ---
    cfg["strategy"] = _choose("Strategy", [
        ("SMA Crossover",  "sma_cross"),
        ("FVG Ladder",     "fvg"),
        ("EMA Pullback",   "ema_pullback"),
    ], default_index=0)

    if cfg["strategy"] == "sma_cross":
        print("\n── SMA Parameters ──")
        cfg["fast"] = _prompt("Fast SMA period", cfg["fast"], cast=int)
        cfg["slow"] = _prompt("Slow SMA period", cfg["slow"], cast=int)

    elif cfg["strategy"] == "fvg":
        print("\n── FVG Parameters ──")
        cfg["fvg_direction"] = _choose("Direction", [
            ("Long only",  "long"),
            ("Short only", "short"),
            ("Both",       "both"),
        ], default_index=0)
        cfg["fvg_atr_stop_mult"] = _prompt("ATR stop multiplier (below gap_low)", cfg["fvg_atr_stop_mult"], cast=float)
        cfg["fvg_tp_atr_mult"]   = _prompt("ATR take-profit multiplier", cfg["fvg_tp_atr_mult"], cast=float)
        cfg["fvg_ema200_filter"]      = _prompt_bool("EMA200 filter (only trade with trend)", cfg["fvg_ema200_filter"])
        cfg["fvg_order_block_filter"] = _prompt_bool("Order block filter (opposing candle at gap origin)", cfg["fvg_order_block_filter"])
        cfg["fvg_min_gap_atr"] = _prompt("Min gap size as ATR multiple (0 = off)", cfg["fvg_min_gap_atr"], cast=float)
        cfg["fvg_max_gap_age"] = _prompt("Max gap age in bars before discarding", cfg["fvg_max_gap_age"], cast=int)

    elif cfg["strategy"] == "ema_pullback":
        print("\n── EMA Pullback Parameters ──")
        cfg["ep_direction"] = _choose("Direction", [
            ("Long only",  "long"),
            ("Short only", "short"),
            ("Both",       "both"),
        ], default_index=0)
        cfg["ep_pullback_ema"] = _choose("Pullback EMA (which EMA must price touch)", [
            ("EMA fast (default, e.g. 20)", "ema_fast"),
            ("EMA slow (deeper, e.g. 50)",  "ema_slow"),
        ], default_index=0)
        cfg["ep_adx_min"]       = _prompt("ADX gate (min ADX for trade)", cfg["ep_adx_min"], cast=float)
        cfg["ep_atr_stop_mult"] = _prompt("ATR stop multiplier (below entry)", cfg["ep_atr_stop_mult"], cast=float)
        cfg["ep_tp1_r"]         = _prompt("TP1 R-multiple", cfg["ep_tp1_r"], cast=float)
        cfg["ep_tp1_ratio"]     = _prompt("Fraction taken off at TP1 (0-1)", cfg["ep_tp1_ratio"], cast=float)
        cfg["ep_runner_mode"]   = _choose("Runner exit mode", [
            ("Structure (TP2 at overhead swing high)", "structure"),
            ("ATR trail (trail stop on runner)",       "atr_trail"),
            ("Fixed R (TP2 at N*R)",                   "fixed_r"),
        ], default_index=0)
        if cfg["ep_runner_mode"] == "fixed_r":
            cfg["ep_runner_fixed_r"] = _prompt("Runner R-multiple", cfg["ep_runner_fixed_r"], cast=float)
        elif cfg["ep_runner_mode"] == "atr_trail":
            cfg["ep_atr_trail_mult"] = _prompt("ATR trail multiplier", cfg["ep_atr_trail_mult"], cast=float)
        cfg["ep_max_hold_bars"] = _prompt("Max hold bars (timeout)", cfg["ep_max_hold_bars"], cast=int)
        cfg["ep_supertrend_filter"] = _prompt_bool(
            "1d Supertrend filter (green→longs only, red→shorts only)",
            cfg["ep_supertrend_filter"],
        )
        if cfg["ep_supertrend_filter"]:
            cfg["ep_st_atr_period"] = _prompt("Supertrend ATR period (daily)", cfg["ep_st_atr_period"], cast=int)
            cfg["ep_st_multiplier"] = _prompt("Supertrend ATR multiplier", cfg["ep_st_multiplier"], cast=float)

    # --- Output ---
    print("\n── Output ──")
    cfg["plot_charts"] = _prompt_bool("Plot charts", default=True)
    if cfg["plot_charts"]:
        save = _prompt_bool("Save charts as PNGs instead of displaying interactively", default=False)
        cfg["chart_output_dir"] = _prompt("Output directory", "charts") if save else None
    else:
        cfg["chart_output_dir"] = None

    # --- Summary ---
    print("\n" + "=" * 50)
    print("  Config Summary")
    print("=" * 50)
    print(f"  Source   : {cfg['data_source']}  |  Timeframe : {cfg['interval']}")
    print(f"  Symbols  : {', '.join(cfg['symbols'])}")
    print(f"  Period   : {cfg['start']} → {cfg['end']}")
    print(f"  Strategy : {cfg['strategy']}")
    print(f"  Capital  : ${cfg['initial_capital']:,.0f}  |  Size : {cfg['position_size_pct']*100:.0f}%")
    print(f"  Slippage : {cfg['slippage_model']}  |  Commission : {cfg['commission_model']}")
    print("=" * 50)

    if not _prompt_bool("Run backtest with these settings?", default=True):
        print("  Aborted.")
        return None

    return cfg


def _prompt_list(label: str, default: list, cast=float) -> list:
    """Prompt for a comma-separated list of values (e.g. '0.5, 0.75, 1.0')."""
    default_str = ", ".join(str(v) for v in default)
    raw = input(f"  {label} [{default_str}]: ").strip()
    if raw == "":
        return default
    try:
        return [cast(v.strip()) for v in raw.split(",") if v.strip()]
    except (ValueError, TypeError):
        print(f"    Invalid input. Using default: {default_str}")
        return default


def _prompt_base_config() -> dict:
    """
    Prompt for the shared parts of any run: data source, symbols, timeframe,
    capital, slippage, commission. Returns a partial cfg dict.
    """
    cfg = dict(CONFIG)

    # --- Data source ---
    cfg["data_source"] = _choose("Data source", [
        ("yfinance",  "yfinance"),
        ("Alpaca",    "alpaca"),
        ("CCXT",      "ccxt"),
        ("Forex",     "forex"),
    ], default_index=0)
    source = cfg["data_source"]

    # --- Symbols ---
    print("\n── Symbols ──")
    if source == "ccxt":
        example = "e.g. BTC/USDT, ETH/USDT"
    elif source == "forex":
        example = "e.g. EURUSD, GBPUSD"
    else:
        example = "e.g. SPY, AAPL, MSFT"
    raw_symbols = _prompt(f"Tickers ({example})", ", ".join(cfg["symbols"]))
    cfg["symbols"] = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]

    # --- Timeframe ---
    if source == "alpaca":
        valid_tf   = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min",
                      "1h":"1hour","4h":"4hour","1d":"1day","1w":"1week"}
        default_tf = "1d"
    elif source == "ccxt":
        valid_tf   = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h",
                      "4h":"4h","12h":"12h","1d":"1d","3d":"3d","1w":"1w"}
        default_tf = "1d"
    else:
        valid_tf   = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h",
                      "4h":"4h","1d":"1d","1wk":"1wk"}
        default_tf = "1d"

    valid_keys = "  |  ".join(valid_tf.keys())
    print(f"\n  Timeframe  [{valid_keys}]")
    while True:
        raw = input(f"  Enter timeframe [default {default_tf}]: ").strip().lower()
        if raw == "":
            raw = default_tf
        if raw in valid_tf:
            cfg["interval"] = valid_tf[raw]
            break
        print(f"    Invalid. Choose from: {valid_keys}")

    # --- Capital & sizing ---
    print("\n── Capital & Sizing ──")
    cfg["initial_capital"] = _prompt("Initial capital ($)", int(cfg["initial_capital"]), cast=float)

    # --- Slippage ---
    cfg["slippage_model"] = _choose("Slippage model", [
        ("Fixed",          "fixed"),
        ("Volatility",     "volatility"),
        ("Volume impact",  "volume_impact"),
    ], default_index=0)
    if cfg["slippage_model"] == "fixed":
        cfg["slippage_pct"] = _prompt("Slippage % per side (e.g. 0.0005)", cfg["slippage_pct"], cast=float)

    # --- Commission ---
    default_comm_idx = {"forex": 3, "ccxt": 2}.get(source, 0)
    cfg["commission_model"] = _choose("Commission model", [
        ("Zero",       "zero"),
        ("Per share",  "per_share"),
        ("Percent",    "percent"),
        ("Spread",     "spread"),
    ], default_index=default_comm_idx)
    if cfg["commission_model"] == "percent":
        cfg["commission_rate"] = _prompt("Rate (e.g. 0.001 = 0.1%)", cfg["commission_rate"], cast=float)
    elif cfg["commission_model"] == "per_share":
        cfg["commission_rate"]    = _prompt("Rate ($/share)", cfg["commission_rate"], cast=float)
        cfg["commission_minimum"] = _prompt("Minimum ($/order)", cfg["commission_minimum"], cast=float)
    elif cfg["commission_model"] == "spread":
        cfg["spread_pips"] = _prompt("Spread in pips", cfg["spread_pips"], cast=float)

    return cfg


def prompt_optimize_config() -> dict:
    """
    Interactively build a config for optimization or walk-forward.
    Returns a dict with keys: base_cfg, param_grid, mode, and mode-specific keys.
    """
    print("\n" + "=" * 50)
    print("  Backtester — Optimizer Setup")
    print("  ( * = default )")
    print("=" * 50)

    base_cfg = _prompt_base_config()

    # --- Strategy ---
    base_cfg["strategy"] = _choose("Strategy to optimize", [
        ("FVG Ladder",    "fvg"),
        ("SMA Crossover", "sma_cross"),
        ("EMA Pullback",  "ema_pullback"),
    ], default_index=0)

    # --- Fixed (non-optimized) strategy params ---
    if base_cfg["strategy"] == "fvg":
        print("\n── FVG Fixed Parameters (not in the search grid) ──")
        base_cfg["fvg_direction"] = _choose("Direction", [
            ("Long only",  "long"),
            ("Short only", "short"),
            ("Both",       "both"),
        ], default_index=0)
        base_cfg["fvg_ema200_filter"]      = _prompt_bool("EMA200 filter", base_cfg["fvg_ema200_filter"])
        base_cfg["fvg_order_block_filter"] = _prompt_bool("Order block filter", base_cfg["fvg_order_block_filter"])
    elif base_cfg["strategy"] == "sma_cross":
        print("\n── SMA Fixed Parameters ──")
        base_cfg["fast"] = _prompt("Fast SMA period", base_cfg["fast"], cast=int)
        base_cfg["slow"] = _prompt("Slow SMA period", base_cfg["slow"], cast=int)
    elif base_cfg["strategy"] == "ema_pullback":
        print("\n── EMA Pullback Fixed Parameters (not in the search grid) ──")
        base_cfg["ep_direction"] = _choose("Direction", [
            ("Long only",  "long"),
            ("Short only", "short"),
            ("Both",       "both"),
        ], default_index=0)
        base_cfg["ep_runner_mode"] = _choose("Runner exit mode", [
            ("Structure",  "structure"),
            ("ATR trail",  "atr_trail"),
            ("Fixed R",    "fixed_r"),
        ], default_index=0)

    # --- Param grid ---
    print("\n── Parameter Grid (comma-separated values to try) ──")
    if base_cfg["strategy"] == "fvg":
        param_grid = {
            "fvg_atr_stop_mult": _prompt_list(
                "ATR stop multiplier values",
                [0.5, 0.75, 1.0, 1.25],
                cast=float,
            ),
            "fvg_tp_atr_mult": _prompt_list(
                "ATR take-profit multiplier values",
                [2.0, 3.0, 4.0, 5.0],
                cast=float,
            ),
            "fvg_min_gap_atr": _prompt_list(
                "Min gap size (ATR multiples) values",
                [0.0, 0.25, 0.5],
                cast=float,
            ),
        }
        # Drop single-value params from grid (no point iterating over one value)
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}
    elif base_cfg["strategy"] == "ema_pullback":
        param_grid = {
            "ep_adx_min": _prompt_list(
                "ADX gate values",
                [20.0, 25.0, 30.0],
                cast=float,
            ),
            "ep_atr_stop_mult": _prompt_list(
                "ATR stop multiplier values",
                [1.0, 1.5, 2.0],
                cast=float,
            ),
            "ep_tp1_r": _prompt_list(
                "TP1 R-multiple values",
                [1.5, 2.0, 2.5],
                cast=float,
            ),
        }
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}
    else:
        # SMA: optimize fast/slow periods
        param_grid = {
            "fast": _prompt_list("Fast SMA periods", [20, 50, 100], cast=int),
            "slow": _prompt_list("Slow SMA periods", [100, 150, 200], cast=int),
        }
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  → {total_combos} combinations to test")

    # --- Optimize metric ---
    metric = _choose("Metric to maximize", [
        ("Sharpe",        "sharpe_ratio"),
        ("Sortino",       "sortino_ratio"),
        ("CAGR",          "cagr"),
        ("Expectancy",    "expectancy"),
        ("Profit Factor", "profit_factor"),
    ], default_index=0)

    # --- Mode: simple split, walk-forward, or per-symbol ---
    mode_options = [
        ("Simple IS/OOS",          "simple"),
        ("Walk-Forward",           "walkforward"),
    ]
    # Per-symbol mode only makes sense when there's more than one symbol in the basket
    if len(base_cfg.get("symbols", [])) > 1:
        mode_options.append(("Per-symbol IS/OOS (one search per coin)", "per_symbol"))
    mode = _choose("Optimization mode", mode_options, default_index=0)

    opt_cfg = {
        "base_cfg":   base_cfg,
        "param_grid": param_grid,
        "metric":     metric,
        "mode":       mode,
    }

    if mode in ("simple", "per_symbol"):
        print("\n── Date Ranges ──")
        valid_periods = ["1y","2y","3y","4y","5y","7y","10y"]
        print(f"\n  Full data range  [{' | '.join(valid_periods)}]")
        while True:
            raw = input("  Enter period [default 5y]: ").strip().lower()
            if raw == "":
                raw = "5y"
            if raw in valid_periods:
                full_period = raw
                break
            print(f"    Invalid. Choose from: {' | '.join(valid_periods)}")
        full_start, full_end = _resolve_lookback(full_period)

        split_pct = _prompt("IS split % (e.g. 0.7 = first 70% is IS)", 0.7, cast=float)
        from datetime import date, timedelta
        fs = date.fromisoformat(full_start)
        fe = date.fromisoformat(full_end)
        total_days = (fe - fs).days
        split_day  = fs + timedelta(days=int(total_days * split_pct))

        opt_cfg["is_start"]  = full_start
        opt_cfg["is_end"]    = (split_day - timedelta(days=1)).isoformat()
        opt_cfg["oos_start"] = split_day.isoformat()
        opt_cfg["oos_end"]   = full_end
        print(f"  IS:  {opt_cfg['is_start']} → {opt_cfg['is_end']}")
        print(f"  OOS: {opt_cfg['oos_start']} → {opt_cfg['oos_end']}")

    else:  # walkforward
        print("\n── Walk-Forward Settings ──")
        valid_periods = ["1y","2y","3y","4y","5y","7y","10y"]
        print(f"\n  Full data range  [{' | '.join(valid_periods)}]")
        while True:
            raw = input("  Enter period [default 5y]: ").strip().lower()
            if raw == "":
                raw = "5y"
            if raw in valid_periods:
                full_period = raw
                break
            print(f"    Invalid. Choose from: {' | '.join(valid_periods)}")
        opt_cfg["wf_start"], opt_cfg["wf_end"] = _resolve_lookback(full_period)
        opt_cfg["train_months"] = _prompt("Training window (months)", 36, cast=int)
        opt_cfg["test_months"]  = _prompt("Test window / step size (months)", 6, cast=int)

        import multiprocessing as _mp
        _cpu = _mp.cpu_count()
        print(f"\n  Parallel workers  [1 = sequential | -1 = all CPUs ({_cpu})]")
        opt_cfg["n_jobs"] = _prompt(f"Workers", 1, cast=int)

        from analytics.optimizer import _build_windows_months
        windows = _build_windows_months(
            opt_cfg["wf_start"], opt_cfg["wf_end"],
            opt_cfg["train_months"], opt_cfg["test_months"],
        )
        print(f"  → {len(windows)} windows × {total_combos} combinations = "
              f"{len(windows) * total_combos} total runs")

    print("\n" + "=" * 50)
    if not _prompt_bool("Start optimizer with these settings?", default=True):
        print("  Aborted.")
        return None

    return opt_cfg


def _simple_optimize_summary_payload(result, opt_cfg: dict, metric: str) -> dict:
    """Build the simple optimizer summary.json payload."""
    base_cfg = opt_cfg["base_cfg"]
    return {
        "mode": "simple",
        "best_params": result.best_params,
        "is_metrics":  {k: v for k, v in result.best_is_metrics.items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
        "oos_metrics": {k: v for k, v in result.oos_metrics.items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
        "overfit_diagnostics": {
            k: v for k, v in result.overfit_diagnostics.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
        "is_start":  opt_cfg["is_start"],  "is_end":  opt_cfg["is_end"],
        "oos_start": opt_cfg["oos_start"], "oos_end": opt_cfg["oos_end"],
        "metric": metric,
        "_config": {k: v for k, v in base_cfg.items()
                    if isinstance(v, (int, float, str, bool, list, type(None)))},
    }


def _write_summary_json(path: str, payload: dict) -> None:
    """Write summary.json using the app's stable JSON format."""
    import json
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _safe_symbol_dir(sym: str) -> str:
    """Filesystem-safe symbol name. 'BTC/USDT' → 'BTC_USDT'."""
    return sym.replace("/", "_").replace(":", "_")


def _run_per_symbol_optimize(opt_cfg: dict, run_dir: str, metric: str) -> None:
    """
    Run the IS/OOS optimizer independently for each symbol in the basket.
    Each symbol gets its own best params + its own sub-folder of results.
    Writes a top-level summary.json and report.txt comparing them.
    """
    import json
    import copy
    from analytics.optimizer import optimize_per_symbol

    base_cfg   = opt_cfg["base_cfg"]
    param_grid = opt_cfg["param_grid"]
    symbols    = list(base_cfg["symbols"])

    print(f"\n  Per-symbol mode: {len(symbols)} symbols × {sum(1 for _ in _iter_combos(param_grid))} combos each\n")

    results_by_sym = optimize_per_symbol(
        base_cfg=base_cfg,
        param_grid=param_grid,
        is_start=opt_cfg["is_start"],
        is_end=opt_cfg["is_end"],
        oos_start=opt_cfg["oos_start"],
        oos_end=opt_cfg["oos_end"],
        metric=metric,
    )

    # Per-symbol sub-folders
    per_symbol_dir = os.path.join(run_dir, "per_symbol")
    os.makedirs(per_symbol_dir, exist_ok=True)

    top_payload = {
        "mode":       "per_symbol",
        "metric":     metric,
        "is_start":   opt_cfg["is_start"],
        "is_end":     opt_cfg["is_end"],
        "oos_start":  opt_cfg["oos_start"],
        "oos_end":    opt_cfg["oos_end"],
        "per_symbol": {},
        "_config": {k: v for k, v in base_cfg.items()
                    if isinstance(v, (int, float, str, bool, list, type(None)))},
    }

    for sym, result in results_by_sym.items():
        sym_dir = os.path.join(per_symbol_dir, _safe_symbol_dir(sym))
        os.makedirs(sym_dir, exist_ok=True)

        if result is None:
            top_payload["per_symbol"][sym] = {"status": "failed"}
            with open(os.path.join(sym_dir, "summary.json"), "w") as f:
                json.dump({"status": "failed", "symbol": sym}, f, indent=2)
            continue

        # Reuse the simple-mode payload by spoofing the sub-cfg
        sub_opt_cfg = copy.deepcopy(opt_cfg)
        sub_base = copy.deepcopy(base_cfg)
        sub_base["symbols"] = [sym]
        sub_opt_cfg["base_cfg"] = sub_base
        payload = _simple_optimize_summary_payload(result, sub_opt_cfg, metric)
        payload["symbol"] = sym
        _write_summary_json(os.path.join(sym_dir, "summary.json"), payload)
        result.all_results.to_csv(os.path.join(sym_dir, "all_runs.csv"), index=False)

        # Per-symbol human-readable report
        from analytics.report import optimize_report, save_report
        try:
            report_text = optimize_report(sub_base, result, sub_opt_cfg)
            save_report(report_text, sym_dir)
        except Exception as exc:
            logger.warning(f"  {sym}: per-symbol report generation failed: {exc}")

        top_payload["per_symbol"][sym] = {
            "status":         "ok",
            "best_params":    result.best_params,
            "is_metrics":     {k: v for k, v in result.best_is_metrics.items()
                               if isinstance(v, (int, float, str, bool, type(None)))},
            "oos_metrics":    {k: v for k, v in result.oos_metrics.items()
                               if isinstance(v, (int, float, str, bool, type(None)))},
        }

    _write_summary_json(os.path.join(run_dir, "summary.json"), top_payload)

    # Top-level comparison table
    report_lines = []
    report_lines.append("=" * 96)
    report_lines.append(f"  PER-SYMBOL OPTIMIZE — IS {opt_cfg['is_start']} → {opt_cfg['is_end']}  |  OOS {opt_cfg['oos_start']} → {opt_cfg['oos_end']}")
    report_lines.append(f"  Metric: {metric}")
    report_lines.append("=" * 96)
    header = f"  {'Symbol':<14}{'Status':<8}{'IS '+metric:<14}{'OOS '+metric:<14}{'OOS trades':<12}{'OOS win%':<10}{'Best params'}"
    report_lines.append(header)
    report_lines.append("-" * 96)
    for sym, result in results_by_sym.items():
        if result is None:
            report_lines.append(f"  {sym:<14}{'FAILED':<8}")
            continue
        is_val  = result.best_is_metrics.get(metric, float("nan"))
        oos_val = result.oos_metrics.get(metric, float("nan"))
        oos_n   = result.oos_metrics.get("total_trades", 0)
        oos_wr  = result.oos_metrics.get("win_rate", 0.0)
        bp_str  = ", ".join(f"{k}={v}" for k, v in result.best_params.items())
        report_lines.append(
            f"  {sym:<14}{'ok':<8}{is_val:<14.4f}{oos_val:<14.4f}{int(oos_n):<12}{oos_wr*100:<10.1f}{bp_str}"
        )
    report_lines.append("=" * 96)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    from analytics.report import save_report
    save_report(report_text, run_dir)

    print(f"\n  Results saved to: {run_dir}/")
    print(f"    summary.json  |  report.txt  |  per_symbol/<SYMBOL>/...")


def _iter_combos(param_grid: dict):
    """Tiny helper just for counting — yields each combo dict."""
    from analytics.optimizer import _param_combinations
    return _param_combinations(param_grid)


def run_optimize(opt_cfg: dict) -> None:
    """Run optimizer or walk-forward from a prompt_optimize_config() result and save output."""
    import json
    from datetime import datetime
    from analytics.optimizer import optimize, walk_forward

    mode      = opt_cfg["mode"]
    base_cfg  = opt_cfg["base_cfg"]
    param_grid = opt_cfg["param_grid"]
    metric    = opt_cfg["metric"]

    # Derive the annualization factor from the chosen interval so optimizer/
    # walk-forward metrics annualize correctly (optimizer.py reads it from cfg).
    _apply_periods_per_year(base_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num   = _next_run_number()
    run_dir   = os.path.join("results", f"run_{run_num}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    if mode == "per_symbol":
        _run_per_symbol_optimize(opt_cfg, run_dir, metric)
        return

    if mode == "simple":
        result = optimize(
            base_cfg=base_cfg,
            param_grid=param_grid,
            is_start=opt_cfg["is_start"],
            is_end=opt_cfg["is_end"],
            oos_start=opt_cfg["oos_start"],
            oos_end=opt_cfg["oos_end"],
            metric=metric,
        )

        print("\n" + "=" * 50)
        print("  Optimization Result")
        print("=" * 50)
        print(f"  Best params:   {result.best_params}")
        print(f"  IS  {metric}: {result.best_is_metrics.get(metric, 'n/a'):.4f}")
        print(f"  OOS {metric}: {result.oos_metrics.get(metric, 'n/a'):.4f}")
        print(f"  OOS trades:    {result.oos_metrics.get('total_trades', 0)}")
        print(f"  OOS win rate:  {result.oos_metrics.get('win_rate', 0):.1%}")
        dsr = result.overfit_diagnostics
        if dsr.get("available"):
            verdict = "WARNING" if dsr.get("warning") else "OK"
            print(f"  DSR prob:      {dsr.get('deflated_sharpe_prob', float('nan')):.1%} ({verdict})")
        else:
            print(f"  DSR prob:      unavailable ({dsr.get('reason', 'not computed')})")

        print(f"\n  Top IS combinations:")
        display_cols = list(param_grid.keys()) + [metric, "total_trades"]
        display_cols = [c for c in display_cols if c in result.all_results.columns]
        print(result.all_results[display_cols].head(10).to_string(index=False))

        # Save
        summary_path = os.path.join(run_dir, "summary.json")
        payload = _simple_optimize_summary_payload(result, opt_cfg, metric)
        _write_summary_json(summary_path, payload)

        all_runs_path = os.path.join(run_dir, "all_runs.csv")
        result.all_results.to_csv(all_runs_path, index=False)

        _persist_to_db(
            "optimize", cfg=base_cfg, result=result, metric=metric,
            is_start=opt_cfg["is_start"], is_end=opt_cfg["is_end"],
            oos_start=opt_cfg["oos_start"], oos_end=opt_cfg["oos_end"],
            run_dir=run_dir,
        )

        # Human-readable report
        from analytics.report import optimize_report, save_report
        report_text = optimize_report(base_cfg, result, opt_cfg)
        print(report_text)
        save_report(report_text, run_dir)

        print(f"\n  Results saved to: {run_dir}/")
        print(f"    summary.json  |  all_runs.csv  |  report.txt")

    else:  # walkforward
        from analytics.optimizer import walk_forward_months
        result = walk_forward_months(
            base_cfg=base_cfg,
            param_grid=param_grid,
            start=opt_cfg["wf_start"],
            end=opt_cfg["wf_end"],
            train_months=opt_cfg["train_months"],
            test_months=opt_cfg["test_months"],
            metric=metric,
            n_jobs=opt_cfg.get("n_jobs", 1),
        )

        print("\n" + "=" * 50)
        print("  Walk-Forward Result")
        print("=" * 50)
        print(f"  OOS Sharpe (weighted): {result.oos_sharpe:.4f}")
        print(f"  OOS Win Rate:          {result.oos_win_rate:.1%}")
        print(f"  Total OOS trades:      {result.oos_total_trades}")
        print(f"\n  Per-window summary:")
        print(result.summary.to_string(index=False))

        # Save
        summary_path = os.path.join(run_dir, "summary.csv")
        result.summary.to_csv(summary_path, index=False)

        json_path = os.path.join(run_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump({
                "mode":             "walkforward",
                "oos_sharpe":       result.oos_sharpe,
                "oos_win_rate":     result.oos_win_rate,
                "oos_total_trades": result.oos_total_trades,
                "train_months":     opt_cfg["train_months"],
                "test_months":      opt_cfg["test_months"],
                "start": opt_cfg["wf_start"], "end": opt_cfg["wf_end"],
                "metric": metric,
                "_config": {k: v for k, v in base_cfg.items()
                            if isinstance(v, (int, float, str, bool, list, type(None)))},
            }, f, indent=2, default=str)

        # Combine OOS trades from all folds and save
        import pandas as pd
        fold_trades = [w.oos_trades for w in result.windows
                       if w.oos_trades is not None and not w.oos_trades.empty]
        all_oos_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
        if not all_oos_trades.empty:
            all_oos_trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False)

        _persist_to_db(
            "walkforward", cfg=base_cfg, result=result, metric=metric,
            train_months=opt_cfg["train_months"], test_months=opt_cfg["test_months"],
            start=opt_cfg["wf_start"], end=opt_cfg["wf_end"], run_dir=run_dir,
        )

        # Human-readable report
        from analytics.report import walkforward_report, save_report
        report_text = walkforward_report(base_cfg, result, opt_cfg, trades=all_oos_trades)
        print(report_text)
        save_report(report_text, run_dir)

        saved_files = "summary.csv  |  summary.json  |  report.txt"
        if not all_oos_trades.empty:
            saved_files += "  |  trades.csv"
        print(f"\n  Results saved to: {run_dir}/")
        print(f"    {saved_files}")


if __name__ == "__main__":
    import sys
    if "--refresh-cache" in sys.argv:
        CONFIG["refresh_cache"] = True
        logger.info("Cache refresh enabled for this run.")

    # Pass --no-prompt (or -y) to skip interactive setup and use CONFIG defaults
    if "--no-prompt" in sys.argv or "-y" in sys.argv:
        run()
    else:
        print("\n" + "=" * 50)
        print("  Backtester")
        print("=" * 50)
        mode = _choose("What would you like to do?", [
            ("Backtest",       "backtest"),
            ("Optimize IS/OOS","optimize"),
            ("Walk-Forward",   "walkforward"),
        ], default_index=0)

        if mode == "backtest":
            cfg = prompt_config()
            if cfg:
                run(cfg)
        else:
            # Both optimize modes go through the same prompt; mode is stored inside opt_cfg
            if mode == "walkforward":
                # Pre-select walk-forward in the prompt by monkey-patching isn't clean —
                # just let prompt_optimize_config() ask. User already knows what they want.
                pass
            opt_cfg = prompt_optimize_config()
            if opt_cfg:
                run_optimize(opt_cfg)
