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
    "strategy":    "sma_cross",      # "sma_cross" | "fvg"
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

    print(f"\n  Results saved to: {run_dir}/")
    print(f"    trades.csv  |  equity.csv  |  metrics.json  |  report.txt")


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
        ("SMA Crossover", "sma_cross"),
        ("FVG Ladder",    "fvg"),
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

    # --- Output ---
    print("\n── Output ──")
    save = _prompt_bool("Save charts as PNGs instead of displaying interactively", default=False)
    cfg["chart_output_dir"] = _prompt("Output directory", "charts") if save else None

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
        valid_tf   = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h",
                      "12h":"12h","1d":"1d","3d":"3d","1w":"1w"}
        default_tf = "1d"
    else:
        valid_tf   = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h",
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
    cfg["initial_capital"]   = _prompt("Initial capital ($)", int(cfg["initial_capital"]), cast=float)
    cfg["position_size_pct"] = _prompt("Position size % of equity (e.g. 0.95)", cfg["position_size_pct"], cast=float)

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

    # --- Strategy (only FVG supported for optimization) ---
    base_cfg["strategy"] = _choose("Strategy to optimize", [
        ("FVG Ladder",    "fvg"),
        ("SMA Crossover", "sma_cross"),
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

    # --- Mode: simple split or walk-forward ---
    mode = _choose("Optimization mode", [
        ("Simple IS/OOS", "simple"),
        ("Walk-Forward",  "walkforward"),
    ], default_index=0)

    opt_cfg = {
        "base_cfg":   base_cfg,
        "param_grid": param_grid,
        "metric":     metric,
        "mode":       mode,
    }

    if mode == "simple":
        print("\n── Date Ranges ──")
        valid_periods = ["5y","7y","10y"]
        print(f"\n  Full data range  [{' | '.join(valid_periods)}]")
        while True:
            raw = input("  Enter period [default 10y]: ").strip().lower()
            if raw == "":
                raw = "10y"
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
        valid_periods = ["5y","7y","10y"]
        print(f"\n  Full data range  [{' | '.join(valid_periods)}]")
        while True:
            raw = input("  Enter period [default 10y]: ").strip().lower()
            if raw == "":
                raw = "10y"
            if raw in valid_periods:
                full_period = raw
                break
            print(f"    Invalid. Choose from: {' | '.join(valid_periods)}")
        opt_cfg["wf_start"], opt_cfg["wf_end"] = _resolve_lookback(full_period)
        opt_cfg["train_years"] = _prompt("Training window (years)", 3, cast=int)
        opt_cfg["test_years"]  = _prompt("Test window / step size (years)", 1, cast=int)

        from analytics.optimizer import _build_windows
        windows = _build_windows(
            opt_cfg["wf_start"], opt_cfg["wf_end"],
            opt_cfg["train_years"], opt_cfg["test_years"],
        )
        print(f"  → {len(windows)} windows × {total_combos} combinations = "
              f"{len(windows) * total_combos} total runs")

    print("\n" + "=" * 50)
    if not _prompt_bool("Start optimizer with these settings?", default=True):
        print("  Aborted.")
        return None

    return opt_cfg


def run_optimize(opt_cfg: dict) -> None:
    """Run optimizer or walk-forward from a prompt_optimize_config() result and save output."""
    import json
    from datetime import datetime
    from analytics.optimizer import optimize, walk_forward

    mode      = opt_cfg["mode"]
    base_cfg  = opt_cfg["base_cfg"]
    param_grid = opt_cfg["param_grid"]
    metric    = opt_cfg["metric"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num   = _next_run_number()
    run_dir   = os.path.join("results", f"run_{run_num}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

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

        print(f"\n  Top IS combinations:")
        display_cols = list(param_grid.keys()) + [metric, "total_trades"]
        display_cols = [c for c in display_cols if c in result.all_results.columns]
        print(result.all_results[display_cols].head(10).to_string(index=False))

        # Save
        summary_path = os.path.join(run_dir, "summary.json")
        payload = {
            "mode": "simple",
            "best_params": result.best_params,
            "is_metrics":  {k: v for k, v in result.best_is_metrics.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
            "oos_metrics": {k: v for k, v in result.oos_metrics.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
            "is_start":  opt_cfg["is_start"],  "is_end":  opt_cfg["is_end"],
            "oos_start": opt_cfg["oos_start"], "oos_end": opt_cfg["oos_end"],
            "metric": metric,
            "_config": {k: v for k, v in base_cfg.items()
                        if isinstance(v, (int, float, str, bool, list, type(None)))},
        }
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        all_runs_path = os.path.join(run_dir, "all_runs.csv")
        result.all_results.to_csv(all_runs_path, index=False)

        # Human-readable report
        from analytics.report import optimize_report, save_report
        report_text = optimize_report(base_cfg, result, opt_cfg)
        print(report_text)
        save_report(report_text, run_dir)

        print(f"\n  Results saved to: {run_dir}/")
        print(f"    summary.json  |  all_runs.csv  |  report.txt")

    else:  # walkforward
        result = walk_forward(
            base_cfg=base_cfg,
            param_grid=param_grid,
            start=opt_cfg["wf_start"],
            end=opt_cfg["wf_end"],
            train_years=opt_cfg["train_years"],
            test_years=opt_cfg["test_years"],
            metric=metric,
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
                "mode": "walkforward",
                "oos_sharpe":       result.oos_sharpe,
                "oos_win_rate":     result.oos_win_rate,
                "oos_total_trades": result.oos_total_trades,
                "train_years":      opt_cfg["train_years"],
                "test_years":       opt_cfg["test_years"],
                "start": opt_cfg["wf_start"], "end": opt_cfg["wf_end"],
                "metric": metric,
                "_config": {k: v for k, v in base_cfg.items()
                            if isinstance(v, (int, float, str, bool, list, type(None)))},
            }, f, indent=2, default=str)

        # Human-readable report
        from analytics.report import walkforward_report, save_report
        report_text = walkforward_report(base_cfg, result, opt_cfg)
        print(report_text)
        save_report(report_text, run_dir)

        print(f"\n  Results saved to: {run_dir}/")
        print(f"    summary.csv  |  summary.json  |  report.txt")


if __name__ == "__main__":
    import sys
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
