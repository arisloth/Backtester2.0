"""
cli.py — Interactive terminal prompts for configuring backtest runs.

All user-facing prompt logic lives here so main.py stays focused on
orchestration. Import prompt_config() or prompt_optimize_config() from here.
"""

from main import CONFIG  # CONFIG is the single source of defaults


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

def valid_timeframes(source: str) -> dict:
    """Return the short-key → internal-value timeframe map for a data source."""
    if source == "alpaca":
        return {"1m":"1min","5m":"5min","15m":"15min","30m":"30min",
                "1h":"1hour","4h":"4hour","1d":"1day","1w":"1week"}
    if source == "ccxt":
        return {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h",
                "12h":"12h","1d":"1d","3d":"3d","1w":"1w"}
    # yfinance / forex
    return {"1m":"1m","5m":"5m","15m":"15m","1h":"1h",
            "4h":"4h","1d":"1d","1wk":"1wk"}


# ---------------------------------------------------------------------------
# Low-level prompt primitives
# ---------------------------------------------------------------------------

def _choose(label: str, options: list, default_index: int = 0) -> str:
    """Print a numbered horizontal menu and return the chosen value."""
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


def _resolve_lookback(period: str) -> tuple:
    """Convert a lookback string like '3y' to (start, end) date strings."""
    from datetime import date
    years = int(period.replace("y", ""))
    end   = date.today()
    start = date(end.year - years, end.month, end.day)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Shared base config prompt (data source, symbols, timeframe, costs)
# ---------------------------------------------------------------------------

def _prompt_base_config() -> dict:
    """Prompt for the parts common to all run modes. Returns a partial cfg dict."""
    cfg = dict(CONFIG)

    cfg["data_source"] = _choose("Data source", [
        ("yfinance",  "yfinance"),
        ("Alpaca",    "alpaca"),
        ("CCXT",      "ccxt"),
        ("Forex",     "forex"),
    ], default_index=0)
    source = cfg["data_source"]

    print("\n── Symbols ──")
    if source == "ccxt":
        example = "e.g. BTC/USDT, ETH/USDT"
    elif source == "forex":
        example = "e.g. EURUSD, GBPUSD"
    else:
        example = "e.g. SPY, AAPL, MSFT"
    raw_symbols = _prompt(f"Tickers ({example})", ", ".join(cfg["symbols"]))
    cfg["symbols"] = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]

    valid_tf   = valid_timeframes(source)
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

    print("\n── Capital & Sizing ──")
    cfg["initial_capital"] = _prompt("Initial capital ($)", int(cfg["initial_capital"]), cast=float)

    cfg["slippage_model"] = _choose("Slippage model", [
        ("Fixed",          "fixed"),
        ("Volatility",     "volatility"),
        ("Volume impact",  "volume_impact"),
    ], default_index=0)
    if cfg["slippage_model"] == "fixed":
        cfg["slippage_pct"] = _prompt("Slippage % per side (e.g. 0.0005)", cfg["slippage_pct"], cast=float)

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


# ---------------------------------------------------------------------------
# Public: backtest config prompt
# ---------------------------------------------------------------------------

def prompt_config() -> dict:
    """Interactively build a config dict for a single backtest run."""
    print("\n" + "=" * 50)
    print("  Backtester — Backtest Setup")
    print("  ( * = default )")
    print("=" * 50)

    cfg = _prompt_base_config()

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

    print("\n── Output ──")
    save = _prompt_bool("Save charts as PNGs instead of displaying interactively", default=False)
    cfg["chart_output_dir"] = _prompt("Output directory", "charts") if save else None

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


# ---------------------------------------------------------------------------
# Public: optimizer / walk-forward config prompt
# ---------------------------------------------------------------------------

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

    base_cfg["strategy"] = _choose("Strategy to optimize", [
        ("FVG Ladder",    "fvg"),
        ("SMA Crossover", "sma_cross"),
    ], default_index=0)

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
            "fvg_max_hold_bars": _prompt_list(
                "Max hold bars values",
                [25, 50, 100],
                cast=int,
            ),
            "fvg_tp1_ratio": _prompt_list(
                "TP1 ratio values (TP1 distance = ratio × stop distance)",
                [0.5, 1.0, 1.5],
                cast=float,
            ),
        }
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}
    else:
        param_grid = {
            "fast": _prompt_list("Fast SMA periods", [20, 50, 100], cast=int),
            "slow": _prompt_list("Slow SMA periods", [100, 150, 200], cast=int),
        }
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}

    # Optional timeframe grid axis
    vtf        = valid_timeframes(base_cfg["data_source"])
    valid_keys = "  |  ".join(vtf.keys())
    print(f"\n── Timeframe Grid (optional) ──")
    print(f"  Valid keys: {valid_keys}")
    raw_tf = input(f"  Timeframes to search (comma-separated, Enter to keep single [{base_cfg['interval']}]): ").strip().lower()
    if raw_tf:
        parsed   = [t.strip() for t in raw_tf.split(",") if t.strip()]
        resolved = [vtf[t] for t in parsed if t in vtf]
        invalid  = [t for t in parsed if t not in vtf]
        if invalid:
            print(f"    Skipping unrecognised keys: {', '.join(invalid)}")
        if len(resolved) > 1:
            param_grid["interval"] = resolved
            print(f"  → Adding {len(resolved)} timeframes to grid: {', '.join(resolved)}")
        elif len(resolved) == 1:
            base_cfg["interval"] = resolved[0]
            print(f"  → Single timeframe — keeping as fixed: {resolved[0]}")

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  → {total_combos} combinations to test")

    metric = _choose("Metric to maximize", [
        ("Sharpe",        "sharpe_ratio"),
        ("Sortino",       "sortino_ratio"),
        ("CAGR",          "cagr"),
        ("Expectancy",    "expectancy"),
        ("Profit Factor", "profit_factor"),
    ], default_index=0)

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
