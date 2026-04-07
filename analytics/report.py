"""
analytics/report.py — Human-readable text reports for backtest and walk-forward results.

Generates formatted reports similar to professional backtest output.
Reports are printed to terminal and saved as report.txt in the run folder.

Usage:
    from analytics.report import backtest_report, walkforward_report

    text = backtest_report(cfg, metrics, eq, trades, mc=mc)
    print(text)
    with open("results/run_1/report.txt", "w") as f:
        f.write(text)
"""

from datetime import datetime
from typing import Optional

import pandas as pd

W = 96  # report width


def _divider(char="=") -> str:
    return char * W


def _row(label: str, value: str, width: int = 28) -> str:
    return f"  {label:<{width}}: {value}"


def _pct(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"


def _money(v) -> str:
    if v is None:
        return "n/a"
    return f"${v:,.2f}"


def _f(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Buy & hold benchmark
# ---------------------------------------------------------------------------

def _fetch_buyhold(symbol: str, start: str, end: str, cfg: dict) -> float:
    """
    Fetch the simple buy-and-hold return for `symbol` over [start, end],
    using the same data source as the backtest. Returns 0.0 on any failure.
    """
    source = cfg.get("data_source", "yfinance")
    if source == "alpaca":
        return _fetch_buyhold_alpaca(symbol, start, end, cfg)
    elif source == "ccxt":
        return _fetch_buyhold_ccxt(symbol, start, end, cfg)
    else:  # yfinance, forex
        return _fetch_buyhold_yfinance(symbol, start, end)


def _fetch_buyhold_yfinance(symbol: str, start: str, end: str) -> float:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return 0.0
        return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
    except Exception:
        return 0.0


def _fetch_buyhold_alpaca(symbol: str, start: str, end: str, cfg: dict) -> float:
    try:
        import pandas as pd
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

        client = StockHistoricalDataClient(
            api_key=cfg.get("alpaca_api_key") or ALPACA_API_KEY,
            secret_key=cfg.get("alpaca_api_secret") or ALPACA_API_SECRET,
        )
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=pd.Timestamp(start, tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
            feed=cfg.get("alpaca_feed", "iex"),
        )
        df = client.get_stock_bars(req).df
        if df.empty or len(df) < 2:
            return 0.0
        closes = df.reset_index()["close"].values
        return float(closes[-1] / closes[0] - 1)
    except Exception:
        return 0.0


def _fetch_buyhold_ccxt(symbol: str, start: str, end: str, cfg: dict) -> float:
    try:
        import ccxt
        import pandas as pd

        exchange_id = cfg.get("exchange_id", "binance")
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            return 0.0
        exchange = exchange_class({"enableRateLimit": True})

        start_ms     = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms       = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)
        ms_per_day   = 86_400_000

        first = exchange.fetch_ohlcv(symbol, "1d", since=start_ms, limit=1)
        if not first:
            return 0.0

        last_since = max(start_ms, end_ms - 2 * ms_per_day)
        last_batch = exchange.fetch_ohlcv(symbol, "1d", since=last_since, limit=10)
        last_batch = [b for b in last_batch if b[0] < end_ms]
        if not last_batch:
            return 0.0

        return float(last_batch[-1][4] / first[0][4] - 1)  # close[-1] / close[0] - 1
    except Exception:
        return 0.0


def _buyhold_section(symbol: str, start: str, end: str, initial: float,
                     strategy_return: float, cfg: dict) -> list:
    """Return formatted lines for a buy-and-hold comparison block."""
    bh_ret   = _fetch_buyhold(symbol, start, end, cfg)
    bh_final = initial * (1 + bh_ret)
    alpha    = strategy_return - bh_ret
    return [
        "",
        f"  BUY & HOLD COMPARISON  ({symbol})",
        _divider("-"),
        _row("B&H Return",  f"{_pct(bh_ret)}  →  Final: {_money(bh_final)}"),
        _row("Strategy",    f"{_pct(strategy_return)}  →  Final: {_money(initial * (1 + strategy_return))}"),
        _row("Alpha",       _pct(alpha)),
        _divider("="),
    ]


# ---------------------------------------------------------------------------
# PDT rule check
# ---------------------------------------------------------------------------

def _pdt_warning(trades: pd.DataFrame, cfg: dict) -> list:
    """
    Return warning lines if the strategy would trigger the Pattern Day Trader
    rule (>3 same-day round-trips in any rolling 5-trading-day window with
    equity < $25,000). Returns empty list if no violation or not applicable.
    """
    initial = cfg.get("initial_capital", 0)
    source  = cfg.get("data_source", "")

    # PDT only applies to US stock trading under $25k
    if source not in ("yfinance", "alpaca") or initial >= 25_000:
        return []
    if trades is None or trades.empty:
        return []
    if "entry_time" not in trades.columns or "exit_time" not in trades.columns:
        return []

    # Identify same-day round-trips
    entry_dates = pd.to_datetime(trades["entry_time"]).dt.date
    exit_dates  = pd.to_datetime(trades["exit_time"]).dt.date
    day_trades  = trades[entry_dates == exit_dates].copy()

    if day_trades.empty:
        return []

    # Count per exit date, then check rolling 5-day windows
    daily_counts = day_trades.groupby(exit_dates[day_trades.index]).size()
    dates = sorted(daily_counts.index)
    max_in_window = 0
    for i, d in enumerate(dates):
        window = [dt for dt in dates if 0 <= (d - dt).days < 5]
        total  = sum(daily_counts[dt] for dt in window)
        max_in_window = max(max_in_window, total)

    total_day_trades = len(day_trades)
    if max_in_window <= 3:
        return []

    return [
        "",
        "  PDT WARNING",
        _divider("-"),
        f"  Detected {total_day_trades} same-day round-trip(s) — max {max_in_window} in a 5-day window.",
        f"  US brokers limit accounts under $25,000 to 3 day trades per rolling 5 days.",
        f"  Live trading may block orders the strategy expects to execute.",
        _divider("="),
    ]


# ---------------------------------------------------------------------------
# Backtest report
# ---------------------------------------------------------------------------

def backtest_report(
    cfg: dict,
    metrics: dict,
    eq: pd.Series,
    trades: pd.DataFrame,
    mc=None,
) -> str:
    """
    Generate a human-readable backtest report string.

    Parameters
    ----------
    cfg     : config dict from main.py
    metrics : dict from analytics.metrics.compute_all
    eq      : equity Series from Portfolio.equity_series()
    trades  : trade DataFrame from Portfolio.trade_dataframe()
    mc      : MonteCarloResults object (optional)

    Returns
    -------
    Formatted report as a single string.
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    initial = cfg.get("initial_capital", 100_000)
    final   = float(eq.iloc[-1]) if eq is not None and not eq.empty else initial
    total_return = (final / initial) - 1

    symbols_str  = ", ".join(cfg.get("symbols", []))
    strategy_str = {"fvg": "FVG Ladder", "sma_cross": "SMA Crossover"}.get(cfg.get("strategy", ""), cfg.get("strategy", ""))
    direction_str = cfg.get("fvg_direction", "n/a").capitalize() + " only" if cfg.get("strategy") == "fvg" else "n/a"

    lines += [
        _divider("="),
        f"  BACKTEST — {now}",
        _divider("-"),
        _row("Ticker(s)",    symbols_str),
        _row("Source",       cfg.get("data_source", "n/a").capitalize() + "  |  Interval: " + cfg.get("interval", "n/a")),
        _row("Period",       f"{cfg.get('start', '?')} → {cfg.get('end', '?')}"),
        _row("Strategy",     strategy_str + (f"  |  Direction: {direction_str}" if cfg.get("strategy") == "fvg" else "")),
        _row("Starting cap", _money(initial)),
        _row("Slippage",     cfg.get("slippage_model", "n/a") + "  |  Commission: " + cfg.get("commission_model", "n/a")),
        _divider("="),
    ]

    # --- Performance ---
    lines += [
        "",
        "  PERFORMANCE",
        _divider("-"),
        _row("Total Return",   f"{_pct(total_return)}  →  Final: {_money(final)}"),
        _row("CAGR",           _pct(metrics.get("cagr"))),
        _row("Sharpe Ratio",   _f(metrics.get("sharpe_ratio"))),
        _row("Sortino Ratio",  _f(metrics.get("sortino_ratio"))),
        _row("Max Drawdown",   _pct(metrics.get("max_drawdown_pct")) + f"  ({metrics.get('max_drawdown_bars', 0)} bars)"),
        _divider("-"),
        _row("Total Trades",   str(metrics.get("total_trades", 0)) +
             f"  (L: {metrics.get('long_trades', 0)}  S: {metrics.get('short_trades', 0)})"),
        _row("Win Rate",       _pct(metrics.get("win_rate")) +
             f"  ({int(metrics.get('win_rate', 0) * metrics.get('total_trades', 0))}W / "
             f"{metrics.get('total_trades', 0) - int(metrics.get('win_rate', 0) * metrics.get('total_trades', 0))}L)"),
        _row("Profit Factor",  _f(metrics.get("profit_factor"))),
        _row("Avg Win",        _money(metrics.get("avg_win"))),
        _row("Avg Loss",       _money(metrics.get("avg_loss"))),
        _row("Expectancy",     _money(metrics.get("expectancy")) + " per trade"),
        _divider("="),
    ]

    # --- Per-symbol breakdown (only when multiple symbols) ---
    symbols = cfg.get("symbols", [])
    if len(symbols) > 1 and trades is not None and not trades.empty and "symbol" in trades.columns:
        col = [8, 8, 10, 12, 10, 10, 10]
        hdr = (
            f"  {'Symbol':<{col[0]}}  {'Trades':>{col[1]}}  {'Win Rate':>{col[2]}}  "
            f"{'Total P&L':>{col[3]}}  {'Avg P&L':>{col[4]}}  {'Best':>{col[5]}}  {'Worst':>{col[6]}}"
        )
        lines += ["", "  PER-SYMBOL BREAKDOWN", _divider("-"), hdr, _divider("-")]
        for sym, grp in trades.groupby("symbol"):
            n       = len(grp)
            wr      = (grp["pnl"] > 0).mean()
            total   = grp["pnl"].sum()
            avg     = grp["pnl"].mean()
            best    = grp["pnl"].max()
            worst   = grp["pnl"].min()
            lines.append(
                f"  {sym:<{col[0]}}  {n:>{col[1]}}  {_pct(wr):>{col[2]}}  "
                f"{_money(total):>{col[3]}}  {_money(avg):>{col[4]}}  "
                f"{_money(best):>{col[5]}}  {_money(worst):>{col[6]}}"
            )
        lines.append(_divider("="))

    # --- Buy & Hold comparison (one row per symbol) ---
    start, end = cfg.get("start", ""), cfg.get("end", "")
    if symbols and start and end:
        if len(symbols) == 1:
            lines += _buyhold_section(symbols[0], start, end, initial, total_return, cfg)
        else:
            lines += ["", "  BUY & HOLD COMPARISON", _divider("-")]
            for sym in symbols:
                bh_ret   = _fetch_buyhold(sym, start, end, cfg)
                bh_final = initial * (1 + bh_ret)
                alpha    = total_return - bh_ret
                lines.append(
                    f"  {sym:<10}  B&H: {_pct(bh_ret)}  →  {_money(bh_final)}"
                    f"    Alpha: {_pct(alpha)}"
                )
            lines.append(_divider("="))

    # --- Monte Carlo ---
    if mc is not None:
        pct5  = _pct((mc.pct5_equity  / initial) - 1)
        pct50 = _pct((mc.median_equity / initial) - 1)
        pct95 = _pct((mc.pct95_equity  / initial) - 1)
        dd5   = _pct(float(pd.Series(mc.max_drawdowns).quantile(0.95)))
        dd50  = _pct(float(pd.Series(mc.max_drawdowns).quantile(0.50)))
        dd95  = _pct(float(pd.Series(mc.max_drawdowns).quantile(0.05)))

        lines += [
            "",
            f"  MONTE CARLO  (N={mc.n_iterations:,})",
            _divider("-"),
            f"  {'Method':<18}  {'Return 5/50/95':<30}  {'Max DD 5/50/95':<30}  {'P(Profit)':>10}",
            _divider("-"),
            f"  {'Bootstrap':<18}  {pct5}/{pct50}/{pct95:<18}  {dd5}/{dd50}/{dd95:<18}  {mc.p_profit*100:>9.1f}%",
            _divider("="),
        ]

    # --- PDT warning (US stocks, equity < $25k) ---
    pdt_warn = _pdt_warning(trades, cfg)
    if pdt_warn:
        lines += pdt_warn

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward report
# ---------------------------------------------------------------------------

def walkforward_report(
    cfg: dict,
    wf_result,
    opt_cfg: dict,
) -> str:
    """
    Generate a human-readable walk-forward report string.

    Parameters
    ----------
    cfg        : base config dict
    wf_result  : WalkForwardResult from analytics.optimizer.walk_forward
    opt_cfg    : full optimizer config dict (has wf_start, wf_end, train_years, etc.)

    Returns
    -------
    Formatted report as a single string.
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    symbols_str  = ", ".join(cfg.get("symbols", []))
    strategy_str = {"fvg": "FVG Ladder", "sma_cross": "SMA Crossover"}.get(cfg.get("strategy", ""), cfg.get("strategy", ""))
    direction_str = cfg.get("fvg_direction", "n/a").capitalize() + " only" if cfg.get("strategy") == "fvg" else "n/a"
    initial      = cfg.get("initial_capital", 100_000)
    train_y      = opt_cfg.get("train_months", opt_cfg.get("train_years", "?"))
    test_y       = opt_cfg.get("test_months",  opt_cfg.get("test_years",  "?"))
    unit         = "month(s)" if "train_months" in opt_cfg else "year(s)"
    metric       = opt_cfg.get("metric", "sharpe_ratio")

    lines += [
        _divider("="),
        f"  WALK-FORWARD — {now}",
        _divider("-"),
        _row("Ticker(s)",     symbols_str),
        _row("Source",        cfg.get("data_source", "n/a").capitalize() + "  |  Interval: " + cfg.get("interval", "n/a")),
        _row("Full period",   f"{opt_cfg.get('wf_start', '?')} → {opt_cfg.get('wf_end', '?')}"),
        _row("Train / Test",  f"{train_y} {unit} train  |  {test_y} {unit} test"),
        _row("Strategy",      strategy_str + (f"  |  Direction: {direction_str}" if cfg.get("strategy") == "fvg" else "")),
        _row("Optimize on",   metric.replace("_", " ").title()),
        _row("Starting cap",  _money(initial)),
        _row("Slippage",      cfg.get("slippage_model", "n/a") + "  |  Commission: " + cfg.get("commission_model", "n/a")),
        _row("Total folds",   str(len(wf_result.windows))),
        _divider("="),
    ]

    # --- OOS summary ---
    windows  = wf_result.windows
    symbols  = cfg.get("symbols", [])
    symbol   = symbols[0] if symbols else ""

    # Chain OOS returns using actual equity curves (compound fold returns)
    cum_equity = initial
    for w in windows:
        if w.oos_equity is not None and not w.oos_equity.empty:
            fold_return = float(w.oos_equity.iloc[-1]) / initial - 1
        else:
            # Fallback if equity wasn't captured
            n_tr = w.oos_metrics.get("total_trades", 0)
            fold_return = n_tr * w.oos_metrics.get("expectancy", 0.0) / initial
        cum_equity *= (1 + fold_return)
    chained_return = (cum_equity / initial) - 1

    # Buy & hold over full OOS span (first OOS start → last OOS end)
    full_oos_start = windows[0].oos_start if windows else opt_cfg.get("wf_start", "")
    full_oos_end   = windows[-1].oos_end  if windows else opt_cfg.get("wf_end", "")

    bh_rows = []
    for sym in symbols:
        bh_ret   = _fetch_buyhold(sym, full_oos_start, full_oos_end, cfg) if sym else 0.0
        bh_final = initial * (1 + bh_ret)
        alpha    = chained_return - bh_ret
        bh_rows.append(_row(f"B&H {sym}", f"{_pct(bh_ret)}  →  Final: {_money(bh_final)}  |  Alpha: {_pct(alpha)}"))

    lines += [
        "",
        "  SUMMARY  (out-of-sample periods only)",
        _divider("-"),
        _row("Chained OOS return",  f"{_pct(chained_return)}  →  Final: {_money(cum_equity)}"),
        *bh_rows,
        _row("OOS Sharpe (wtd)",    _f(wf_result.oos_sharpe)),
        _row("OOS Win Rate (wtd)",  _pct(wf_result.oos_win_rate)),
        _row("Total OOS trades",    str(wf_result.oos_total_trades)),
        _divider("="),
    ]

    # --- Folds table ---
    col_w = [4, 27, 24, 10, 10, 10, 8, 10, 14]
    header = (
        f"  {'#':>{col_w[0]}}  "
        f"{'Test Period':<{col_w[1]}}  "
        f"{'Best Params':<{col_w[2]}}  "
        f"{'OOS Ret':>{col_w[3]}}  "
        f"{'B&H Ret':>{col_w[4]}}  "
        f"{'OOS Sh':>{col_w[5]}}  "
        f"{'Trades':>{col_w[6]}}  "
        f"{'Max DD':>{col_w[7]}}  "
        f"{'End Cap':>{col_w[8]}}"
    )
    lines += ["", "  PORTFOLIO FOLDS", _divider("-"), header, _divider("-")]

    cum_eq       = initial
    total_trades = 0
    sharpe_sum   = 0.0
    ret_sum      = 0.0
    bh_sum       = 0.0
    dd_sum       = 0.0
    n_windows    = len(windows)

    for i, w in enumerate(windows, 1):
        m    = w.oos_metrics
        n_tr = m.get("total_trades", 0)
        total_trades += n_tr

        # Compound actual fold return into running equity
        if w.oos_equity is not None and not w.oos_equity.empty:
            fold_return = float(w.oos_equity.iloc[-1]) / initial - 1
        else:
            fold_return = n_tr * m.get("expectancy", 0.0) / initial
        cum_eq *= (1 + fold_return)

        oos_ret = m.get("cagr", 0.0)
        oos_sh  = m.get("sharpe_ratio", 0.0)
        oos_dd  = m.get("max_drawdown_pct", 0.0)
        # Average B&H across all symbols for this fold
        bh_rets = [_fetch_buyhold(sym, w.oos_start, w.oos_end, cfg) for sym in symbols if sym]
        bh_ret  = sum(bh_rets) / len(bh_rets) if bh_rets else 0.0

        sharpe_sum += oos_sh
        ret_sum    += oos_ret
        bh_sum     += bh_ret
        dd_sum     += oos_dd

        params_str = "  ".join(
            f"{k.replace('fvg_','').replace('atr_','').replace('_mult','')}={v}"
            for k, v in w.best_params.items()
        )

        lines.append(
            f"  {i:>{col_w[0]}}  "
            f"{w.oos_start} → {w.oos_end}  "
            f"{params_str:<{col_w[2]}}  "
            f"{_pct(oos_ret):>{col_w[3]}}  "
            f"{_pct(bh_ret):>{col_w[4]}}  "
            f"{_f(oos_sh):>{col_w[5]}}  "
            f"{n_tr:>{col_w[6]}}  "
            f"{_pct(oos_dd):>{col_w[7]}}  "
            f"{_money(cum_eq):>{col_w[8]}}"
        )

    lines += [
        _divider("-"),
        f"  {'AVG':<{col_w[0]+col_w[1]+col_w[2]+6}}  "
        f"{_pct(ret_sum/n_windows):>{col_w[3]}}  "
        f"{_pct(bh_sum/n_windows):>{col_w[4]}}  "
        f"{_f(sharpe_sum/n_windows):>{col_w[5]}}  "
        f"{total_trades//n_windows:>{col_w[6]}}  "
        f"{_pct(dd_sum/n_windows):>{col_w[7]}}",
        f"  {'TOTAL':<{col_w[0]+col_w[1]+col_w[2]+6}}  "
        f"{'':>{col_w[3]}}  "
        f"{'':>{col_w[4]}}  "
        f"{'':>{col_w[5]}}  "
        f"{total_trades:>{col_w[6]}}  "
        f"{'':>{col_w[7]}}  "
        f"{_money(cum_eq):>{col_w[8]}}",
        _divider("="),
    ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IS/OOS optimize report
# ---------------------------------------------------------------------------

def optimize_report(
    cfg: dict,
    result,
    opt_cfg: dict,
) -> str:
    """Generate a human-readable IS/OOS optimization report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    symbols_str  = ", ".join(cfg.get("symbols", []))
    strategy_str = {"fvg": "FVG Ladder", "sma_cross": "SMA Crossover"}.get(cfg.get("strategy", ""), cfg.get("strategy", ""))
    initial      = cfg.get("initial_capital", 100_000)
    metric       = opt_cfg.get("metric", "sharpe_ratio")
    metric_label = metric.replace("_", " ").title()

    lines += [
        _divider("="),
        f"  OPTIMIZATION (IS/OOS) — {now}",
        _divider("-"),
        _row("Ticker(s)",    symbols_str),
        _row("Source",       cfg.get("data_source", "n/a").capitalize() + "  |  Interval: " + cfg.get("interval", "n/a")),
        _row("IS period",    f"{opt_cfg.get('is_start', '?')} → {opt_cfg.get('is_end', '?')}"),
        _row("OOS period",   f"{opt_cfg.get('oos_start', '?')} → {opt_cfg.get('oos_end', '?')}"),
        _row("Strategy",     strategy_str),
        _row("Optimize on",  metric_label),
        _row("Starting cap", _money(initial)),
        _divider("="),
    ]

    # Best params
    lines += [
        "",
        "  BEST PARAMS  (selected on IS)",
        _divider("-"),
    ]
    for k, v in result.best_params.items():
        label = k.replace("fvg_", "").replace("_", " ").title()
        lines.append(_row(label, str(v)))

    # IS vs OOS comparison
    is_m  = result.best_is_metrics
    oos_m = result.oos_metrics
    lines += [
        _divider("-"),
        f"  {'Metric':<28}  {'In-Sample':>12}  {'Out-of-Sample':>14}",
        _divider("-"),
        f"  {metric_label:<28}  {_f(is_m.get(metric)):>12}  {_f(oos_m.get(metric)):>14}",
        f"  {'Sharpe Ratio':<28}  {_f(is_m.get('sharpe_ratio')):>12}  {_f(oos_m.get('sharpe_ratio')):>14}",
        f"  {'CAGR':<28}  {_pct(is_m.get('cagr')):>12}  {_pct(oos_m.get('cagr')):>14}",
        f"  {'Max Drawdown':<28}  {_pct(is_m.get('max_drawdown_pct')):>12}  {_pct(oos_m.get('max_drawdown_pct')):>14}",
        f"  {'Win Rate':<28}  {_pct(is_m.get('win_rate')):>12}  {_pct(oos_m.get('win_rate')):>14}",
        f"  {'Profit Factor':<28}  {_f(is_m.get('profit_factor')):>12}  {_f(oos_m.get('profit_factor')):>14}",
        f"  {'Total Trades':<28}  {is_m.get('total_trades', 0):>12}  {oos_m.get('total_trades', 0):>14}",
        _divider("="),
    ]

    # --- Buy & Hold comparison (OOS period, all symbols) ---
    oos_symbols = cfg.get("symbols", [])
    oos_strategy_return = oos_m.get("cagr", 0.0)
    if oos_symbols and opt_cfg.get("oos_start") and opt_cfg.get("oos_end"):
        if len(oos_symbols) == 1:
            lines += _buyhold_section(
                oos_symbols[0],
                opt_cfg["oos_start"],
                opt_cfg["oos_end"],
                initial,
                oos_strategy_return,
                cfg,
            )
        else:
            lines += ["", "  BUY & HOLD COMPARISON  (OOS period)", _divider("-")]
            for sym in oos_symbols:
                bh_ret   = _fetch_buyhold(sym, opt_cfg["oos_start"], opt_cfg["oos_end"], cfg)
                bh_final = initial * (1 + bh_ret)
                alpha    = oos_strategy_return - bh_ret
                lines.append(
                    f"  {sym:<10}  B&H: {_pct(bh_ret)}  →  {_money(bh_final)}"
                    f"    Alpha: {_pct(alpha)}"
                )
            lines.append(_divider("="))

    # All IS runs ranked
    if result.all_results is not None and not result.all_results.empty:
        lines += ["", "  ALL IS COMBINATIONS (ranked by " + metric_label + ")", _divider("-")]
        param_keys = list(result.best_params.keys())
        show_cols  = param_keys + [metric, "total_trades"]
        show_cols  = [c for c in show_cols if c in result.all_results.columns]
        df         = result.all_results[show_cols].head(20)
        lines.append("  " + df.to_string(index=False).replace("\n", "\n  "))
        lines.append(_divider("="))

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_report(text: str, run_dir: str) -> str:
    """Write report text to report.txt in run_dir. Returns the file path."""
    import os
    path = os.path.join(run_dir, "report.txt")
    with open(path, "w") as f:
        f.write(text)
    return path
