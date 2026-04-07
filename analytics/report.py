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

def _fetch_buyhold(symbol: str, start: str, end: str) -> float:
    """
    Fetch actual price data and return the simple buy-and-hold return
    for `symbol` over [start, end]. Returns 0.0 on any failure.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return 0.0
        return float((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1)
    except Exception:
        return 0.0


def _buyhold_section(symbol: str, start: str, end: str, initial: float, strategy_return: float) -> list:
    """Return formatted lines for a buy-and-hold comparison block."""
    bh_ret   = _fetch_buyhold(symbol, start, end)
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

    # --- Buy & Hold comparison ---
    symbol = cfg.get("symbols", [""])[0]
    if symbol:
        lines += _buyhold_section(
            symbol,
            cfg.get("start", ""),
            cfg.get("end", ""),
            initial,
            total_return,
        )

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
    windows = wf_result.windows
    symbol  = cfg.get("symbols", [""])[0]

    # Chain OOS returns via expectancy
    cum_equity = initial
    for w in windows:
        cum_equity += w.oos_metrics.get("total_trades", 0) * w.oos_metrics.get("expectancy", 0.0)
    chained_return = (cum_equity / initial) - 1

    # Buy & hold over full OOS span (first OOS start → last OOS end)
    full_oos_start = windows[0].oos_start  if windows else opt_cfg.get("wf_start", "")
    full_oos_end   = windows[-1].oos_end   if windows else opt_cfg.get("wf_end", "")
    bh_total       = _fetch_buyhold(symbol, full_oos_start, full_oos_end) if symbol else 0.0
    bh_final       = initial * (1 + bh_total)
    alpha          = chained_return - bh_total

    lines += [
        "",
        "  SUMMARY  (out-of-sample periods only)",
        _divider("-"),
        _row("Chained OOS return",  f"{_pct(chained_return)}  →  Final: {_money(cum_equity)}"),
        _row("Buy & Hold return",   f"{_pct(bh_total)}  →  Final: {_money(bh_final)}  ({symbol})"),
        _row("Alpha vs B&H",        _pct(alpha)),
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
        m       = w.oos_metrics
        n_tr    = m.get("total_trades", 0)
        cum_eq += n_tr * m.get("expectancy", 0.0)
        total_trades += n_tr

        oos_ret = m.get("cagr", 0.0)
        oos_sh  = m.get("sharpe_ratio", 0.0)
        oos_dd  = m.get("max_drawdown_pct", 0.0)
        bh_ret  = _fetch_buyhold(symbol, w.oos_start, w.oos_end) if symbol else 0.0

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

    # --- Buy & Hold comparison (OOS period) ---
    symbol = cfg.get("symbols", [""])[0]
    if symbol and opt_cfg.get("oos_start") and opt_cfg.get("oos_end"):
        oos_strategy_return = oos_m.get("cagr", 0.0)
        lines += _buyhold_section(
            symbol,
            opt_cfg["oos_start"],
            opt_cfg["oos_end"],
            initial,
            oos_strategy_return,
        )

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
