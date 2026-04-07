"""
analytics/metrics.py — Performance metrics for a completed backtest.

All functions accept the outputs of Portfolio.equity_series() and
Portfolio.trade_dataframe() and return plain floats or dicts.

Metrics computed:
  - Sharpe Ratio (annualized)
  - Sortino Ratio (annualized)
  - Max Drawdown (% and duration in bars)
  - CAGR
  - Win Rate
  - Profit Factor
  - Average Win / Average Loss
  - Expectancy per trade
  - Total / long / short trade counts
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_all(
    equity: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute the full suite of metrics and return them as a dict.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed equity curve from Portfolio.equity_series().
    trades : pd.DataFrame
        Completed trade log from Portfolio.trade_dataframe().
    risk_free_rate : float
        Annualized risk-free rate (default 0.0).
    periods_per_year : int
        Trading periods in a year. 252 for daily stocks, 365 for crypto,
        52 for weekly, etc.

    Returns
    -------
    dict with all metric keys.
    """
    results = {}

    results["sharpe_ratio"]      = sharpe_ratio(equity, risk_free_rate, periods_per_year)
    results["sortino_ratio"]     = sortino_ratio(equity, risk_free_rate, periods_per_year)
    results["max_drawdown_pct"]  = max_drawdown(equity)
    results["max_drawdown_bars"] = max_drawdown_duration(equity)
    results["cagr"]              = cagr(equity, periods_per_year)
    results.update(trade_metrics(trades))

    return results


# ------------------------------------------------------------------
# Equity-curve metrics
# ------------------------------------------------------------------

def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe Ratio."""
    returns = equity.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino Ratio. Uses downside deviation (returns below
    risk-free rate) as the denominator instead of total std dev.
    """
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = np.sqrt((downside ** 2).mean())  # RMS of negative excess returns
    return float((excess.mean() / downside_std) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction (e.g. -0.25 = 25% drawdown).
    Returns 0.0 if the equity curve never draws down.
    """
    if equity.empty:
        return 0.0
    rolling_peak = equity.cummax()
    drawdowns = (equity - rolling_peak) / rolling_peak
    return float(drawdowns.min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """
    Longest drawdown duration in bars (from peak to recovery or end of series).
    Returns 0 if no drawdown occurred.
    """
    if equity.empty:
        return 0

    rolling_peak = equity.cummax()
    in_drawdown = equity < rolling_peak

    max_dur = 0
    current_dur = 0
    for flag in in_drawdown:
        if flag:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0

    return max_dur


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    Returns 0.0 if fewer than 2 data points.
    """
    if len(equity) < 2:
        return 0.0
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    return float(total_return ** (1 / n_years) - 1)


# ------------------------------------------------------------------
# Trade-level metrics
# ------------------------------------------------------------------

def trade_metrics(trades: pd.DataFrame) -> dict:
    """
    Compute all trade-level metrics from the trade log.
    Returns zeroed dict if no trades have been completed.
    """
    empty = {
        "total_trades": 0,
        "long_trades": 0,
        "short_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
    }

    if trades is None or trades.empty:
        return empty

    pnl = trades["pnl"]
    winners = pnl[pnl > 0]
    losers  = pnl[pnl <= 0]

    gross_profit = winners.sum()
    gross_loss   = abs(losers.sum())

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate      = len(winners) / len(pnl) if len(pnl) > 0 else 0.0
    avg_win       = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loss      = float(losers.mean())  if len(losers)  > 0 else 0.0

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    long_trades  = int((trades["side"] == "LONG").sum())  if "side" in trades.columns else 0
    short_trades = int((trades["side"] == "SHORT").sum()) if "side" in trades.columns else 0

    return {
        "total_trades":  len(trades),
        "long_trades":   long_trades,
        "short_trades":  short_trades,
        "win_rate":      float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "expectancy":    float(expectancy),
    }


def print_summary(metrics: dict, initial_capital: float) -> None:
    """Pretty-print a metrics dict to stdout."""
    print("\n" + "=" * 40)
    print("  Backtest Performance Summary")
    print("=" * 40)
    print(f"  CAGR              : {metrics['cagr']*100:>8.2f}%")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio     : {metrics['sortino_ratio']:>8.2f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown_pct']*100:>8.2f}%")
    print(f"  Max DD Duration   : {metrics['max_drawdown_bars']:>8d} bars")
    print(f"  Total Trades      : {metrics['total_trades']:>8d}")
    print(f"  Long / Short      : {metrics['long_trades']} / {metrics['short_trades']}")
    print(f"  Win Rate          : {metrics['win_rate']*100:>8.2f}%")
    print(f"  Profit Factor     : {metrics['profit_factor']:>8.2f}")
    print(f"  Avg Win           : ${metrics['avg_win']:>10,.2f}")
    print(f"  Avg Loss          : ${metrics['avg_loss']:>10,.2f}")
    print(f"  Expectancy/Trade  : ${metrics['expectancy']:>10,.2f}")
    print("=" * 40)
