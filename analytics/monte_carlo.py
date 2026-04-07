"""
analytics/monte_carlo.py — Monte Carlo simulation for strategy validation.

Resamples trade returns with replacement (bootstrap) over N iterations to
estimate the distribution of outcomes. This is the key validation gate —
a strategy should not go to paper trading without Monte Carlo confirmation.

Outputs:
  - P(Profit)                     probability terminal equity > initial
  - P(Max Drawdown > threshold)   probability of exceeding a DD limit
  - Median terminal equity
  - 5th / 95th percentile equity range
  - Full distribution arrays for custom analysis

Usage:
    from analytics.monte_carlo import run_monte_carlo, plot_monte_carlo

    results = run_monte_carlo(trades, initial_capital=100_000, n=1000)
    print(results.summary())
    fig = plot_monte_carlo(results)
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation outputs."""

    n_iterations: int
    initial_capital: float
    terminal_equities: np.ndarray    # shape (n,) — final equity of each sim
    max_drawdowns: np.ndarray        # shape (n,) — max DD fraction each sim (negative)

    # Derived stats (computed at construction)
    p_profit: float
    p_dd_exceed: float               # P(max DD < -dd_threshold)
    dd_threshold: float              # the threshold used
    median_equity: float
    pct5_equity: float
    pct95_equity: float

    # Full equity paths (optional — only stored if return_paths=True)
    equity_paths: Optional[np.ndarray] = None  # shape (n, n_trades+1)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 44,
            "  Monte Carlo Results  ({:,} iterations)".format(self.n_iterations),
            "=" * 44,
            f"  P(Profit)            : {self.p_profit*100:>7.1f}%",
            f"  P(MaxDD > {self.dd_threshold*100:.0f}%)       : {self.p_dd_exceed*100:>7.1f}%",
            f"  Median terminal eq.  : ${self.median_equity:>12,.0f}",
            f"  5th pct terminal eq. : ${self.pct5_equity:>12,.0f}",
            f"  95th pct terminal eq.: ${self.pct95_equity:>12,.0f}",
            "=" * 44,
        ]
        return "\n".join(lines)


def run_monte_carlo(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n: int = 1000,
    dd_threshold: float = 0.20,
    return_paths: bool = False,
    seed: Optional[int] = None,
) -> MonteCarloResults:
    """
    Bootstrap Monte Carlo simulation over completed trades.

    Each iteration:
      1. Resample the trade P&L series with replacement (same length).
      2. Build a cumulative equity path from initial_capital.
      3. Record terminal equity and max drawdown.

    Parameters
    ----------
    trades : pd.DataFrame
        Completed trade log from Portfolio.trade_dataframe(). Must have a
        "pnl" column. Needs at least 1 trade.
    initial_capital : float
        Starting equity for each simulated path.
    n : int
        Number of bootstrap iterations (default 1000).
    dd_threshold : float
        Drawdown fraction for P(DD > threshold) calculation (default 0.20 = 20%).
    return_paths : bool
        If True, store all equity paths in results (memory-intensive for large n).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    MonteCarloResults
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        raise ValueError("trades must be a non-empty DataFrame with a 'pnl' column.")

    pnl_array = trades["pnl"].values.astype(float)
    n_trades = len(pnl_array)
    rng = np.random.default_rng(seed)

    terminal_equities = np.empty(n)
    max_drawdowns     = np.empty(n)
    paths             = np.empty((n, n_trades + 1)) if return_paths else None

    for i in range(n):
        # Resample with replacement
        sampled = rng.choice(pnl_array, size=n_trades, replace=True)

        # Build cumulative equity path
        path = np.empty(n_trades + 1)
        path[0] = initial_capital
        for j, pnl in enumerate(sampled):
            path[j + 1] = path[j] + pnl

        terminal_equities[i] = path[-1]
        max_drawdowns[i]     = _max_drawdown(path)

        if return_paths:
            paths[i] = path

    p_profit   = float(np.mean(terminal_equities > initial_capital))
    p_dd_exceed = float(np.mean(max_drawdowns < -dd_threshold))

    return MonteCarloResults(
        n_iterations=n,
        initial_capital=initial_capital,
        terminal_equities=terminal_equities,
        max_drawdowns=max_drawdowns,
        p_profit=p_profit,
        p_dd_exceed=p_dd_exceed,
        dd_threshold=dd_threshold,
        median_equity=float(np.median(terminal_equities)),
        pct5_equity=float(np.percentile(terminal_equities, 5)),
        pct95_equity=float(np.percentile(terminal_equities, 95)),
        equity_paths=paths,
    )


def plot_monte_carlo(
    results: MonteCarloResults,
    title: str = "Monte Carlo Simulation",
    max_paths_shown: int = 200,
) -> plt.Figure:
    """
    Plot Monte Carlo results: equity path fan + terminal equity distribution.

    Parameters
    ----------
    results : MonteCarloResults
    max_paths_shown : int
        Max number of individual paths to draw (keeps the chart readable).
    """
    has_paths = results.equity_paths is not None

    fig, axes = plt.subplots(
        1, 2 if has_paths else 1,
        figsize=(14 if has_paths else 8, 6),
        facecolor="#0f0f0f",
    )
    if not has_paths:
        axes = [axes]

    # --- Left: equity paths fan ---
    if has_paths:
        ax = axes[0]
        ax.set_facecolor("#1a1a1a")
        paths = results.equity_paths
        n_show = min(max_paths_shown, len(paths))
        idx = np.random.choice(len(paths), size=n_show, replace=False)
        x = np.arange(paths.shape[1])
        for i in idx:
            color = "#00c8ff" if paths[i, -1] > results.initial_capital else "#ff4444"
            ax.plot(x, paths[i], color=color, alpha=0.08, linewidth=0.6)
        ax.axhline(results.initial_capital, color="#ffffff", linewidth=0.8,
                   linestyle="--", label="Initial capital")
        ax.set_title("Equity Paths", fontsize=12, color="#ffffff")
        ax.set_xlabel("Trade #", color="#888888")
        ax.set_ylabel("Equity ($)", color="#888888")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(colors="#888888")
        ax.grid(True, alpha=0.2, color="#333333")
        ax.legend(fontsize=8)

    # --- Right: terminal equity distribution ---
    ax = axes[-1]
    ax.set_facecolor("#1a1a1a")
    te = results.terminal_equities

    bins = min(60, results.n_iterations // 10)
    ax.hist(te[te > results.initial_capital],  bins=bins, color="#00e676", alpha=0.7, label="Profit")
    ax.hist(te[te <= results.initial_capital], bins=bins, color="#ff4444", alpha=0.7, label="Loss")

    ax.axvline(results.initial_capital, color="#ffffff", linewidth=1.0,
               linestyle="--", label="Initial capital")
    ax.axvline(results.median_equity,   color="#ffff00", linewidth=1.0,
               linestyle="-",  label=f"Median ${results.median_equity:,.0f}")
    ax.axvline(results.pct5_equity,     color="#ff8c00", linewidth=0.8,
               linestyle=":",  label=f"5th pct ${results.pct5_equity:,.0f}")
    ax.axvline(results.pct95_equity,    color="#00c8ff", linewidth=0.8,
               linestyle=":",  label=f"95th pct ${results.pct95_equity:,.0f}")

    ax.set_title("Terminal Equity Distribution", fontsize=12, color="#ffffff")
    ax.set_xlabel("Terminal Equity ($)", color="#888888")
    ax.set_ylabel("Frequency", color="#888888")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors="#888888")
    ax.grid(True, alpha=0.2, color="#333333")
    ax.legend(fontsize=8)

    # Stats annotation
    stats = (
        f"P(Profit) = {results.p_profit*100:.1f}%\n"
        f"P(DD>{results.dd_threshold*100:.0f}%) = {results.p_dd_exceed*100:.1f}%\n"
        f"n = {results.n_iterations:,}"
    )
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="#cccccc",
            bbox=dict(facecolor="#2a2a2a", edgecolor="#444444", boxstyle="round,pad=0.4"))

    fig.suptitle(title, fontsize=14, color="#ffffff", y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _max_drawdown(path: np.ndarray) -> float:
    """Max drawdown fraction for a 1D equity path array. Returns negative float."""
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / peak
    return float(dd.min())
