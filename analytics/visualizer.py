"""
analytics/visualizer.py — Charts and tables for backtest results.

Four outputs:
  1. Equity curve vs benchmark (SPY buy-and-hold)
  2. Drawdown chart (time series of drawdown from peak)
  3. Trade log table (entry/exit/PnL/side per trade)
  4. Monthly returns heatmap

All functions accept Portfolio outputs and return matplotlib Figure objects
so callers can save, display, or embed them as needed.

Usage:
    from analytics.visualizer import plot_equity, plot_drawdown, plot_monthly_returns
    fig = plot_equity(equity_series, benchmark_series)
    fig.savefig("equity.png")
    fig.show()
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Suppress noisy matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Consistent style across all charts
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#cccccc",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "text.color":       "#cccccc",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#444444",
})

_STRATEGY_COLOR  = "#00c8ff"
_BENCHMARK_COLOR = "#ff8c00"
_DD_COLOR        = "#ff4444"
_WIN_COLOR       = "#00e676"
_LOSS_COLOR      = "#ff4444"


# ------------------------------------------------------------------
# 1. Equity curve
# ------------------------------------------------------------------

def plot_equity(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
) -> plt.Figure:
    """
    Plot the strategy equity curve, optionally vs a benchmark.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed equity curve from Portfolio.equity_series().
    benchmark : pd.Series | None
        Optional benchmark equity curve (same index scale — normalized to
        same starting value as the strategy).
    title : str
        Chart title.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize both to 100 for a fair comparison
    norm_eq = equity / equity.iloc[0] * 100
    ax.plot(norm_eq.index, norm_eq.values, color=_STRATEGY_COLOR, linewidth=1.5, label="Strategy")

    if benchmark is not None and not benchmark.empty:
        # Align benchmark to strategy index
        bench_aligned = benchmark.reindex(equity.index, method="ffill").dropna()
        if not bench_aligned.empty:
            norm_bench = bench_aligned / bench_aligned.iloc[0] * 100
            ax.plot(norm_bench.index, norm_bench.values,
                    color=_BENCHMARK_COLOR, linewidth=1.2, linestyle="--", label="Benchmark")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("Normalized Value (start = 100)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2. Drawdown chart
# ------------------------------------------------------------------

def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> plt.Figure:
    """
    Plot the rolling drawdown from peak as a filled area chart.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed equity curve from Portfolio.equity_series().
    """
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak * 100  # in %

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=_DD_COLOR, alpha=0.6)
    ax.plot(drawdown.index, drawdown.values, color=_DD_COLOR, linewidth=0.8)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 3. Trade log table
# ------------------------------------------------------------------

def plot_trade_log(trades: pd.DataFrame, title: str = "Trade Log") -> plt.Figure:
    """
    Render the trade log as a formatted table figure.

    Parameters
    ----------
    trades : pd.DataFrame
        Completed trade log from Portfolio.trade_dataframe().
    """
    if trades is None or trades.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No completed trades.", ha="center", va="center",
                fontsize=12, color="#888888")
        ax.set_title(title, fontsize=14)
        fig.tight_layout()
        return fig

    # Format for display
    display = trades.copy()
    for col in ("entry_time", "exit_time"):
        if col in display.columns:
            display[col] = pd.to_datetime(display[col]).dt.strftime("%Y-%m-%d")
    for col in ("entry_price", "exit_price"):
        if col in display.columns:
            display[col] = display[col].map("${:,.2f}".format)
    if "pnl" in display.columns:
        display["pnl"] = display["pnl"].map("${:,.2f}".format)
    if "commission" in display.columns:
        display["commission"] = display["commission"].map("${:,.2f}".format)
    if "quantity" in display.columns:
        display["quantity"] = display["quantity"].map("{:,.4f}".format)

    cols = ["symbol", "side", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "pnl"]
    cols = [c for c in cols if c in display.columns]
    display = display[cols]

    n_rows = len(display)
    fig_height = max(2.5, 0.35 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Colour header row
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2a2a2a")
        tbl[0, j].set_text_props(color="#ffffff", fontweight="bold")

    # Colour P&L cells
    pnl_col_idx = cols.index("pnl") if "pnl" in cols else None
    for i in range(1, n_rows + 1):
        raw_pnl = trades["pnl"].iloc[i - 1]
        color = "#0d2b0d" if raw_pnl >= 0 else "#2b0d0d"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(color)
            tbl[i, j].set_text_props(color="#cccccc")

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 4. Monthly returns heatmap
# ------------------------------------------------------------------

def plot_monthly_returns(equity: pd.Series, title: str = "Monthly Returns (%)") -> plt.Figure:
    """
    Plot a calendar heatmap of monthly returns.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed equity curve from Portfolio.equity_series().
    """
    # Resample to month-end and compute monthly returns
    monthly = equity.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100

    # Pivot into year × month matrix
    df = pd.DataFrame({
        "year":  monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret":   monthly_ret.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.7 + 1)))

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()), 1)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=14, pad=12)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Return (%)", shrink=0.8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Convenience: plot all four in one call
# ------------------------------------------------------------------

def plot_all(
    equity: pd.Series,
    trades: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Generate and display (or save) all four charts.

    Parameters
    ----------
    save_dir : str | None
        If provided, saves PNGs to this directory instead of displaying.
    """
    charts = {
        "equity":          plot_equity(equity, benchmark),
        "drawdown":        plot_drawdown(equity),
        "trade_log":       plot_trade_log(trades),
        "monthly_returns": plot_monthly_returns(equity),
    }

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in charts.items():
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")
    else:
        for fig in charts.values():
            fig.show()
