"""
db/models.py — SQLAlchemy ORM models for backtester results.

Schema mirrors the existing in-memory data shapes so persistence is a direct
mapping, not a translation:
  - Trade        ← core.portfolio.TradeRecord (the 15 round-trip fields)
  - MetricSet    ← analytics.metrics.compute_all() keys
  - OptimizerResult / WfWindow ← analytics.optimizer dataclasses

Market timestamps (trade entry/exit, equity points) are stored as ISO strings
to preserve timezone info losslessly under SQLite and to feed Plotly directly.
created_at is a real DateTime so runs sort chronologically.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
)


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One backtest, optimize, or walk-forward execution."""
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_num: Mapped[Optional[int]] = mapped_column(Integer)          # sequential CLI run number
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    run_type: Mapped[str] = mapped_column(String, index=True)        # backtest | optimize | walkforward
    status: Mapped[str] = mapped_column(String, default="done")      # done | error

    # Denormalized config essentials for fast list/filter without parsing JSON.
    data_source: Mapped[Optional[str]] = mapped_column(String)
    symbols: Mapped[Optional[list]] = mapped_column(JSON)
    start: Mapped[Optional[str]] = mapped_column(String)
    end: Mapped[Optional[str]] = mapped_column(String)
    interval: Mapped[Optional[str]] = mapped_column(String)
    strategy: Mapped[Optional[str]] = mapped_column(String, index=True)

    config: Mapped[Optional[dict]] = mapped_column(JSON)             # full cfg snapshot
    results_dir: Mapped[Optional[str]] = mapped_column(String)      # legacy results/run_* folder, if any

    metrics: Mapped[Optional["MetricSet"]] = relationship(
        back_populates="run", uselist=False,
        cascade="all, delete-orphan", passive_deletes=True,
    )
    trades: Mapped[list["Trade"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True,
    )
    equity_points: Mapped[list["EquityPoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True,
    )
    optimizer_result: Mapped[Optional["OptimizerResult"]] = relationship(
        back_populates="run", uselist=False,
        cascade="all, delete-orphan", passive_deletes=True,
    )
    wf_windows: Mapped[list["WfWindow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True,
    )


class MetricSet(Base):
    """Headline performance metrics for a run (one-to-one with Run)."""
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), unique=True, index=True,
    )

    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown_pct: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown_bars: Mapped[Optional[int]] = mapped_column(Integer)
    cagr: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[Optional[int]] = mapped_column(Integer)
    long_trades: Mapped[Optional[int]] = mapped_column(Integer)
    short_trades: Mapped[Optional[int]] = mapped_column(Integer)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    avg_win: Mapped[Optional[float]] = mapped_column(Float)
    avg_loss: Mapped[Optional[float]] = mapped_column(Float)
    expectancy: Mapped[Optional[float]] = mapped_column(Float)

    monte_carlo: Mapped[Optional[dict]] = mapped_column(JSON)  # p_profit, percentiles, etc.

    run: Mapped["Run"] = relationship(back_populates="metrics")


class Trade(Base):
    """One completed round-trip trade — mirrors TradeRecord."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True,
    )

    symbol: Mapped[Optional[str]] = mapped_column(String)
    side: Mapped[Optional[str]] = mapped_column(String)          # LONG | SHORT
    entry_time: Mapped[Optional[str]] = mapped_column(String)    # ISO
    exit_time: Mapped[Optional[str]] = mapped_column(String)     # ISO
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float)
    quantity: Mapped[Optional[float]] = mapped_column(Float)
    pnl: Mapped[Optional[float]] = mapped_column(Float)
    pnl_pct: Mapped[Optional[float]] = mapped_column(Float)
    commission: Mapped[Optional[float]] = mapped_column(Float)
    slippage: Mapped[Optional[float]] = mapped_column(Float)
    stop_price: Mapped[Optional[float]] = mapped_column(Float)
    tp_price: Mapped[Optional[float]] = mapped_column(Float)
    exit_reason: Mapped[Optional[str]] = mapped_column(String)
    hold_bars: Mapped[Optional[int]] = mapped_column(Integer)

    run: Mapped["Run"] = relationship(back_populates="trades")


class EquityPoint(Base):
    """One mark-to-market point on the equity curve."""
    __tablename__ = "equity_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True,
    )
    timestamp: Mapped[str] = mapped_column(String)   # ISO
    equity: Mapped[float] = mapped_column(Float)

    run: Mapped["Run"] = relationship(back_populates="equity_points")


# Composite index to read a run's curve in time order quickly.
Index("ix_equity_run_ts", EquityPoint.run_id, EquityPoint.timestamp)


class OptimizerResult(Base):
    """IS/OOS optimizer outcome (one-to-one with an optimize Run)."""
    __tablename__ = "optimizer_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), unique=True, index=True,
    )

    mode: Mapped[Optional[str]] = mapped_column(String)        # simple | per_symbol
    metric: Mapped[Optional[str]] = mapped_column(String)
    best_params: Mapped[Optional[dict]] = mapped_column(JSON)
    is_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    oos_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    overfit_diagnostics: Mapped[Optional[dict]] = mapped_column(JSON)
    is_start: Mapped[Optional[str]] = mapped_column(String)
    is_end: Mapped[Optional[str]] = mapped_column(String)
    oos_start: Mapped[Optional[str]] = mapped_column(String)
    oos_end: Mapped[Optional[str]] = mapped_column(String)

    trials: Mapped[list["OptimizerTrial"]] = relationship(
        back_populates="optimizer_result",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    run: Mapped["Run"] = relationship(back_populates="optimizer_result")


class OptimizerTrial(Base):
    """One grid combination from the IS leaderboard (all_results row)."""
    __tablename__ = "optimizer_trials"

    id: Mapped[int] = mapped_column(primary_key=True)
    optimizer_result_id: Mapped[int] = mapped_column(
        ForeignKey("optimizer_results.id", ondelete="CASCADE"), index=True,
    )
    rank: Mapped[Optional[int]] = mapped_column(Integer)
    row: Mapped[dict] = mapped_column(JSON)   # the full all_results row (params + IS metrics)

    optimizer_result: Mapped["OptimizerResult"] = relationship(back_populates="trials")


class WfWindow(Base):
    """One rolling window of a walk-forward run."""
    __tablename__ = "wf_windows"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True,
    )
    window_num: Mapped[Optional[int]] = mapped_column(Integer)
    is_start: Mapped[Optional[str]] = mapped_column(String)
    is_end: Mapped[Optional[str]] = mapped_column(String)
    oos_start: Mapped[Optional[str]] = mapped_column(String)
    oos_end: Mapped[Optional[str]] = mapped_column(String)
    best_params: Mapped[Optional[dict]] = mapped_column(JSON)
    is_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    oos_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    oos_win_rate: Mapped[Optional[float]] = mapped_column(Float)
    oos_profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    oos_max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    oos_trades: Mapped[Optional[int]] = mapped_column(Integer)

    run: Mapped["Run"] = relationship(back_populates="wf_windows")
