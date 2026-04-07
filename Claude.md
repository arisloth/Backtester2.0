# CLAUDE.md — Backtester Project Context

This file gives Claude Code full context on this project. Read it before writing any code.

---

## Project Goal

Build a robust, realistic event-driven backtester for testing trading strategies across **US stocks, crypto, and forex**. The north star is **realism** — slippage, commissions, and fill logic must be modeled properly, not tacked on as flat adjustments. This is not a vectorized backtester; it is event-driven to prevent lookahead bias and to mirror how live trading actually works.

Data sources: **Alpaca** (stocks), **yfinance** (stocks + crypto), **CCXT** (crypto), **yfinance** (forex via `EURUSD=X` style tickers).

---

## Design Decisions (already made — do not re-litigate these)

- **Event-driven, not vectorized.** One bar processed at a time. No strategy code ever sees future data.
- **Built from scratch.** No backtrader, vectorbt, or other frameworks. Full control over every component.
- **Realism over speed.** Slippage, commissions, and fill assumptions are first-class citizens.
- **Asset-agnostic strategies.** All data feeds normalize to the same OHLCV + timestamp schema so one strategy can run on stocks, crypto, or forex without modification.
- **Modular and extensible.** New strategies, data sources, or execution models can be added without touching core engine logic.
- **Fill timing:** signals emit at bar `t` close; market orders fill at bar `t+1` open (next bar). Known simplification, documented in fvg.py.

---

## Current Status — UPDATED 2026-04-07

### What's built and working

| Component | File | Status |
|-----------|------|--------|
| Event types | `core/event.py` | Done |
| Engine loop | `core/engine.py` | Done |
| Portfolio + trade log | `core/portfolio.py` | Done |
| Broker (slippage/fills) | `core/broker.py` | Done |
| Fill models | `execution/fill_model.py` | Done — Fixed, Volatility-scaled, Volume-impact |
| Cost models | `execution/cost_model.py` | Done — Zero, PerShare, Percent, Spread |
| Data base class | `data/base.py` | Done |
| yfinance feed | `data/yfinance_feed.py` | Done — stocks, crypto, forex |
| Alpaca feed | `data/alpaca_feed.py` | Done — uses `ALPACA_SECRET_KEY` env var |
| CCXT feed | `data/ccxt_feed.py` | Done — broader crypto coverage |
| Forex feed | `data/forex_feed.py` | Done — via yfinance |
| Strategy base | `strategy/base.py` | Done |
| SMA crossover | `strategy/examples/sma_cross.py` | Done — smoke test strategy |
| FVG strategy | `strategy/examples/fvg.py` | Done — ported and fully wired in |
| Metrics | `analytics/metrics.py` | Done — Sharpe, Sortino, MDD, CAGR, win rate, PF, expectancy |
| Monte Carlo | `analytics/monte_carlo.py` | Done — N=1000, P(Profit), percentiles |
| Visualizer | `analytics/visualizer.py` | Done — equity curve, drawdown, heatmap |
| Optimizer | `analytics/optimizer.py` | Done — IS/OOS split + walk-forward (month-based windows) |
| Report generator | `analytics/report.py` | Done — human-readable report.txt per run |
| Config | `config/settings.py` | Done — API keys, defaults |
| Tests | `tests/` | Done — 65 tests, all passing |
| Entry point | `main.py` | Done — interactive CLI |

---

## main.py — Interactive CLI

Run with `python main.py`. Three modes selectable at startup:

```
1) Backtest        — single run with fixed parameters
2) Optimize IS/OOS — grid search params, validate on held-out OOS period
3) Walk-Forward    — rolling IS/OOS windows across full date range
```

### Prompt style
- **Data source, strategy, slippage, commission** — numbered horizontal menus (`1) yfinance | 2) Alpaca | ...`)
- **Timeframe** — free-text input (`4h`, `1d`, `1wk`, etc.) with valid options shown inline
- **Period** — free-text input (`1y`, `3y`, `5y`, `10y`, etc.)
- **Walk-forward windows** — entered in **months** (e.g. train=36, test=6)
- `--no-prompt` flag skips interactive setup and uses CONFIG defaults

### Results — per-run folder
Each run saves to its own numbered folder:
```
results/
    run_1_20260407_141247/
        trades.csv      — full trade log
        equity.csv      — equity curve
        metrics.json    — all metrics + config snapshot
        report.txt      — human-readable formatted report (printed to terminal + saved)
    run_2_20260407_153012/
        summary.json    — optimizer/WF summary
        all_runs.csv    — all IS param combinations ranked (IS/OOS mode)
        summary.csv     — per-window results table (walk-forward mode)
        report.txt
```

### Trade log fields (per completed round-trip)
`symbol`, `side`, `entry_time`, `exit_time`, `entry_price`, `exit_price`, `quantity`,
`pnl`, `pnl_pct`, `commission`, `slippage`, `stop_price`, `tp_price`, `exit_reason`, `hold_bars`

### report.txt format
- **Backtest:** header (config), performance table, Monte Carlo summary
- **Walk-forward:** header, OOS summary, per-fold table (best params + OOS metrics per window), totals/averages
- **IS/OOS optimize:** header, best params, IS vs OOS comparison table, top-10 IS combinations

---

## Timeframe support

| Source | Supported intervals |
|--------|-------------------|
| yfinance | 1m, 5m, 15m, 1h, 4h, 1d, 1wk |
| Alpaca | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w |
| CCXT/Binance | 1m, 5m, 15m, 1h, 4h, 12h, 1d, 3d, 1w |

Note: yfinance intraday data (< 1d) is limited to the last 60 days.

---

## Architecture

### Event Loop

```
MarketEvent → Strategy → SignalEvent → Portfolio → OrderEvent → Broker → FillEvent → Portfolio
```

Each bar fires a `MarketEvent`. Strategy emits a `SignalEvent`. Portfolio converts to `OrderEvent` with sizing. Broker simulates execution and returns `FillEvent` with realistic fill price. Portfolio updates positions and equity curve.

### File Structure

```
backtester/
├── core/
│   ├── event.py           # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── engine.py          # Main event loop
│   ├── portfolio.py       # Positions, cash, equity curve, detailed trade log
│   └── broker.py          # Simulated execution: slippage, commissions, fills
│
├── data/
│   ├── base.py            # Abstract DataHandler interface
│   ├── yfinance_feed.py   # yfinance feed (stocks + crypto + forex)
│   ├── alpaca_feed.py     # Alpaca historical data feed
│   ├── ccxt_feed.py       # CCXT feed (broader crypto coverage)
│   └── forex_feed.py      # Forex via yfinance
│
├── strategy/
│   ├── base.py            # Abstract Strategy interface
│   └── examples/
│       ├── sma_cross.py   # Simple MA crossover — smoke test
│       └── fvg.py         # Fair Value Gap strategy (ICT-style, ATR stops/TP)
│
├── execution/
│   ├── fill_model.py      # Fixed, volatility-scaled, volume-impact slippage
│   └── cost_model.py      # Zero, per-share, percent, spread commissions
│
├── analytics/
│   ├── metrics.py         # Sharpe, Sortino, MDD, CAGR, win rate, PF, expectancy
│   ├── monte_carlo.py     # Bootstrap resampling, P(Profit), equity percentiles
│   ├── visualizer.py      # Equity curve, drawdown, monthly heatmap
│   ├── optimizer.py       # IS/OOS grid search + walk-forward (month-based)
│   └── report.py          # Human-readable text report generator
│
├── config/
│   └── settings.py        # API keys (reads ALPACA_KEY / ALPACA_SECRET_KEY env vars)
│
├── tests/
│   ├── test_engine.py     # Engine integration tests
│   ├── test_broker.py     # Broker/fill/slippage tests
│   └── test_metrics.py    # Metrics + Monte Carlo tests (65 total, all passing)
│
├── results/               # Auto-created, one numbered folder per run
├── .venv/                 # Virtual environment (not committed)
├── .vscode/settings.json  # Auto-activates .venv in VS Code terminals
├── main.py                # Interactive CLI entry point
├── requirements.txt       # yfinance, pandas, numpy, matplotlib, alpaca-py, ccxt, pytest
└── CLAUDE.md              # This file
```

---

## Realism Requirements

These are non-negotiable. Every backtest must reflect them.

### Slippage Models (`execution/fill_model.py`)
1. **Fixed** — flat % per side (default 0.05%)
2. **Volatility-scaled** — slippage proportional to ATR
3. **Volume-impact** — order size relative to ADV drives slippage

### Commission Models (`execution/cost_model.py`)
- **Stocks:** $0 default (Alpaca) or per-share for IB-style
- **Crypto:** % of trade value (default 0.1%)
- **Forex:** spread-based (pips)

### Fill Assumptions
- Market orders fill at next bar's open (not current bar's close)
- Limit orders only fill if price trades *through* the limit
- Partial fills supported
- No lookahead — strategy only ever sees data up to bar `t`

---

## FVG Strategy Details (`strategy/examples/fvg.py`)

Ported from a prior project (Sharpe ~2.45, +25% OOS on equities). Clean rewrite for this event-driven framework.

**Detection:** 3-bar imbalance. Bullish: `high[i-2] < low[i]`. Bearish: `low[i-2] > high[i]`.
**Entry:** price retraces into the gap zone and closes inside it.
**Stop:** `gap_low - atr_stop_mult × ATR` (longs).
**TP:** `fill_price + tp_atr_mult × ATR` (longs) — set in `on_fill()` once fill price is known.
**Filters:** EMA200, order block (opposing candle at bar i-2), min gap width in ATR multiples.
**Parameters:** `direction`, `atr_period`, `atr_stop_mult`, `tp_atr_mult`, `ema200_filter`, `order_block_filter`, `min_gap_atr`, `max_gap_age`.

---

## Optimizer Details (`analytics/optimizer.py`)

### IS/OOS split (`optimize()`)
Grid search all param combinations on the in-sample window. Pick best by chosen metric. Validate once on OOS. Saves: `summary.json`, `all_runs.csv`, `report.txt`.

### Walk-forward (`walk_forward_months()`)
Roll a fixed-size IS window forward by `test_months` each step. For each window: grid search IS → best params → OOS test. Stitch OOS results for realistic performance estimate. Saves: `summary.json`, `summary.csv`, `report.txt`.

**Key params:**
- `train_months` — IS window length in months (default 36)
- `test_months` — OOS window + step size in months (default 6)
- `metric` — what to maximize on IS (sharpe_ratio, sortino_ratio, cagr, expectancy, profit_factor)
- `min_trades` — minimum IS trades for a combo to be eligible (default 5)

**Known bottleneck:** walk-forward is slow because data is re-downloaded for every run and all runs are sequential. Data caching and parallelism are the next performance wins.

---

## Ideas for What to Do Next

### High priority

1. **Data caching** — download each symbol/timeframe/date combo once and reuse across all optimizer runs. Biggest single speedup for walk-forward (currently re-downloads data for every combination). Save cache to disk as parquet so it persists across sessions.

2. **Parallel grid search** — run param combinations concurrently using `multiprocessing.Pool`. Combined with caching, this could make walk-forward 10-20x faster.

3. **Benchmark comparison in metrics** — compute buy-and-hold return for the same period and include alpha, beta, information ratio in `metrics.json` and `report.txt`. The report already has a placeholder for B&H return but it's estimated from expectancy, not actual price data.

### Medium priority

4. **Commission/slippage sensitivity analysis** — run the same backtest at multiple cost levels (e.g. 0x, 1x, 2x, 3x) and show how Sharpe and expectancy degrade. Answers "how much edge does the strategy actually have above costs?"

5. **Portfolio-level risk controls** — max concurrent positions, drawdown kill switch (halt if equity drops X% from peak), max daily loss. Currently the engine has no circuit breakers.

6. **Multiple timeframe support in FVG** — detect FVG on 4H but use 1D for EMA200 trend filter. Requires the engine to handle two feeds at different frequencies for the same symbol.

7. **Live paper trading bridge** — thin adapter replacing the broker with real Alpaca paper orders, keeping the same Strategy/Portfolio/event flow. Architecture was designed for this.

### Lower priority / exploratory

8. **Additional strategies** — VWAP reversion, breakout (52-week high), RSI mean reversion. Stress-tests the engine against different trade profiles.

9. **Regime filter** — classify market as trending vs ranging using ADX or VIX, only take FVG trades in the right regime.

10. **SQLite backend for results** — replace per-run folders with a database so runs can be queried, filtered, and compared across sessions.

11. **Web UI dashboard** — minimal Flask/FastAPI + Plotly frontend that reads results and renders equity curves and trade logs interactively.

---

## Key Principles

- **Never use lookahead.** If you're unsure whether something introduces it, it does.
- **Costs are mandatory.** A backtest without slippage + commission is meaningless.
- **Strategy code is isolated.** It communicates only via events — no direct imports from `core/`.
- **One bar at a time.** The engine loop is sequential. No batch operations in the hot path.
- **Test before shipping.** All 65 unit tests must pass before committing new engine or analytics code.
