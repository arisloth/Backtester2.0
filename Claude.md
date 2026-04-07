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
- **Fill timing:** signals emit at bar `t` close; market orders fill at bar `t+1` open (next bar). This is a known simplification documented in fvg.py.

---

## Current Status — COMPLETE AS OF 2026-04-07

The full build order has been completed. The engine is production-ready for backtesting.

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
| Config | `config/settings.py` | Done — API keys, defaults |
| Tests | `tests/` | Done — 65 tests, all passing |
| Entry point | `main.py` | Done — interactive CLI with numbered menus |

### main.py interactive features
- Numbered menus for: data source (yfinance/Alpaca/CCXT/forex), strategy (SMA/FVG), symbols, timeframe
- Lookback presets: 1y, 2y, 3y, 4y, 5y, 10y — resolves to absolute dates at runtime
- Auto-saves results to separate folders:
  - `results/trades/` — trade log CSV
  - `results/equity/` — equity curve CSV
  - `results/metrics/` — metrics + config JSON
- Filename format: `{symbols}_{strategy}_{start}_{end}_{timestamp}`
- `--no-prompt` flag to run non-interactively

### Trade log fields (per completed round-trip)
`symbol`, `side`, `entry_time`, `exit_time`, `entry_price`, `exit_price`, `quantity`,
`pnl`, `pnl_pct`, `commission`, `slippage`, `stop_price`, `tp_price`, `exit_reason`, `hold_bars`

### Timeframe support
| Source | Supported intervals |
|--------|-------------------|
| yfinance | 1m, 2m, 5m, 15m, 30m, 60m/1h, 90m, 1d, 5d, 1wk, 1mo, 3mo |
| Alpaca | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo |
| CCXT/Binance | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo |

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
│   └── visualizer.py      # Equity curve, drawdown, monthly heatmap
│
├── config/
│   └── settings.py        # API keys (reads ALPACA_KEY / ALPACA_SECRET_KEY env vars)
│
├── tests/
│   ├── test_engine.py     # Engine integration tests
│   ├── test_broker.py     # Broker/fill/slippage tests
│   └── test_metrics.py    # Metrics + Monte Carlo tests (65 total, all passing)
│
├── results/               # Auto-created by main.py
│   ├── trades/
│   ├── equity/
│   └── metrics/
│
├── main.py                # Interactive CLI entry point
├── requirements.txt
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

## Data Layer

All feeds output normalized OHLCV:

```python
{
    "timestamp": pd.Timestamp,  # UTC
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "symbol": str,
    "asset_class": str  # "stock" | "crypto" | "forex"
}
```

### Data Source Priority
| Asset | Primary | Fallback |
|-------|---------|---------|
| US Stocks | Alpaca | yfinance |
| Crypto | yfinance | CCXT |
| Forex | yfinance (`EURUSD=X`) | — |

Multi-symbol runs inner-join bars across all symbols so every bar has data for every symbol.

---

## Strategy Interface

```python
class Strategy(ABC):
    def on_bar(self, market_event: MarketEvent) -> Optional[SignalEvent]:
        """Called once per bar. Returns a signal or None."""

    def on_fill(self, fill_event: FillEvent) -> None:
        """Called when an order is filled. Use for position state tracking."""
```

Strategies must not access any data outside what is passed into `on_bar`. No global state, no future data. Strategies import only from `core/event.py` and `strategy/base.py`.

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

## Analytics

### Metrics (`analytics/metrics.py`)
Sharpe, Sortino, Max Drawdown (% + bars), CAGR, Win Rate, Profit Factor, Avg Win, Avg Loss, Expectancy, trade counts (total/long/short).

### Monte Carlo (`analytics/monte_carlo.py`)
N=1000 bootstrap resample of trade P&Ls. Outputs: `p_profit`, `p_max_dd_exceeds`, `median_equity`, `pct5_equity`, `pct95_equity`, optional `equity_paths` matrix.

### Visualizer (`analytics/visualizer.py`)
Equity curve vs SPY benchmark, drawdown time series, monthly returns heatmap.

---

## Ideas for What to Do Next

These are ordered roughly by impact vs effort. None are committed to yet.

### High priority

1. **Parameter optimization / walk-forward** — grid search over FVG params (atr_stop_mult, tp_atr_mult, min_gap_atr) with in-sample/out-of-sample split to find robust settings without overfitting.

2. **Benchmark comparison in metrics** — compute SPY buy-and-hold returns for the same period and include alpha, beta, information ratio alongside the existing metrics. Already plotted in the visualizer but not in the metrics JSON.

3. **Commission/slippage sensitivity analysis** — run the same backtest at multiple cost levels and show how Sharpe and expectancy degrade. Helps understand how much edge the strategy actually has.

### Medium priority

4. **Multiple timeframe support in FVG** — detect FVG on 4H but use 1D for EMA200 trend filter. Requires the engine to handle two data feeds at different frequencies for the same symbol.

5. **Portfolio-level risk controls** — max concurrent positions, max drawdown kill switch (halt trading if equity drops X% from peak), max daily loss. Currently the engine has no circuit breakers.

6. **Strategy combiners / multi-strategy runs** — run SMA + FVG simultaneously on different symbols with shared capital. The engine already supports multiple strategies; the portfolio sizing needs to account for concurrent exposure.

7. **Live paper trading bridge** — thin adapter that replaces the broker with real Alpaca paper orders, keeping the same Strategy/Portfolio/event flow. The architecture was designed for this.

### Lower priority / exploratory

8. **Additional strategies** — VWAP reversion, breakout (52-week high), mean reversion on RSI extremes. Useful for stress-testing the engine against different trade profiles.

9. **Regime filter** — classify market as trending vs ranging using ADX or VIX level, only take FVG trades in the right regime.

10. **Database backend for results** — replace CSV/JSON file dumps with SQLite so runs can be queried, compared, and charted across sessions.

11. **Web UI dashboard** — minimal Flask/FastAPI + Plotly frontend that reads results from the database and renders equity curves and trade logs interactively.

---

## Key Principles

- **Never use lookahead.** If you're unsure whether something introduces it, it does.
- **Costs are mandatory.** A backtest without slippage + commission is meaningless.
- **Strategy code is isolated.** It communicates only via events — no direct imports from `core/`.
- **One bar at a time.** The engine loop is sequential. No batch operations in the hot path.
- **Test before shipping.** All 65 unit tests must pass before committing new engine or analytics code.
