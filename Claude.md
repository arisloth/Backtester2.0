# CLAUDE.md — Backtester Project Context

This file gives Claude Code full context on this project. Read it before writing any code.

---

## Project Goal

Build a robust, realistic event-driven backtester for testing trading strategies across **US stocks, crypto, and forex**. The north star is **realism** — slippage, commissions, and fill logic must be modeled properly, not tacked on as flat adjustments. This is not a vectorized backtester; it is event-driven to prevent lookahead bias and to mirror how live trading actually works.

Data sources: **Alpaca** (stocks), **yfinance** (stocks + crypto), **CCXT** (crypto), **exchangerate-api or yfinance** (forex).

---

## Design Decisions (already made — do not re-litigate these)

- **Event-driven, not vectorized.** One bar processed at a time. No strategy code ever sees future data.
- **Built from scratch.** No backtrader, vectorbt, or other frameworks. Full control over every component.
- **Realism over speed.** Slippage, commissions, and fill assumptions are first-class citizens.
- **Asset-agnostic strategies.** All data feeds normalize to the same OHLCV + timestamp schema so one strategy can run on stocks, crypto, or forex without modification.
- **Modular and extensible.** New strategies, data sources, or execution models can be added without touching core engine logic.

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
│   ├── event.py           # Event types: MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── engine.py          # Main event loop
│   ├── portfolio.py       # Positions, cash, equity curve, P&L
│   └── broker.py          # Simulated execution: slippage, commissions, fills
│
├── data/
│   ├── base.py            # Abstract DataHandler interface
│   ├── yfinance_feed.py   # yfinance feed (stocks + crypto) — build first
│   ├── alpaca_feed.py     # Alpaca historical data feed
│   ├── ccxt_feed.py       # CCXT feed (broader crypto coverage)
│   └── forex_feed.py      # Forex via yfinance or exchangerate-api
│
├── strategy/
│   ├── base.py            # Abstract Strategy interface
│   └── examples/
│       ├── sma_cross.py   # Simple MA crossover — smoke test strategy
│       └── fvg.py         # FVG Ladder (port later)
│
├── execution/
│   ├── fill_model.py      # Slippage models: fixed, volatility-scaled, volume-impact
│   └── cost_model.py      # Commission models: per-share, percentage, zero, spread-based
│
├── analytics/
│   ├── metrics.py         # Sharpe, Sortino, max drawdown, win rate, profit factor, CAGR
│   ├── monte_carlo.py     # Monte Carlo P(Profit), drawdown distribution, confidence intervals
│   └── visualizer.py      # Equity curve, drawdown chart, trade log, monthly returns heatmap
│
├── config/
│   └── settings.py        # API keys, asset class defaults, slippage/commission defaults
│
├── tests/
│   ├── test_engine.py
│   ├── test_broker.py
│   └── test_metrics.py
│
├── main.py                # Entry point — configure and run a backtest
├── requirements.txt
└── CLAUDE.md              # This file
```

---

## Realism Requirements

These are non-negotiable. Every backtest must reflect them.

### Slippage Models (in `execution/fill_model.py`)
Three models, selectable per backtest run:
1. **Fixed** — flat % per side (e.g. 0.05%). Simple baseline.
2. **Volatility-scaled** — slippage proportional to ATR. More realistic in trending/choppy conditions.
3. **Volume-impact** — larger order size relative to average daily volume = more slippage. Relevant for small/mid cap stocks.

### Commission Models (in `execution/cost_model.py`)
Configurable per asset class:
- **Stocks:** $0 default (Alpaca is commission-free) or per-share mode for IB-style simulation
- **Crypto:** % of trade value (e.g. 0.1% Binance taker fee)
- **Forex:** spread-based (e.g. 1–3 pip spread on majors)

### Fill Assumptions
- **Market orders** fill at next bar's open, not current bar's close
- **Limit orders** only fill if price trades *through* the limit (not just touches it)
- **Partial fills** supported — configurable fill ratio for illiquid assets
- **No lookahead** — strategy only ever sees data up to and including bar `t`

---

## Data Layer

All feeds must output a normalized format:

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
| Forex | yfinance (`EURUSD=X`) | exchangerate-api |

All feeds implement the abstract `DataHandler` interface in `data/base.py`. The engine only talks to `DataHandler` — never directly to yfinance or Alpaca.

---

## Strategy Interface

All strategies inherit from `strategy/base.py`. The interface exposes:

```python
class Strategy(ABC):
    def on_bar(self, market_event: MarketEvent) -> Optional[SignalEvent]:
        """Called once per bar. Returns a signal or None."""
        pass

    def on_fill(self, fill_event: FillEvent) -> None:
        """Called when an order is filled. Use for state tracking."""
        pass
```

Strategies must not access any data outside of what is passed into `on_bar`. No global state, no future data.

---

## Analytics Requirements

### Metrics (`analytics/metrics.py`)
Must compute:
- Sharpe Ratio (annualized, risk-free rate configurable — default 0%)
- Sortino Ratio
- Max Drawdown (% and duration in bars)
- CAGR
- Win Rate
- Profit Factor (gross profit / gross loss)
- Average Win / Average Loss
- Expectancy per trade
- Total trades, long trades, short trades

### Monte Carlo (`analytics/monte_carlo.py`)
- Resample trade returns with replacement, N=1000 iterations
- Output: P(Profit), P(Max Drawdown > X%), median terminal equity, 5th/95th percentile equity range
- Key validation gate — a strategy should not go to paper trading without Monte Carlo confirmation

### Visualizer (`analytics/visualizer.py`)
- Equity curve vs benchmark (SPY buy-and-hold where applicable)
- Drawdown chart (time series of drawdown from peak)
- Trade log table (entry/exit/PnL/side per trade)
- Monthly returns heatmap

---

## Build Order

Build in this exact sequence. Do not skip ahead.

1. `core/event.py` — define all event dataclasses
2. `core/engine.py` — the main loop (stub strategy/data interfaces first)
3. `core/portfolio.py` — position tracking, cash, equity curve
4. `core/broker.py` — order execution simulation
5. `execution/fill_model.py` + `execution/cost_model.py`
6. `data/base.py` + `data/yfinance_feed.py` — first working data feed
7. `strategy/base.py` + `strategy/examples/sma_cross.py` — smoke test strategy
8. **Smoke test:** run SMA crossover on SPY 2020–2024, verify equity curve looks sane
9. `analytics/metrics.py`
10. `analytics/visualizer.py`
11. `analytics/monte_carlo.py`
12. `data/alpaca_feed.py`
13. `data/ccxt_feed.py` + `data/forex_feed.py`
14. `tests/` — unit tests for engine, broker, metrics
15. `main.py` — clean entry point tying everything together

---

## Background & Prior Work

This backtester is being built to test trading strategies across multiple asset classes. Previous strategy work includes an FVG Ladder strategy (4H, longs-only on equities) with strong OOS backtest results (Sharpe ~2.45, +25% OOS, high Monte Carlo P(Profit)). That strategy will eventually be ported into `strategy/examples/fvg.py` once the engine is stable.

The developer has experience with Python, Alpaca paper trading, and backtesting. Code should be clean and well-commented but not over-engineered. Prefer explicit over clever.

---

## Key Principles

- **Never use lookahead.** If you're unsure whether something introduces it, it does.
- **Costs are mandatory.** A backtest result without slippage + commission applied is meaningless.
- **Strategy code is isolated.** It communicates only via events — no direct imports from `core/`.
- **One bar at a time.** The engine loop is sequential. No batch operations in the hot path.
- **Smoke test before anything real.** SMA crossover on SPY must run cleanly before any other strategy is ported in.