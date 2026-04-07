"""
data/forex_feed.py — Forex data feed via yfinance.

Downloads historical OHLCV bars for forex pairs using yfinance's
`EURUSD=X` ticker convention, normalizes to the standard bar schema,
and replays one bar at a time through the engine.

yfinance forex tickers use the format: "EURUSD=X", "GBPUSD=X", etc.
This feed accepts either format:
    "EURUSD"   → converted to "EURUSD=X" for yfinance
    "EURUSD=X" → used as-is

Commissions should be modelled with SpreadCommission from cost_model.py.

Usage:
    from data.forex_feed import ForexFeed

    feed = ForexFeed(
        symbols=["EURUSD", "GBPUSD"],
        start="2022-01-01",
        end="2023-01-01",
        interval="1d",
    )
"""

import logging
from queue import Queue
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from data.base import DataHandler
from core.event import MarketEvent

logger = logging.getLogger(__name__)


def _to_yf_ticker(symbol: str) -> str:
    """Normalize a forex symbol to the yfinance =X format."""
    s = symbol.upper().replace("/", "")
    if not s.endswith("=X"):
        s += "=X"
    return s


class ForexFeed(DataHandler):
    """
    Replays historical forex bars sourced from yfinance.

    Parameters
    ----------
    symbols : list[str]
        Forex pairs in any of these formats: "EURUSD", "EUR/USD", "EURUSD=X".
        Internally normalized to yfinance format (e.g. "EURUSD=X").
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date "YYYY-MM-DD".
    interval : str
        yfinance interval. Daily ("1d") is the most reliable for forex.
        Intraday data is limited to the last 60 days.
    """

    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ):
        # Store canonical user-facing symbol → yfinance ticker mapping
        self._symbol_map: Dict[str, str] = {s: _to_yf_ticker(s) for s in symbols}
        self.symbols  = symbols
        self.start    = start
        self.end      = end
        self.interval = interval

        self._data: Dict[str, pd.DataFrame] = {}
        self._index: int = 0
        self._length: int = 0
        self._current_bars: Dict[str, dict] = {}

        self._load()

    # ------------------------------------------------------------------
    # DataHandler interface
    # ------------------------------------------------------------------

    def has_more(self) -> bool:
        return self._index < self._length

    def update_bars(self, events: Queue) -> None:
        for symbol, df in self._data.items():
            if self._index >= len(df):
                continue

            row = df.iloc[self._index]
            bar = self._row_to_bar(symbol, row)
            self._current_bars[symbol] = bar

            events.put(MarketEvent(
                symbol=symbol,
                asset_class="forex",
                timestamp=bar["timestamp"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
            ))

        self._index += 1

    def current_bars(self) -> Dict[str, dict]:
        return self._current_bars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info(
            f"Downloading forex: {self.symbols} | {self.interval} | {self.start} → {self.end}"
        )

        for symbol in self.symbols:
            yf_ticker = self._symbol_map[symbol]
            df = self._download(symbol, yf_ticker)
            if df is None or df.empty:
                raise ValueError(
                    f"No data returned for {symbol} (yfinance ticker: {yf_ticker}). "
                    "Check the pair and date range."
                )
            self._data[symbol] = df
            logger.info(f"  {symbol} ({yf_ticker}): {len(df)} bars loaded.")

        if len(self._data) > 1:
            self._align_symbols()

        self._length = len(next(iter(self._data.values())))

    def _download(self, symbol: str, yf_ticker: str) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(
                yf_ticker,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"yfinance download failed for {yf_ticker}: {e}")
            return None

        if raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]

        # Forex volume is often 0 or unreliable from yfinance — keep it but don't rely on it
        raw = raw[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )

        # Ensure UTC
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")

        raw = raw.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
        if "timestamp" not in raw.columns:
            raw = raw.rename(columns={raw.columns[0]: "timestamp"})

        raw["symbol"]      = symbol
        raw["asset_class"] = "forex"
        return raw.reset_index(drop=True)

    def _align_symbols(self) -> None:
        """Inner-join all symbols to common timestamps."""
        common_ts = None
        for df in self._data.values():
            ts_set = set(df["timestamp"])
            common_ts = ts_set if common_ts is None else common_ts & ts_set

        if not common_ts:
            raise ValueError("No overlapping timestamps across symbols after alignment.")

        for symbol, df in self._data.items():
            self._data[symbol] = df[df["timestamp"].isin(common_ts)].reset_index(drop=True)

        logger.info(f"Symbols aligned to {len(common_ts)} common bars.")

    @staticmethod
    def _row_to_bar(symbol: str, row: pd.Series) -> dict:
        return {
            "timestamp":   row["timestamp"],
            "open":        float(row["open"]),
            "high":        float(row["high"]),
            "low":         float(row["low"]),
            "close":       float(row["close"]),
            "volume":      float(row.get("volume", 0.0)),
            "symbol":      symbol,
            "asset_class": "forex",
        }
