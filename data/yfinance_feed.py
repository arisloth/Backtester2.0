"""
data/yfinance_feed.py — yfinance data feed for stocks and crypto.

Downloads historical OHLCV data via yfinance, normalizes it to the
standard bar schema, and replays it one bar at a time through the engine.

Supports multiple symbols in a single feed (multi-asset backtests).
All timestamps are converted to UTC.
"""

import logging
from queue import Queue
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from data.base import DataHandler
from core.event import MarketEvent

logger = logging.getLogger(__name__)

# yfinance ticker suffixes that indicate crypto (e.g. BTC-USD)
_CRYPTO_SUFFIXES = ("-USD", "-USDT", "-BTC", "-ETH")


def _infer_asset_class(symbol: str) -> str:
    """Infer asset class from ticker format."""
    upper = symbol.upper()
    if any(upper.endswith(s) for s in _CRYPTO_SUFFIXES):
        return "crypto"
    if "=X" in upper:
        return "forex"
    return "stock"


class YFinanceFeed(DataHandler):
    """
    Replays historical data downloaded from yfinance, bar by bar.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols, e.g. ["SPY", "AAPL"] or ["BTC-USD"].
    start : str
        Start date, "YYYY-MM-DD".
    end : str
        End date, "YYYY-MM-DD" (exclusive in yfinance).
    interval : str
        Bar interval. yfinance supports: "1m","2m","5m","15m","30m",
        "60m","90m","1h","1d","5d","1wk","1mo","3mo".
        Note: intraday data is limited to 60 days of history.
    asset_class : str | None
        Override the inferred asset class ("stock", "crypto", "forex").
        If None, it is inferred from the ticker format.
    """

    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
        asset_class: Optional[str] = None,
    ):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.interval = interval

        # Per-symbol asset class
        self._asset_classes: Dict[str, str] = {
            s: (asset_class if asset_class else _infer_asset_class(s))
            for s in symbols
        }

        # symbol → DataFrame of normalized bars (indexed 0..N-1)
        self._data: Dict[str, pd.DataFrame] = {}

        # Current bar index (same for all symbols — bars are aligned)
        self._index: int = 0
        self._length: int = 0

        # Most recently emitted bars (used by broker)
        self._current_bars: Dict[str, dict] = {}

        self._load()

    # ------------------------------------------------------------------
    # DataHandler interface
    # ------------------------------------------------------------------

    def has_more(self) -> bool:
        return self._index < self._length

    def update_bars(self, events: Queue) -> None:
        """Push one MarketEvent per symbol for the current bar index."""
        for symbol, df in self._data.items():
            if self._index >= len(df):
                continue  # symbol has fewer bars (shouldn't happen after alignment)

            row = df.iloc[self._index]
            bar = self._row_to_bar(symbol, row)
            self._current_bars[symbol] = bar

            events.put(MarketEvent(
                symbol=symbol,
                asset_class=bar["asset_class"],
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
        """Download and normalize data for all symbols."""
        logger.info(
            f"Downloading {self.symbols} | {self.interval} | {self.start} → {self.end}"
        )

        for symbol in self.symbols:
            df = self._download(symbol)
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}. Check symbol and date range.")
            self._data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} bars loaded.")

        # Align all symbols to the same timestamp index (inner join)
        if len(self._data) > 1:
            self._align_symbols()

        # All DataFrames now share the same length
        lengths = [len(df) for df in self._data.values()]
        self._length = lengths[0] if lengths else 0

    def _download(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download from yfinance and normalize to standard bar schema."""
        try:
            raw = yf.download(
                symbol,
                start=self.start,
                end=self.end,
                interval=self.interval,
                auto_adjust=True,   # adjust for splits/dividends
                progress=False,
            )
        except Exception as e:
            logger.error(f"yfinance download failed for {symbol}: {e}")
            return None

        if raw.empty:
            return None

        # yfinance may return a MultiIndex if multiple symbols were requested
        # in a single call; we always call it per-symbol so flatten if needed.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalize column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]

        # Ensure UTC timestamps
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")

        # Drop rows with any NaN in OHLCV
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()

        # Reset to integer index; keep timestamp as a column
        raw = raw.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
        if "timestamp" not in raw.columns:
            # Fallback: the index column might have a different name
            raw = raw.rename(columns={raw.columns[0]: "timestamp"})

        raw["symbol"] = symbol
        raw["asset_class"] = self._asset_classes[symbol]

        return raw.reset_index(drop=True)

    def _align_symbols(self) -> None:
        """
        Align all symbol DataFrames to the same set of timestamps (inner join).
        Bars missing for any symbol on a given timestamp are dropped for all.
        """
        # Build a common timestamp set
        common_ts = None
        for df in self._data.values():
            ts_set = set(df["timestamp"])
            common_ts = ts_set if common_ts is None else common_ts & ts_set

        if not common_ts:
            raise ValueError("No overlapping timestamps across symbols after alignment.")

        for symbol, df in self._data.items():
            aligned = df[df["timestamp"].isin(common_ts)].reset_index(drop=True)
            self._data[symbol] = aligned

        dropped = sum(1 for df in self._data.values() for _ in [None]) - 1  # just log
        logger.info(f"Symbols aligned to {len(common_ts)} common bars.")

    @staticmethod
    def _row_to_bar(symbol: str, row: pd.Series) -> dict:
        return {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "symbol": symbol,
            "asset_class": row["asset_class"],
        }
