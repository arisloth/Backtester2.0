"""
data/alpaca_feed.py — Alpaca historical data feed for US stocks.

Uses the Alpaca Data API v2 to download historical OHLCV bars, normalizes
them to the standard bar schema, and replays them one bar at a time.

Requires the `alpaca-py` package:
    pip install alpaca-py

API keys are read from config/settings.py or passed directly. Free Alpaca
accounts have access to IEX data (15-min delayed); paid SIP feed gives full
market data. The feed works with either.

Usage:
    from data.alpaca_feed import AlpacaFeed
    feed = AlpacaFeed(
        symbols=["SPY", "AAPL"],
        start="2022-01-01",
        end="2023-01-01",
        timeframe="1Day",
    )
"""

import logging
from queue import Queue
from typing import Dict, List, Optional

import pandas as pd

from data.base import DataHandler
from core.event import MarketEvent

logger = logging.getLogger(__name__)


def _get_alpaca_client(api_key: Optional[str], api_secret: Optional[str]):
    """
    Build an Alpaca StockHistoricalDataClient. Falls back to config/settings.py
    if keys are not passed directly.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError:
        raise ImportError(
            "alpaca-py is not installed. Run: pip install alpaca-py"
        )

    if api_key is None or api_secret is None:
        try:
            from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
            api_key    = api_key    or ALPACA_API_KEY
            api_secret = api_secret or ALPACA_API_SECRET
        except ImportError:
            pass  # keys may still be None — Alpaca allows unauthenticated for some endpoints

    return StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)


class AlpacaFeed(DataHandler):
    """
    Replays historical stock bars from Alpaca Data API v2.

    Parameters
    ----------
    symbols : list[str]
        US stock tickers, e.g. ["SPY", "AAPL"].
    start : str
        Start date "YYYY-MM-DD" (UTC).
    end : str
        End date "YYYY-MM-DD" (UTC).
    timeframe : str
        Bar timeframe. Common values:
            "1Min", "5Min", "15Min", "1Hour", "1Day", "1Week"
        Passed directly to alpaca-py TimeFrame.
    api_key : str | None
        Alpaca API key. If None, reads from config/settings.py.
    api_secret : str | None
        Alpaca API secret. If None, reads from config/settings.py.
    feed : str
        Data feed: "iex" (free, delayed) or "sip" (paid, full market).
    adjustment : str
        Price adjustment: "raw", "split", "dividend", or "all".
        Defaults to "split" so stock splits don't create fake price gaps.
    """

    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: str = "iex",
        adjustment: str = "split",
    ):
        self.symbols    = symbols
        self.start      = start
        self.end        = end
        self.timeframe  = timeframe
        self.feed       = feed
        self.adjustment = adjustment

        self._data: Dict[str, pd.DataFrame] = {}
        self._index: int = 0
        self._length: int = 0
        self._current_bars: Dict[str, dict] = {}

        self._client = _get_alpaca_client(api_key, api_secret)
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
                asset_class="stock",
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
        from data.cache import load as cache_load, save as cache_save

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py")

        # Load each symbol from cache where possible; collect uncached symbols.
        symbols_to_fetch = []
        for symbol in self.symbols:
            cached = cache_load("alpaca", symbol, self.timeframe, self.start, self.end,
                                self.adjustment)
            if cached is not None:
                self._data[symbol] = cached
                logger.info(f"  {symbol}: loaded from cache.")
            else:
                symbols_to_fetch.append(symbol)

        if symbols_to_fetch:
            tf = self._parse_timeframe(self.timeframe)
            request = StockBarsRequest(
                symbol_or_symbols=symbols_to_fetch,
                timeframe=tf,
                start=pd.Timestamp(self.start, tz="UTC"),
                end=pd.Timestamp(self.end,   tz="UTC"),
                feed=self.feed,
                adjustment=self.adjustment,
            )

            logger.info(
                f"Fetching Alpaca bars: {symbols_to_fetch} | {self.timeframe} "
                f"| {self.start} → {self.end} | feed={self.feed}"
            )

            raw = self._client.get_stock_bars(request).df

            if raw.empty:
                raise ValueError(
                    f"No data returned from Alpaca for {symbols_to_fetch}. "
                    "Check symbols, date range, and API keys."
                )

            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]

            if raw["timestamp"].dt.tz is None:
                raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
            else:
                raw["timestamp"] = raw["timestamp"].dt.tz_convert("UTC")

            for symbol in symbols_to_fetch:
                sym_df = raw[raw["symbol"] == symbol].copy()
                if sym_df.empty:
                    raise ValueError(f"No bars returned for {symbol}.")
                sym_df = sym_df[["timestamp","open","high","low","close","volume"]].dropna()
                sym_df["symbol"]      = symbol
                sym_df["asset_class"] = "stock"
                sym_df = sym_df.reset_index(drop=True)
                self._data[symbol] = sym_df
                cache_save(sym_df, "alpaca", symbol, self.timeframe, self.start, self.end,
                           self.adjustment)
                logger.info(f"  {symbol}: {len(sym_df)} bars loaded.")

        if len(self._data) > 1:
            self._align_symbols()

        self._length = len(next(iter(self._data.values())))

    def _align_symbols(self) -> None:
        """Inner-join all symbols on timestamp so every bar has data for all symbols."""
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
    def _parse_timeframe(tf_str: str):
        """Convert a string like '1Day' or '15Min' to an alpaca-py TimeFrame."""
        try:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py")

        mapping = {
            "1min":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
            "1hour": TimeFrame(1,  TimeFrameUnit.Hour),
            "4hour": TimeFrame(4,  TimeFrameUnit.Hour),
            "1day":  TimeFrame(1,  TimeFrameUnit.Day),
            "1week": TimeFrame(1,  TimeFrameUnit.Week),
        }
        key = tf_str.lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown timeframe '{tf_str}'. "
                f"Supported: {list(mapping.keys())}"
            )
        return mapping[key]

    @staticmethod
    def _row_to_bar(symbol: str, row: pd.Series) -> dict:
        return {
            "timestamp":   row["timestamp"],
            "open":        float(row["open"]),
            "high":        float(row["high"]),
            "low":         float(row["low"]),
            "close":       float(row["close"]),
            "volume":      float(row["volume"]),
            "symbol":      symbol,
            "asset_class": "stock",
        }
