"""
data/ccxt_feed.py — Crypto data feed via CCXT.

Downloads historical OHLCV bars from any CCXT-supported exchange
(Binance, Coinbase, Kraken, etc.), normalizes to the standard bar schema,
and replays one bar at a time through the engine.

Requires the `ccxt` package:
    pip install ccxt

Usage:
    from data.ccxt_feed import CCXTFeed

    feed = CCXTFeed(
        symbols=["BTC/USDT", "ETH/USDT"],
        start="2022-01-01",
        end="2023-01-01",
        timeframe="1d",
        exchange_id="binance",
    )
"""

import logging
import time
from queue import Queue
from typing import Dict, List, Optional

import pandas as pd

from data.base import DataHandler
from core.event import MarketEvent

logger = logging.getLogger(__name__)

# CCXT returns timestamps in milliseconds
_MS_PER_S = 1000


class CCXTFeed(DataHandler):
    """
    Replays historical crypto OHLCV bars from a CCXT exchange.

    Parameters
    ----------
    symbols : list[str]
        CCXT-format trading pairs, e.g. ["BTC/USDT", "ETH/USDT"].
    start : str
        Start date "YYYY-MM-DD" (UTC).
    end : str
        End date "YYYY-MM-DD" (UTC).
    timeframe : str
        CCXT timeframe string: "1m", "5m", "15m", "1h", "4h", "1d", "1w".
    exchange_id : str
        CCXT exchange ID (default "binance"). Must support `fetch_ohlcv`.
    api_key : str | None
        Exchange API key (optional — public endpoints work without one).
    api_secret : str | None
        Exchange API secret (optional).
    request_delay : float
        Seconds to wait between paginated requests (avoid rate limits).
    """

    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1d",
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        request_delay: float = 0.2,
    ):
        self.symbols       = symbols
        self.start         = start
        self.end           = end
        self.timeframe     = timeframe
        self.exchange_id   = exchange_id
        self.request_delay = request_delay

        self._data: Dict[str, pd.DataFrame] = {}
        self._index: int = 0
        self._length: int = 0
        self._current_bars: Dict[str, dict] = {}

        self._exchange = self._build_exchange(exchange_id, api_key, api_secret)
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
                asset_class="crypto",
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
        if not self._exchange.has.get("fetchOHLCV"):
            raise ValueError(
                f"Exchange '{self.exchange_id}' does not support fetchOHLCV."
            )

        start_ms = int(pd.Timestamp(self.start, tz="UTC").timestamp() * _MS_PER_S)
        end_ms   = int(pd.Timestamp(self.end,   tz="UTC").timestamp() * _MS_PER_S)

        for symbol in self.symbols:
            df = self._fetch_ohlcv(symbol, start_ms, end_ms)
            if df.empty:
                raise ValueError(
                    f"No data returned for {symbol} on {self.exchange_id}. "
                    "Check the symbol format (e.g. 'BTC/USDT') and date range."
                )
            self._data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} bars loaded.")

        if len(self._data) > 1:
            self._align_symbols()

        self._length = len(next(iter(self._data.values())))

    def _fetch_ohlcv(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """
        Paginate through CCXT fetchOHLCV until we have all bars in range.
        CCXT exchanges cap results per request (usually 500–1000 bars).
        """
        logger.info(
            f"Fetching {symbol} from {self.exchange_id} | "
            f"{self.timeframe} | {self.start} → {self.end}"
        )

        all_bars = []
        since = start_ms

        while True:
            batch = self._exchange.fetch_ohlcv(
                symbol, timeframe=self.timeframe, since=since, limit=1000
            )
            if not batch:
                break

            # Filter out bars beyond the end date
            batch = [b for b in batch if b[0] < end_ms]
            if not batch:
                break

            all_bars.extend(batch)

            # If the exchange returned fewer bars than requested, we're done
            if len(batch) < 1000:
                break

            # Advance since to the last bar's timestamp + 1ms
            since = batch[-1][0] + 1
            time.sleep(self.request_delay)

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["symbol"]      = symbol
        df["asset_class"] = "crypto"
        df = df.dropna().reset_index(drop=True)
        return df

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
    def _build_exchange(exchange_id: str, api_key: Optional[str], api_secret: Optional[str]):
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt is not installed. Run: pip install ccxt")

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown CCXT exchange: '{exchange_id}'.")

        config = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        return exchange_class(config)

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
            "asset_class": "crypto",
        }
