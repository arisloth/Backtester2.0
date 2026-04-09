"""
data/cache.py — On-disk parquet cache for OHLCV data.

Caches per-symbol normalized DataFrames keyed by
(source, symbol, interval, start, end) to avoid re-downloading
the same data across optimizer runs. Cache lives in data_cache/
at the project root and persists across sessions.

Historical price data is immutable so entries never expire.
Delete the data_cache/ directory to clear all cached data.
"""

import hashlib
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")


def _key(source: str, symbol: str, interval: str, start: str, end: str,
         adjustment: str = "raw") -> str:
    raw = f"{source}|{symbol}|{interval}|{start}|{end}|{adjustment}"
    return hashlib.md5(raw.encode()).hexdigest()


def load(source: str, symbol: str, interval: str, start: str, end: str,
         adjustment: str = "raw") -> Optional[pd.DataFrame]:
    """Return cached DataFrame or None if not cached."""
    path = os.path.join(_CACHE_DIR, _key(source, symbol, interval, start, end, adjustment) + ".parquet")
    try:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.debug(f"Cache hit: {source}|{symbol}|{interval}|{start}→{end}|{adjustment}")
            return df
    except Exception as exc:
        logger.warning(f"Cache read failed ({path}): {exc} — will re-fetch.")
    return None


def save(df: pd.DataFrame, source: str, symbol: str, interval: str, start: str, end: str,
         adjustment: str = "raw") -> None:
    """Persist DataFrame to cache. Silently skips on failure."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, _key(source, symbol, interval, start, end, adjustment) + ".parquet")
        df.to_parquet(path, index=False)
        logger.debug(f"Cache saved: {source}|{symbol}|{interval}|{start}→{end}|{adjustment}")
    except Exception as exc:
        logger.warning(f"Cache write failed: {exc}")
