"""
data/cache.py -- On-disk parquet cache for OHLCV data.

Caches per-symbol normalized DataFrames keyed by
(source, symbol, interval, start, end) to avoid re-downloading
the same data across optimizer runs. Cache lives in data_cache/
at the project root and persists across sessions.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
_DEFAULT_MAX_AGE_DAYS = 7


def _key(source: str, symbol: str, interval: str, start: str, end: str) -> str:
    raw = f"{source}|{symbol}|{interval}|{start}|{end}"
    return hashlib.md5(raw.encode()).hexdigest()


def _paths(source: str, symbol: str, interval: str, start: str, end: str) -> Tuple[str, str]:
    base = os.path.join(_CACHE_DIR, _key(source, symbol, interval, start, end))
    return base + ".parquet", base + ".json"


def _last_bar_hash(df: pd.DataFrame) -> str:
    if df.empty:
        payload = {}
    else:
        payload = df.iloc[-1].to_dict()
    encoded = json.dumps(payload, default=str, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metadata(df: pd.DataFrame) -> dict:
    last_timestamp = None
    if not df.empty and "timestamp" in df.columns:
        last_timestamp = str(df.iloc[-1]["timestamp"])

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "last_timestamp": last_timestamp,
        "last_bar_hash": _last_bar_hash(df),
    }


def _read_metadata(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Cache metadata read failed ({path}): {exc} — will re-fetch.")
        return None


def _is_expired(meta: dict, max_age_days: Optional[float]) -> bool:
    if max_age_days is None:
        return False

    try:
        created_at = datetime.fromisoformat(meta["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.warning(f"Cache metadata has invalid created_at ({meta.get('created_at')}): {exc}")
        return True

    return datetime.now(timezone.utc) - created_at > timedelta(days=max_age_days)


def _matches_metadata(df: pd.DataFrame, meta: dict) -> bool:
    current = _metadata(df)
    for key in ("row_count", "last_timestamp", "last_bar_hash"):
        if current.get(key) != meta.get(key):
            logger.warning(
                f"Cache sanity check failed for {key}: "
                f"metadata={meta.get(key)} parquet={current.get(key)}"
            )
            return False
    return True


def load(
    source: str,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    max_age_days: Optional[float] = _DEFAULT_MAX_AGE_DAYS,
    refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Return cached DataFrame or None when absent, stale, invalid, or bypassed."""
    path, meta_path = _paths(source, symbol, interval, start, end)
    label = f"{source}|{symbol}|{interval}|{start}→{end}"

    if refresh:
        logger.info(f"Cache refresh requested: {label}")
        return None

    try:
        if not os.path.exists(path):
            return None
        if not os.path.exists(meta_path):
            logger.warning(f"Cache metadata missing for {label} — will re-fetch.")
            return None

        meta = _read_metadata(meta_path)
        if meta is None:
            return None
        if _is_expired(meta, max_age_days):
            logger.info(f"Cache expired: {label}")
            return None

        df = pd.read_parquet(path)
        if not _matches_metadata(df, meta):
            return None

        logger.debug(f"Cache hit: {label}")
        return df
    except Exception as exc:
        logger.warning(f"Cache read failed ({path}): {exc} — will re-fetch.")
    return None


def save(df: pd.DataFrame, source: str, symbol: str, interval: str, start: str, end: str) -> None:
    """Persist DataFrame and metadata to cache. Silently skips on failure."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path, meta_path = _paths(source, symbol, interval, start, end)
        old_meta = _read_metadata(meta_path) if os.path.exists(meta_path) else None
        new_meta = _metadata(df)

        df.to_parquet(path, index=False)
        with open(meta_path, "w") as f:
            json.dump(new_meta, f, indent=2, sort_keys=True)

        if old_meta and (
            old_meta.get("last_timestamp") != new_meta.get("last_timestamp")
            or old_meta.get("last_bar_hash") != new_meta.get("last_bar_hash")
        ):
            logger.warning(
                f"Cache last bar changed for {source}|{symbol}|{interval}|{start}→{end}: "
                f"{old_meta.get('last_timestamp')} -> {new_meta.get('last_timestamp')}"
            )

        logger.debug(f"Cache saved: {source}|{symbol}|{interval}|{start}→{end}")
    except Exception as exc:
        logger.warning(f"Cache write failed: {exc}")
