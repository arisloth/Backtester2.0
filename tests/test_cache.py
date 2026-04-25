"""
tests/test_cache.py -- Cache invalidation and refresh controls.
"""

from datetime import datetime, timedelta, timezone
import json
import queue
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pandas.testing import assert_frame_equal

from data import cache


def _df(close=101.0):
    return pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ],
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, close],
        "volume": [1_000_000.0, 1_100_000.0],
        "symbol": ["SPY", "SPY"],
        "asset_class": ["stock", "stock"],
    })


def _raw_yfinance_df(index=None):
    if index is None:
        index = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
        ], name="Date")
    return pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1_000_000, 1_100_000],
    }, index=index)


class TestCacheInvalidation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self.tmp.name
        self.cache_patch = patch.object(cache, "_CACHE_DIR", self.cache_dir)
        self.cache_patch.start()
        self.args = ("yfinance", "SPY", "1d", "2024-01-01", "2024-01-03")

    def tearDown(self):
        self.cache_patch.stop()
        self.tmp.cleanup()

    def _paths(self):
        return cache._paths(*self.args)

    def test_fresh_cache_load_succeeds(self):
        df = _df()
        cache.save(df, *self.args)

        loaded = cache.load(*self.args)

        self.assertIsNotNone(loaded)
        assert_frame_equal(loaded, df)

    def test_expired_cache_returns_none(self):
        cache.save(_df(), *self.args)
        _, meta_path = self._paths()
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["created_at"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        self.assertIsNone(cache.load(*self.args, max_age_days=7))

    def test_refresh_returns_none(self):
        cache.save(_df(), *self.args)

        self.assertIsNone(cache.load(*self.args, refresh=True))

    def test_last_bar_hash_mismatch_returns_none(self):
        cache.save(_df(), *self.args)
        parquet_path, _ = self._paths()
        changed = _df(close=125.0)
        changed.to_parquet(parquet_path, index=False)

        self.assertIsNone(cache.load(*self.args))

    def test_changed_last_bar_save_logs_warning(self):
        cache.save(_df(close=101.0), *self.args)

        with self.assertLogs("data.cache", level="WARNING") as captured:
            cache.save(_df(close=125.0), *self.args)

        self.assertIn("Cache last bar changed", "\n".join(captured.output))


class TestCacheConfigPlumbing(unittest.TestCase):

    def test_build_data_handler_passes_cache_settings_to_yfinance_feed(self):
        import main

        cfg = dict(main.CONFIG)
        cfg.update({
            "data_source": "yfinance",
            "symbols": ["SPY"],
            "start": "2024-01-01",
            "end": "2024-01-03",
            "interval": "1d",
            "cache_ttl_days": 3,
            "refresh_cache": True,
        })

        module = types.ModuleType("data.yfinance_feed")
        module.YFinanceFeed = Mock()
        with patch.dict("sys.modules", {"data.yfinance_feed": module}):
            main.build_data_handler(cfg)

        feed_cls = module.YFinanceFeed
        feed_cls.assert_called_once()
        self.assertEqual(feed_cls.call_args.kwargs["cache_ttl_days"], 3)
        self.assertTrue(feed_cls.call_args.kwargs["refresh_cache"])


class TestYFinanceFeedCacheCorrectness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_patch = patch.object(cache, "_CACHE_DIR", self.tmp.name)
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop()
        self.tmp.cleanup()

    def test_cached_fetch_matches_fresh_fetch_exactly(self):
        from data.yfinance_feed import YFinanceFeed

        raw = _raw_yfinance_df()
        with patch("data.yfinance_feed.yf.download", return_value=raw.copy()) as download:
            fresh = YFinanceFeed(["SPY"], "2024-01-01", "2024-01-04", interval="1d")
            cached = YFinanceFeed(["SPY"], "2024-01-01", "2024-01-04", interval="1d")

        self.assertEqual(download.call_count, 1)
        assert_frame_equal(
            cached._data["SPY"],
            fresh._data["SPY"],
            check_exact=True,
        )
        self.assertTrue(
            pd.util.hash_pandas_object(cached._data["SPY"], index=True).equals(
                pd.util.hash_pandas_object(fresh._data["SPY"], index=True)
            )
        )

    def test_yfinance_dst_spring_forward_does_not_drop_or_duplicate_bars(self):
        from data.yfinance_feed import YFinanceFeed

        eastern = "America/New_York"
        raw = _raw_yfinance_df(pd.DatetimeIndex([
            pd.Timestamp("2024-03-08 09:30", tz=eastern),
            pd.Timestamp("2024-03-11 09:30", tz=eastern),
        ], name="Datetime"))

        with patch("data.cache.load", return_value=None), \
             patch("data.cache.save"), \
             patch("data.yfinance_feed.yf.download", return_value=raw):
            feed = YFinanceFeed(["SPY"], "2024-03-08", "2024-03-12", interval="1h")

        events = queue.Queue()
        emitted = []
        while feed.has_more():
            feed.update_bars(events)
            emitted.append(events.get_nowait())

        timestamps = [event.timestamp for event in emitted]
        self.assertEqual(len(timestamps), 2)
        self.assertEqual(len(set(timestamps)), 2)
        self.assertEqual(timestamps, [
            pd.Timestamp("2024-03-08 14:30", tz="UTC"),
            pd.Timestamp("2024-03-11 13:30", tz="UTC"),
        ])


if __name__ == "__main__":
    unittest.main()
