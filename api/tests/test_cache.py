"""
tests/test_cache.py — Unit tests for QueryCache and QuotaTracker.

Pure unit tests — no API, no LLM needed.

Run:
    pytest tests/test_cache.py -v
"""

import time
import tempfile
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cache import QueryCache, QuotaTracker


class TestQueryCache:

    def test_miss_on_empty_cache(self):
        cache = QueryCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = QueryCache()
        key   = cache.make_key("chicken dinner")
        cache.set(key, {"answer": "Here is a recipe..."})
        result = cache.get(key)
        assert result is not None
        assert result["answer"] == "Here is a recipe..."

    def test_key_normalized(self):
        cache = QueryCache()
        k1 = cache.make_key("Chicken Dinner")
        k2 = cache.make_key("chicken  dinner")
        k3 = cache.make_key("  chicken dinner  ")
        assert k1 == k2 == k3

    def test_different_queries_different_keys(self):
        cache = QueryCache()
        k1 = cache.make_key("chicken dinner")
        k2 = cache.make_key("beef stew")
        assert k1 != k2

    def test_ttl_expiry(self):
        cache = QueryCache(ttl=1)  # 1 second TTL
        key   = cache.make_key("test query")
        cache.set(key, {"answer": "result"})
        assert cache.get(key) is not None
        time.sleep(1.1)
        assert cache.get(key) is None

    def test_lru_eviction(self):
        cache = QueryCache(max_size=3)
        for i in range(3):
            key = cache.make_key(f"query {i}")
            cache.set(key, {"answer": f"result {i}"})
        # Add 4th — should evict oldest
        key4 = cache.make_key("query 3")
        cache.set(key4, {"answer": "result 3"})
        assert len(cache._store) == 3

    def test_clear(self):
        cache = QueryCache()
        cache.set(cache.make_key("q1"), {"answer": "a"})
        cache.set(cache.make_key("q2"), {"answer": "b"})
        cache.clear()
        assert len(cache._store) == 0

    def test_hit_rate_tracked(self):
        cache = QueryCache()
        key   = cache.make_key("chicken")
        cache.set(key, {"answer": "result"})
        cache.get(key)   # hit
        cache.get("xxx") # miss
        stats = cache.stats
        assert stats["hits"]   == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_stats_structure(self):
        cache = QueryCache()
        stats = cache.stats
        for field in ["size", "max_size", "hits", "misses", "hit_rate", "ttl_sec"]:
            assert field in stats


class TestQuotaTracker:

    def _make_tracker(self, limit=10):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        return QuotaTracker(daily_limit=limit, quota_file=tmp)

    def test_fresh_tracker_allows_calls(self):
        tracker = self._make_tracker(limit=10)
        allowed, remaining = tracker.can_call()
        assert allowed
        assert remaining == 10

    def test_record_call_increments(self):
        tracker = self._make_tracker(limit=10)
        tracker.record_call()
        assert tracker.status["calls_used"] == 1
        assert tracker.status["calls_remaining"] == 9

    def test_exhausted_blocks_calls(self):
        tracker = self._make_tracker(limit=3)
        for _ in range(3):
            tracker.record_call()
        allowed, remaining = tracker.can_call()
        assert not allowed
        assert remaining == 0

    def test_status_structure(self):
        tracker = self._make_tracker()
        status  = tracker.status
        for field in ["date", "calls_used", "calls_remaining", "daily_limit", "usage_pct", "exhausted"]:
            assert field in status

    def test_usage_pct_calculated(self):
        tracker = self._make_tracker(limit=10)
        tracker.record_call()
        tracker.record_call()
        assert tracker.status["usage_pct"] == 20.0

    def test_not_exhausted_when_within_limit(self):
        tracker = self._make_tracker(limit=10)
        tracker.record_call()
        assert not tracker.status["exhausted"]

    def test_exhausted_flag_set_at_limit(self):
        tracker = self._make_tracker(limit=2)
        tracker.record_call()
        tracker.record_call()
        assert tracker.status["exhausted"]

    def test_persists_across_instances(self):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        t1 = QuotaTracker(daily_limit=10, quota_file=tmp)
        t1.record_call()
        t1.record_call()
        # New instance reads same file
        t2 = QuotaTracker(daily_limit=10, quota_file=tmp)
        assert t2.status["calls_used"] == 2