"""
cache.py — Query-level response cache + YouTube API quota tracker.

QueryCache
----------
In-memory LRU cache for full pipeline responses.
Key: SHA256 of normalized query string.
TTL: 1 hour (recipes don't change often).
Max: 200 entries — evicts oldest on overflow.

QuotaTracker
------------
Persists daily YouTube API call count to logs/youtube_quota.json.
Resets automatically at midnight UTC.
Warns at 80% usage, blocks at 100%.
Free tier: 100 searches/day (each search = 100 units, daily limit = 10,000 units).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 3600        # 1 hour
CACHE_MAX_ENTRIES = 200
QUOTA_DAILY_LIMIT = 100         # free YouTube searches/day
QUOTA_WARN_AT     = 80          # warn at 80%
QUOTA_FILE        = Path("logs/youtube_quota.json")


# ── Query cache ────────────────────────────────────────────────────────────────

class QueryCache:
    """
    Thread-safe in-memory LRU cache for pipeline responses.

    Usage:
        cache = QueryCache()
        key   = cache.make_key(query)
        hit   = cache.get(key)
        if hit:
            return hit
        result = await pipeline.run(query)
        cache.set(key, result)
    """

    def __init__(self, ttl: int = CACHE_TTL_SECONDS, max_size: int = CACHE_MAX_ENTRIES):
        self.ttl      = ttl
        self.max_size = max_size
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._lock    = Lock()
        self._hits    = 0
        self._misses  = 0

    @staticmethod
    def make_key(query: str) -> str:
        """Normalize query and return SHA256 hash as cache key."""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, key: str) -> dict | None:
        """Return cached response or None if missing/expired."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            value, expires_at = self._store[key]
            if time.time() > expires_at:
                del self._store[key]
                self._misses += 1
                log.debug("[cache] Expired: %s", key)
                return None

            # Move to end (LRU — most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            log.info("[cache] HIT key=%s (hits=%d misses=%d)", key, self._hits, self._misses)
            return value

    def set(self, key: str, value: dict) -> None:
        """Store response with TTL. Evicts oldest entry if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time() + self.ttl)

            # Evict oldest if over capacity
            while len(self._store) > self.max_size:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("[cache] Evicted: %s", evicted_key)

        log.info("[cache] SET key=%s (size=%d/%d)", key, len(self._store), self.max_size)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        log.info("[cache] Cleared all entries")

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":      len(self._store),
                "max_size":  self.max_size,
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  round(self._hits / max(total, 1) * 100, 1),
                "ttl_sec":   self.ttl,
            }


# ── YouTube quota tracker ──────────────────────────────────────────────────────

class QuotaTracker:
    """
    Tracks daily YouTube API search calls.
    Persists count to logs/youtube_quota.json so it survives restarts.

    Usage:
        tracker = QuotaTracker()
        if not tracker.can_call():
            raise Exception("Daily YouTube quota exhausted")
        tracker.record_call()
    """

    def __init__(self, daily_limit: int = QUOTA_DAILY_LIMIT, quota_file: Path = QUOTA_FILE):
        self.daily_limit = daily_limit
        self.quota_file  = quota_file
        self._lock       = Lock()
        self.quota_file.parent.mkdir(parents=True, exist_ok=True)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            if self.quota_file.exists():
                data = json.loads(self.quota_file.read_text())
                # Reset if it's a new day
                if data.get("date") != self._today():
                    return {"date": self._today(), "calls": 0}
                return data
        except Exception:
            pass
        return {"date": self._today(), "calls": 0}

    def _save(self, data: dict) -> None:
        self.quota_file.write_text(json.dumps(data, indent=2))

    def can_call(self) -> tuple[bool, int]:
        """
        Check if a YouTube API call is allowed.

        Returns
        -------
        (allowed, calls_remaining)
        """
        with self._lock:
            data      = self._load()
            calls     = data.get("calls", 0)
            remaining = self.daily_limit - calls

            if calls >= self.daily_limit:
                log.error(
                    "[quota] YouTube daily limit EXHAUSTED: %d/%d calls used",
                    calls, self.daily_limit,
                )
                return False, 0

            if calls >= self.daily_limit * QUOTA_WARN_AT / 100:
                log.warning(
                    "[quota] YouTube quota WARNING: %d/%d calls used (%.0f%%)",
                    calls, self.daily_limit, calls / self.daily_limit * 100,
                )

            return True, remaining

    def record_call(self) -> int:
        """Increment call counter. Returns updated count."""
        with self._lock:
            data = self._load()
            data["calls"] += 1
            data["date"]   = self._today()
            self._save(data)
            log.info(
                "[quota] YouTube call recorded: %d/%d used today",
                data["calls"], self.daily_limit,
            )
            return data["calls"]

    @property
    def status(self) -> dict:
        with self._lock:
            data  = self._load()
            calls = data.get("calls", 0)
            return {
                "date":           data.get("date"),
                "calls_used":     calls,
                "calls_remaining": max(self.daily_limit - calls, 0),
                "daily_limit":    self.daily_limit,
                "usage_pct":      round(calls / self.daily_limit * 100, 1),
                "exhausted":      calls >= self.daily_limit,
            }


# ── Singletons ─────────────────────────────────────────────────────────────────

query_cache   = QueryCache()
quota_tracker = QuotaTracker()