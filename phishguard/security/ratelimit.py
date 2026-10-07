"""In-memory sliding-window rate limiting.

Deliberately dependency-free: a single-process counter is the correct tool for
the current single-worker deployment, and it avoids adding Redis before there
is a second process that needs to share state.

Security properties
-------------------
* Keys are the client IP. ``X-Forwarded-For`` is only trusted when
  ``TRUST_PROXY=1`` -- otherwise a caller could spoof the header and mint a
  fresh bucket per request to bypass the limit.
* Memory is bounded by ``max_keys``: when full, the oldest bucket is evicted.
  A caller who can rotate many keys could evict their own bucket (documented
  residual risk); callers behind one IP cannot.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after: int
    remaining: int


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float, max_keys: int = 10_000) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._window = window_seconds
        self._max_keys = max_keys
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def check(self, key: str, now: float | None = None) -> RateLimitResult:
        """Consume one token for ``key`` (allowed only when under the limit)."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    oldest_key = next(iter(self._buckets))
                    del self._buckets[oldest_key]
                bucket = deque()
                self._buckets[key] = bucket

            while bucket and moment - bucket[0] >= self._window:
                bucket.popleft()

            if len(bucket) >= self._limit:
                retry_after = max(int(self._window - (moment - bucket[0])) + 1, 1)
                return RateLimitResult(False, retry_after, 0)

            bucket.append(moment)
            return RateLimitResult(True, 0, self._limit - len(bucket))

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
