"""Sliding-window rate limiter tests."""

import pytest

from phishguard.security.ratelimit import SlidingWindowRateLimiter


def test_allows_up_to_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
    assert all(limiter.check("ip").allowed for _ in range(3))
    result = limiter.check("ip")
    assert not result.allowed
    assert result.retry_after >= 1
    assert result.remaining == 0


def test_window_slides_and_frees_tokens():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    start = 1_000.0
    assert limiter.check("ip", now=start).allowed
    assert limiter.check("ip", now=start).allowed
    assert not limiter.check("ip", now=start + 30).allowed
    # after the window passes, the old tokens expire
    assert limiter.check("ip", now=start + 61).allowed


def test_keys_are_independent():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    assert limiter.check("a").allowed
    assert not limiter.check("a").allowed
    assert limiter.check("b").allowed


def test_reset_clears_state():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    limiter.check("ip")
    limiter.reset()
    assert limiter.check("ip").allowed


def test_memory_is_bounded_by_max_keys():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60, max_keys=2)
    limiter.check("a")
    limiter.check("b")
    limiter.check("c")  # evicts the oldest bucket instead of growing forever
    assert limiter.check("a").allowed


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(limit=0, window_seconds=60)
