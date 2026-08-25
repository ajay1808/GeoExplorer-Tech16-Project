"""The TTL cache that keeps repeated questions off the HERE quota."""

from __future__ import annotations

import time

from geoexplorer.cache import TTLCache, make_key


def test_entries_expire(monkeypatch):
    cache = TTLCache(ttl_seconds=10)
    cache.set("k", "v")
    assert cache.get("k") == "v"

    later = time.monotonic() + 11
    monkeypatch.setattr(time, "monotonic", lambda: later)
    assert cache.get("k") is None


def test_cache_is_bounded_and_evicts_least_recently_used():
    cache = TTLCache(ttl_seconds=60, max_entries=3)
    for key in "abc":
        cache.set(key, key)
    cache.get("a")  # 'b' is now the least recently used
    cache.set("d", "d")

    assert cache.get("b") is None
    assert cache.get("a") == "a"
    assert len(cache) == 3


def test_hit_and_miss_counters():
    cache = TTLCache(ttl_seconds=60)
    cache.set("k", 1)
    cache.get("k")
    cache.get("nope")

    assert (cache.hits, cache.misses) == (1, 1)


def test_key_ignores_the_api_key_and_parameter_order():
    a = make_key("/v1/geocode", {"q": "x", "limit": 1, "apiKey": "secret-1"})
    b = make_key("/v1/geocode", {"limit": 1, "q": "x", "apiKey": "secret-2"})

    assert a == b
    assert "secret" not in a
