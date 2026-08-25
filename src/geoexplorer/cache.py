"""A small TTL cache for HERE responses.

Neighbourhood data does not change between two questions asked a minute apart, but the
agent will happily re-geocode the same address on every turn. Caching cuts both latency
and API quota. Deliberately dependency-free and synchronous — the working set is a
handful of entries per session.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """A thread-safe, size-bounded mapping whose entries expire after `ttl` seconds."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def make_key(endpoint: str, params: dict[str, Any]) -> str:
    """Build a stable cache key, excluding the API key so it never lands in memory twice."""
    parts = sorted(f"{k}={v}" for k, v in params.items() if k != "apiKey")
    return endpoint + "?" + "&".join(parts)
