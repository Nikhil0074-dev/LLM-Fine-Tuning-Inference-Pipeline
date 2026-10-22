from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

log = logging.getLogger("llmp.cache")


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: int) -> None: ...

    @abstractmethod
    def incr(self, key: str, ttl: int) -> int:
        """Atomically add 1 and return the new value. The key expires after ttl seconds."""

    def ping(self) -> bool:
        return True


class MemoryCache(Cache):
    def __init__(self, max_items: int = 10000):
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._max = max_items

    def _purge(self, now: float) -> None:
        for k in [k for k, (exp, _) in self._data.items() if exp <= now]:
            del self._data[k]
        while len(self._data) > self._max:
            self._data.pop(next(iter(self._data)))

    def get(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] <= now:
                self._data.pop(key, None)
                return None
            return item[1]

    def set(self, key: str, value: str, ttl: int) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now + ttl, value)
            if len(self._data) > self._max:
                self._purge(now)

    def incr(self, key: str, ttl: int) -> int:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            count = int(item[1]) + 1 if item and item[0] > now else 1
            exp = item[0] if item and item[0] > now else now + ttl
            self._data[key] = (exp, str(count))
            return count


class RedisCache(Cache):
    """Redis cache. Cache *reads/writes* never break inference: errors are logged and skipped.
    incr() raises on failure; the rate limiter catches it and fails open."""

    def __init__(self, url: str | None = None, client=None):
        if client is None:
            import redis  # lazy import

            client = redis.Redis.from_url(url or "redis://localhost:6379/0", socket_timeout=0.25, socket_connect_timeout=0.5, decode_responses=True)
        self.client = client

    def get(self, key: str) -> str | None:
        try:
            return self.client.get(key)
        except Exception as exc:
            log.warning("cache get failed: %s", exc)
            return None

    def set(self, key: str, value: str, ttl: int) -> None:
        try:
            self.client.set(key, value, ex=ttl)
        except Exception as exc:
            log.warning("cache set failed: %s", exc)

    def incr(self, key: str, ttl: int) -> int:
        pipe = self.client.pipeline()  # MULTI/EXEC: both commands run atomically
        pipe.set(key, 0, ex=ttl, nx=True)  # create with TTL only if missing (works on all Redis versions)
        pipe.incr(key)
        _, count = pipe.execute()
        return int(count)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False
