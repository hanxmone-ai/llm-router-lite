"""Who is calling (admin key, client key, or no auth) and per-client limits."""

from __future__ import annotations

import hmac
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import ClientConfig, Config


@dataclass(frozen=True)
class Principal:
    #: ``admin`` for admin keys and when auth is off, otherwise the client name from the config.
    name: str
    is_admin: bool
    client: ClientConfig | None = None

    @property
    def client_name(self) -> str | None:
        """Name recorded with usage; ``None`` for admin calls."""
        return None if self.is_admin else self.name

    def can_use(self, model: str) -> bool:
        if self.client is None or not self.client.routes:
            return True
        return model in self.client.routes

    def allowed_routes(self, all_routes: list[str]) -> list[str]:
        if self.client is None or not self.client.routes:
            return all_routes
        return [r for r in all_routes if r in self.client.routes]


ADMIN = Principal("admin", is_admin=True)


def bearer_token(headers: Mapping[str, str]) -> str:
    header = headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return headers.get("x-api-key", "").strip()


class Authenticator:
    def __init__(self, config: Config) -> None:
        self._admin_keys = [k for k in config.server.api_keys if k]
        # Clients whose key resolved to "" (env var not set) can never log in.
        self._clients = [(name, c) for name, c in config.clients.items() if c.api_key]

    @property
    def enabled(self) -> bool:
        return bool(self._admin_keys or self._clients)

    def authenticate(self, headers: Mapping[str, str]) -> Principal | None:
        if not self.enabled:
            return ADMIN
        token = bearer_token(headers)
        if not token:
            return None
        # Compare against every key so timing does not reveal which one matched.
        matched: Principal | None = None
        for key in self._admin_keys:
            if hmac.compare_digest(token.encode(), key.encode()):
                matched = ADMIN
        for name, client in self._clients:
            if hmac.compare_digest(token.encode(), client.api_key.encode()):
                matched = Principal(name, is_admin=False, client=client)
        return matched


class RateLimiter:
    """Sliding one-minute window per client, kept in memory."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, window_seconds: float = 60.0) -> None:
        self._clock = clock
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, limit: int) -> float | None:
        """Counts one request. Returns ``None`` if allowed, else seconds until a slot frees up."""
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= limit:
            return max(self._window - (now - hits[0]), 0.0)
        hits.append(now)
        return None


def month_start(now: float) -> float:
    """Unix time of the start of the current calendar month (UTC)."""
    current = datetime.fromtimestamp(now, tz=timezone.utc)
    return current.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
