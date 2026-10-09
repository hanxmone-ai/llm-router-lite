"""A small circuit breaker so a failing upstream is skipped instead of retried on every request."""

from __future__ import annotations

import time
from collections.abc import Callable


class CircuitBreaker:
    """closed → (N consecutive failures) → open → (cooldown) → half-open → closed/open.

    In the half-open state requests are let through again; the first success closes the
    breaker and the next failure reopens it for another cooldown.
    """

    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.cooldown_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self._failures}
