"""Error types shared by providers, the router and the HTTP layer."""

from __future__ import annotations

from enum import Enum


class ErrorKind(str, Enum):
    """How the router should react to a failed upstream call."""

    #: Timeouts, connection errors, HTTP 429 and 5xx: retry the same target, then fall back.
    RETRYABLE = "retryable"
    #: Bad credentials, unknown model, feature the adapter cannot translate: skip to the next target.
    PROVIDER = "provider"
    #: The request itself is invalid (HTTP 400/413/422): return the error, do not fall back.
    CLIENT = "client"


class ProviderError(Exception):
    def __init__(self, message: str, kind: ErrorKind, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status

    @classmethod
    def from_status(cls, status: int, body: str) -> ProviderError:
        snippet = body.strip()[:300]
        message = f"upstream returned HTTP {status}" + (f": {snippet}" if snippet else "")
        if status == 429 or status >= 500:
            kind = ErrorKind.RETRYABLE
        elif status in (401, 403, 404):
            kind = ErrorKind.PROVIDER
        else:
            kind = ErrorKind.CLIENT
        return cls(message, kind, status)


class RouteNotFound(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"unknown model or route: {model!r}")
        self.model = model


class AllTargetsFailed(Exception):
    def __init__(self, route: str, attempts: list) -> None:
        super().__init__(f"all targets for route {route!r} failed")
        self.route = route
        self.attempts = attempts
