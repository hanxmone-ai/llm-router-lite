"""Chooses a target for each request, with retries, fallback, circuit breaking and accounting."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .breaker import CircuitBreaker
from .config import Config, Target
from .errors import AllTargetsFailed, ErrorKind, ProviderError, RouteNotFound
from .providers.base import ChatResult, Provider, Usage
from .usage import UsageRecord, UsageStore, compute_cost

log = logging.getLogger("llm_router")


@dataclass
class Attempt:
    target: str
    ok: bool
    error: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Routed:
    route: str
    target: Target
    attempts: list[Attempt]

    def headers(self) -> dict[str, str]:
        return {
            "x-router-route": self.route,
            "x-router-target": self.target.key,
            "x-router-attempts": str(len([a for a in self.attempts if not a.skipped])),
        }


@dataclass
class RoutedResult(Routed):
    body: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)


@dataclass
class RoutedStream(Routed):
    chunks: AsyncIterator[dict[str, Any]] | None = None


class Router:
    def __init__(
        self,
        config: Config,
        providers: dict[str, Provider],
        usage: UsageStore,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.providers = providers
        self.usage = usage
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self._rand = rand
        self._breakers: dict[str, CircuitBreaker] = {}
        for targets in config.routes.values():
            for target in targets:
                self._breaker(target)

    # ---- routing --------------------------------------------------------------------

    def resolve(self, model: str) -> tuple[str, list[Target]]:
        """``model`` is a route name from the config, or ``provider/model`` to bypass routing."""
        if model in self.config.routes:
            return model, self.config.routes[model]
        provider, sep, name = model.partition("/")
        if sep and name and provider in self.providers:
            return model, [Target(provider=provider, model=name)]
        raise RouteNotFound(model)

    def status(self) -> dict:
        return {
            "providers": {
                name: {"type": p.config.type, "enabled": p.enabled, "disabled_reason": p.disabled_reason}
                for name, p in self.providers.items()
            },
            "routes": {
                route: [{"target": t.key, **self._breaker(t).snapshot()} for t in targets]
                for route, targets in self.config.routes.items()
            },
        }

    def _breaker(self, target: Target) -> CircuitBreaker:
        breaker = self._breakers.get(target.key)
        if breaker is None:
            cb = self.config.circuit_breaker
            breaker = CircuitBreaker(cb.failure_threshold, cb.cooldown_seconds, clock=self._clock)
            self._breakers[target.key] = breaker
        return breaker

    def _backoff(self, attempt: int) -> float:
        retry = self.config.retry
        delay = min(retry.max_delay, retry.base_delay * (2**attempt))
        return delay * (0.5 + self._rand() / 2)  # jitter: 50-100% of the delay

    async def _try_targets(
        self,
        route: str,
        targets: list[Target],
        call: Callable[[Provider, Target], Awaitable[Any]],
    ) -> tuple[Target, Any, list[Attempt]]:
        """Runs ``call`` against each target in order until one succeeds."""
        attempts: list[Attempt] = []
        for target in targets:
            provider = self.providers[target.provider]
            if not provider.enabled:
                attempts.append(Attempt(target.key, False, provider.disabled_reason, skipped=True))
                continue
            breaker = self._breaker(target)
            if not breaker.allow():
                attempts.append(Attempt(target.key, False, "circuit open", skipped=True))
                continue

            for n in range(self.config.retry.max_attempts):
                try:
                    value = await call(provider, target)
                except ProviderError as exc:
                    attempts.append(Attempt(target.key, False, str(exc)))
                    if exc.kind is ErrorKind.CLIENT:
                        raise
                    breaker.record_failure()
                    log.warning("route=%s target=%s attempt=%d failed: %s", route, target.key, n + 1, exc)
                    last_try = n + 1 >= self.config.retry.max_attempts
                    if exc.kind is ErrorKind.RETRYABLE and not last_try and breaker.allow():
                        await self._sleep(self._backoff(n))
                        continue
                    break
                else:
                    breaker.record_success()
                    attempts.append(Attempt(target.key, True))
                    return target, value, attempts
        raise AllTargetsFailed(route, attempts)

    # ---- public API -----------------------------------------------------------------

    async def complete(self, request: dict[str, Any], client: str | None = None) -> RoutedResult:
        return await self._unary(request, client, "chat", lambda p, t: p.complete(t.model, request))

    async def embed(self, request: dict[str, Any], client: str | None = None) -> RoutedResult:
        """``POST /v1/embeddings``: same routing, retries and fallback as chat."""
        return await self._unary(request, client, "embeddings", lambda p, t: p.embed(t.model, request))

    async def _unary(
        self,
        request: dict[str, Any],
        client: str | None,
        endpoint: str,
        call: Callable[[Provider, Target], Awaitable[ChatResult]],
    ) -> RoutedResult:
        route, targets = self.resolve(request["model"])
        started = self._clock()
        meta = {"client": client, "endpoint": endpoint, "stream": False}
        try:
            target, result, attempts = await self._try_targets(route, targets, call)
        except AllTargetsFailed as exc:
            self._record(route, None, started, ok=False, attempts=exc.attempts, error=str(exc), **meta)
            raise
        except ProviderError as exc:
            self._record(route, None, started, ok=False, attempts=[], error=str(exc), **meta)
            raise
        self._record(route, target, started, ok=True, attempts=attempts, usage=result.usage, **meta)
        return RoutedResult(route=route, target=target, attempts=attempts, body=result.body, usage=result.usage)

    async def open_stream(self, request: dict[str, Any], client: str | None = None) -> RoutedStream:
        """Starts a stream on the first healthy target.

        The first chunk is read before returning, so a target that fails up front is
        retried or skipped. Once a chunk has been sent to the client there is no fallback;
        a later upstream error is reported as a final error chunk.
        """
        route, targets = self.resolve(request["model"])
        started = self._clock()
        meta = {"client": client, "endpoint": "chat", "stream": True}

        async def call(provider: Provider, target: Target):
            chunks = provider.stream(target.model, request)
            try:
                first = await anext(chunks)
            except StopAsyncIteration:
                first = None
            except BaseException:
                await chunks.aclose()
                raise
            return chunks, first

        try:
            target, (chunks, first), attempts = await self._try_targets(route, targets, call)
        except AllTargetsFailed as exc:
            self._record(route, None, started, ok=False, attempts=exc.attempts, error=str(exc), **meta)
            raise
        except ProviderError as exc:
            self._record(route, None, started, ok=False, attempts=[], error=str(exc), **meta)
            raise
        relay = self._relay(route, target, attempts, started, chunks, first, meta)
        return RoutedStream(route=route, target=target, attempts=attempts, chunks=relay)

    async def _relay(
        self,
        route: str,
        target: Target,
        attempts: list[Attempt],
        started: float,
        chunks: AsyncIterator[dict[str, Any]],
        first: dict[str, Any] | None,
        meta: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        usage = Usage()
        error: str | None = "client disconnected"
        try:
            if first is not None:
                if first.get("usage"):
                    usage = Usage.from_openai(first["usage"])
                yield first
            async for chunk in chunks:
                if chunk.get("usage"):
                    usage = Usage.from_openai(chunk["usage"])
                yield chunk
            error = None
        except ProviderError as exc:
            error = str(exc)
            yield {"error": {"message": f"upstream failed mid-stream: {exc}", "type": "upstream_error"}}
        finally:
            await chunks.aclose()
            self._record(route, target, started, ok=error is None, attempts=attempts, usage=usage, error=error, **meta)

    # ---- accounting -----------------------------------------------------------------

    def _record(
        self,
        route: str,
        target: Target | None,
        started: float,
        *,
        ok: bool,
        stream: bool,
        attempts: list[Attempt],
        usage: Usage | None = None,
        error: str | None = None,
        client: str | None = None,
        endpoint: str = "chat",
    ) -> None:
        usage = usage or Usage()
        price = self.config.prices.get(target.model) if target else None
        try:
            self.usage.record(
                UsageRecord(
                    ts=self._wall_clock(),
                    route=route,
                    provider=target.provider if target else None,
                    model=target.model if target else None,
                    ok=ok,
                    stream=stream,
                    attempts=len([a for a in attempts if not a.skipped]),
                    latency_ms=int((self._clock() - started) * 1000),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    cost_usd=compute_cost(usage, price),
                    error=error,
                    client=client,
                    endpoint=endpoint,
                )
            )
        except Exception:  # accounting must never break a response
            log.exception("failed to record usage")
