"""HTTP layer: an OpenAI-compatible API plus usage, status and a small dashboard."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

from .auth import Authenticator, Principal, RateLimiter, month_start
from .config import Config, load_config
from .errors import AllTargetsFailed, ProviderError, RouteNotFound
from .providers import build_providers
from .router import Router
from .usage import UsageStore

log = logging.getLogger("llm_router")


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = None


class ChatRequest(BaseModel):
    """Only the fields the router needs are validated; everything else is passed through."""

    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    messages: list[_Message] = Field(min_length=1)
    stream: bool = False


def _error(status: int, message: str, kind: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, **extra}}, status_code=status)


def _sse(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    input: str | list[str]


def create_app(
    config: Config,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    usage_store: UsageStore | None = None,
    router: Router | None = None,
    clock: Callable[[], float] = time.time,
) -> Starlette:
    providers = build_providers(config, transport)
    usage = usage_store or UsageStore(config.server.usage_db)
    router = router or Router(config, providers, usage)
    auth = Authenticator(config)
    limiter = RateLimiter()
    dashboard_html = resources.files("llm_router").joinpath("static/dashboard.html").read_text(encoding="utf-8")

    def unauthorized() -> JSONResponse:
        return _error(401, "missing or invalid API key", "authentication_error")

    def check_limits(principal: Principal, model: str) -> JSONResponse | None:
        """Route allowlist, requests per minute and monthly budget for client keys."""
        if not principal.can_use(model):
            return _error(
                403,
                f"this API key may not use {model!r}",
                "permission_error",
                allowed_routes=principal.client.routes if principal.client else [],
            )
        client = principal.client
        if client is None:
            return None
        if client.requests_per_minute:
            wait = limiter.check(principal.name, client.requests_per_minute)
            if wait is not None:
                response = _error(429, "rate limit exceeded", "rate_limit_error")
                response.headers["retry-after"] = str(max(1, math.ceil(wait)))
                return response
        if client.monthly_budget_usd:
            spent = usage.cost_since(principal.name, month_start(clock()))
            if spent >= client.monthly_budget_usd:
                return _error(
                    429,
                    "monthly budget exhausted",
                    "budget_exceeded",
                    budget_usd=client.monthly_budget_usd,
                    spent_usd=round(spent, 4),
                )
        return None

    async def read_json(request: Request) -> dict | JSONResponse:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "request body must be JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return _error(400, "request body must be a JSON object", "invalid_request_error")
        return payload

    def upstream_error(exc: Exception) -> JSONResponse:
        if isinstance(exc, RouteNotFound):
            return _error(404, str(exc), "invalid_request_error", available_routes=sorted(config.routes))
        if isinstance(exc, AllTargetsFailed):
            return _error(502, str(exc), "upstream_error", attempts=[a.to_dict() for a in exc.attempts])
        assert isinstance(exc, ProviderError)
        return _error(exc.status or 400, str(exc), "invalid_request_error")

    async def chat_completions(request: Request) -> Response:
        principal = auth.authenticate(request.headers)
        if principal is None:
            return unauthorized()
        payload = await read_json(request)
        if isinstance(payload, JSONResponse):
            return payload
        try:
            parsed = ChatRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(400, "invalid chat request", "invalid_request_error", details=exc.errors(include_url=False))
        if (refused := check_limits(principal, parsed.model)) is not None:
            return refused

        try:
            if parsed.stream:
                routed = await router.open_stream(payload, client=principal.client_name)

                async def body() -> AsyncIterator[str]:
                    assert routed.chunks is not None
                    async for chunk in routed.chunks:
                        yield _sse(chunk)
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    body(),
                    media_type="text/event-stream",
                    headers={**routed.headers(), "cache-control": "no-cache"},
                )
            result = await router.complete(payload, client=principal.client_name)
            return JSONResponse(result.body, headers=result.headers())
        except (RouteNotFound, AllTargetsFailed, ProviderError) as exc:
            return upstream_error(exc)

    async def embeddings(request: Request) -> Response:
        principal = auth.authenticate(request.headers)
        if principal is None:
            return unauthorized()
        payload = await read_json(request)
        if isinstance(payload, JSONResponse):
            return payload
        try:
            parsed = EmbeddingRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                400, "invalid embeddings request", "invalid_request_error", details=exc.errors(include_url=False)
            )
        if (refused := check_limits(principal, parsed.model)) is not None:
            return refused
        try:
            result = await router.embed(payload, client=principal.client_name)
        except (RouteNotFound, AllTargetsFailed, ProviderError) as exc:
            return upstream_error(exc)
        return JSONResponse(result.body, headers=result.headers())

    async def list_models(request: Request) -> Response:
        principal = auth.authenticate(request.headers)
        if principal is None:
            return unauthorized()
        names = principal.allowed_routes(list(config.routes))
        data = [{"id": name, "object": "model", "owned_by": "llm-router-lite"} for name in names]
        return JSONResponse({"object": "list", "data": data})

    async def usage_summary(request: Request) -> Response:
        principal = auth.authenticate(request.headers)
        if principal is None:
            return unauthorized()
        try:
            hours = min(max(float(request.query_params.get("hours", "24")), 0.0), 24 * 365)
        except ValueError:
            return _error(400, "hours must be a number", "invalid_request_error")
        # Client keys only ever see their own usage; admins may filter with ?client=.
        scope = principal.client_name if not principal.is_admin else request.query_params.get("client") or None
        summary = usage.summary(since_ts=clock() - hours * 3600, client=scope)
        body = {"hours": hours, "client": scope, **summary, "recent": usage.recent(20, client=scope)}
        if principal.client and principal.client.monthly_budget_usd:
            body["budget"] = {
                "monthly_usd": principal.client.monthly_budget_usd,
                "spent_this_month_usd": round(usage.cost_since(principal.name, month_start(clock())), 6),
            }
        return JSONResponse(body)

    async def status(request: Request) -> Response:
        principal = auth.authenticate(request.headers)
        if principal is None:
            return unauthorized()
        if not principal.is_admin:
            return _error(403, "admin key required", "permission_error")
        clients = {
            name: {
                "enabled": bool(c.api_key),
                "routes": c.routes or "all",
                "monthly_budget_usd": c.monthly_budget_usd,
                "requests_per_minute": c.requests_per_minute,
            }
            for name, c in config.clients.items()
        }
        return JSONResponse({**router.status(), "clients": clients})

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def dashboard(_: Request) -> Response:
        return HTMLResponse(dashboard_html)

    async def index(_: Request) -> Response:
        return RedirectResponse("/dashboard")

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
        for provider in providers.values():
            await provider.aclose()
        usage.close()

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/healthz", health),
            Route("/dashboard", dashboard),
            Route("/v1/models", list_models),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/embeddings", embeddings, methods=["POST"]),
            Route("/v1/usage", usage_summary),
            Route("/v1/status", status),
        ],
        lifespan=lifespan,
    )
    app.state.router = router
    app.state.usage = usage
    return app


def app_from_env() -> Starlette:
    """ASGI factory: ``uvicorn llm_router.app:app_from_env --factory``."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    return create_app(load_config(os.environ.get("ROUTER_CONFIG", "config.yaml")))
