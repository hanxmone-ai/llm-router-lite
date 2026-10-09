"""HTTP layer: an OpenAI-compatible API plus usage, status and a small dashboard."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

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


def create_app(
    config: Config,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    usage_store: UsageStore | None = None,
    router: Router | None = None,
) -> Starlette:
    providers = build_providers(config, transport)
    usage = usage_store or UsageStore(config.server.usage_db)
    router = router or Router(config, providers, usage)
    dashboard_html = resources.files("llm_router").joinpath("static/dashboard.html").read_text(encoding="utf-8")

    def authorized(request: Request) -> bool:
        keys = config.server.api_keys
        if not keys:
            return True
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        return any(hmac.compare_digest(token.encode(), key.encode()) for key in keys)

    def unauthorized() -> JSONResponse:
        return _error(401, "missing or invalid API key", "authentication_error")

    async def chat_completions(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "request body must be JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return _error(400, "request body must be a JSON object", "invalid_request_error")
        try:
            parsed = ChatRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(400, "invalid chat request", "invalid_request_error", details=exc.errors(include_url=False))

        try:
            if parsed.stream:
                routed = await router.open_stream(payload)

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
            result = await router.complete(payload)
            return JSONResponse(result.body, headers=result.headers())
        except RouteNotFound as exc:
            return _error(404, str(exc), "invalid_request_error", available_routes=sorted(config.routes))
        except AllTargetsFailed as exc:
            return _error(502, str(exc), "upstream_error", attempts=[a.to_dict() for a in exc.attempts])
        except ProviderError as exc:
            return _error(exc.status or 400, str(exc), "invalid_request_error")

    async def list_models(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        data = [{"id": name, "object": "model", "owned_by": "llm-router-lite"} for name in config.routes]
        return JSONResponse({"object": "list", "data": data})

    async def usage_summary(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        try:
            hours = min(max(float(request.query_params.get("hours", "24")), 0.0), 24 * 365)
        except ValueError:
            return _error(400, "hours must be a number", "invalid_request_error")
        summary = usage.summary(since_ts=time.time() - hours * 3600)
        return JSONResponse({"hours": hours, **summary, "recent": usage.recent(20)})

    async def status(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        return JSONResponse(router.status())

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
