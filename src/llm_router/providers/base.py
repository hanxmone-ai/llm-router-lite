"""Provider interface. Every adapter speaks the OpenAI chat-completions format to the router."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import ProviderConfig
from ..errors import ErrorKind, ProviderError


@dataclass
class Usage:
    """Token counts in OpenAI terms: ``prompt_tokens`` includes ``cached_tokens``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: dict[str, Any] | None) -> Usage:
        if not usage:
            return cls()
        details = usage.get("prompt_tokens_details") or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }


@dataclass
class ChatResult:
    body: dict[str, Any]
    usage: Usage


def completion_chunk(
    chunk_id: str,
    model: str,
    created: int,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: Usage | None = None,
) -> dict[str, Any]:
    """Builds an OpenAI ``chat.completion.chunk`` object."""
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage.to_openai()
    return chunk


def completion_body(chat_id: str, model: str, text: str, finish_reason: str, usage: Usage) -> dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": usage.to_openai(),
    }


async def iter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yields the payload of each ``data:`` line of a server-sent-events stream."""
    async for line in lines:
        if line.startswith("data:"):
            yield line[5:].strip()


class Provider(ABC):
    def __init__(self, name: str, config: ProviderConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.name = name
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout, transport=transport)

    @property
    def disabled_reason(self) -> str | None:
        return self.config.disabled_reason()

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    @abstractmethod
    async def complete(self, model: str, request: dict[str, Any]) -> ChatResult:
        """Non-streaming chat completion. ``request`` is an OpenAI-style payload."""

    @abstractmethod
    def stream(self, model: str, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion as OpenAI ``chat.completion.chunk`` dicts.

        Implementations are async generators. They raise :class:`ProviderError` before the
        first chunk when the upstream rejects the request, which lets the router fall back.
        """

    async def embed(self, model: str, request: dict[str, Any]) -> ChatResult:
        """OpenAI-style embeddings. Adapters without embeddings fall back to the next target."""
        raise ProviderError(f"{self.name}: embeddings are not supported by this provider", ErrorKind.PROVIDER)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- helpers for HTTP adapters ---------------------------------------------------

    async def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: request timed out", ErrorKind.RETRYABLE) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"{self.name}: connection failed ({exc})", ErrorKind.RETRYABLE) from exc
        if response.status_code >= 400:
            raise ProviderError.from_status(response.status_code, response.text)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name}: invalid JSON in response", ErrorKind.RETRYABLE) from exc

    async def _stream_sse(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        """POSTs ``payload`` and yields each JSON event of the SSE response."""
        try:
            async with self._client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise ProviderError.from_status(response.status_code, body)
                async for data in iter_sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        return
                    if data:
                        yield json.loads(data)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: stream timed out", ErrorKind.RETRYABLE) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"{self.name}: connection failed ({exc})", ErrorKind.RETRYABLE) from exc
