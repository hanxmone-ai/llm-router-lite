"""Adapter for Anthropic's Messages API, translated to and from the OpenAI format.

Supported: system prompts, text and image content, temperature/top_p/stop/max_tokens,
streaming, token usage including prompt-cache reads. Tool calling is not translated; a
request with tools makes this target fail with a provider error so the router falls back
to an OpenAI-compatible target.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ErrorKind, ProviderError
from .base import ChatResult, Provider, Usage, completion_body, completion_chunk

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _convert_part(part: dict[str, Any]) -> dict[str, Any]:
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": part.get("text", "")}
    if kind == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            media_type, data = url[5:].split(";base64,", 1)
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ProviderError(f"anthropic adapter: unsupported content part {kind!r}", ErrorKind.PROVIDER)


def to_anthropic(model: str, request: dict[str, Any]) -> dict[str, Any]:
    """Converts an OpenAI chat request into a Messages API payload."""
    if request.get("tools") or request.get("functions"):
        raise ProviderError("anthropic adapter: tool calling is not translated", ErrorKind.PROVIDER)

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in request.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            system_parts.append(_text_of(content))
        elif role in ("user", "assistant"):
            converted = content if isinstance(content, str) else [_convert_part(p) for p in content or []]
            messages.append({"role": role, "content": converted})
        else:
            raise ProviderError(f"anthropic adapter: unsupported message role {role!r}", ErrorKind.PROVIDER)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.get("max_tokens") or request.get("max_completion_tokens") or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        payload["system"] = "\n\n".join(p for p in system_parts if p)
    for key in ("temperature", "top_p"):
        if request.get(key) is not None:
            payload[key] = request[key]
    stop = request.get("stop")
    if stop:
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return payload


def _usage(raw: dict[str, Any], output_tokens: int | None = None) -> Usage:
    cached = int(raw.get("cache_read_input_tokens") or 0)
    created = int(raw.get("cache_creation_input_tokens") or 0)
    fresh = int(raw.get("input_tokens") or 0)
    return Usage(
        # Anthropic reports cache reads separately; OpenAI's prompt_tokens includes them.
        prompt_tokens=fresh + cached + created,
        completion_tokens=int(raw.get("output_tokens") or 0) if output_tokens is None else output_tokens,
        cached_tokens=cached,
    )


def from_anthropic(model: str, response: dict[str, Any]) -> ChatResult:
    text = "".join(b.get("text", "") for b in response.get("content", []) if b.get("type") == "text")
    usage = _usage(response.get("usage") or {})
    body = completion_body(
        chat_id=response.get("id", "msg"),
        model=response.get("model", model),
        text=text,
        finish_reason=STOP_REASONS.get(response.get("stop_reason") or "", "stop"),
        usage=usage,
    )
    return ChatResult(body=body, usage=usage)


class AnthropicProvider(Provider):
    @property
    def _url(self) -> str:
        return (self.config.base_url or DEFAULT_BASE_URL).rstrip("/") + "/messages"

    def _headers(self) -> dict[str, str]:
        return {
            **self.config.headers,
            "x-api-key": self.config.api_key or "",
            "anthropic-version": API_VERSION,
        }

    async def complete(self, model: str, request: dict[str, Any]) -> ChatResult:
        payload = to_anthropic(model, request)
        response = await self._post_json(self._url, payload, self._headers())
        return from_anthropic(model, response)

    async def stream(self, model: str, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        payload = {**to_anthropic(model, request), "stream": True}
        created = int(time.time())
        chunk_id, model_name = "msg", model
        usage = Usage()
        role_sent = False

        def role_chunk() -> dict[str, Any]:
            return completion_chunk(chunk_id, model_name, created, {"role": "assistant", "content": ""})

        async for event in self._stream_sse(self._url, payload, self._headers()):
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
                chunk_id = message.get("id", chunk_id)
                model_name = message.get("model", model_name)
                usage = _usage(message.get("usage") or {})
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    # The role chunk is held back until real content arrives, so an
                    # "overloaded" error right after message_start can still fall back.
                    if not role_sent:
                        role_sent = True
                        yield role_chunk()
                    yield completion_chunk(chunk_id, model_name, created, {"content": delta.get("text", "")})
            elif kind == "message_delta":
                if not role_sent:
                    role_sent = True
                    yield role_chunk()
                output = (event.get("usage") or {}).get("output_tokens")
                if output is not None:
                    usage.completion_tokens = int(output)
                reason = STOP_REASONS.get((event.get("delta") or {}).get("stop_reason") or "", "stop")
                yield completion_chunk(chunk_id, model_name, created, {}, finish_reason=reason)
            elif kind == "error":
                error = event.get("error") or {}
                retryable = error.get("type") in ("overloaded_error", "api_error", "rate_limit_error")
                raise ProviderError(
                    f"anthropic stream error: {error.get('message', 'unknown')}",
                    ErrorKind.RETRYABLE if retryable else ErrorKind.PROVIDER,
                )
            elif kind == "message_stop":
                break
        yield completion_chunk(chunk_id, model_name, created, usage=usage)
