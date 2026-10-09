"""Adapter for any OpenAI-compatible endpoint: OpenAI, DeepSeek, Ollama, vLLM, LM Studio, ..."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .base import ChatResult, Provider, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(Provider):
    @property
    def _url(self) -> str:
        return (self.config.base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = dict(self.config.headers)
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def complete(self, model: str, request: dict[str, Any]) -> ChatResult:
        payload = {**request, "model": model, "stream": False}
        payload.pop("stream_options", None)
        body = await self._post_json(self._url, payload, self._headers())
        return ChatResult(body=body, usage=Usage.from_openai(body.get("usage")))

    async def stream(self, model: str, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        # include_usage makes the server send token counts in a final chunk.
        payload = {**request, "model": model, "stream": True, "stream_options": {"include_usage": True}}
        async for chunk in self._stream_sse(self._url, payload, self._headers()):
            yield chunk
