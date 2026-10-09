"""Offline provider for demos and tests: echoes the last user message, or always fails."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ErrorKind, ProviderError
from .anthropic import _text_of
from .base import ChatResult, Provider, Usage, completion_body, completion_chunk

EMBEDDING_DIMENSIONS = 256


def hashed_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic bag-of-character-n-grams vector: similar texts get similar vectors.

    Not semantic, but enough to demo and test retrieval without any API key.
    """
    vector = [0.0] * dimensions
    normalized = " ".join(text.lower().split())
    for n in (1, 2, 3):
        for i in range(len(normalized) - n + 1):
            gram = normalized[i : i + n]
            if gram.strip():
                digest = hashlib.blake2b(gram.encode(), digest_size=4).digest()
                vector[int.from_bytes(digest, "little") % dimensions] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / norm, 6) for v in vector]


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token) so the usage dashboard has numbers to show."""
    return max(1, len(text) // 4)


class MockProvider(Provider):
    def _check(self) -> None:
        if self.config.mode == "fail":
            raise ProviderError("mock provider is configured to fail", ErrorKind.RETRYABLE, status=503)

    @staticmethod
    def _answer(request: dict[str, Any]) -> tuple[str, Usage]:
        messages = request.get("messages", [])
        last_user = next((_text_of(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
        prompt_text = " ".join(_text_of(m.get("content")) for m in messages)
        answer = f"echo: {last_user}" if last_user else "echo: (empty)"
        return answer, Usage(prompt_tokens=estimate_tokens(prompt_text), completion_tokens=estimate_tokens(answer))

    async def complete(self, model: str, request: dict[str, Any]) -> ChatResult:
        self._check()
        answer, usage = self._answer(request)
        return ChatResult(
            body=completion_body(f"mock-{uuid.uuid4().hex[:12]}", model, answer, "stop", usage), usage=usage
        )

    async def stream(self, model: str, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self._check()
        answer, usage = self._answer(request)
        chunk_id, created = f"mock-{uuid.uuid4().hex[:12]}", int(time.time())
        yield completion_chunk(chunk_id, model, created, {"role": "assistant", "content": ""})
        words = answer.split(" ")
        for i, word in enumerate(words):
            await asyncio.sleep(0)
            yield completion_chunk(chunk_id, model, created, {"content": word + (" " if i < len(words) - 1 else "")})
        yield completion_chunk(chunk_id, model, created, {}, finish_reason="stop")
        yield completion_chunk(chunk_id, model, created, usage=usage)

    async def embed(self, model: str, request: dict[str, Any]) -> ChatResult:
        self._check()
        raw = request.get("input", "")
        texts = [raw] if isinstance(raw, str) else [str(t) for t in raw]
        usage = Usage(prompt_tokens=sum(estimate_tokens(t) for t in texts))
        body = {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": hashed_embedding(t)} for i, t in enumerate(texts)
            ],
            "usage": {"prompt_tokens": usage.prompt_tokens, "total_tokens": usage.prompt_tokens},
        }
        return ChatResult(body=body, usage=usage)
