from __future__ import annotations

import httpx

from ..config import Config
from .anthropic import AnthropicProvider
from .base import ChatResult, Provider, Usage
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

_TYPES: dict[str, type[Provider]] = {
    "openai": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def build_providers(config: Config, transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Provider]:
    return {name: _TYPES[pc.type](name, pc, transport) for name, pc in config.providers.items()}


__all__ = ["ChatResult", "Provider", "Usage", "build_providers"]
