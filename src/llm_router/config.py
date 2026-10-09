"""YAML configuration with ${ENV_VAR} interpolation, validated by pydantic."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any, environ: Mapping[str, str]) -> Any:
    """Replace ``${NAME}`` and ``${NAME:-default}`` in every string of a nested structure.

    A variable that is not set (and has no default) becomes an empty string, which lets a
    provider be switched off simply by not exporting its API key.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v, environ) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, environ) for k, v in value.items()}
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(_Strict):
    type: Literal["openai", "anthropic", "mock"]
    #: For ``openai``: any OpenAI-compatible endpoint (OpenAI, DeepSeek, Ollama, vLLM, ...).
    base_url: str | None = None
    #: ``None`` = not configured; ``""`` = configured from an environment variable that is not set.
    api_key: str | None = None
    timeout: float = Field(60.0, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)
    #: Mock provider only: ``echo`` answers, ``fail`` always returns HTTP 503 (to demo fallback).
    mode: Literal["echo", "fail"] = "echo"

    def disabled_reason(self) -> str | None:
        if self.type == "anthropic" and not self.api_key:
            return "api_key is not set"
        if self.api_key == "":
            return "api_key is not set"
        return None


class Target(_Strict):
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"


class RetryConfig(_Strict):
    #: Attempts per target (1 = no retry) before falling back to the next target.
    max_attempts: int = Field(2, ge=1, le=10)
    base_delay: float = Field(0.5, ge=0)
    max_delay: float = Field(8.0, ge=0)


class BreakerConfig(_Strict):
    failure_threshold: int = Field(3, ge=1)
    cooldown_seconds: float = Field(30.0, ge=0)


class Price(_Strict):
    """USD per 1M tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: float | None = Field(None, ge=0)


class ServerConfig(_Strict):
    #: Admin keys, sent as ``Authorization: Bearer <key>``: every route, every client's usage, /v1/status.
    #: With no admin keys and no clients, auth is off (local use only).
    api_keys: list[str] = Field(default_factory=list)
    usage_db: str = "data/usage.db"

    @field_validator("api_keys")
    @classmethod
    def _drop_empty(cls, keys: list[str]) -> list[str]:
        return [k for k in keys if k]


class ClientConfig(_Strict):
    """One API key per customer or application, with optional limits."""

    #: An empty key (environment variable not set) disables the client.
    api_key: str
    #: Routes (or ``provider/model`` targets) this client may call. Empty = all routes.
    routes: list[str] = Field(default_factory=list)
    #: Requests are refused with HTTP 429 once the month-to-date cost reaches this (needs ``prices``).
    monthly_budget_usd: float | None = Field(None, gt=0)
    requests_per_minute: int | None = Field(None, ge=1)


class Config(_Strict):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: dict[str, ProviderConfig]
    routes: dict[str, list[Target]]
    clients: dict[str, ClientConfig] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    prices: dict[str, Price] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_routes(self) -> Config:
        for name, targets in self.routes.items():
            if not targets:
                raise ValueError(f"route {name!r} has no targets")
            if "/" in name:
                raise ValueError(f"route name {name!r} must not contain '/' (reserved for provider/model)")
            for target in targets:
                if target.provider not in self.providers:
                    raise ValueError(f"route {name!r} uses unknown provider {target.provider!r}")
        keys: dict[str, str] = {}
        for name, client in self.clients.items():
            if name == "admin":
                raise ValueError("client name 'admin' is reserved")
            for route in client.routes:
                provider = route.partition("/")[0] if "/" in route else None
                if route not in self.routes and provider not in self.providers:
                    raise ValueError(f"client {name!r} allows unknown route {route!r}")
            if client.api_key:
                if client.api_key in keys or client.api_key in self.server.api_keys:
                    raise ValueError(f"client {name!r} reuses an API key")
                keys[client.api_key] = name
        return self


def load_config(path: str | Path, environ: Mapping[str, str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(expand_env(raw, os.environ if environ is None else environ))
