import pytest
from pydantic import ValidationError

from llm_router.breaker import CircuitBreaker
from llm_router.config import Config, expand_env, load_config


def test_expand_env_with_defaults_and_missing_values():
    env = {"KEY": "secret"}
    data = {"a": "${KEY}", "b": ["x-${MISSING}", "${MISSING:-fallback}"], "c": 3}
    assert expand_env(data, env) == {"a": "secret", "b": ["x-", "fallback"], "c": 3}


def test_missing_api_key_disables_provider(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
server:
  api_keys: ["${ROUTER_KEY}"]
providers:
  claude: {type: anthropic, api_key: "${ANTHROPIC_API_KEY}"}
  local: {type: openai, base_url: "http://localhost:11434/v1"}
routes:
  smart:
    - {provider: claude, model: some-model}
    - {provider: local, model: llama}
""",
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.server.api_keys == []  # empty keys are dropped, so auth is off
    assert config.providers["claude"].disabled_reason() == "api_key is not set"
    assert config.providers["local"].disabled_reason() is None  # Ollama needs no key


def test_route_with_unknown_provider_is_rejected():
    with pytest.raises(ValidationError, match="unknown provider"):
        Config.model_validate({"providers": {}, "routes": {"r": [{"provider": "nope", "model": "x"}]}})


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"providers": {"p": {"type": "mock", "typo": 1}}, "routes": {}})


def test_breaker_opens_and_recovers_after_cooldown():
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=10, clock=lambda: now[0])
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow()

    now[0] = 10.0
    assert breaker.state == "half_open" and breaker.allow()
    breaker.record_failure()  # failed probe: open again
    assert breaker.state == "open"

    now[0] = 25.0
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.snapshot() == {"state": "closed", "consecutive_failures": 0}
