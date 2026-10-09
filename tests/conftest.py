import pytest

from llm_router.config import Config


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_config(**overrides) -> Config:
    data = {
        "providers": {
            "up": {"type": "mock"},
            "down": {"type": "mock", "mode": "fail"},
        },
        "routes": {
            "demo": [{"provider": "down", "model": "m1"}, {"provider": "up", "model": "m2"}],
            "solo": [{"provider": "up", "model": "m2"}],
        },
        "retry": {"max_attempts": 2, "base_delay": 0, "max_delay": 0},
        "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 30},
        "prices": {"m2": {"input": 1.0, "output": 2.0, "cached_input": 0.1}},
        "server": {"usage_db": ":memory:"},
    }
    data.update(overrides)
    return Config.model_validate(data)


@pytest.fixture
def config() -> Config:
    return make_config()
