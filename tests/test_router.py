import pytest

from llm_router.errors import AllTargetsFailed, ErrorKind, ProviderError, RouteNotFound
from llm_router.providers import build_providers
from llm_router.providers.base import ChatResult, Provider, Usage
from llm_router.router import Router
from llm_router.usage import UsageStore

from .conftest import make_config

REQUEST = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}


class ScriptedProvider(Provider):
    """Fails with the given errors in order, then succeeds."""

    def __init__(self, name, config, errors=()):
        super().__init__(name, config)
        self.errors = list(errors)
        self.calls = 0

    async def complete(self, model, request):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        usage = Usage(prompt_tokens=1000, completion_tokens=500, cached_tokens=400)
        return ChatResult(body={"model": model, "choices": []}, usage=usage)

    async def stream(self, model, request):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        yield {"choices": [{"index": 0, "delta": {"content": "a"}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def make_router(config, providers=None):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    store = UsageStore(":memory:")
    router = Router(config, providers or build_providers(config), store, sleep=fake_sleep, rand=lambda: 1.0)
    return router, store, sleeps


@pytest.mark.anyio
async def test_falls_back_after_retrying_a_failing_target(config):
    router, store, sleeps = make_router(config)
    result = await router.complete(REQUEST)

    assert result.target.key == "up/m2"
    assert [a.target for a in result.attempts] == ["down/m1", "down/m1", "up/m2"]
    assert len(sleeps) == 1  # one retry on the failing target before falling back
    assert result.headers()["x-router-attempts"] == "3"
    assert store.recent(1)[0]["provider"] == "up"


@pytest.mark.anyio
async def test_client_errors_are_returned_without_fallback(config):
    bad_request = ProviderError("bad request", ErrorKind.CLIENT, status=400)
    first = ScriptedProvider("down", config.providers["down"], errors=[bad_request])
    second = ScriptedProvider("up", config.providers["up"])
    router, _, _ = make_router(config, {"down": first, "up": second})

    with pytest.raises(ProviderError) as exc:
        await router.complete(REQUEST)
    assert exc.value.status == 400
    assert second.calls == 0


@pytest.mark.anyio
async def test_provider_errors_skip_to_next_target_without_retry(config):
    auth = ProviderError("401", ErrorKind.PROVIDER, status=401)
    first = ScriptedProvider("down", config.providers["down"], errors=[auth])
    second = ScriptedProvider("up", config.providers["up"])
    router, _, sleeps = make_router(config, {"down": first, "up": second})

    result = await router.complete(REQUEST)
    assert first.calls == 1 and second.calls == 1 and sleeps == []
    assert result.target.provider == "up"


@pytest.mark.anyio
async def test_open_circuit_is_skipped(config):
    router, _, _ = make_router(config)
    for _ in range(2):  # 2 requests x 2 attempts = 4 failures >= threshold 3
        await router.complete(REQUEST)
    assert router.status()["routes"]["demo"][0]["state"] == "open"

    result = await router.complete(REQUEST)
    assert result.attempts[0].skipped and result.attempts[0].error == "circuit open"
    assert result.headers()["x-router-attempts"] == "1"


@pytest.mark.anyio
async def test_disabled_provider_is_skipped():
    config = make_config(
        providers={"keyless": {"type": "anthropic", "api_key": ""}, "up": {"type": "mock"}},
        routes={"r": [{"provider": "keyless", "model": "x"}, {"provider": "up", "model": "m2"}]},
    )
    router, _, _ = make_router(config)
    result = await router.complete({**REQUEST, "model": "r"})
    assert result.attempts[0].skipped and result.attempts[0].error == "api_key is not set"
    assert result.target.provider == "up"


@pytest.mark.anyio
async def test_all_targets_failing_raises_and_is_recorded():
    config = make_config(routes={"dead": [{"provider": "down", "model": "m1"}]})
    router, store, _ = make_router(config)
    with pytest.raises(AllTargetsFailed) as exc:
        await router.complete({**REQUEST, "model": "dead"})
    assert len(exc.value.attempts) == 2
    row = store.recent(1)[0]
    assert row["ok"] is False and row["provider"] is None


@pytest.mark.anyio
async def test_direct_provider_model_and_unknown_route(config):
    router, _, _ = make_router(config)
    result = await router.complete({**REQUEST, "model": "up/any-model"})
    assert result.target.key == "up/any-model"
    with pytest.raises(RouteNotFound):
        await router.complete({**REQUEST, "model": "nope"})


@pytest.mark.anyio
async def test_cost_uses_cached_rate(config):
    provider = ScriptedProvider("up", config.providers["up"])
    router, store, _ = make_router(config, {"up": provider, "down": provider})
    await router.complete({**REQUEST, "model": "solo"})
    # 600 fresh x $1 + 400 cached x $0.1 + 500 output x $2, per 1M tokens
    assert store.recent(1)[0]["cost_usd"] == pytest.approx((600 * 1.0 + 400 * 0.1 + 500 * 2.0) / 1_000_000)


@pytest.mark.anyio
async def test_stream_falls_back_before_first_chunk_and_records_usage(config):
    failing = ScriptedProvider("down", config.providers["down"], errors=[ProviderError("503", ErrorKind.RETRYABLE)] * 2)
    healthy = ScriptedProvider("up", config.providers["up"])
    router, store, _ = make_router(config, {"down": failing, "up": healthy})

    routed = await router.open_stream(REQUEST)
    chunks = [c async for c in routed.chunks]
    assert routed.target.key == "up/m2"
    assert chunks[0]["choices"][0]["delta"]["content"] == "a"
    row = store.recent(1)[0]
    assert row["stream"] is True and row["ok"] is True and row["prompt_tokens"] == 3
