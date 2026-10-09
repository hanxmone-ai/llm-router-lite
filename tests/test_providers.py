import json

import httpx
import pytest

from llm_router.config import ProviderConfig
from llm_router.errors import ErrorKind, ProviderError
from llm_router.providers.anthropic import AnthropicProvider, to_anthropic
from llm_router.providers.openai_compat import OpenAICompatProvider

REQUEST = {"model": "alias", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}


def sse(*events) -> bytes:
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events).encode()


def openai_provider(handler):
    config = ProviderConfig(type="openai", base_url="https://llm.example/v1", api_key="k")
    return OpenAICompatProvider("oa", config, transport=httpx.MockTransport(handler))


def anthropic_provider(handler):
    config = ProviderConfig(type="anthropic", api_key="k")
    return AnthropicProvider("an", config, transport=httpx.MockTransport(handler))


# ---- OpenAI-compatible --------------------------------------------------------------


@pytest.mark.anyio
async def test_openai_complete_sends_target_model_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}},
            },
        )

    result = await openai_provider(handler).complete("gpt-x", REQUEST)
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "gpt-x" and seen["body"]["temperature"] == 0.2
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.cached_tokens) == (10, 2, 4)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,kind",
    [(429, ErrorKind.RETRYABLE), (503, ErrorKind.RETRYABLE), (401, ErrorKind.PROVIDER), (400, ErrorKind.CLIENT)],
)
async def test_openai_status_codes_map_to_error_kinds(status, kind):
    provider = openai_provider(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(ProviderError) as exc:
        await provider.complete("gpt-x", REQUEST)
    assert exc.value.kind is kind and exc.value.status == status


@pytest.mark.anyio
async def test_openai_connection_errors_are_retryable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as exc:
        await openai_provider(handler).complete("gpt-x", REQUEST)
    assert exc.value.kind is ErrorKind.RETRYABLE


@pytest.mark.anyio
async def test_openai_stream_requests_usage_and_parses_events():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        body = sse(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            "[DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in openai_provider(handler).stream("gpt-x", REQUEST)]
    assert seen["body"]["stream"] is True and seen["body"]["stream_options"] == {"include_usage": True}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Hello"
    assert chunks[-1]["usage"]["prompt_tokens"] == 5


@pytest.mark.anyio
async def test_openai_stream_error_status_raises_before_first_chunk():
    provider = openai_provider(lambda r: httpx.Response(529, text="overloaded"))
    with pytest.raises(ProviderError) as exc:
        await anext(provider.stream("gpt-x", REQUEST))
    assert exc.value.kind is ErrorKind.RETRYABLE


# ---- Anthropic -------------------------------------------------------------------------


def test_to_anthropic_moves_system_prompt_and_maps_options():
    payload = to_anthropic(
        "claude-x",
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                },
            ],
            "stop": "END",
            "temperature": 0,
        },
    )
    assert payload["system"] == "Be brief."
    assert payload["max_tokens"] == 1024
    assert payload["stop_sequences"] == ["END"]
    assert payload["temperature"] == 0
    assert payload["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


def test_to_anthropic_rejects_tools_so_router_falls_back():
    with pytest.raises(ProviderError) as exc:
        to_anthropic("claude-x", {**REQUEST, "tools": [{"type": "function"}]})
    assert exc.value.kind is ErrorKind.PROVIDER


@pytest.mark.anyio
async def test_anthropic_complete_converts_response_and_usage():
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-x",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}],
                "usage": {"input_tokens": 3, "cache_read_input_tokens": 7, "output_tokens": 4},
            },
        )

    result = await anthropic_provider(handler).complete("claude-x", REQUEST)
    assert seen["headers"]["x-api-key"] == "k" and seen["headers"]["anthropic-version"] == "2023-06-01"
    choice = result.body["choices"][0]
    assert choice["message"]["content"] == "Hi there" and choice["finish_reason"] == "length"
    assert (result.usage.prompt_tokens, result.usage.cached_tokens, result.usage.completion_tokens) == (10, 7, 4)


@pytest.mark.anyio
async def test_anthropic_stream_is_translated_to_openai_chunks():
    events = sse(
        {"type": "message_start", "message": {"id": "msg_2", "model": "claude-x", "usage": {"input_tokens": 9}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )
    provider = anthropic_provider(lambda r: httpx.Response(200, content=events))
    chunks = [c async for c in provider.stream("claude-x", REQUEST)]

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens"] == 9 and chunks[-1]["usage"]["completion_tokens"] == 2


@pytest.mark.anyio
async def test_anthropic_overloaded_before_content_can_still_fall_back():
    events = sse(
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    provider = anthropic_provider(lambda r: httpx.Response(200, content=events))
    with pytest.raises(ProviderError) as exc:
        await anext(provider.stream("claude-x", REQUEST))  # nothing was yielded yet
    assert exc.value.kind is ErrorKind.RETRYABLE
