import json

import pytest
from starlette.testclient import TestClient

from llm_router.app import create_app
from llm_router.usage import UsageStore

from .conftest import make_config


@pytest.fixture
def client():
    config = make_config(server={"api_keys": ["test-key"], "usage_db": ":memory:"})
    with TestClient(create_app(config, usage_store=UsageStore(":memory:"))) as c:
        c.headers["Authorization"] = "Bearer test-key"
        yield c


def chat(client, **body):
    payload = {"model": "demo", "messages": [{"role": "user", "content": "ping pong"}], **body}
    return client.post("/v1/chat/completions", json=payload)


def test_requires_api_key(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "demo", "messages": [{"role": "user", "content": "x"}]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
    assert client.get("/healthz").json() == {"status": "ok"}  # health check stays public


def test_chat_completion_falls_back_and_reports_target(client):
    response = chat(client)
    assert response.status_code == 200
    assert response.headers["x-router-target"] == "up/m2"
    assert response.headers["x-router-attempts"] == "3"
    assert response.json()["choices"][0]["message"]["content"] == "echo: ping pong"


def test_streaming_uses_openai_sse_format(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "demo",
            "stream": True,
            "messages": [{"role": "user", "content": "ping pong"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-router-target"] == "up/m2"
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "echo: ping pong"
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_validation_and_routing_errors(client):
    assert client.post("/v1/chat/completions", content=b"not json").status_code == 400
    assert chat(client, messages=[]).status_code == 400

    missing = chat(client, model="nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["available_routes"] == ["demo", "solo"]


def test_all_targets_failing_returns_502_with_attempts():
    config = make_config(routes={"dead": [{"provider": "down", "model": "m1"}]})
    with TestClient(create_app(config, usage_store=UsageStore(":memory:"))) as c:
        response = c.post(
            "/v1/chat/completions", json={"model": "dead", "messages": [{"role": "user", "content": "x"}]}
        )
    assert response.status_code == 502
    attempts = response.json()["error"]["attempts"]
    assert [a["target"] for a in attempts] == ["down/m1", "down/m1"]


def test_models_usage_status_and_dashboard(client):
    chat(client)
    chat(client, model="solo", stream=False)

    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["demo", "solo"]

    usage = client.get("/v1/usage?hours=1").json()
    assert usage["requests"] == 2 and usage["failures"] == 0
    assert usage["by_model"][0]["provider"] == "up" and usage["by_model"][0]["requests"] == 2
    assert usage["cost_usd"] > 0
    assert len(usage["recent"]) == 2

    status = client.get("/v1/status").json()
    assert status["routes"]["demo"][0]["target"] == "down/m1"
    assert status["providers"]["up"]["enabled"] is True

    page = client.get("/dashboard")
    assert page.status_code == 200 and "llm-router-lite" in page.text
