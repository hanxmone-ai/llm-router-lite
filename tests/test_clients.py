import math
import sqlite3

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_router.app import create_app
from llm_router.auth import RateLimiter, month_start
from llm_router.config import Config
from llm_router.providers.mock import hashed_embedding
from llm_router.usage import UsageStore

from .conftest import make_config

ADMIN = {"Authorization": "Bearer admin-key"}
ACME = {"Authorization": "Bearer acme-key"}
GLOBEX = {"Authorization": "Bearer globex-key"}


def clients_config(**client_overrides):
    clients = {
        "acme": {"api_key": "acme-key", "routes": ["solo", "embed"]},
        "globex": {"api_key": "globex-key"},
        "unset": {"api_key": ""},  # env var not exported
    }
    for name, extra in client_overrides.items():
        clients[name] = {**clients[name], **extra}
    return make_config(
        server={"api_keys": ["admin-key"], "usage_db": ":memory:"},
        clients=clients,
        providers={
            "up": {"type": "mock"},
            "down": {"type": "mock", "mode": "fail"},
            "claude": {"type": "anthropic", "api_key": "x"},
        },
        routes={
            "demo": [{"provider": "down", "model": "m1"}, {"provider": "up", "model": "m2"}],
            "solo": [{"provider": "up", "model": "m2"}],
            "embed": [{"provider": "claude", "model": "no-embeddings"}, {"provider": "up", "model": "hash-256"}],
        },
    )


def app_client(config, clock=None):
    kwargs = {"usage_store": UsageStore(":memory:")}
    if clock is not None:
        kwargs["clock"] = clock
    return TestClient(create_app(config, **kwargs))


def ask(client, headers, model="solo"):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hello there"}]},
        headers=headers,
    )


def test_admin_and_client_keys_authenticate_and_empty_keys_never_match():
    with app_client(clients_config()) as c:
        assert ask(c, ADMIN).status_code == 200
        assert ask(c, ACME).status_code == 200
        assert ask(c, {"Authorization": "Bearer nope"}).status_code == 401
        # The "unset" client has an empty key: an empty or missing token must not log in as it.
        assert ask(c, {}).status_code == 401
        assert ask(c, {"Authorization": "Bearer "}).status_code == 401


def test_route_allowlist_and_model_list():
    with app_client(clients_config()) as c:
        refused = ask(c, ACME, model="demo")
        assert refused.status_code == 403
        assert refused.json()["error"]["allowed_routes"] == ["solo", "embed"]
        assert [m["id"] for m in c.get("/v1/models", headers=ACME).json()["data"]] == ["solo", "embed"]
        assert ask(c, GLOBEX, model="demo").status_code == 200  # no allowlist = every route


def test_requests_per_minute_limit():
    with app_client(clients_config(acme={"requests_per_minute": 2})) as c:
        assert ask(c, ACME).status_code == 200
        assert ask(c, ACME).status_code == 200
        limited = ask(c, ACME)
        assert limited.status_code == 429
        assert limited.json()["error"]["type"] == "rate_limit_error"
        assert 1 <= int(limited.headers["retry-after"]) <= 60
        assert ask(c, GLOBEX).status_code == 200  # limits are per client


def test_monthly_budget_blocks_once_spent():
    # Price for m2 in conftest: $1 in / $2 out per 1M tokens; a tiny budget is used up by one call.
    with app_client(clients_config(acme={"monthly_budget_usd": 0.000001})) as c:
        assert ask(c, ACME).status_code == 200
        blocked = ask(c, ACME)
        assert blocked.status_code == 429
        assert blocked.json()["error"]["type"] == "budget_exceeded"
        usage = c.get("/v1/usage", headers=ACME).json()
        assert usage["budget"]["spent_this_month_usd"] > 0


def test_usage_is_scoped_per_client_and_status_is_admin_only():
    with app_client(clients_config()) as c:
        ask(c, ACME)
        ask(c, GLOBEX)
        ask(c, GLOBEX)
        ask(c, ADMIN)

        own = c.get("/v1/usage", headers=GLOBEX).json()
        assert own["client"] == "globex" and own["requests"] == 2
        assert {r["client"] for r in own["recent"]} == {"globex"}

        everything = c.get("/v1/usage", headers=ADMIN).json()
        assert everything["requests"] == 4
        by_client = {row["client"]: row["requests"] for row in everything["by_client"]}
        assert by_client == {"acme": 1, "globex": 2, "admin": 1}
        assert c.get("/v1/usage?client=acme", headers=ADMIN).json()["requests"] == 1

        assert c.get("/v1/status", headers=ACME).status_code == 403
        clients = c.get("/v1/status", headers=ADMIN).json()["clients"]
        assert clients["unset"]["enabled"] is False and clients["acme"]["routes"] == ["solo", "embed"]


def test_embeddings_fall_back_from_providers_without_embeddings():
    with app_client(clients_config()) as c:
        response = c.post(
            "/v1/embeddings", json={"model": "embed", "input": ["red apple", "green apple"]}, headers=ACME
        )
        assert response.status_code == 200
        assert response.headers["x-router-target"] == "up/hash-256"  # anthropic target skipped
        data = response.json()["data"]
        assert len(data) == 2 and len(data[0]["embedding"]) == 256
        row = c.get("/v1/usage", headers=ACME).json()["recent"][0]
        assert row["endpoint"] == "embeddings"
        assert c.post("/v1/embeddings", json={"model": "embed"}, headers=ACME).status_code == 400


def test_hashed_embeddings_are_normalized_and_similarity_aware():
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    apple, apples, engine = (
        hashed_embedding("apple pie recipe"),
        hashed_embedding("apple pies"),
        hashed_embedding("diesel engine"),
    )
    assert math.isclose(math.sqrt(sum(v * v for v in apple)), 1.0, rel_tol=1e-3)
    assert cosine(apple, apples) > cosine(apple, engine)


def test_client_config_validation():
    base = {"providers": {"p": {"type": "mock"}}, "routes": {"r": [{"provider": "p", "model": "m"}]}}
    with pytest.raises(ValidationError, match="unknown route"):
        Config.model_validate({**base, "clients": {"a": {"api_key": "k", "routes": ["nope"]}}})
    with pytest.raises(ValidationError, match="reuses an API key"):
        Config.model_validate({**base, "clients": {"a": {"api_key": "k"}, "b": {"api_key": "k"}}})
    with pytest.raises(ValidationError, match="reserved"):
        Config.model_validate({**base, "clients": {"admin": {"api_key": "k"}}})
    ok = Config.model_validate({**base, "clients": {"a": {"api_key": "k", "routes": ["r", "p/any"]}}})
    assert ok.clients["a"].routes == ["r", "p/any"]


def test_rate_limiter_window_and_month_start():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert limiter.check("a", 1) is None
    assert limiter.check("a", 1) == pytest.approx(60.0)
    now[0] = 60.0
    assert limiter.check("a", 1) is None
    assert month_start(1_791_513_162) == 1_790_812_800  # 2026-10-09 → 2026-10-01T00:00:00Z


def test_usage_store_upgrades_an_old_database(tmp_path):
    path = tmp_path / "usage.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, route TEXT NOT NULL,"
        " provider TEXT, model TEXT, ok INTEGER NOT NULL, stream INTEGER NOT NULL, attempts INTEGER NOT NULL,"
        " latency_ms INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,"
        " cached_tokens INTEGER NOT NULL, cost_usd REAL, error TEXT)"
    )
    db.execute(
        "INSERT INTO requests (ts, route, ok, stream, attempts, latency_ms, prompt_tokens, completion_tokens,"
        " cached_tokens) VALUES (1, 'old', 1, 0, 1, 5, 1, 1, 0)"
    )
    db.commit()
    db.close()

    store = UsageStore(str(path))
    rows = store.recent(5)
    assert rows[0]["route"] == "old" and rows[0]["client"] is None and rows[0]["endpoint"] == "chat"
    store.close()
