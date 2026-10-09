"""Per-request token and cost accounting, stored in SQLite."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Price
from .providers.base import Usage


def compute_cost(usage: Usage, price: Price | None) -> float | None:
    """Cost in USD, or ``None`` when no price is configured for the model."""
    if price is None:
        return None
    cached_rate = price.input if price.cached_input is None else price.cached_input
    fresh = max(usage.prompt_tokens - usage.cached_tokens, 0)
    total = fresh * price.input + usage.cached_tokens * cached_rate + usage.completion_tokens * price.output
    return round(total / 1_000_000, 8)


@dataclass
class UsageRecord:
    ts: float
    route: str
    provider: str | None
    model: str | None
    ok: bool
    stream: bool
    attempts: int
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    route TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    ok INTEGER NOT NULL,
    stream INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cost_usd REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
"""

_COLUMNS = list(UsageRecord.__dataclass_fields__)


class UsageStore:
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def record(self, record: UsageRecord) -> None:
        values = asdict(record)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock:
            self._db.execute(
                f"INSERT INTO requests ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                [values[c] for c in _COLUMNS],
            )
            self._db.commit()

    def summary(self, since_ts: float) -> dict:
        with self._lock:
            totals = self._db.execute(
                """
                SELECT COUNT(*) AS requests,
                       COALESCE(SUM(1 - ok), 0) AS failures,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                       COALESCE(SUM(cost_usd), 0) AS cost_usd
                FROM requests WHERE ts >= ?
                """,
                (since_ts,),
            ).fetchone()
            by_model = self._db.execute(
                """
                SELECT provider, model,
                       COUNT(*) AS requests,
                       SUM(1 - ok) AS failures,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(cached_tokens) AS cached_tokens,
                       COALESCE(SUM(cost_usd), 0) AS cost_usd,
                       CAST(AVG(latency_ms) AS INTEGER) AS avg_latency_ms
                FROM requests WHERE ts >= ? AND provider IS NOT NULL
                GROUP BY provider, model ORDER BY requests DESC
                """,
                (since_ts,),
            ).fetchall()
        result = dict(totals)
        result["cost_usd"] = round(result["cost_usd"], 6)
        result["by_model"] = [dict(row) for row in by_model]
        for row in result["by_model"]:
            row["cost_usd"] = round(row["cost_usd"], 6)
        return result

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(r), "ok": bool(r["ok"]), "stream": bool(r["stream"])} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
