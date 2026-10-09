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
    #: Client name from the config; ``None`` for admin keys or when auth is off.
    client: str | None = None
    #: ``chat`` or ``embeddings``.
    endpoint: str = "chat"


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

#: Columns added after 0.1.0; existing databases are upgraded in place.
_MIGRATIONS = {
    "client": "ALTER TABLE requests ADD COLUMN client TEXT",
    "endpoint": "ALTER TABLE requests ADD COLUMN endpoint TEXT NOT NULL DEFAULT 'chat'",
}

_COLUMNS = list(UsageRecord.__dataclass_fields__)

_TOTALS = """
    COUNT(*) AS requests,
    COALESCE(SUM(1 - ok), 0) AS failures,
    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
    COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
    COALESCE(SUM(cost_usd), 0) AS cost_usd
"""


class UsageStore:
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(requests)")}
        for column, statement in _MIGRATIONS.items():
            if column not in existing:
                self._db.execute(statement)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_requests_client_ts ON requests (client, ts)")
        self._db.commit()
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

    def cost_since(self, client: str, since_ts: float) -> float:
        """Total recorded cost for ``client`` since ``since_ts`` (used for monthly budgets)."""
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE client = ? AND ts >= ?",
                (client, since_ts),
            ).fetchone()
        return float(row[0])

    def summary(self, since_ts: float, client: str | None = None) -> dict:
        """Totals plus breakdowns by model and by client. ``client`` restricts everything to one client."""
        where, params = "ts >= ?", [since_ts]
        if client is not None:
            where += " AND client = ?"
            params.append(client)
        with self._lock:
            totals = self._db.execute(f"SELECT {_TOTALS} FROM requests WHERE {where}", params).fetchone()
            by_model = self._db.execute(
                f"""
                SELECT provider, model, {_TOTALS}, CAST(AVG(latency_ms) AS INTEGER) AS avg_latency_ms
                FROM requests WHERE {where} AND provider IS NOT NULL
                GROUP BY provider, model ORDER BY requests DESC
                """,
                params,
            ).fetchall()
            by_client = self._db.execute(
                f"""
                SELECT COALESCE(client, 'admin') AS client, {_TOTALS}
                FROM requests WHERE {where}
                GROUP BY COALESCE(client, 'admin') ORDER BY cost_usd DESC, requests DESC
                """,
                params,
            ).fetchall()
        result = _rounded(dict(totals))
        result["by_model"] = [_rounded(dict(row)) for row in by_model]
        result["by_client"] = [_rounded(dict(row)) for row in by_client]
        return result

    def recent(self, limit: int = 20, client: str | None = None) -> list[dict]:
        query = f"SELECT {', '.join(_COLUMNS)} FROM requests"
        params: list = []
        if client is not None:
            query += " WHERE client = ?"
            params.append(client)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [{**dict(r), "ok": bool(r["ok"]), "stream": bool(r["stream"])} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _rounded(row: dict) -> dict:
    row["cost_usd"] = round(row["cost_usd"], 6)
    return row
