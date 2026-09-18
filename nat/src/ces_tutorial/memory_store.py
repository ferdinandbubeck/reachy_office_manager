"""Simple local, persistent memory for the agent - not tied to any single
conversation. Backed by SQLite (one file, one table) rather than a hosted
memory platform: this is a single-robot office assistant, not a multi-user
SaaS product, so there's no need for embeddings/retrieval or a remote
service with its own API key - a flat list the LLM reads in full and
reasons over itself is simpler and just as effective at this scale.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "agent_memory.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def remember(fact: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO facts (fact, created_at) VALUES (?, ?)",
            (fact, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def recall_all() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM facts ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def forget(fact_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0


init_db()
