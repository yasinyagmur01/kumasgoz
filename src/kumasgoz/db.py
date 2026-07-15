"""SQLite offline buffer for detections, heartbeats, and profiles."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    image_path TEXT,
    score REAL,
    is_anomaly INTEGER,
    line_state TEXT,
    reason TEXT,
    profile_id TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_name TEXT,
    category TEXT,
    checkpoint_path TEXT,
    threshold REAL,
    max_fpr REAL,
    meta TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EventStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_event(
        self,
        event_type: str,
        *,
        image_path: str | None = None,
        score: float | None = None,
        is_anomaly: bool | None = None,
        line_state: str | None = None,
        reason: str | None = None,
        profile_id: str | None = None,
        payload: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (ts, event_type, image_path, score, is_anomaly, line_state, reason, profile_id, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts or utc_now(),
                    event_type,
                    image_path,
                    score,
                    None if is_anomaly is None else int(is_anomaly),
                    line_state,
                    reason,
                    profile_id,
                    json.dumps(payload) if payload is not None else None,
                ),
            )
            return int(cur.lastrowid)

    def add_heartbeat(self, source: str, ok: bool = True, detail: str | None = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO heartbeats (ts, source, ok, detail) VALUES (?, ?, ?, ?)",
                (utc_now(), source, int(ok), detail),
            )
            return int(cur.lastrowid)

    def upsert_profile(
        self,
        profile_id: str,
        name: str,
        *,
        model_name: str | None = None,
        category: str | None = None,
        checkpoint_path: str | None = None,
        threshold: float | None = None,
        max_fpr: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles (id, name, created_at, model_name, category, checkpoint_path, threshold, max_fpr, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    model_name=excluded.model_name,
                    category=excluded.category,
                    checkpoint_path=excluded.checkpoint_path,
                    threshold=excluded.threshold,
                    max_fpr=excluded.max_fpr,
                    meta=excluded.meta
                """,
                (
                    profile_id,
                    name,
                    utc_now(),
                    model_name,
                    category,
                    checkpoint_path,
                    threshold,
                    max_fpr,
                    json.dumps(meta or {}),
                ),
            )

    def list_events(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM events WHERE event_type=? ORDER BY id DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def list_profiles(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        return dict(row) if row else None

    def latest_heartbeat(self, source: str | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            if source:
                row = conn.execute(
                    "SELECT * FROM heartbeats WHERE source=? ORDER BY id DESC LIMIT 1",
                    (source,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM heartbeats ORDER BY id DESC LIMIT 1"
                ).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            n_events = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            n_anom = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE is_anomaly=1"
            ).fetchone()["c"]
            n_stop = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE line_state='STOP'"
            ).fetchone()["c"]
            n_profiles = conn.execute("SELECT COUNT(*) AS c FROM profiles").fetchone()["c"]
            n_hb = conn.execute("SELECT COUNT(*) AS c FROM heartbeats").fetchone()["c"]
        return {
            "events": n_events,
            "anomalies": n_anom,
            "stops": n_stop,
            "profiles": n_profiles,
            "heartbeats": n_hb,
        }
