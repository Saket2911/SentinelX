"""SQLite-backed application state.

Research artifacts stay read-only; everything an analyst changes lives here:
case status, notes, assignment, an append-only audit trail, model activation
history and saved settings are all stored in ``var/sentinelx.db``.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS event_state (
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    status TEXT NOT NULL,
    assignee TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id)
);
CREATE TABLE IF NOT EXISTS event_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    from_value TEXT,
    to_value TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_activation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    previous_run_id TEXT,
    actor TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        raise NotImplementedError

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, params)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------ events

    def event_states(self, run_id: str) -> dict[str, dict[str, Any]]:
        rows = self.query(
            "SELECT event_id, status, assignee, notes, updated_at "
            "FROM event_state WHERE run_id = ?",
            (run_id,),
        )
        return {row["event_id"]: row for row in rows}

    def event_state(self, run_id: str, event_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT event_id, status, assignee, notes, updated_at FROM event_state "
            "WHERE run_id = ? AND event_id = ?",
            (run_id, event_id),
        )
        return rows[0] if rows else None

    def save_event_state(
        self,
        run_id: str,
        event_id: str,
        status: str,
        assignee: str | None,
        notes: str | None,
        actor: str,
        changes: list[tuple[str, str | None, str | None]],
    ) -> dict[str, Any]:
        timestamp = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO event_state (run_id, event_id, status, assignee, notes, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, event_id) DO UPDATE SET "
                "status = excluded.status, assignee = excluded.assignee, "
                "notes = excluded.notes, updated_at = excluded.updated_at",
                (run_id, event_id, status, assignee, notes, timestamp),
            )
            for field, before, after in changes:
                self._connection.execute(
                    "INSERT INTO event_history "
                    "(run_id, event_id, actor, action, from_value, to_value, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, event_id, actor, field, before, after, timestamp),
                )
            self._connection.execute(
                "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (?, ?, ?, ?)",
                (actor, "event_update", f"{run_id}:{event_id}", timestamp),
            )
        return {
            "event_id": event_id,
            "status": status,
            "assignee": assignee,
            "notes": notes,
            "updated_at": timestamp,
        }

    def event_history(self, run_id: str, event_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT actor, action, from_value, to_value, created_at FROM event_history "
            "WHERE run_id = ? AND event_id = ? ORDER BY id ASC",
            (run_id, event_id),
        )

    # ------------------------------------------------------------------ models

    def record_activation(
        self,
        run_id: str,
        previous_run_id: str | None,
        actor: str,
        reason: str | None,
    ) -> str:
        timestamp = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO model_activation "
                "(run_id, previous_run_id, actor, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, previous_run_id, actor, reason, timestamp),
            )
        return timestamp

    def activation_history(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT run_id, previous_run_id, actor, reason, created_at "
            "FROM model_activation ORDER BY id DESC LIMIT 50"
        )

    def audit(self, actor: str, action: str, detail: str | None = None) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (?, ?, ?, ?)",
                (actor, action, detail, utc_now()),
            )

    def audit_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.query(
            "SELECT actor, action, detail, created_at FROM audit_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
