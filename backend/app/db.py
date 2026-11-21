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
CREATE TABLE IF NOT EXISTS playback_alert (
    alert_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    target_step INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    timestamp TEXT,
    segment_id TEXT,
    dataset_label TEXT,
    capture_day TEXT,
    anomaly_score REAL NOT NULL,
    threshold REAL,
    ratio REAL,
    severity TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    assignee TEXT,
    notes TEXT,
    features_json TEXT NOT NULL,
    prediction_source TEXT NOT NULL,
    predicted_ref TEXT,
    actual_ref TEXT,
    context_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, model_version, dataset_id, target_step)
);
CREATE INDEX IF NOT EXISTS idx_playback_alert_run
    ON playback_alert (run_id, target_step);
CREATE TABLE IF NOT EXISTS playback_alert_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    from_value TEXT,
    to_value TEXT,
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

    # ---------------------------------------------------------- playback alerts

    def insert_playback_alert(self, values: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Insert idempotently; return the stored row and whether it was created."""
        columns = (
            "alert_id",
            "run_id",
            "dataset_id",
            "model_version",
            "target_step",
            "target_index",
            "timestamp",
            "segment_id",
            "dataset_label",
            "capture_day",
            "anomaly_score",
            "threshold",
            "ratio",
            "severity",
            "trigger_reason",
            "status",
            "assignee",
            "notes",
            "features_json",
            "prediction_source",
            "predicted_ref",
            "actual_ref",
            "context_ref",
            "created_at",
            "updated_at",
        )
        placeholders = ", ".join("?" for _ in columns)
        with self._lock:
            cursor = self._connection.execute(
                f"INSERT OR IGNORE INTO playback_alert ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                tuple(values[column] for column in columns),
            )
            created = cursor.rowcount > 0
        row = self.query(
            "SELECT * FROM playback_alert WHERE alert_id = ?", (values["alert_id"],)
        )
        if not row:
            row = self.query(
                "SELECT * FROM playback_alert WHERE run_id = ? AND model_version = ? "
                "AND dataset_id = ? AND target_step = ?",
                (
                    values["run_id"],
                    values["model_version"],
                    values["dataset_id"],
                    values["target_step"],
                ),
            )
        return row[0], created

    def playback_alerts(
        self,
        run_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        dataset_id: str | None = None,
        target_step: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if target_step is not None:
            clauses.append("target_step = ?")
            params.append(int(target_step))
        if since:
            clauses.append("COALESCE(timestamp, created_at) >= ?")
            params.append(since)
        if until:
            clauses.append("COALESCE(timestamp, created_at) <= ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        return self.query(
            f"SELECT * FROM playback_alert {where} ORDER BY target_step ASC, created_at ASC "
            "LIMIT ? OFFSET ?",
            tuple(params),
        )

    def playback_alert(self, alert_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM playback_alert WHERE alert_id = ?", (alert_id,))
        return rows[0] if rows else None

    def playback_alerts_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM playback_alert WHERE run_id = ? ORDER BY target_step ASC",
            (run_id,),
        )

    def count_playback_alerts(self, run_id: str) -> int:
        rows = self.query(
            "SELECT COUNT(*) AS total FROM playback_alert WHERE run_id = ?", (run_id,)
        )
        return int(rows[0]["total"]) if rows else 0

    def update_playback_alert(
        self,
        alert_id: str,
        status: str,
        assignee: str | None,
        notes: str | None,
        actor: str,
        changes: list[tuple[str, str | None, str | None]],
    ) -> dict[str, Any] | None:
        timestamp = utc_now()
        with self._lock:
            self._connection.execute(
                "UPDATE playback_alert SET status = ?, assignee = ?, notes = ?, "
                "updated_at = ? WHERE alert_id = ?",
                (status, assignee, notes, timestamp, alert_id),
            )
            for field, before, after in changes:
                self._connection.execute(
                    "INSERT INTO playback_alert_history "
                    "(alert_id, actor, action, from_value, to_value, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (alert_id, actor, field, before, after, timestamp),
                )
            self._connection.execute(
                "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (?, ?, ?, ?)",
                (actor, "playback_alert_update", alert_id, timestamp),
            )
        return self.playback_alert(alert_id)

    def playback_alert_history(self, alert_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT actor, action, from_value, to_value, created_at "
            "FROM playback_alert_history WHERE alert_id = ? ORDER BY id ASC",
            (alert_id,),
        )

    def playback_alert_status_counts(self, run_id: str) -> dict[str, int]:
        rows = self.query(
            "SELECT status, COUNT(*) AS total FROM playback_alert WHERE run_id = ? "
            "GROUP BY status",
            (run_id,),
        )
        return {row["status"]: int(row["total"]) for row in rows}

    def playback_alert_severity_counts(self, run_id: str) -> dict[str, int]:
        rows = self.query(
            "SELECT severity, COUNT(*) AS total FROM playback_alert WHERE run_id = ? "
            "GROUP BY severity",
            (run_id,),
        )
        return {row["severity"]: int(row["total"]) for row in rows}

    def playback_alert_daily_counts(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT substr(COALESCE(timestamp, created_at), 1, 10) AS day, "
            "COUNT(*) AS total FROM playback_alert WHERE run_id = ? "
            "GROUP BY day ORDER BY day ASC",
            (run_id,),
        )
        return [{"day": row["day"], "count": int(row["total"])} for row in rows]
