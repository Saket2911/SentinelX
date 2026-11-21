"""Persisted application settings (``var/settings.json``).

Settings never modify research artifacts. The review threshold recorded here is
an analyst-side display preference; the calibrated value from the run is always
reported separately so the two cannot be confused.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "default_run_id": None,
    "review_threshold": None,
    "page_size": 25,
    "timezone": "UTC",
    "notify_on_review_required": True,
    "analyst_name": "analyst",
    "data_mode": "dataset",
    "active_run_id": None,
    "active_run_changed_at": None,
}


class SettingsStore:
    def __init__(self, path: Path, default_run_id: str) -> None:
        self.path = Path(path)
        self.default_run_id = default_run_id
        self._lock = threading.RLock()
        self._cache: dict[str, Any] | None = None

    def _read(self) -> dict[str, Any]:
        data = dict(DEFAULTS)
        data["default_run_id"] = self.default_run_id
        if self.path.is_file():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                if isinstance(stored, dict):
                    data.update(
                        {
                            key: value
                            for key, value in stored.items()
                            if key in DEFAULTS
                        }
                    )
            except (OSError, json.JSONDecodeError):
                pass
        return data

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self._cache is None:
                self._cache = self._read()
            return dict(self._cache)

    def _write(self, data: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        temporary.replace(self.path)
        self._cache = data
        return dict(data)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self.load()
            for key, value in changes.items():
                if key in DEFAULTS and value is not None:
                    data[key] = value
            return self._write(data)

    def active_run_id(self) -> str:
        data = self.load()
        return data.get("active_run_id") or data.get("default_run_id") or self.default_run_id

    def set_active_run(self, run_id: str, changed_at: str) -> dict[str, Any]:
        with self._lock:
            data = self.load()
            data["active_run_id"] = run_id
            data["active_run_changed_at"] = changed_at
            return self._write(data)
