"""Read/write runtime state: settings and the active-model pointer.

Both files live under ``backend/var`` and are created on first write. Writes are
atomic (tmp file + rename) so a crash cannot leave a corrupt config behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .config import (
    ACTIVE_MODEL_FILE,
    DEFAULT_ACTIVE_MODEL,
    MODEL_VERSIONS,
    SETTINGS_FILE,
    VAR_DIR,
)
from .schemas import SettingsPayload

DEFAULT_SETTINGS = SettingsPayload()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):  # pragma: no cover
            os.unlink(tmp)


def load_settings() -> SettingsPayload:
    if not SETTINGS_FILE.is_file():
        return DEFAULT_SETTINGS
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return SettingsPayload(**raw)
    except (OSError, ValueError, TypeError):
        return DEFAULT_SETTINGS


def save_settings(payload: SettingsPayload) -> SettingsPayload:
    _atomic_write(SETTINGS_FILE, payload.model_dump())
    return payload


def get_active_model() -> str:
    if ACTIVE_MODEL_FILE.is_file():
        try:
            data = json.loads(ACTIVE_MODEL_FILE.read_text(encoding="utf-8"))
            mid = data.get("active_model")
            if mid in MODEL_VERSIONS:
                return mid
        except (OSError, ValueError, TypeError):
            pass
    return DEFAULT_ACTIVE_MODEL


def set_active_model(model_id: str) -> str:
    if model_id not in MODEL_VERSIONS:
        raise KeyError(model_id)
    _atomic_write(ACTIVE_MODEL_FILE, {"active_model": model_id})
    return model_id


def ensure_var_dirs() -> None:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
