"""Shared request dependencies."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Header, HTTPException, Request, status

from .config import Settings
from .db import Database
from .services.artifacts import ArtifactStore
from .services.settings_store import SettingsStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.store


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_settings_store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def active_run(request: Request) -> str:
    settings_store: SettingsStore = request.app.state.settings_store
    store: ArtifactStore = request.app.state.store
    configured = settings_store.active_run_id()
    return store.resolve_run_id(configured, request.app.state.settings.default_run_id)


def require_write_access(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "detail": "A valid X-API-Key header is required."},
        )


def payload(data: Any, source: dict[str, Any], generated_at: str) -> dict[str, Any]:
    return {"data": data, "source": source, "generated_at": generated_at}
