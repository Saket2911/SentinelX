from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from .. import schemas
from ..deps import active_run, get_database, get_settings, get_settings_store, get_store
from ..services import system as system_service
from ..services.artifacts import ArtifactStore
from ..services.settings_store import SettingsStore

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    summary="Liveness and deployment health",
    responses={200: {"model": schemas.HealthResponse}},
)
def health(request: Request) -> dict:
    uptime = time.monotonic() - request.app.state.started_at
    return system_service.health(request.app.state.settings, request.app.state.store, uptime)


@router.get(
    "/system/status",
    summary="Artifact inventory and runtime status",
    responses={200: {"model": schemas.SystemStatusResponse}},
)
def system_status(
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    settings=Depends(get_settings),
    settings_store: SettingsStore = Depends(get_settings_store),
) -> dict:
    resolved = store.resolve_run_id(run_id or settings_store.active_run_id(), settings.default_run_id)
    return system_service.status(settings, store, resolved)


@router.get(
    "/capabilities",
    summary="Implemented, conditional and planned capabilities",
    responses={200: {"model": schemas.CapabilitiesResponse}},
)
def capabilities() -> dict:
    return system_service.capabilities()


@router.get("/audit", summary="Recent audited actions")
def audit(database=Depends(get_database)) -> dict:
    return {"entries": database.audit_entries(100)}


@router.get("/runs", summary="Available model run identifiers")
def runs(request: Request, store: ArtifactStore = Depends(get_store)) -> dict:
    active = active_run(request)
    return {"active_run_id": active, "runs": sorted(store.runs)}
