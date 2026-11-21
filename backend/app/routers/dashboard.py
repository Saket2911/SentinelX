from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .. import schemas
from ..deps import active_run, get_database, get_store
from ..services import dashboard as dashboard_service
from ..services.artifacts import ArtifactStore
from ..services.dashboard import overview

router = APIRouter(tags=["dashboard"])


@router.get(
    "/dashboard",
    summary="SOC overview: posture, metrics, activity, model and dataset health",
    responses={200: {"model": schemas.DashboardResponse}},
)
def dashboard(request: Request, run_id: str | None = None, store: ArtifactStore = Depends(get_store)) -> dict:
    resolved = store.resolve_run_id(run_id or active_run(request), request.app.state.settings.default_run_id)
    return overview(store, request.app.state.database, resolved)
