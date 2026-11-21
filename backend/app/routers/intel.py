from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import schemas
from ..deps import active_run, get_database, get_store, require_write_access
from ..services import events as events_service
from ..services import explainability as explainability_service
from ..services import network as network_service
from ..services import predictions as predictions_service
from ..services.artifacts import ArtifactStore
from ..schemas import EventStatus

router = APIRouter(tags=["intelligence"])


def _run(request: Request, run_id: str | None, store: ArtifactStore) -> str:
    return store.resolve_run_id(run_id or active_run(request), request.app.state.settings.default_run_id)


@router.get(
    "/network",
    summary="Network activity, label mix, segments and flow-plane analytics",
    responses={200: {"model": schemas.NetworkOverviewResponse}},
)
def network(request: Request, run_id: str | None = None, store: ArtifactStore = Depends(get_store)) -> dict:
    return network_service.overview(store, _run(request, run_id, store))


@router.get(
    "/predictions/summary",
    summary="Forecast quality, error distribution and multi-step evaluation",
    responses={200: {"model": schemas.PredictionSummaryResponse}},
)
def prediction_summary(request: Request, run_id: str | None = None, store: ArtifactStore = Depends(get_store)) -> dict:
    return predictions_service.summary(store, _run(request, run_id, store))


@router.get(
    "/predictions/series",
    summary="Actual versus predicted state for one feature",
    responses={200: {"model": schemas.PredictionSeriesResponse}},
)
def prediction_series(
    request: Request,
    feature: str | None = None,
    start: int = Query(default=0, ge=0),
    limit: int = Query(default=240, ge=1, le=2000),
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    return predictions_service.series(store, _run(request, run_id, store), feature, start, limit)


@router.get("/predictions/top-residuals", summary="Highest prediction residuals (review context)")
def top_residuals(
    request: Request,
    limit: int = Query(default=25, ge=1, le=200),
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    resolved = _run(request, run_id, store)
    return {
        "run_id": resolved,
        "residuals": predictions_service.top_residuals(store, resolved, limit),
        "note": "Highest mean absolute residuals. Not incidents and not attack confirmations.",
    }


@router.get(
    "/events",
    summary="Residual review candidates with analyst workflow state",
    responses={200: {"model": schemas.EventListResponse}},
)
def list_events(
    request: Request,
    status: EventStatus | None = None,
    priority: str | None = None,
    label: str | None = None,
    segment: str | None = None,
    search: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    include_top: bool = False,
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    return events_service.list_events(
        store,
        database,
        _run(request, run_id, store),
        status=status,
        priority=priority,
        label=label,
        segment=segment,
        search=search,
        page=page,
        page_size=page_size,
        include_top=include_top,
    )


@router.get(
    "/events/{event_id}",
    summary="Single review candidate with case history",
    responses={200: {"model": schemas.EventDetailResponse}},
)
def get_event(
    event_id: str,
    request: Request,
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    detail = events_service.get_event(store, database, _run(request, run_id, store), event_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown event {event_id}"})
    return detail


@router.patch(
    "/events/{event_id}",
    summary="Update case status, assignee or notes (writes an audit trail)",
    dependencies=[Depends(require_write_access)],
    responses={200: {"model": schemas.EventDetailResponse}},
)
def update_event(
    event_id: str,
    payload: schemas.EventUpdateRequest,
    request: Request,
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    try:
        return events_service.update_event(
            store,
            database,
            _run(request, run_id, store),
            event_id,
            payload.status,
            payload.assignee,
            payload.notes,
            payload.actor,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown event {event_id}"})


@router.get(
    "/timeline",
    summary="Chronological view of label runs and residual activity",
    responses={200: {"model": schemas.TimelineResponse}},
)
def timeline(
    request: Request,
    include_top: bool = True,
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    return events_service.timeline(store, database, _run(request, run_id, store), include_top)


@router.get(
    "/explainability",
    summary="Global, temporal and local model explanations",
    responses={200: {"model": schemas.ExplainabilityResponse}},
)
def explainability(
    request: Request,
    label: str | None = None,
    output: str | None = None,
    sample_index: int = Query(default=0, ge=0),
    top_n: int = Query(default=10, ge=1, le=50),
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    return explainability_service.overview(
        store,
        _run(request, run_id, store),
        label=label,
        output=output,
        sample_index=sample_index,
        top_n=top_n,
    )
