from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import schemas
from ..deps import active_run, get_database, get_store, require_write_access
from ..services import alerts as alerts_service
from ..services.artifacts import ArtifactStore

router = APIRouter(tags=["prediction-alerts"])


def _run(request: Request, run_id: str | None, store: ArtifactStore) -> str:
    return store.resolve_run_id(
        run_id or active_run(request), request.app.state.settings.default_run_id
    )


@router.post(
    "/playback/evaluate",
    summary="Score one replay state against the LSTM forecast and raise an alert",
    dependencies=[Depends(require_write_access)],
)
def evaluate_state(
    payload: schemas.PlaybackEvaluateRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    resolved = store.resolve_run_id(
        payload.run_id or active_run(request), request.app.state.settings.default_run_id
    )
    try:
        return alerts_service.evaluate_target(store, database, resolved, payload.step)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "detail": f"Unknown replay step {payload.step}"},
        )
    except alerts_service.ModelContractError as exc:
        raise HTTPException(
            status_code=409, detail={"error": "model_contract_mismatch", "detail": str(exc)}
        )
    except alerts_service.InsufficientContextError as exc:
        raise HTTPException(
            status_code=409, detail={"error": "insufficient_context", "detail": str(exc)}
        )


@router.get("/playback/alerts", summary="Persisted prediction alerts for playback")
def list_alerts(
    request: Request,
    status: str | None = None,
    severity: str | None = None,
    dataset_id: str | None = None,
    target_step: int | None = Query(default=None, ge=1),
    since: str | None = Query(default=None, description="ISO timestamp lower bound"),
    until: str | None = Query(default=None, description="ISO timestamp upper bound"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    resolved = _run(request, run_id, store)
    rows = database.playback_alerts(
        run_id=resolved,
        status=status,
        severity=severity,
        dataset_id=dataset_id,
        target_step=target_step,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return {
        "run_id": resolved,
        "alerts": [alerts_service.serialize_alert(row) for row in rows],
        "total": len(rows),
        "rules": alerts_service.rules(),
        "limitations": alerts_service.LIMITATIONS,
    }


@router.post(
    "/playback/alerts/scan",
    summary="Scan the whole replay for rule-matching alerts (idempotent)",
    dependencies=[Depends(require_write_access)],
)
def scan_alerts(
    request: Request,
    run_id: str | None = None,
    limit: int = Query(default=500, ge=1, le=2000),
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    return alerts_service.scan_replay(
        store, database, _run(request, run_id, store), limit=limit
    )


@router.get("/playback/alerts/{alert_id}", summary="Alert detail with evidence and audit trail")
def get_alert(
    alert_id: str,
    request: Request,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    detail = alerts_service.get_alert(database, alert_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "detail": f"Unknown alert {alert_id}"},
        )
    alert = detail["alert"]
    top = [entry["feature"] for entry in alert["features"]]
    detail["explainability"] = {
        "shap": alerts_service.shap_context(
            store, str(alert["run_id"]), top, alert.get("dataset_label")
        )
    }
    return detail


@router.patch(
    "/playback/alerts/{alert_id}",
    summary="Update playback alert investigation status",
    dependencies=[Depends(require_write_access)],
)
def update_alert(
    alert_id: str,
    payload: schemas.PlaybackAlertUpdateRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    detail = alerts_service.update_alert(
        store,
        database,
        alert_id,
        payload.status,
        payload.assignee,
        payload.notes,
        payload.actor,
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "detail": f"Unknown alert {alert_id}"},
        )
    return detail


@router.get("/playback/anomalies", summary="Per-state anomaly scores, threshold and flagged states")
def anomalies(
    request: Request,
    run_id: str | None = None,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
) -> dict:
    return alerts_service.anomaly_series(store, database, _run(request, run_id, store))
