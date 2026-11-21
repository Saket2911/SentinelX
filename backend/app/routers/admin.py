from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from .. import schemas
from ..deps import get_database, get_settings, get_settings_store, get_store, require_write_access
from ..services import reports as reports_service
from ..services.anomaly import calibration_for
from ..services.artifacts import ArtifactStore
from ..services.settings_store import SettingsStore

router = APIRouter(tags=["administration"])


@router.get("/reports", summary="Generated reports", responses={200: {"model": schemas.ReportListResponse}})
def list_reports(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "reports": reports_service.list_reports(settings.generated_reports_dir),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


@router.post(
    "/reports",
    summary="Generate a report from live service data",
    dependencies=[Depends(require_write_access)],
    responses={200: {"model": schemas.ReportSummary}},
)
def create_report(
    payload: schemas.ReportRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    settings = request.app.state.settings
    run_id = store.resolve_run_id(payload.run_id or request.app.state.settings_store.active_run_id(), settings.default_run_id)
    try:
        return reports_service.create_report(
            store,
            request.app.state.database,
            settings.generated_reports_dir,
            run_id,
            payload.title,
            payload.sections,
            payload.author,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_report", "detail": str(exc)})


@router.get("/reports/{report_id}", summary="Report metadata and Markdown content")
def report_content(report_id: str, request: Request) -> dict:
    settings = request.app.state.settings
    try:
        metadata_path = reports_service.report_path(settings.generated_reports_dir, report_id, "json")
        markdown_path = reports_service.report_path(settings.generated_reports_dir, report_id, "md")
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown report {report_id}"})
    return {
        "metadata": __import__("json").loads(metadata_path.read_text(encoding="utf-8")),
        "markdown": markdown_path.read_text(encoding="utf-8"),
    }


@router.get("/reports/{report_id}/download", summary="Download a generated report")
def download_report(report_id: str, request: Request, format: str = "md") -> FileResponse:
    settings = request.app.state.settings
    if format not in {"md", "json"}:
        raise HTTPException(status_code=422, detail={"error": "invalid_format", "detail": "format must be md or json."})
    try:
        path = reports_service.report_path(settings.generated_reports_dir, report_id, format)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown report {report_id}"})
    media = "text/markdown" if format == "md" else "application/json"
    return FileResponse(path, media_type=media, filename=path.name)


@router.get(
    "/settings",
    summary="Analyst preferences and calibration provenance",
    responses={200: {"model": schemas.SettingsResponse}},
)
def get_settings_endpoint(
    request: Request,
    settings_store: SettingsStore = Depends(get_settings_store),
    store: ArtifactStore = Depends(get_store),
) -> dict:
    settings = request.app.state.settings
    values = settings_store.load()
    run_id = store.resolve_run_id(values.get("active_run_id"), settings.default_run_id)
    calibration = calibration_for(store, run_id)
    return {
        "settings": {
            key: values.get(key)
            for key in (
                "default_run_id",
                "review_threshold",
                "page_size",
                "timezone",
                "notify_on_review_required",
                "analyst_name",
                "data_mode",
            )
        },
        "threshold_source": {
            "calibrated_threshold": calibration.threshold,
            "source": calibration.threshold_source,
            "quantiles": calibration.quantiles,
            "analyst_override_in_use": values.get("review_threshold") is not None,
            "note": "An analyst override only affects display and review filters; research artifacts keep "
            "their frozen calibration.",
        },
        "auth_enabled": settings.auth_enabled,
        "editable_keys": [
            "default_run_id",
            "review_threshold",
            "page_size",
            "timezone",
            "notify_on_review_required",
            "analyst_name",
            "data_mode",
        ],
    }


@router.put(
    "/settings",
    summary="Update analyst preferences",
    dependencies=[Depends(require_write_access)],
    responses={200: {"model": schemas.SettingsResponse}},
)
def update_settings_endpoint(
    payload: schemas.SettingsPayload,
    request: Request,
) -> dict:
    settings_store: SettingsStore = request.app.state.settings_store
    settings_store.update(payload.model_dump(exclude_none=True))
    request.app.state.database.audit(payload.analyst_name, "settings_updated", None)
    return get_settings_endpoint(request, settings_store, request.app.state.store)
