from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from .. import schemas
from ..deps import active_run, get_database, get_settings, get_settings_store, get_store, require_write_access
from ..services import datasets as datasets_service
from ..services import inference as inference_service
from ..services import models as models_service
from ..services.artifacts import ArtifactStore
from ..services.settings_store import SettingsStore

router = APIRouter(tags=["registry"])


class ModelSpacePredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(
        description="10 consecutive standardized states, each with 209 features."
    )


class RawPredictionRequest(BaseModel):
    states: list[dict[str, Any]] = Field(
        description="10 consecutive raw state rows including the 209 model features, Timestamp and Segment ID."
    )


def _resolve(request: Request, run_id: str | None, store: ArtifactStore) -> str:
    return store.resolve_run_id(run_id or active_run(request), request.app.state.settings.default_run_id)


@router.get(
    "/models",
    summary="Available model runs with metrics and integrity status",
    responses={200: {"model": schemas.ModelListResponse}},
)
def list_models(request: Request, store: ArtifactStore = Depends(get_store), settings_store: SettingsStore = Depends(get_settings_store)) -> dict:
    active = _resolve(request, None, store)
    return {
        "source": {"mode": "recorded_artifact", "label": "Model registry", "detail": str(store.models_dir)},
        "generated_at": __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "active_run_id": active,
        "models": models_service.list_models(store, active),
        "limitations": [
            "Only runs with a config.json are registered.",
            "Checkpoints are git-ignored; a registered run may have recorded metrics without a local checkpoint.",
            "V1/V2 (212-feature) runs are retained under .quarantine and are not registered as production candidates.",
        ],
    }


@router.get(
    "/models/compare",
    summary="Compare two runs without blending their feature contracts",
    responses={200: {"model": schemas.ModelComparisonResponse}},
)
def compare_models(
    request: Request,
    left: str = Query(min_length=1),
    right: str = Query(min_length=1),
    store: ArtifactStore = Depends(get_store),
) -> dict:
    try:
        active = _resolve(request, None, store)
        result = models_service.compare(store, left, right, active)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {exc}"})
    return {
        "source": {"mode": "recorded_artifact", "label": "Model comparison", "detail": str(store.models_dir)},
        "generated_at": __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        **result,
    }


@router.get(
    "/models/{run_id}",
    summary="Full run detail: architecture, artifacts, evaluation, explanations",
    responses={200: {"model": schemas.ModelDetailResponse}},
)
def model_detail(run_id: str, request: Request, store: ArtifactStore = Depends(get_store)) -> dict:
    try:
        active = _resolve(request, None, store)
        detail = models_service.detail(store, run_id, active)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {run_id}"})
    return {
        "source": {"mode": "recorded_artifact", "label": "Model run", "detail": str(store.models_dir)},
        "generated_at": __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        **detail,
    }


@router.post(
    "/models/{run_id}/activate",
    summary="Activate a run (requires its checkpoint and the V3 contract)",
    dependencies=[Depends(require_write_access)],
    responses={200: {"model": schemas.ActivationResponse}},
)
def activate_model(
    run_id: str,
    payload: schemas.ActivationRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
    database=Depends(get_database),
    settings_store: SettingsStore = Depends(get_settings_store),
) -> dict:
    if payload.run_id != run_id:
        raise HTTPException(status_code=400, detail={"error": "invalid_request", "detail": "run_id mismatch."})
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail={"error": "confirmation_required", "detail": "Set confirm=true to activate a model run."},
        )
    try:
        return models_service.activate(
            store, database, settings_store, run_id, actor="analyst", reason=payload.reason
        )
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {run_id}"})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"error": "activation_blocked", "detail": str(exc)})


@router.get("/models/{run_id}/inference-status", summary="Whether live inference can run here")
def inference_status(run_id: str, request: Request, store: ArtifactStore = Depends(get_store)) -> dict:
    try:
        return inference_service.runtime_status(request.app.state.settings, store, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {run_id}"})


@router.post(
    "/models/{run_id}/predict",
    summary="Predict the next state from 10 standardized states",
    dependencies=[Depends(require_write_access)],
)
def predict(
    run_id: str,
    payload: ModelSpacePredictionRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    try:
        return inference_service.predict_model_space(
            request.app.state.settings, store, run_id, payload.sequence
        )
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {run_id}"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_sequence", "detail": str(exc)})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"error": "inference_unavailable", "detail": str(exc)})


@router.post(
    "/models/{run_id}/predict/raw",
    summary="Predict from 10 raw state rows using frozen preprocessing",
    dependencies=[Depends(require_write_access)],
)
def predict_raw(
    run_id: str,
    payload: RawPredictionRequest,
    request: Request,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    try:
        return inference_service.predict_raw_state(
            request.app.state.settings, store, run_id, payload.states
        )
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown run {run_id}"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_sequence", "detail": str(exc)})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"error": "inference_unavailable", "detail": str(exc)})


@router.get(
    "/datasets",
    summary="Dataset directory, artifacts, uploads and recorded label coverage",
    responses={200: {"model": schemas.DatasetListResponse}},
)
def list_datasets(request: Request, run_id: str | None = None, store: ArtifactStore = Depends(get_store)) -> dict:
    resolved = _resolve(request, run_id, store)
    overview = datasets_service.overview(store, request.app.state.settings.uploads_dir, resolved)
    return {
        "source": overview["source"],
        "generated_at": __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        **overview,
    }


def _resolve_dataset_path(request: Request, store: ArtifactStore, dataset_id: str) -> Path:
    settings = request.app.state.settings
    kind, _, name = dataset_id.partition(":")
    name = Path(name).name
    if kind == "upload":
        path = settings.uploads_dir / name
        if path.is_file() and settings.uploads_dir.resolve() in path.resolve().parents:
            return path
    if kind == "research":
        for candidate in store.dataset_files():
            if candidate.name == name:
                return candidate
    raise HTTPException(status_code=404, detail={"error": "not_found", "detail": f"Unknown dataset {dataset_id}"})


@router.get(
    "/datasets/validate",
    summary="Validate a dataset against the model feature contract",
    responses={200: {"model": schemas.DatasetValidationResponse}},
)
def validate_dataset(
    dataset_id: str = Query(min_length=1),
    request: Request = None,
    store: ArtifactStore = Depends(get_store),
) -> dict:
    path = _resolve_dataset_path(request, store, dataset_id)
    return datasets_service.analyse_file(path, store.model_feature_names())


@router.post(
    "/datasets/upload",
    summary="Upload a flow CSV for validation (never replaces research data)",
    dependencies=[Depends(require_write_access)],
)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    store: ArtifactStore = Depends(get_store),
) -> dict:
    settings = request.app.state.settings
    content = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(content) == 0:
        raise HTTPException(status_code=400, detail={"error": "empty_file", "detail": "Uploaded file is empty."})
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": f"Upload exceeds the {settings.max_upload_mb} MB limit.",
            },
        )
    path = datasets_service.save_upload(settings.uploads_dir, file.filename or "dataset.csv", content)
    validation = datasets_service.analyse_file(path, store.model_feature_names())
    request.app.state.database.audit("analyst", "dataset_uploaded", path.name)
    return {
        "dataset_id": f"upload:{path.name}",
        "path": str(path),
        "validation": validation,
        "note": "Stored under var/uploads for validation only. Research artifacts were not modified.",
    }
