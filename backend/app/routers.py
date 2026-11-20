"""HTTP routers. Thin wrappers over :mod:`app.services`."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response

from . import services
from .config import GENERATED_REPORTS_DIR, MODEL_VERSIONS, REPO_ROOT
from .schemas import SettingsPayload
from .state import get_active_model, load_settings, save_settings, set_active_model

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- system ----

@router.get("/health", tags=["system"])
def get_health():
    return services.health()


@router.get("/settings", tags=["system"])
def get_settings():
    return {
        "settings": load_settings().model_dump(),
        "defaults": SettingsPayload().model_dump(),
        "note": (
            "Settings persist server-side under backend/var. The anomaly review "
            "threshold is a UI preference and never rewrites recorded research "
            "results or the frozen calibration."
        ),
    }


@router.put("/settings", tags=["system"])
def put_settings(payload: SettingsPayload):
    saved = save_settings(payload)
    return {"settings": saved.model_dump(), "status": "saved"}


# --------------------------------------------------------------- dashboard --

@router.get("/dashboard", tags=["dashboard"])
def get_dashboard():
    return services.dashboard()


# ----------------------------------------------------------------- network --

@router.get("/network/overview", tags=["network"])
def get_network():
    return services.network_overview()


# ------------------------------------------------------------- predictions --

@router.get("/predictions", tags=["predictions"])
def get_predictions():
    return services.predictions()


@router.get("/predictions/multi-step", tags=["predictions"])
def get_multi_step(
    horizon: int | None = Query(default=None, ge=1, le=3600),
    label: str | None = Query(default=None, max_length=40),
):
    data = services.predictions()["multi_step"]
    rows = data["rows"]
    if horizon is not None:
        rows = [r for r in rows if r["horizon_seconds"] == horizon]
    if label is not None:
        rows = [r for r in rows if r["target_label"].lower() == label.lower()]
    return {
        **data,
        "rows": rows,
        "filtered_count": len(rows),
        "horizons": data["horizons"],
        "labels": sorted({r["target_label"] for r in data["rows"]}),
    }


# ------------------------------------------------------------------ triage --

@router.get("/triage", tags=["triage"])
def get_triage():
    return services.triage()


# ---------------------------------------------------------- explainability --

@router.get("/explainability", tags=["explainability"])
def get_explainability():
    return services.explainability()


# ---------------------------------------------------------------- timeline --

@router.get("/timeline", tags=["timeline"])
def get_timeline():
    return services.timeline()


# ---------------------------------------------------------------- datasets --

@router.get("/datasets", tags=["datasets"])
def get_datasets():
    return services.datasets()


# ------------------------------------------------------------------ models --

@router.get("/models", tags=["models"])
def get_models():
    data = services.models_overview()
    data["active_model"] = get_active_model()
    return data


@router.post("/models/{model_id}/activate", tags=["models"])
def activate_model(model_id: str):
    if model_id not in MODEL_VERSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown model version: {model_id}")
    set_active_model(model_id)
    status = next(m for m in services.model_statuses() if m["id"] == model_id)
    return {
        "active_model": model_id,
        "status": "activated",
        "inference_available": status["loadable"],
        "warning": None
        if status["loadable"]
        else (
            "Activated for reporting context only. The checkpoint for this version is "
            "not present in this checkout, so on-demand inference remains unavailable."
        ),
    }


# ----------------------------------------------------------------- reports --

@router.get("/reports/catalog", tags=["reports"])
def reports_catalog():
    return {"artifacts": services.report_catalog()}


@router.get("/reports/artifact/{report_id}", tags=["reports"])
def reports_artifact(report_id: str):
    entry = next((r for r in services.report_catalog() if r["id"] == report_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown report artifact")
    if not entry["present"]:
        raise HTTPException(
            status_code=409,
            detail=f"Artifact not present in this checkout: {entry['path']}",
        )
    path = REPO_ROOT / entry["path"]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if entry["kind"] == "json":
        try:
            return {"id": report_id, "kind": "json", "content": json.loads(raw)}
        except json.JSONDecodeError:
            return {"id": report_id, "kind": "text", "content": raw}
    return {"id": report_id, "kind": "text", "content": raw}


@router.post("/reports/generate", tags=["reports"])
def generate_report():
    result = services.generate_summary_report()
    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result["generated_at"].lower())[:40]
    out = GENERATED_REPORTS_DIR / f"{slug}.md"
    out.write_text(result["markdown"], encoding="utf-8")
    result["id"] = out.stem
    return result


@router.get("/reports/generated", tags=["reports"])
def list_generated():
    if not GENERATED_REPORTS_DIR.is_dir():
        return {"reports": []}
    out = []
    for p in sorted(GENERATED_REPORTS_DIR.glob("*.md"), reverse=True):
        out.append({"id": p.stem, "bytes": p.stat().st_size})
    return {"reports": out}


@router.get("/reports/generated/{report_id}", tags=["reports"])
def download_generated(report_id: str):
    if not re.fullmatch(r"[a-z0-9\-]+", report_id):
        raise HTTPException(status_code=400, detail="Invalid report id")
    path = GENERATED_REPORTS_DIR / f"{report_id}.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated report not found")
    return Response(
        content=path.read_text(encoding="utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{report_id}.md"'},
    )
