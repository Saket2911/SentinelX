"""System status, capabilities and deployment health."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from ..config import Settings
from .artifacts import ArtifactStore

VERSION = "4.0.0"


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def health(settings: Settings, store: ArtifactStore, uptime_seconds: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, label: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "label": label, "status": status, "detail": detail})

    add(
        "models_dir",
        "Models directory",
        "ok" if settings.models_dir.is_dir() else "failed",
        str(settings.models_dir),
    )
    add(
        "runs",
        "Model runs discovered",
        "ok" if store.runs else "failed",
        f"{len(store.runs)} run(s): {', '.join(sorted(store.runs)) or 'none'}",
    )
    checkpoints = [
        run_id for run_id, run in store.runs.items() if run.has_checkpoint
    ]
    add(
        "checkpoints",
        "Inference checkpoints",
        "ok" if checkpoints else "warning",
        (
            f"Checkpoint present for {', '.join(sorted(checkpoints))}"
            if checkpoints
            else "No .pt checkpoint is present in this checkout; inference endpoints will report unavailable."
        ),
    )
    add(
        "torch",
        "PyTorch runtime",
        "ok" if _torch_available() else "warning",
        "torch importable." if _torch_available() else "torch is not installed in this environment.",
    )
    try:
        settings.var_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.var_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    add(
        "state",
        "Writable application state",
        "ok" if writable else "failed",
        str(settings.var_dir),
    )
    add(
        "datasets",
        "Dataset directory",
        "ok" if settings.datasets_dir.is_dir() else "warning",
        (
            str(settings.datasets_dir)
            if settings.datasets_dir.is_dir()
            else f"{settings.datasets_dir} is absent; dataset-dependent features report unavailable."
        ),
    )
    free = shutil.disk_usage(settings.var_dir if settings.var_dir.exists() else settings.root).free
    add(
        "disk",
        "Free disk space",
        "ok" if free > 512 * 1024 * 1024 else "warning",
        f"{free / 1024 ** 3:.1f} GiB free",
    )

    statuses = {check["status"] for check in checks}
    overall = "failed" if "failed" in statuses else ("degraded" if "warning" in statuses else "ok")
    return {
        "status": overall,
        "version": VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "checks": checks,
        "runtime": {
            "root": str(settings.root),
            "models_dir": str(settings.models_dir),
            "datasets_dir": str(settings.datasets_dir),
            "var_dir": str(settings.var_dir),
            "auth_enabled": settings.auth_enabled,
            "torch_available": _torch_available(),
            "python": _python_version(),
        },
    }


def _python_version() -> str:
    import platform

    return platform.python_version()


def capabilities() -> dict[str, Any]:
    capabilities = [
        {
            "id": "offline_dataset_mode",
            "label": "Offline dataset workflow",
            "state": "implemented",
            "detail": "Reads recorded state tables, sequence metadata and evaluation artifacts.",
        },
        {
            "id": "one_step_forecasting",
            "label": "One-step state forecasting",
            "state": "conditional",
            "detail": "Requires best_model.pt plus the frozen 209-feature preprocessing artifacts.",
        },
        {
            "id": "residual_triage",
            "label": "Residual review prioritisation",
            "state": "conditional",
            "detail": "Requires a replay or explainable triage table with residual columns.",
        },
        {
            "id": "shap_local",
            "label": "Local SHAP explanations",
            "state": "conditional",
            "detail": "Requires shap_explainability_v3/shap_values.npz.",
        },
        {
            "id": "analyst_workflow",
            "label": "Analyst workflow and audit trail",
            "state": "implemented",
            "detail": "Case status, assignment and notes are persisted in SQLite.",
        },
        {
            "id": "reporting",
            "label": "Report generation",
            "state": "implemented",
            "detail": "Markdown and JSON reports assembled from the same services as the dashboard.",
        },
        {
            "id": "live_telemetry",
            "label": "Live network telemetry",
            "state": "planned",
            "detail": "Not implemented. No packet capture or Zeek ingestion is present in this release.",
        },
        {
            "id": "pcap_ingestion",
            "label": "PCAP ingestion",
            "state": "planned",
            "detail": "Planned. The feature contract expects CICFlowMeter-compatible flow records.",
        },
        {
            "id": "model_training_ui",
            "label": "Training from the UI",
            "state": "planned",
            "detail": "Training stays a CLI workflow (experiments/run_experiment.py) to keep runs reproducible.",
        },
    ]
    return {
        "capabilities": capabilities,
        "limitations": [
            "Live capture is not implemented; all analytics are derived from recorded artifacts.",
            "Model inference requires local artifacts that are excluded from version control.",
            "Residual review is a heuristic and the recorded detection recall is low.",
        ],
    }


def status(settings: Settings, store: ArtifactStore, run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    artifacts = [
        {
            "key": artifact.key,
            "label": artifact.label,
            "path": artifact.display_path,
            "kind": artifact.kind,
            "present": artifact.present,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in list(run.artifacts.values()) + list(run.tables.values())
    ]
    artifacts.extend(
        {
            "key": f"dataset:{artifact.key}",
            "label": artifact.label,
            "path": artifact.display_path,
            "kind": "table",
            "present": artifact.present,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in store.dataset_artifacts()
    )
    present = sum(1 for artifact in artifacts if artifact["present"])
    from datetime import datetime, timezone

    return {
        "source": {
            "mode": "recorded_artifact" if present else "unavailable",
            "label": "Artifact inventory",
            "detail": str(run.directory),
        },
        "availability": {
            "available": present > 0,
            "reason": None if present else "No artifacts present.",
            "required_artifacts": [artifact["path"] for artifact in artifacts if not artifact["present"]][:12],
        },
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "active_run_id": run_id,
        "artifacts": artifacts,
        "runtime": {
            "root": str(settings.root),
            "models_dir": str(settings.models_dir),
            "datasets_dir": str(settings.datasets_dir),
            "var_dir": str(settings.var_dir),
            "auth_enabled": settings.auth_enabled,
            "torch_available": _torch_available(),
            "artifacts_present": present,
            "artifacts_total": len(artifacts),
        },
        "limitations": [
            "Absent artifacts are reported explicitly; the API never substitutes generated values.",
            "Artifact presence reflects this checkout, not the original research environment.",
        ],
    }
