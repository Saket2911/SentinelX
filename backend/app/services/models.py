"""Model registry: run discovery, integrity checks, comparisons, activation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..db import Database
from .anomaly import calibration_for
from .artifacts import ArtifactStore, ModelRun
from .research_text import parse_feature_analysis_report, parse_multi_step_report, parse_training_report
from .settings_store import SettingsStore

V3_CONTRACT = {
    "input_features": 209,
    "hidden_size": 128,
    "num_layers": 1,
    "sequence_length": 10,
}

HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    if cache_key in HASH_CACHE:
        return HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    HASH_CACHE[cache_key] = value
    return value


def _availability(run: ModelRun) -> dict[str, Any]:
    required = [run.artifacts["config"].path.name]
    if run.artifacts["config"].present:
        return {
            "available": True,
            "reason": None,
            "required_artifacts": required,
        }
    return {
        "available": False,
        "reason": "Run directory exists but config.json is missing.",
        "required_artifacts": required,
    }


def _architecture(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", "SentinelXLSTM"),
        "input_features": config.get("input_features"),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_layers"),
        "dropout": config.get("dropout"),
        "sequence_length": config.get("sequence_length"),
        "output_features": config.get("input_features"),
        "loss": config.get("loss"),
        "optimizer": config.get("optimizer"),
    }


def _training(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_samples": config.get("training_samples"),
        "validation_samples": config.get("validation_samples"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "max_epochs": config.get("max_epochs"),
        "patience": config.get("patience"),
        "seed": config.get("seed"),
        "device": config.get("device"),
        "gpu_name": config.get("gpu_name"),
        "gpu_memory_gb": config.get("gpu_memory_gb"),
        "best_epoch": config.get("best_epoch"),
        "best_validation_huber": config.get("best_validation_huber"),
        "training_seconds": config.get("training_seconds"),
        "test_data_used_for_training": config.get("test_data_used", False),
    }


def _checkpoint(run: ModelRun) -> dict[str, Any]:
    files = []
    for path in run.checkpoints:
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": file_sha256(path),
            }
        )
    return {
        "present": bool(files),
        "files": files,
        "count": len(files),
        "primary": files[0] if files else None,
    }


def _test_metrics(store: ArtifactStore, run_id: str) -> dict[str, Any] | None:
    summary = store.read_json(run_id, "test_evaluation")
    if not summary:
        return None
    return {
        "samples": summary.get("test_samples"),
        "lstm": summary.get("lstm_metrics") or {},
        "persistence": summary.get("persistence_baseline_metrics") or {},
        "improvement_percent": summary.get("improvement_over_baseline_percent") or {},
        "best_epoch": summary.get("best_epoch"),
        "evaluation_seconds": summary.get("evaluation_seconds"),
        "test_used_for_training": summary.get("test_data_used_for_training"),
        "test_used_for_selection": summary.get("test_data_used_for_model_selection"),
        "space": "Preprocessed model feature space",
    }


def _validation_metrics(store: ArtifactStore, run_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    anomaly = store.read_json(run_id, "validation_anomaly") or {}
    evaluation = store.read_json(run_id, "test_evaluation") or {}
    value = config.get("best_validation_huber")
    if value is None and not anomaly and not evaluation:
        return None
    return {
        "best_validation_huber": value,
        "source": "Run configuration (training-time validation)",
        "samples": config.get("validation_samples"),
        "anomaly_samples": anomaly.get("validation_samples"),
        "space": "Preprocessed model feature space",
    }


def health_checks(store: ArtifactStore, run_id: str, run: ModelRun, config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "id": "config",
            "label": "Run configuration",
            "status": "ok" if run.artifacts["config"].present else "failed",
            "detail": str(run.artifacts["config"].path),
        }
    )
    contract_ok = all(config.get(key) == value for key, value in V3_CONTRACT.items())
    checks.append(
        {
            "id": "feature_contract",
            "label": "Feature contract (209 / seq 10 / hidden 128 / 1 layer)",
            "status": "ok" if contract_ok else "warning",
            "detail": (
                "Matches the V3 inference contract."
                if contract_ok
                else "Run does not match the V3 inference contract; the API will not run inference on it."
            ),
        }
    )
    checkpoint = _checkpoint(run)
    checks.append(
        {
            "id": "checkpoint",
            "label": "Checkpoint present",
            "status": "ok" if checkpoint["present"] else "warning",
            "detail": (
                checkpoint["primary"]["name"]
                if checkpoint["present"]
                else "No .pt checkpoint in this checkout (git-ignored). Inference is unavailable until it is supplied."
            ),
        }
    )
    recorded_hash = (store.read_json(run_id, "replay_audit") or {}).get("checkpoint_sha256")
    if not recorded_hash:
        import json

        reliability_path = store.reports_dir / "v3_reliability_audit_v1" / "reliability_summary.json"
        if reliability_path.is_file():

            try:
                with reliability_path.open("r", encoding="utf-8") as handle:
                    recorded_hash = json.load(handle).get("checkpoint_sha256")
            except (OSError, json.JSONDecodeError):
                recorded_hash = None
    if checkpoint["present"] and recorded_hash:
        actual = checkpoint["primary"]["sha256"]
        checks.append(
            {
                "id": "checkpoint_hash",
                "label": "Checkpoint hash matches recorded audit",
                "status": "ok" if actual == recorded_hash else "warning",
                "detail": (
                    "SHA-256 matches the recorded reliability audit."
                    if actual == recorded_hash
                    else "Checkpoint bytes differ from the hash recorded in the reliability audit."
                ),
            }
        )
    checks.append(
        {
            "id": "evaluation",
            "label": "Held-out evaluation present",
            "status": "ok" if run.artifacts["test_evaluation"].present else "warning",
            "detail": "Recorded one-step test metrics." if run.artifacts["test_evaluation"].present else "No test evaluation summary.",
        }
    )

    epochs = {
        "test_evaluation": (store.read_json(run_id, "test_evaluation") or {}).get("best_epoch"),
        "test_anomaly": (store.read_json(run_id, "test_anomaly") or {}).get("checkpoint_epoch"),
        "validation_anomaly": (store.read_json(run_id, "validation_anomaly") or {}).get("checkpoint_epoch"),
        "shap": (store.read_json(run_id, "shap") or {}).get("checkpoint_epoch"),
    }
    known = {value for value in epochs.values() if value is not None}
    checks.append(
        {
            "id": "epoch_consistency",
            "label": "Artifact epoch consistency",
            "status": "ok" if len(known) <= 1 else "warning",
            "detail": (
                "All summaries reference the same checkpoint epoch."
                if len(known) <= 1
                else f"Summaries reference different epochs: {epochs}"
            ),
        }
    )
    return checks


def summary(store: ArtifactStore, run_id: str, active_run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    config = store.read_json(run_id, "config") or {}
    return {
        "run_id": run_id,
        "name": str(config.get("model") or run_id),
        "active": run_id == active_run_id,
        "available": run.artifacts["config"].present,
        "availability": _availability(run),
        "architecture": _architecture(config),
        "training": _training(config),
        "checkpoint": _checkpoint(run),
        "validation_metrics": _validation_metrics(store, run_id, config),
        "test_metrics": _test_metrics(store, run_id),
    }


def list_models(store: ArtifactStore, active_run_id: str) -> list[dict[str, Any]]:
    return [summary(store, run_id, active_run_id) for run_id in store.runs]


def detail(store: ArtifactStore, run_id: str, active_run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    config = store.read_json(run_id, "config") or {}
    model = summary(store, run_id, active_run_id)
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
    multi_step_text = store.read_text(run_id, "multi_step_report")
    multi_step = parse_multi_step_report(multi_step_text) if multi_step_text else []
    feature_text = store.read_text(run_id, "feature_analysis_report")
    feature_analysis = parse_feature_analysis_report(feature_text) if feature_text else {"horizons": [], "verification_passed": None}
    training_text = store.read_text(run_id, "training_report")
    training_facts = parse_training_report(training_text) if training_text else {}

    shap = store.read_json(run_id, "shap") or {}
    triage = store.read_json(run_id, "integrated_triage") or {}
    test_anomaly = store.read_json(run_id, "test_anomaly") or {}
    validation_anomaly = store.read_json(run_id, "validation_anomaly") or {}
    evaluation = {
        "test": model["test_metrics"],
        "validation": model["validation_metrics"],
        "test_anomaly": test_anomaly or None,
        "validation_anomaly": validation_anomaly or None,
        "triage": triage or None,
        "feature_analysis": feature_analysis,
        "training_facts": training_facts,
        "calibration": calibration_for(store, run_id).as_dict(),
        "limitations": [
            "Metrics are in standardized preprocessed model feature space, not raw units.",
            "The test partition was previously used for exploratory analysis and is not an untouched independent test.",
            "Residual thresholds are review heuristics, not validated attack detectors.",
        ],
    }
    training_curve: list[dict[str, Any]] = []
    for line in training_text.splitlines() if training_text else []:
        stripped = line.strip()
        if not stripped.lower().startswith("epoch"):
            continue
        parts = [part for part in stripped.replace("\t", " ").split("  ") if part.strip()]
        entry: dict[str, Any] = {"epoch": len(training_curve) + 1}
        for part in parts:
            if ":" not in part:
                continue
            key, _, value = part.partition(":")
            try:
                entry[key.strip().lower().replace(" ", "_")] = float(value)
            except ValueError:
                continue
        training_curve.append(entry)

    explanations = {
        "method": shap.get("method"),
        "output_features": shap.get("output_features") or [],
        "explanation_samples": (shap.get("benign_explanation_samples") or 0)
        + (shap.get("bot_explanation_samples") or 0),
        "background_size": shap.get("background_size"),
        "interpretation": shap.get("interpretation"),
        "available": bool(shap),
    }
    availability = {
        "available": True,
        "reason": None,
        "required_artifacts": [run.artifacts["config"].path.name],
    }
    if not model["checkpoint"]["present"]:
        availability["reason"] = (
            "The checkpoint (.pt) is not present in this checkout; recorded metrics are shown, "
            "live inference is unavailable."
        )
    return {
        "model": model,
        "artifacts": artifacts,
        "training_curve": training_curve,
        "evaluation": evaluation,
        "multi_step": multi_step,
        "explanations": explanations,
        "health": health_checks(store, run_id, run, config),
        "availability": availability,
    }


def compare(store: ArtifactStore, left_id: str, right_id: str, active_run_id: str) -> dict[str, Any]:
    left = summary(store, left_id, active_run_id)
    right = summary(store, right_id, active_run_id)
    metrics: list[dict[str, Any]] = []

    def add(metric_id: str, label: str, unit: str, left_value: Any, right_value: Any, higher_is_better: bool) -> None:
        delta = None
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            delta = float(left_value) - float(right_value)
        metrics.append(
            {
                "id": metric_id,
                "label": label,
                "unit": unit,
                "left": left_value,
                "right": right_value,
                "delta": delta,
                "higher_is_better": higher_is_better,
            }
        )

    left_test = (left["test_metrics"] or {}).get("lstm") or {}
    right_test = (right["test_metrics"] or {}).get("lstm") or {}
    add("mae", "Test MAE", "model space", left_test.get("mae"), right_test.get("mae"), False)
    add("rmse", "Test RMSE", "model space", left_test.get("rmse"), right_test.get("rmse"), False)
    add("huber", "Test Huber", "model space", left_test.get("huber_loss"), right_test.get("huber_loss"), False)
    add(
        "val_huber",
        "Best validation Huber",
        "model space",
        (left["validation_metrics"] or {}).get("best_validation_huber"),
        (right["validation_metrics"] or {}).get("best_validation_huber"),
        False,
    )
    add(
        "features",
        "Input features",
        "features",
        left["architecture"].get("input_features"),
        right["architecture"].get("input_features"),
        True,
    )
    add(
        "params_hidden",
        "Hidden size",
        "units",
        left["architecture"].get("hidden_size"),
        right["architecture"].get("hidden_size"),
        True,
    )
    warnings: list[str] = []
    left_features = left["architecture"].get("input_features")
    right_features = right["architecture"].get("input_features")
    if left_features != right_features:
        warnings.append(
            "The runs use different feature counts. Their metrics, scalers and sequence arrays are "
            "not interchangeable; compare them only as separate experiments."
        )
    if left_test and right_test:
        warnings.append(
            "Test metrics were recorded after exploratory use of the test partition; no controlled "
            "retraining comparison is implied by this table."
        )
    return {"left": left, "right": right, "metrics": metrics, "warnings": warnings}


def activate(
    store: ArtifactStore,
    database: Database,
    settings: SettingsStore,
    run_id: str,
    actor: str,
    reason: str | None,
) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    config = store.read_json(run_id, "config") or {}
    warnings: list[str] = []
    if not run.artifacts["config"].present:
        raise ValueError("The selected run has no config.json and cannot be activated.")
    if not run.has_checkpoint:
        raise ValueError(
            "The selected run has no checkpoint in this checkout. Supply best_model.pt before "
            "activating it; recorded metrics alone cannot serve inference."
        )
    mismatched = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in V3_CONTRACT.items()
        if config.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "The selected run does not satisfy the V3 inference contract: "
            + ", ".join(f"{key}={value['actual']} (expected {value['expected']})" for key, value in mismatched.items())
        )
    previous = settings.active_run_id()
    if previous == run_id:
        warnings.append("The selected run was already active.")
    timestamp = database.record_activation(run_id, previous, actor, reason)
    settings.set_active_run(run_id, timestamp)
    if not (store.read_json(run_id, "test_evaluation") or {}):
        warnings.append("This run has no recorded test evaluation.")
    return {
        "active_run_id": run_id,
        "previous_run_id": previous,
        "activated_at": timestamp,
        "reason": reason,
        "warnings": warnings,
    }
