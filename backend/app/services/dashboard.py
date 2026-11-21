"""SOC dashboard composition."""

from __future__ import annotations

from typing import Any

from ..db import Database, utc_now
from . import alerts as alerts_service
from . import events as events_service
from . import models as models_service
from . import network as network_service
from . import predictions as predictions_service
from .anomaly import calibration_for
from .artifacts import ArtifactStore


def overview(store: ArtifactStore, database: Database, run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    network = network_service.overview(store, run_id)
    predictions = predictions_service.summary(store, run_id)
    event_summary = events_service.summary_counts(store, database, run_id)
    alert_summary = alerts_service.alert_summary(store, database, run_id)
    event_page = events_service.list_events(
        store, database, run_id, page=1, page_size=5
    )
    model = models_service.summary(store, run_id, run_id)
    calibration = calibration_for(store, run_id)
    config = store.read_json(run_id, "config") or {}
    dataset_artifacts = store.dataset_artifacts()
    present_dataset_artifacts = sum(1 for artifact in dataset_artifacts if artifact.present)

    telemetry_available = bool(network["availability"]["available"])
    observations = (network.get("totals") or {}).get("observations")
    candidates = event_summary.get("candidates") or 0
    checkpoint_present = bool(model["checkpoint"]["present"])
    evaluation = store.read_json(run_id, "test_evaluation") or {}

    if candidates:
        posture = "review_required"
        headline = f"{candidates} observation(s) crossed the frozen review threshold"
        posture_detail = (
            "These are residual review candidates, not confirmed incidents. Corroborate with "
            "independent network or host evidence before escalating."
        )
    elif not network["availability"]["available"]:
        posture = "degraded"
        headline = "Telemetry artifacts are not available in this deployment"
        posture_detail = (
            "The research summaries are loaded, but row-level replay data is missing, so no "
            "live posture can be computed."
        )
    elif not checkpoint_present:
        posture = "degraded"
        headline = "Recorded evaluation loaded; checkpoint not present"
        posture_detail = (
            "Metrics and triage statistics come from committed research artifacts. Live inference "
            "is disabled until best_model.pt is supplied."
        )
    else:
        posture = "nominal"
        headline = "No observation crossed the frozen review threshold"
        posture_detail = (
            "A low residual does not prove benign activity: the recorded evaluation recalled 0 of "
            "the labelled Bot observations at this threshold."
        )

    metrics: list[dict[str, Any]] = [
        {
            "id": "observations",
            "label": "States observed",
            "value": observations,
            "display": f"{observations:,}" if isinstance(observations, int) else "n/a",
            "unit": "1 s states",
            "status": "neutral",
            "detail": "Aggregated one-second network states in the replay window.",
        },
        {
            "id": "benign",
            "label": "Dataset-labelled benign",
            "value": (network.get("totals") or {}).get("benign_observations"),
            "display": f"{(network.get('totals') or {}).get('benign_observations'):,}"
            if isinstance((network.get("totals") or {}).get("benign_observations"), int)
            else "n/a",
            "unit": "states",
            "status": "neutral",
            "detail": "Reference labels from the capture, not model output.",
        },
        {
            "id": "attack_labels",
            "label": "Dataset-labelled attack",
            "value": (network.get("totals") or {}).get("attack_labelled_observations"),
            "display": f"{(network.get('totals') or {}).get('attack_labelled_observations'):,}"
            if isinstance((network.get("totals") or {}).get("attack_labelled_observations"), int)
            else "n/a",
            "unit": "states",
            "status": "warning" if (network.get("totals") or {}).get("attack_labelled_observations") else "neutral",
            "detail": "Reference annotations for the window; the LSTM does not emit attack labels.",
        },
        {
            "id": "review_candidates",
            "label": "Review candidates",
            "value": candidates if telemetry_available else None,
            "display": f"{candidates:,}" if telemetry_available else "n/a",
            "unit": "states",
            "status": ("critical" if candidates else "good") if telemetry_available else "unknown",
            "detail": (
                f"Residual >= frozen threshold {calibration.threshold}"
                if calibration.threshold is not None
                else "Frozen threshold not recorded."
            )
            if telemetry_available
            else "No row-level telemetry is available, so no review queue can be computed.",
        },
        {
            "id": "playback_alerts",
            "label": "Playback alerts",
            "value": alert_summary["total"],
            "display": f"{alert_summary['total']:,}",
            "unit": "alerts",
            "status": "critical" if alert_summary["high_priority"] else ("warning" if alert_summary["total"] else "good"),
            "detail": "Generated by the playback anomaly rule set and persisted.",
        },
        {
            "id": "high_priority_alerts",
            "label": "High-priority alerts",
            "value": alert_summary["high_priority"],
            "display": f"{alert_summary['high_priority']:,}",
            "unit": "alerts",
            "status": "critical" if alert_summary["high_priority"] else "good",
            "detail": "Severity high or critical by the documented threshold-ratio rule.",
        },
        {
            "id": "unresolved_alerts",
            "label": "Unresolved alerts",
            "value": alert_summary["unresolved"],
            "display": f"{alert_summary['unresolved']:,}",
            "unit": "alerts",
            "status": "warning" if alert_summary["unresolved"] else "good",
            "detail": "New, investigating or escalated alerts.",
        },
        {
            "id": "mae",
            "label": "Model MAE",
            "value": ((evaluation.get("lstm_metrics") or {}).get("mae")),
            "display": f"{((evaluation.get('lstm_metrics') or {}).get('mae')):.4f}"
            if isinstance((evaluation.get("lstm_metrics") or {}).get("mae"), (int, float))
            else "n/a",
            "unit": "model space",
            "status": "neutral",
            "detail": "Held-out test partition, preprocessed feature space.",
        },
        {
            "id": "segments",
            "label": "Capture segments",
            "value": (network.get("totals") or {}).get("segments"),
            "display": str((network.get("totals") or {}).get("segments") or "n/a"),
            "unit": "segments",
            "status": "neutral",
            "detail": "Independent contiguous captures in the replay window.",
        },
        {
            "id": "dataset_artifacts",
            "label": "Dataset artifacts present",
            "value": present_dataset_artifacts,
            "display": f"{present_dataset_artifacts}/{len(dataset_artifacts)}",
            "unit": "files",
            "status": "good" if present_dataset_artifacts == len(dataset_artifacts) else "warning",
            "detail": "Frozen preprocessing and sequence artifacts required for inference.",
        },
    ]

    test_anomaly = store.read_json(run_id, "test_anomaly") or {}
    validation_anomaly = store.read_json(run_id, "validation_anomaly") or {}
    prediction_quality = {
        "metrics": predictions["summary"].get("metrics") or [],
        "per_label": predictions["summary"].get("per_label") or [],
        "space": "Preprocessed model feature space",
        "test_partition_note": predictions["summary"].get("test_partition_note"),
        "detection": {
            "test": test_anomaly.get("detection_metrics_benign_vs_bot"),
            "validation": validation_anomaly.get("detection_metrics_benign_infilteration")
            or validation_anomaly.get("detection_metrics_benign_vs_infilteration"),
            "warning": "Detection metrics at a frozen residual threshold are descriptive, not a validated detector.",
        },
    }

    system_health = {
        "checkpoint_present": checkpoint_present,
        "evaluation_present": bool(evaluation),
        "dataset_artifacts_present": present_dataset_artifacts,
        "dataset_artifacts_total": len(dataset_artifacts),
        "preprocessing_ready": present_dataset_artifacts == len(dataset_artifacts),
        "device_recorded": config.get("device"),
        "gpu_recorded": config.get("gpu_name"),
    }

    limitations = []
    limitations.extend(network.get("limitations") or [])
    limitations.extend(predictions.get("limitations") or [])
    limitations.append(
        "Dashboard figures are derived from recorded research artifacts, not from live network capture."
    )
    limitations.append(
        "Dataset labels are reference annotations; prediction residuals are not attack confirmations."
    )

    return {
        "source": {
            "mode": "recorded_artifact" if evaluation or network["availability"]["available"] else "unavailable",
            "label": "Recorded research artifacts",
            "detail": str(run.directory),
        },
        "availability": network["availability"],
        "status": {
            "posture": posture,
            "headline": headline,
            "detail": posture_detail,
            "run_id": run_id,
            "generated_at": utc_now(),
            "threshold": calibration.threshold,
            "threshold_source": calibration.threshold_source,
        },
        "metrics": metrics,
        "activity_series": network.get("activity_series") or [],
        "label_distribution": network.get("label_distribution") or [],
        "model_health": {
            **model,
            "calibration": calibration.as_dict(),
        },
        "prediction_quality": prediction_quality,
        "dataset_status": {
            "artifacts": [
                {
                    "key": artifact.key,
                    "label": artifact.label,
                    "present": artifact.present,
                    "path": artifact.display_path,
                }
                for artifact in dataset_artifacts
            ],
            "dataset_directory": str(store.datasets_dir),
            "label_coverage": store.label_coverage(),
        },
        "system_health": system_health,
        "recent_events": event_page["events"],
        "event_summary": event_summary,
        "alerts": alert_summary,
        "limitations": limitations,
    }
