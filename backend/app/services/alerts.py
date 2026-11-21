"""Prediction-residual anomaly scoring and playback alert generation.

The LSTM world model forecasts the next one-second network state. The anomaly
score is the mean absolute residual between the observed state and that
forecast, computed in the standardized model feature space used by the frozen
V3 preprocessing pipeline. This is the same score that the recorded evaluation
used to calibrate the frozen review threshold (validation benign p99), so no new
uncalibrated metric is introduced.

Alerts are persisted idempotently per (run, model version, dataset, target
state). They are *review alerts*: a large residual means the model found the
state hard to predict, not that an attack occurred.

Playback can run the world model today in two ways:

* the historical replay artifacts already contain the LSTM forecast for every
  state (``replay_arrays.npz``), and that is the default source; or
* a live checkpoint with PyTorch can be supplied, in which case the same
  service can be extended to recompute the forecast. This deployment uses the
  recorded forecast and labels it explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import numpy as np

from ..db import Database, utc_now
from .anomaly import calibration_for
from .artifacts import ArtifactError, ArtifactStore
from .playback import ReplayBundle, bundle as replay_bundle

SCORE_METHOD = "mean_absolute_residual"
SCORE_SPACE = "Preprocessed model feature space (V3 StandardScaler)"
RULE_VERSION = "playback-alert-rules-v1"
PREDICTION_SOURCE = "recorded_lstm_replay"
PREDICTION_SOURCE_DETAIL = (
    "LSTM V3 next-state forecasts stored by the historical replay pipeline "
    "(replay_arrays.npz). The same frozen checkpoint and 10-state input window "
    "produced these predictions; this process does not invent or re-simulate them."
)

LIMITATIONS = [
    "An alert means the observed state deviated strongly from the LSTM forecast. "
    "It is a review signal, not an attack classification.",
    "Scores are computed in standardized model feature space, not raw network units.",
    "The frozen threshold was calibrated on benign validation states and recalled 0 of "
    "the labelled Bot observations in the recorded test evaluation; it is not a "
    "validated intrusion detector.",
    "Playback evaluates held-out historical states. It is not live network monitoring.",
    "Dataset labels are reference annotations and are never used to trigger an alert.",
]

SEVERITIES = ("medium", "high", "critical")
STATUSES = ("new", "investigating", "escalated", "benign", "closed")


class ModelContractError(ValueError):
    """The selected model, config and arrays do not describe one V3 bundle."""


class InsufficientContextError(ValueError):
    """The target state does not have a complete ten-state input window."""


def rules() -> dict[str, Any]:
    """Alert rules, configurable through the environment.

    The defaults are conservative: an alert requires either a score at the
    calibrated validation threshold, or three consecutive states at 80% of it.
    """

    def _float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    persistence = max(1, min(_int("SENTINELX_ALERT_PERSISTENCE", 3), 20))
    elevated = min(max(_float("SENTINELX_ALERT_ELEVATED_RATIO", 0.8), 0.1), 1.0)
    high = max(_float("SENTINELX_ALERT_HIGH_RATIO", 1.5), 1.0)
    critical = max(_float("SENTINELX_ALERT_CRITICAL_RATIO", 2.0), high)
    return {
        "rule_version": RULE_VERSION,
        "persistence_window": persistence,
        "elevated_ratio": elevated,
        "high_ratio": high,
        "critical_ratio": critical,
    }


def _digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def model_version(store: ArtifactStore, run_id: str, bundle: ReplayBundle) -> str:
    config = store.read_json(run_id, "config") or {}
    epoch = bundle.manifest.get("checkpoint_epoch") or config.get("best_epoch")
    features = config.get("input_features") or len(bundle.feature_names)
    sequence = config.get("sequence_length") or bundle.sequence_length
    return f"{run_id}@epoch{epoch or 'unknown'}-{features}f-seq{sequence}"


def dataset_id(bundle: ReplayBundle) -> str:
    """Stable identity of the replay artifact pair used for deduplication."""
    try:
        arrays_size = bundle.arrays_path.stat().st_size
    except OSError:
        arrays_size = 0
    try:
        sequences_size = bundle.sequences_path.stat().st_size
    except OSError:
        sequences_size = 0
    return _digest(
        f"{bundle.arrays_path.name}:{arrays_size}:{bundle.sequences_path.name}:{sequences_size}",
        12,
    )


def _ensure_contract(store: ArtifactStore, run_id: str, bundle: ReplayBundle) -> dict[str, Any]:
    config = store.read_json(run_id, "config") or {}
    expected_features = config.get("input_features")
    expected_sequence = config.get("sequence_length")
    if isinstance(expected_features, int) and expected_features != len(bundle.feature_names):
        raise ModelContractError(
            f"Run config declares {expected_features} features but the replay arrays hold "
            f"{len(bundle.feature_names)}; refusing to mix model versions."
        )
    if isinstance(expected_sequence, int) and expected_sequence != bundle.sequence_length:
        raise ModelContractError(
            f"Run config declares sequence length {expected_sequence} but the arrays hold "
            f"{bundle.sequence_length}; refusing to mix model versions."
        )
    if bundle.sequences.ndim != 3 or bundle.sequences.shape[2] != len(bundle.feature_names):
        raise ModelContractError("Sequence feature width does not match the replay arrays.")
    return config


def _validate_sequence(
    sequence: np.ndarray, feature_count: int, sequence_length: int
) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)
    if array.shape != (sequence_length, feature_count):
        raise InsufficientContextError(
            f"Expected a {sequence_length} x {feature_count} input window, received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise InsufficientContextError("The ten-state input window contains non-finite values.")
    return array


def score_state(
    actual: np.ndarray,
    predicted: np.ndarray,
    feature_names: list[str],
    limit: int = 12,
) -> dict[str, Any]:
    """Mean absolute residual with feature-level explanation.

    Features are already standardized by the training-fitted scaler, so each
    feature contributes on a comparable scale and the mean is not dominated by
    a single raw-unit outlier. Ranking for explanation uses absolute error.
    """

    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted states differ in shape.")
    if actual_array.shape[0] != len(feature_names):
        raise ValueError("Feature names do not match the state width.")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("State contains non-finite values.")

    residual = actual_array - predicted_array
    absolute = np.abs(residual)
    mse = float(np.mean(residual**2))
    order = np.argsort(-absolute, kind="stable")[: max(1, int(limit))]
    total_absolute = float(absolute.sum()) or 1.0
    return {
        "method": SCORE_METHOD,
        "space": SCORE_SPACE,
        "anomaly_score": float(absolute.mean()),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mean_signed_residual": float(residual.mean()),
        "max_absolute_error": float(absolute.max()),
        "feature_count": int(actual_array.shape[0]),
        "feature_residuals": [
            {
                "feature": feature_names[int(index)],
                "actual": float(actual_array[int(index)]),
                "predicted": float(predicted_array[int(index)]),
                "signed_residual": float(residual[int(index)]),
                "absolute_error": float(absolute[int(index)]),
                "share_percent": round(100.0 * float(absolute[int(index)]) / total_absolute, 2),
            }
            for index in order
        ],
    }


def evaluate_conditions(
    scores: list[float],
    segments: list[str],
    threshold: float | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate single-state and persistence alert rules.

    ``scores`` and ``segments`` are chronological and end at the target state.
    Persistence never spans a segment boundary, and no alert can be generated
    without a calibrated threshold.
    """

    config = config or rules()
    window = int(config["persistence_window"])
    elevated_ratio = float(config["elevated_ratio"])
    high_ratio = float(config["high_ratio"])
    critical_ratio = float(config["critical_ratio"])

    if threshold is None or not scores:
        return {
            "available": False,
            "reason": "No calibrated residual threshold is recorded for this run.",
            "triggered": False,
            "single_threshold": False,
            "persistence": False,
            "severity": None,
            "trigger": None,
            "reasons": ["No calibrated threshold is recorded."],
            "recent_scores": scores,
            "rules": config,
        }

    current = float(scores[-1])
    recent = scores[-window:]
    recent_segments = segments[-window:]
    same_segment = len(set(str(segment) for segment in recent_segments)) <= 1
    persistence = (
        len(recent) >= window
        and same_segment
        and all(float(value) >= threshold * elevated_ratio for value in recent)
    )
    single = current >= threshold
    triggered = bool(single or persistence)
    ratio = current / threshold if threshold else 0.0

    severity: str | None = None
    if triggered:
        if ratio >= critical_ratio or (persistence and ratio >= high_ratio):
            severity = "critical"
        elif ratio >= high_ratio or persistence:
            severity = "high"
        else:
            severity = "medium"

    trigger_parts: list[str] = []
    reasons: list[str] = []
    if single:
        trigger_parts.append("score_threshold")
        reasons.append(
            f"Anomaly score {current:.4f} crossed the calibrated threshold {threshold:.4f}."
        )
    if persistence:
        trigger_parts.append("sustained_elevated")
        reasons.append(
            f"{len(recent)} consecutive states scored at or above "
            f"{threshold * elevated_ratio:.4f} ({elevated_ratio:.0%} of the threshold)."
        )
    if not triggered:
        reasons.append(
            f"Score {current:.4f} is below the threshold {threshold:.4f} and the "
            "persistence rule was not satisfied."
        )

    return {
        "available": True,
        "reason": None,
        "triggered": triggered,
        "single_threshold": bool(single),
        "persistence": bool(persistence),
        "same_segment": same_segment,
        "severity": severity,
        "trigger": "+".join(trigger_parts) if trigger_parts else None,
        "reasons": reasons,
        "recent_scores": [float(value) for value in recent],
        "ratio": ratio,
        "elevated_threshold": threshold * elevated_ratio,
        "rules": config,
        "threshold": threshold,
        "score": current,
    }


def _serialize_features(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def serialize_alert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "alert_id": row["alert_id"],
        "run_id": row["run_id"],
        "session_id": row["run_id"],
        "dataset_id": row["dataset_id"],
        "model_version": row["model_version"],
        "target_step": int(row["target_step"]),
        "target_index": int(row["target_index"]),
        "timestamp": row["timestamp"],
        "segment_id": row["segment_id"],
        "dataset_label": row["dataset_label"],
        "capture_day": row["capture_day"],
        "anomaly_score": row["anomaly_score"],
        "threshold": row["threshold"],
        "ratio": row["ratio"],
        "severity": row["severity"],
        "trigger_reason": row["trigger_reason"],
        "status": row["status"],
        "assignee": row["assignee"],
        "notes": row["notes"],
        "features": _serialize_features(row["features_json"]),
        "prediction_source": row["prediction_source"],
        "predicted_ref": row["predicted_ref"],
        "actual_ref": row["actual_ref"],
        "context_ref": row["context_ref"],
        "source": {
            "mode": "dataset",
            "label": "Prediction-residual playback alert",
            "detail": row["actual_ref"],
        },
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "limitations": LIMITATIONS,
    }


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        lowered_column = str(column).lower()
        if any(candidate in lowered_column for candidate in candidates):
            return str(column)
    return None


def shap_context(
    store: ArtifactStore, run_id: str, features: list[str], label: str | None = None
) -> dict[str, Any]:
    """Global SHAP magnitudes for the residual features, when recorded.

    This is supporting context only: SHAP attributes the model's output on the
    explanation sample, whereas an alert comes from the prediction residual on
    the target state. They are not interchangeable evidence.
    """

    table = store.read_table(run_id, "shap_importance")
    if table is None or table.empty:
        return {
            "available": False,
            "reason": "No recorded SHAP feature-importance table for this run.",
            "note": "SHAP attribution is not attached to this alert.",
        }
    columns = [str(column) for column in table.columns]
    feature_column = _find_column(columns, ("input feature", "feature"))
    value_column = _find_column(columns, ("mean absolute shap", "absolute shap", "mean_abs"))
    label_column = _find_column(columns, ("target label", "label"))
    if not feature_column or not value_column:
        return {
            "available": False,
            "reason": "The SHAP table has an unrecognised schema.",
            "note": "SHAP attribution is not attached to this alert.",
        }
    frame = table
    if label and label_column:
        lowered = frame[label_column].astype(str).str.strip().str.lower()
        filtered = frame[lowered == str(label).strip().lower()]
        if not filtered.empty:
            frame = filtered
    grouped: dict[str, float] = {}
    for _, row in frame.iterrows():
        name = str(row[feature_column])
        value = _finite(row[value_column])
        if value is None:
            continue
        grouped[name] = grouped.get(name, 0.0) + abs(value)
    requested = [name for name in features if name in grouped]
    matches = [
        {"feature": name, "mean_absolute_shap": round(grouped[name], 6)}
        for name in requested[:12]
    ]
    top = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "available": True,
        "source": "Recorded SHAP feature importance table",
        "label": label,
        "matches": matches,
        "top_features": [
            {"feature": name, "mean_absolute_shap": round(value, 6)} for name, value in top
        ],
        "note": (
            "Global SHAP magnitudes describe the model output on the explanation "
            "sample; they are not the alert's residual and do not prove an attack."
        ),
    }


def evaluate_target(
    store: ArtifactStore,
    database: Database,
    run_id: str,
    step: int,
) -> dict[str, Any]:
    """Score one playback target state and persist an alert when triggered."""

    bundle = replay_bundle(store, run_id)
    config = _ensure_contract(store, run_id, bundle)
    position = bundle.position_for(int(step))
    feature_count = len(bundle.feature_names)
    sequence_length = bundle.sequence_length
    if sequence_length <= 0:
        raise InsufficientContextError("The replay sequences have no time dimension.")
    sequence = _validate_sequence(
        bundle.sequences[position], feature_count, sequence_length
    )
    actual = np.asarray(bundle.actual[position], dtype=np.float64)
    predicted = np.asarray(bundle.predictions[position], dtype=np.float64)
    score = score_state(actual, predicted, bundle.feature_names)

    threshold = bundle.threshold
    config_rules = rules()
    window = int(config_rules["persistence_window"])
    first = max(0, position - window + 1)
    recent_positions = list(range(first, position + 1))
    scores = [
        float(np.mean(np.abs(bundle.actual[i] - bundle.predictions[i])))
        for i in recent_positions
    ]
    segments = [bundle.segments[i] for i in recent_positions]
    conditions = evaluate_conditions(scores, segments, threshold, config_rules)

    input_positions = list(range(max(0, position - sequence_length + 1), position + 1))
    context_segments = [bundle.segments[i] for i in input_positions]
    segment_consistent = len(set(context_segments)) <= 1
    model = model_version(store, run_id, bundle)
    dataset = dataset_id(bundle)
    timestamp = bundle.timestamps[position]

    alert: dict[str, Any] | None = None
    deduplicated = False
    if conditions.get("triggered"):
        identity = f"{run_id}|{model}|{dataset}|{int(step)}"
        alert_id = f"alert-{_digest(identity)}"
        now = utc_now()
        features_json = json.dumps(score["feature_residuals"])
        values = {
            "alert_id": alert_id,
            "run_id": run_id,
            "dataset_id": dataset,
            "model_version": model,
            "target_step": int(step),
            "target_index": position,
            "timestamp": timestamp,
            "segment_id": bundle.segments[position],
            "dataset_label": bundle.labels[position],
            "capture_day": bundle.capture_days[position] if bundle.capture_days else None,
            "anomaly_score": score["anomaly_score"],
            "threshold": threshold,
            "ratio": conditions.get("ratio"),
            "severity": conditions["severity"],
            "trigger_reason": _trigger_text(conditions),
            "status": "new",
            "assignee": None,
            "notes": None,
            "features_json": features_json,
            "prediction_source": PREDICTION_SOURCE,
            "predicted_ref": f"{bundle.arrays_path.name}#predictions[{position}]",
            "actual_ref": f"{bundle.arrays_path.name}#actual[{position}]",
            "context_ref": (
                f"{bundle.sequences_path.name}[{position}] "
                f"({sequence_length}x{feature_count} states)"
            ),
            "created_at": now,
            "updated_at": now,
        }
        row, created = database.insert_playback_alert(values)
        deduplicated = not created
        alert = serialize_alert(row)

    shap = shap_context(
        store,
        run_id,
        [entry["feature"] for entry in score["feature_residuals"]],
        bundle.labels[position],
    )

    return {
        "source": {
            "mode": "dataset",
            "label": "LSTM prediction residual evaluation",
            "detail": str(bundle.arrays_path),
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "run_id": run_id,
        "model_version": model,
        "dataset_id": dataset,
        "step": int(step),
        "state": {
            "step": int(step),
            "index": position,
            "total_states": bundle.total,
            "timestamp": timestamp,
            "capture_day": bundle.capture_days[position] if bundle.capture_days else None,
            "segment_id": bundle.segments[position],
            "dataset_label": bundle.labels[position],
            "threshold": threshold,
            "feature_count": feature_count,
        },
        "prediction": {
            "source": PREDICTION_SOURCE,
            "source_detail": PREDICTION_SOURCE_DETAIL,
            "model_version": model,
            "sequence_length": sequence_length,
            "features": feature_count,
            "target_step": int(step),
        },
        "actual": {
            "step": int(step),
            "timestamp": timestamp,
            "reference": f"{bundle.arrays_path.name}#actual[{position}]",
        },
        "context": {
            "states_used": sequence_length,
            "first_input_step": int(bundle.steps[input_positions[0]]),
            "last_input_step": int(bundle.steps[input_positions[-1]]),
            "target_step": int(step),
            "segment_ids": sorted(set(context_segments)),
            "segment_consistent": segment_consistent,
            "reference": f"{bundle.sequences_path.name}[{position}]",
        },
        "score": score,
        "conditions": conditions,
        "alert": alert,
        "deduplicated": deduplicated,
        "explainability": {"shap": shap},
        "limitations": LIMITATIONS,
    }


def _trigger_text(conditions: dict[str, Any]) -> str:
    parts = []
    if conditions.get("single_threshold"):
        parts.append("calibrated threshold crossed")
    if conditions.get("persistence"):
        parts.append(
            f"{len(conditions.get('recent_scores') or [])} consecutive elevated states"
        )
    return " + ".join(parts) or "alert condition met"


def anomaly_series(
    store: ArtifactStore, database: Database, run_id: str
) -> dict[str, Any]:
    """Per-state anomaly scores for charts and the flagged-state list."""

    try:
        bundle = replay_bundle(store, run_id)
    except ArtifactError as exc:
        return {
            "source": {
                "mode": "unavailable",
                "label": "LSTM residual anomaly scores",
                "detail": str(exc),
            },
            "availability": {
                "available": False,
                "reason": str(exc),
                "required_artifacts": [],
            },
            "run_id": run_id,
            "threshold": None,
            "threshold_source": "Unavailable",
            "method": SCORE_METHOD,
            "space": SCORE_SPACE,
            "flagged_steps": [],
            "alert_steps": [],
            "points": [],
            "summary": {
                "states": 0,
                "mean_score": 0.0,
                "max_score": 0.0,
                "flagged": 0,
                "persisted_alerts": 0,
            },
            "limitations": LIMITATIONS,
        }
    _ensure_contract(store, run_id, bundle)
    absolute = np.abs(bundle.actual - bundle.predictions)
    scores = absolute.mean(axis=1)
    threshold = bundle.threshold
    flagged = (
        [int(bundle.steps[i]) for i in np.nonzero(scores >= threshold)[0]]
        if threshold is not None
        else []
    )
    persisted = {
        int(row["target_step"]): row["alert_id"]
        for row in database.playback_alerts_for_run(run_id)
    }
    points = [
        {
            "index": position,
            "step": int(bundle.steps[position]),
            "timestamp": bundle.timestamps[position],
            "segment_id": bundle.segments[position],
            "label": bundle.labels[position],
            "anomaly_score": float(scores[position]),
            "flagged": bool(threshold is not None and scores[position] >= threshold),
            "alert_id": persisted.get(int(bundle.steps[position])),
        }
        for position in range(bundle.total)
    ]
    ordered = np.argsort(-scores, kind="stable")
    benign_names = {"benign", "normal", "no flows", "unknown", ""}
    label_counts: dict[str, int] = {}
    for label in bundle.labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    attack_labelled = sum(
        count
        for label, count in label_counts.items()
        if label.strip().lower() not in benign_names
    )
    top_states = [
        {
            "step": int(bundle.steps[int(position)]),
            "timestamp": bundle.timestamps[int(position)],
            "segment_id": bundle.segments[int(position)],
            "label": bundle.labels[int(position)],
            "anomaly_score": float(scores[int(position)]),
            "ratio": float(scores[int(position)] / threshold) if threshold else None,
            "flagged": bool(threshold is not None and scores[int(position)] >= threshold),
            "alert_id": persisted.get(int(bundle.steps[int(position)])),
        }
        for position in ordered[:25]
    ]
    return {
        "source": {
            "mode": "dataset",
            "label": "LSTM residual anomaly scores",
            "detail": str(bundle.arrays_path),
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "run_id": run_id,
        "threshold": threshold,
        "threshold_source": calibration_for(store, run_id).threshold_source,
        "method": SCORE_METHOD,
        "space": SCORE_SPACE,
        "flagged_steps": flagged,
        "alert_steps": sorted(persisted),
        "points": points,
        "top_states": top_states,
        "label_summary": label_counts,
        "attack_labelled_states": attack_labelled,
        "benign_labelled_states": int(bundle.total) - attack_labelled,
        "summary": {
            "states": bundle.total,
            "mean_score": float(scores.mean()),
            "max_score": float(scores.max()),
            "flagged": len(flagged),
            "persisted_alerts": len(persisted),
        },
        "limitations": LIMITATIONS,
    }


def scan_replay(
    store: ArtifactStore,
    database: Database,
    run_id: str,
    limit: int = 500,
) -> dict[str, Any]:
    """Find every state that satisfies the alert rules without visiting them.

    Playback at high speed cannot evaluate all 30k+ states individually, so the
    whole replay is scanned against the same vectorized score and rules, and only
    matching states are persisted (idempotently).
    """

    bundle = replay_bundle(store, run_id)
    _ensure_contract(store, run_id, bundle)
    threshold = bundle.threshold
    config = rules()
    window = int(config["persistence_window"])
    elevated_ratio = float(config["elevated_ratio"])
    scores = np.abs(bundle.actual - bundle.predictions).mean(axis=1)
    candidate_positions: list[int] = []
    for position in range(bundle.total):
        single = threshold is not None and scores[position] >= threshold
        first = max(0, position - window + 1)
        recent = scores[first : position + 1]
        segments = bundle.segments[first : position + 1]
        persistence = (
            threshold is not None
            and len(recent) >= window
            and len(set(segments)) <= 1
            and bool((recent >= threshold * elevated_ratio).all())
        )
        if single or persistence:
            candidate_positions.append(position)
            if len(candidate_positions) >= limit:
                break

    created = 0
    alerts: list[dict[str, Any]] = []
    for position in candidate_positions:
        result = evaluate_target(store, database, run_id, int(bundle.steps[position]))
        alert = result.get("alert")
        if alert:
            alerts.append(alert)
            if not result.get("deduplicated"):
                created += 1
    return {
        "run_id": run_id,
        "threshold": threshold,
        "scanned_states": bundle.total,
        "candidate_states": [int(bundle.steps[p]) for p in candidate_positions],
        "candidates": len(candidate_positions),
        "created": created,
        "alerts": alerts,
        "persisted": len(database.playback_alerts_for_run(run_id)),
        "limited": len(candidate_positions) >= limit,
        "limitations": LIMITATIONS,
    }


def alert_summary(store: ArtifactStore, database: Database, run_id: str) -> dict[str, Any]:
    rows = database.playback_alerts_for_run(run_id)
    severity_counts = database.playback_alert_severity_counts(run_id)
    status_counts = database.playback_alert_status_counts(run_id)
    unresolved = sum(
        count
        for status, count in status_counts.items()
        if status in ("new", "investigating", "escalated")
    )
    high_priority = severity_counts.get("high", 0) + severity_counts.get("critical", 0)
    calibration = calibration_for(store, run_id)
    return {
        "available": True,
        "reason": None,
        "total": len(rows),
        "high_priority": high_priority,
        "unresolved": unresolved,
        "severity_counts": severity_counts,
        "status_counts": status_counts,
        "recent": [serialize_alert(row) for row in rows[-8:]][::-1],
        "daily": database.playback_alert_daily_counts(run_id),
        "threshold": calibration.threshold,
        "threshold_source": calibration.threshold_source,
        "rules": rules(),
        "limitations": LIMITATIONS,
    }


def get_alert(database: Database, alert_id: str) -> dict[str, Any] | None:
    row = database.playback_alert(alert_id)
    if row is None:
        return None
    return {
        "alert": serialize_alert(row),
        "history": database.playback_alert_history(alert_id),
        "limitations": LIMITATIONS,
    }


def update_alert(
    store: ArtifactStore,
    database: Database,
    alert_id: str,
    status: str | None,
    assignee: str | None,
    notes: str | None,
    actor: str,
) -> dict[str, Any] | None:
    row = database.playback_alert(alert_id)
    if row is None:
        return None
    previous_status = str(row["status"] or "new")
    next_status = status or previous_status
    next_assignee = assignee if assignee is not None else row["assignee"]
    next_notes = notes if notes is not None else row["notes"]
    changes: list[tuple[str, str | None, str | None]] = []
    if status is not None and previous_status != next_status:
        changes.append(("status", previous_status, next_status))
    if assignee is not None and row["assignee"] != next_assignee:
        changes.append(("assignee", row["assignee"], next_assignee))
    if notes is not None and row["notes"] != next_notes:
        changes.append(("notes", row["notes"], next_notes))
    database.update_playback_alert(
        alert_id,
        str(next_status),
        next_assignee,
        next_notes,
        actor,
        changes,
    )
    detail = get_alert(database, alert_id)
    if detail is not None:
        detail["explainability"] = {
            "shap": shap_context(
                store,
                str(row["run_id"]),
                [entry["feature"] for entry in _serialize_features(row["features_json"])],
                row["dataset_label"],
            )
        }
    return detail
