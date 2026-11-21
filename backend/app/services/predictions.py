"""Prediction and forecast evaluation services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anomaly import calibration_for
from .artifacts import ArtifactStore
from .replay import load_replay, replay_source_key
from .research_text import parse_multi_step_report

LIMITATIONS = [
    "Values are in standardized preprocessed model feature space (median imputation + signed "
    "log1p + StandardScaler fitted on training states), not raw network units.",
    "The persistence baseline repeats the last observed state; a model is only useful where it "
    "beats that baseline.",
    "Test-partition metrics were recorded after exploratory analysis of that partition and are "
    "not an untouched independent estimate.",
    "Prediction error ranks observations for review. It does not confirm malicious activity.",
]


def _recorded_quantiles(store: ArtifactStore) -> dict[str, float]:
    path = store.reports_dir / "v3_reliability_audit_v1" / "reliability_summary.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    sample_mae = ((data.get("overall_error_quantiles") or {}).get("sample_mae") or {})
    return {
        str(key): float(value)
        for key, value in sample_mae.items()
        if isinstance(value, (int, float))
    }


def _multi_step(store: ArtifactStore, run_id: str) -> list[dict[str, Any]]:
    text = store.read_text(run_id, "multi_step_report")
    return parse_multi_step_report(text) if text else []


def summary(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    evaluation = store.read_json(run_id, "test_evaluation")
    calibration = calibration_for(store, run_id)
    replay = load_replay(store, run_id)
    multi_step = _multi_step(store, run_id)

    missing = run.required(["config"], ["replay"])
    availability = {
        "available": bool(evaluation or multi_step or replay),
        "reason": None if (evaluation or multi_step or replay) else "No evaluation artifacts recorded for this run.",
        "required_artifacts": missing,
    }
    source = {
        "mode": "recorded_artifact" if availability["available"] else "unavailable",
        "label": "Recorded evaluation artifacts",
        "detail": str(run.directory),
    }

    metrics: list[dict[str, Any]] = []
    if evaluation:
        lstm = evaluation.get("lstm_metrics") or {}
        persistence = evaluation.get("persistence_baseline_metrics") or {}
        improvement = evaluation.get("improvement_over_baseline_percent") or {}
        metrics.extend(
            [
                {
                    "id": "mae",
                    "label": "LSTM MAE",
                    "value": lstm.get("mae"),
                    "display": f"{lstm.get('mae'):.4f}" if isinstance(lstm.get("mae"), (int, float)) else "n/a",
                    "unit": "model space",
                    "status": "neutral",
                    "detail": f"Persistence baseline {persistence.get('mae'):.4f}"
                    if isinstance(persistence.get("mae"), (int, float))
                    else None,
                },
                {
                    "id": "rmse",
                    "label": "LSTM RMSE",
                    "value": lstm.get("rmse"),
                    "display": f"{lstm.get('rmse'):.4f}" if isinstance(lstm.get("rmse"), (int, float)) else "n/a",
                    "unit": "model space",
                    "status": "neutral",
                    "detail": f"Persistence baseline {persistence.get('rmse'):.4f}"
                    if isinstance(persistence.get("rmse"), (int, float))
                    else None,
                },
                {
                    "id": "huber",
                    "label": "LSTM Huber",
                    "value": lstm.get("huber_loss"),
                    "display": f"{lstm.get('huber_loss'):.4f}" if isinstance(lstm.get("huber_loss"), (int, float)) else "n/a",
                    "unit": "model space",
                    "status": "neutral",
                    "detail": f"Persistence baseline {persistence.get('huber_loss'):.4f}"
                    if isinstance(persistence.get("huber_loss"), (int, float))
                    else None,
                },
                {
                    "id": "improvement",
                    "label": "MAE improvement vs persistence",
                    "value": improvement.get("mae"),
                    "display": f"{improvement.get('mae'):.2f}%" if isinstance(improvement.get("mae"), (int, float)) else "n/a",
                    "unit": "%",
                    "status": "good" if isinstance(improvement.get("mae"), (int, float)) and improvement["mae"] > 0 else "warning",
                    "detail": "Positive values mean the LSTM predicted better than repeating the last state.",
                },
                {
                    "id": "samples",
                    "label": "Test sequences",
                    "value": evaluation.get("test_samples"),
                    "display": f"{evaluation.get('test_samples'):,}" if isinstance(evaluation.get("test_samples"), int) else "n/a",
                    "unit": "sequences",
                    "status": "neutral",
                    "detail": "Held-out chronological test partition.",
                },
            ]
        )

    per_label: list[dict[str, Any]] = []
    for row in multi_step:
        if row.get("horizon_seconds") == 1:
            per_label.append(
                {
                    "label": row.get("label"),
                    "samples": row.get("valid_origins"),
                    "lstm_mae": row.get("lstm_mae"),
                    "persistence_mae": row.get("persistence_mae"),
                    "mae_improvement_percent": row.get("mae_improvement_percent"),
                    "lstm_rmse": row.get("lstm_rmse"),
                    "persistence_rmse": row.get("persistence_rmse"),
                }
            )

    recorded = _recorded_quantiles(store)
    error_distribution: list[dict[str, Any]] = []
    if replay is not None:
        residuals = replay.residuals()
        if residuals is not None and residuals.notna().any():
            values = residuals.dropna().to_numpy(dtype=float)
            counts, edges = np.histogram(values, bins=24)
            error_distribution = [
                {
                    "bucket_start": float(edges[index]),
                    "bucket_end": float(edges[index + 1]),
                    "count": int(counts[index]),
                }
                for index in range(len(counts))
            ]
    for key, value in recorded.items():
        error_distribution.append(
            {"quantile": key, "value": value, "kind": "recorded_quantile"}
        )

    if not error_distribution and recorded:
        error_distribution = [
            {"quantile": key, "value": value, "kind": "recorded_quantile"}
            for key, value in recorded.items()
        ]

    return {
        "source": source,
        "availability": availability,
        "feature": None,
        "features": replay.columns.features if replay else [],
        "points": [],
        "metrics": metrics,
        "summary": {
            "metrics": metrics,
            "per_label": per_label,
            "multi_step": multi_step,
            "samples": evaluation.get("test_samples") if evaluation else None,
            "space": "Preprocessed model feature space",
            "test_partition_note": "The test partition has been used for exploratory analysis and is not untouched.",
        },
        "calibration": calibration.as_dict(),
        "horizon": {
            "one_step": True,
            "multi_step_horizons": sorted({row.get("horizon_seconds") for row in multi_step if row.get("horizon_seconds")}),
            "method": "Rolling one-step-ahead for the replay; recursive multi-step forecasting evaluated separately.",
        },
        "error_distribution": error_distribution,
        "per_label": per_label,
        "multi_step": multi_step,
        "limitations": LIMITATIONS,
    }


def series(
    store: ArtifactStore,
    run_id: str,
    feature: str | None = None,
    start: int = 0,
    limit: int = 240,
) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    replay = load_replay(store, run_id)
    calibration = calibration_for(store, run_id)
    source_key = replay_source_key(store, run_id)
    required = run.required([], ["replay"])
    if replay is None:
        return {
            "source": {
                "mode": "unavailable",
                "label": "Row-level replay",
                "detail": "No replay, triage or explainable-triage table is present.",
            },
            "availability": {
                "available": False,
                "reason": "Row-level predictions are unavailable. Generate the replay table with the research scripts.",
                "required_artifacts": required,
            },
            "feature": feature,
            "features": [],
            "points": [],
            "summary": {},
            "calibration": calibration.as_dict(),
            "horizon": {},
            "limitations": LIMITATIONS,
        }

    start = max(0, int(start))
    limit = max(1, min(int(limit), 2000))
    feature_name = feature if feature in replay.columns.pairs else None
    steps = replay.steps()
    labels = replay.labels()
    timestamps = replay.timestamps()
    residuals = replay.residuals()

    stop = min(len(replay.frame), start + limit)
    points: list[dict[str, Any]] = []
    for position in range(start, stop):
        actual = predicted = None
        if feature_name:
            actual_column, predicted_column = replay.columns.pairs[feature_name]
            actual_value = replay.frame[actual_column].iloc[position]
            predicted_value = replay.frame[predicted_column].iloc[position]
            actual = float(actual_value) if np.isfinite(actual_value) else None
            predicted = float(predicted_value) if np.isfinite(predicted_value) else None
        residual = None
        if residuals is not None:
            value = residuals.iloc[position]
            residual = float(value) if np.isfinite(value) else None
        timestamp = timestamps.iloc[position]
        points.append(
            {
                "index": start + len(points),
                "step": int(steps.iloc[position]) if np.isfinite(steps.iloc[position]) else None,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else (timestamp or None),
                "actual": actual,
                "predicted": predicted,
                "residual": residual,
                "label": str(labels.iloc[position]),
                "priority": calibration.priority_for(residual),
            }
        )

    all_residuals = residuals.dropna() if residuals is not None else None
    window_summary = {
        "window_start": start,
        "window_end": stop,
        "points": len(points),
        "total_points": len(replay.frame),
        "mean_residual": float(all_residuals.mean()) if all_residuals is not None and len(all_residuals) else None,
        "max_residual": float(all_residuals.max()) if all_residuals is not None and len(all_residuals) else None,
        "feature": feature_name,
        "source_table": source_key,
        "space": "Preprocessed model feature space",
    }
    if feature_name is None and feature:
        window_summary["feature_note"] = (
            f"Feature {feature!r} was not found as an actual/predicted column pair in the "
            f"{source_key} table. Columns are resolved defensively; available pairs: "
            f"{len(replay.columns.pairs)}."
        )
    return {
        "source": {
            "mode": "dataset",
            "label": "Row-level replay",
            "detail": replay.path,
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "feature": feature_name,
        "features": replay.columns.features,
        "points": points,
        "summary": window_summary,
        "calibration": calibration.as_dict(),
        "horizon": {
            "one_step": True,
            "prediction_horizon": "Next one-second network state",
            "predictions_fed_back": False,
        },
        "limitations": LIMITATIONS,
    }


def top_residuals(store: ArtifactStore, run_id: str, limit: int = 25) -> list[dict[str, Any]]:
    replay = load_replay(store, run_id)
    if replay is None:
        return []
    residuals = replay.residuals()
    if residuals is None:
        return []
    calibration = calibration_for(store, run_id)
    steps = replay.steps()
    labels = replay.labels()
    timestamps = replay.timestamps()
    ranked = residuals.dropna().sort_values(ascending=False).head(max(1, min(limit, 200)))
    items = []
    for position, value in ranked.items():
        timestamp = timestamps.loc[position]
        items.append(
            {
                "step": int(steps.loc[position]) if np.isfinite(steps.loc[position]) else None,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else (timestamp or None),
                "label": str(labels.loc[position]),
                "residual": float(value),
                "priority": calibration.priority_for(float(value)),
            }
        )
    return items
