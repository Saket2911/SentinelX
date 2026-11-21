"""Per-state historical replay.

The research pipeline stores the ten-state inputs and the model's next-state
outputs in ``historical_replay_v1/replay_arrays.npz`` and the frozen test
sequences in ``Datasets/sentinelx_test_X_seq10_v3.npy``. This service exposes
them one state at a time so the console can play through held-out network
activity.

Nothing here is simulated: when an array is missing or inconsistent, the service
reports exactly which artifact is required and refuses to serve values.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .anomaly import calibration_for
from .artifacts import ArtifactError, ArtifactStore
from .replay import load_replay

REPLAY_DIR = "historical_replay_v1"
REPLAY_ARRAYS = f"{REPLAY_DIR}/replay_arrays.npz"
SEQUENCE_ARRAYS = "sentinelx_test_X_seq10_v3.npy"
# Name used by earlier console builds; still honoured when present.
SEQUENCE_ALIASES = ("sentinelx_test_sequences_v3.npy",)
MANIFEST = f"{REPLAY_DIR}/replay_manifest.json"

LIMITATIONS = [
    "Playback steps through held-out historical states; it is not live monitoring.",
    "Values are in standardized model feature space (imputation + signed log1p + "
    "StandardScaler fitted on training states), not raw network units.",
    "The dataset label is a reference annotation for retrospective analysis. The LSTM "
    "does not predict labels and the triage rule does not use them.",
    "A large residual means the model found the state hard to predict; it is not an "
    "attack classification.",
]


@dataclass(frozen=True)
class ReplayBundle:
    frame: Any
    steps: np.ndarray
    timestamps: list[str | None]
    labels: list[str]
    segments: list[str]
    capture_days: list[str]
    predictions: np.ndarray
    actual: np.ndarray
    feature_names: list[str]
    sequences: np.ndarray
    manifest: dict[str, Any]
    threshold: float | None
    arrays_path: Path
    sequences_path: Path
    table_path: str

    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def sequence_length(self) -> int:
        return int(self.sequences.shape[1]) if self.sequences.ndim == 3 else 0

    def position_for(self, step: int) -> int:
        matches = np.nonzero(self.steps == step)[0]
        if not len(matches):
            raise KeyError(step)
        return int(matches[0])

    def top_contributors(self, position: int, limit: int = 8) -> list[dict[str, Any]]:
        """Rank residual contributors from either recorded column naming scheme."""
        row = self.frame.iloc[position]
        items: list[dict[str, Any]] = []
        for rank in range(1, limit + 1):
            feature = _value(row, f"Top {rank} Feature", f"Top Contributor {rank}")
            if feature is None:
                continue
            items.append(
                {
                    "rank": rank,
                    "feature": str(feature),
                    "signed_residual": _number(_value(row, f"Top {rank} Signed Residual")),
                    "absolute_error": _number(_value(row, f"Top {rank} Absolute Error")),
                    "contribution_percent": _number(
                        _value(row, f"Top {rank} Contribution Percent")
                    ),
                }
            )
        return items


def _value(row: Any, *names: str) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if value is not None and not (isinstance(value, float) and not np.isfinite(value)):
                return value
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


_cache: dict[str, tuple[tuple[int, int, int, str], ReplayBundle]] = {}
_lock = threading.RLock()


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def _sequences_path(datasets_dir: Path) -> Path:
    """Resolve the frozen test sequences under their real or legacy filename."""
    for name in (SEQUENCE_ARRAYS, *SEQUENCE_ALIASES):
        path = datasets_dir / name
        if path.is_file():
            return path
    return datasets_dir / SEQUENCE_ARRAYS


def _signature(arrays: Path, sequences: Path, table: Path) -> tuple[int, int, int, str]:
    return (
        arrays.stat().st_mtime_ns if arrays.is_file() else 0,
        sequences.stat().st_mtime_ns if sequences.is_file() else 0,
        table.stat().st_mtime_ns if table.is_file() else 0,
        str(sequences),
    )


def _load(store: ArtifactStore, run_id: str) -> tuple[ReplayBundle | None, list[str], str | None]:
    run = store.run(run_id)
    if run is None:
        return None, [], f"Unknown model run {run_id}."

    arrays_path = run.directory / REPLAY_ARRAYS
    sequences_path = _sequences_path(store.datasets_dir)
    frame = load_replay(store, run_id)
    if frame is None:
        return (
            None,
            [
                f"{run_id}/explainable_triage_v2/explainable_triage.csv",
                f"{run_id}/triage_engine_v1/test_triage.csv",
                f"{run_id}/{REPLAY_DIR}/test_replay.csv",
                f"{run_id}/{REPLAY_ARRAYS}",
                f"Datasets/{SEQUENCE_ARRAYS}",
            ],
            "No row-level replay table is present for this run.",
        )

    missing = [
        name
        for name, path in ((f"{run_id}/{REPLAY_ARRAYS}", arrays_path), (f"Datasets/{SEQUENCE_ARRAYS}", sequences_path))
        if not path.is_file()
    ]
    if missing:
        return (
            None,
            missing,
            "Per-state playback needs the replay arrays written by the historical replay "
            "script and the frozen test sequences.",
        )

    signature = _signature(arrays_path, sequences_path, Path(frame.path))
    with _lock:
        cached = _cache.get(run_id)
        if cached and cached[0] == signature:
            return cached[1], [], None

    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            predictions = np.asarray(archive["predictions"], dtype=np.float64)
            actual = np.asarray(archive["actual"], dtype=np.float64)
            feature_names = [str(name) for name in archive["feature_names"]]
    except (OSError, KeyError, ValueError) as exc:
        raise ArtifactError(f"Replay arrays cannot be read: {exc}") from exc

    sequences = np.load(sequences_path, mmap_mode="r")
    total = len(frame.frame)

    if predictions.shape != actual.shape:
        raise ArtifactError("Replay prediction and actual arrays differ in shape.")
    if predictions.ndim != 2:
        raise ArtifactError("Replay arrays must be two-dimensional (states x features).")
    if predictions.shape[0] != total:
        raise ArtifactError(
            f"Replay arrays hold {predictions.shape[0]} states but the table holds {total}."
        )
    if predictions.shape[1] != len(feature_names):
        raise ArtifactError("Feature names do not match the replay array width.")
    if sequences.ndim != 3 or sequences.shape[0] != total or sequences.shape[2] != len(feature_names):
        raise ArtifactError(
            "Test sequences do not align with the replay arrays "
            f"(sequences {sequences.shape}, replay {predictions.shape})."
        )
    if not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ArtifactError("Replay arrays contain non-finite values.")

    steps = frame.steps().to_numpy(dtype=np.int64)
    day_column = next(
        (
            column
            for column in frame.frame.columns
            if str(column).strip().lower() in ("target captureday", "capture day", "captureday")
        ),
        None,
    )
    parsed = frame.timestamps()
    timestamps: list[str | None] = []
    for value in parsed:
        timestamps.append(value.isoformat() if hasattr(value, "isoformat") else (value or None))
    labels = [str(value) for value in frame.labels().tolist()]
    segments = [str(value) for value in frame.segments().tolist()]
    capture_days = [str(value) for value in frame.frame[day_column].tolist()] if day_column else []

    manifest: dict[str, Any] = {}
    manifest_path = run.directory / MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}

    calibration = calibration_for(store, run_id)
    bundle = ReplayBundle(
        frame=frame.frame,
        steps=steps,
        timestamps=timestamps,
        labels=labels,
        segments=segments,
        capture_days=capture_days,
        predictions=predictions,
        actual=actual,
        feature_names=feature_names,
        sequences=sequences,
        manifest=manifest,
        threshold=calibration.threshold,
        arrays_path=arrays_path,
        sequences_path=sequences_path,
        table_path=str(frame.path),
    )
    with _lock:
        _cache[run_id] = (signature, bundle)
    return bundle, [], None


def _unavailable(run_id: str, missing: list[str], reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "source": {
            "mode": "unavailable",
            "label": "Historical replay arrays",
            "detail": reason,
        },
        "availability": {
            "available": False,
            "reason": reason,
            "required_artifacts": missing,
        },
        "run_id": run_id,
        "limitations": LIMITATIONS,
        **extra,
    }


def _segment_ranges(bundle: ReplayBundle) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for segment in dict.fromkeys(bundle.segments):
        positions = [index for index, value in enumerate(bundle.segments) if value == segment]
        if not positions:
            continue
        counts: dict[str, int] = {}
        for position in positions:
            counts[bundle.labels[position]] = counts.get(bundle.labels[position], 0) + 1
        ranges.append(
            {
                "segment_id": segment,
                "first_step": int(bundle.steps[positions[0]]),
                "last_step": int(bundle.steps[positions[-1]]),
                "states": len(positions),
                "labels": counts,
            }
        )
    return ranges


def meta(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    bundle, missing, reason = _load(store, run_id)
    if bundle is None:
        return _unavailable(run_id, missing, reason or "Replay arrays unavailable.", total_states=0, features=[])

    return {
        "source": {
            "mode": "dataset",
            "label": "Historical replay arrays",
            "detail": f"{bundle.arrays_path} · {bundle.sequences_path}",
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "run_id": run_id,
        "total_states": bundle.total,
        "step_min": int(bundle.steps.min()) if bundle.total else 0,
        "step_max": int(bundle.steps.max()) if bundle.total else 0,
        "feature_count": len(bundle.feature_names),
        "features": bundle.feature_names,
        "sequence_length": bundle.sequence_length,
        "threshold": bundle.threshold,
        "segments": _segment_ranges(bundle),
        "capture_days": sorted({day for day in bundle.capture_days if day}),
        "manifest": bundle.manifest,
        "table_path": bundle.table_path,
        "limitations": LIMITATIONS,
    }


def _feature_payload(bundle: ReplayBundle, position: int, name: str) -> dict[str, Any] | None:
    if name not in bundle.feature_names:
        return None
    column = bundle.feature_names.index(name)
    actual = float(bundle.actual[position, column])
    predicted = float(bundle.predictions[position, column])
    history = np.asarray(bundle.sequences[position, :, column], dtype=np.float64)
    return {
        "feature": name,
        "position": column,
        "actual": actual,
        "predicted": predicted,
        "residual": actual - predicted,
        "absolute_error": abs(actual - predicted),
        "history": [float(value) for value in history],
    }


def state(
    store: ArtifactStore,
    run_id: str,
    step: int,
    features: list[str] | None = None,
    contributor_limit: int = 8,
    include_all: bool = False,
) -> dict[str, Any]:
    bundle, missing, reason = _load(store, run_id)
    if bundle is None:
        return _unavailable(run_id, missing, reason or "Replay arrays unavailable.", state=None, features=[])

    position = bundle.position_for(int(step))
    contributors = bundle.top_contributors(position, contributor_limit)

    requested: list[str] = []
    for name in features or []:
        if name and name not in requested:
            requested.append(name)
    if not requested:
        requested = [item["feature"] for item in contributors[:6] if item["feature"] in bundle.feature_names]
        if "Flow Count" in bundle.feature_names:
            requested.insert(0, "Flow Count")
        if not requested:
            requested = bundle.feature_names[:6]
    requested = [name for name in requested if name in bundle.feature_names][:24]

    payload = [entry for entry in (_feature_payload(bundle, position, name) for name in requested) if entry]
    residual = _number(_value(bundle.frame.iloc[position], "Prediction MAE", "Mean Absolute Error"))
    calibration = calibration_for(store, run_id)

    return {
        "source": {
            "mode": "dataset",
            "label": "Historical replay arrays",
            "detail": bundle.arrays_path.as_posix(),
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "run_id": run_id,
        "state": {
            "step": int(step),
            "index": position,
            "total_states": bundle.total,
            "timestamp": bundle.timestamps[position],
            "capture_day": bundle.capture_days[position] if bundle.capture_days else None,
            "segment_id": bundle.segments[position],
            "dataset_label": bundle.labels[position],
            "residual": residual,
            "priority": calibration.priority_for(residual),
            "threshold": bundle.threshold,
            "sequence_length": bundle.sequence_length,
            "feature_count": len(bundle.feature_names),
        },
        "features": payload,
        "available_features": bundle.feature_names,
        "contributors": contributors,
        "feature_table": _feature_table(bundle, position) if include_all else [],
        "limitations": LIMITATIONS,
    }


def _feature_table(bundle: ReplayBundle, position: int) -> list[dict[str, Any]]:
    actual = bundle.actual[position]
    predicted = bundle.predictions[position]
    residual = actual - predicted
    order = np.argsort(-np.abs(residual), kind="stable")
    return [
        {
            "feature": bundle.feature_names[index],
            "actual": float(actual[index]),
            "predicted": float(predicted[index]),
            "residual": float(residual[index]),
            "absolute_error": float(abs(residual[index])),
        }
        for index in order
    ]


def contributors(store: ArtifactStore, run_id: str, limit: int = 15) -> dict[str, Any]:
    """Features most often among the top residual contributors across the replay."""
    bundle, missing, reason = _load(store, run_id)
    if bundle is None:
        return _unavailable(run_id, missing, reason or "Replay arrays unavailable.", contributors=[])

    table = store.read_table(run_id, "explainable_contributors")
    if table is not None and {"Feature", "Appearance Percent"} <= set(table.columns):
        ordered = table.sort_values("Appearance Percent", ascending=False).head(limit)
        return {
            "run_id": run_id,
            "source": "recorded contributor report",
            "total_states": bundle.total,
            "contributors": [
                {
                    "feature": str(row["Feature"]),
                    "appearances": int(row.get("Top 5 Appearance Count", 0) or 0),
                    "appearance_percent": float(row["Appearance Percent"]),
                }
                for _, row in ordered.iterrows()
            ],
        }

    counts: dict[str, int] = {}
    for position in range(bundle.total):
        for item in bundle.top_contributors(position, 5):
            counts[item["feature"]] = counts.get(item["feature"], 0) + 1
    total = bundle.total or 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {
        "run_id": run_id,
        "source": "computed from replay contributor columns",
        "total_states": bundle.total,
        "contributors": [
            {
                "feature": feature,
                "appearances": count,
                "appearance_percent": 100.0 * count / total,
            }
            for feature, count in ranked
        ],
    }


def series(
    store: ArtifactStore,
    run_id: str,
    feature: str | None,
    start: int = 0,
    limit: int = 160,
) -> dict[str, Any]:
    bundle, missing, reason = _load(store, run_id)
    if bundle is None:
        return _unavailable(
            run_id, missing, reason or "Replay arrays unavailable.", feature=feature, points=[], stats={}
        )

    name = feature if feature in bundle.feature_names else bundle.feature_names[0]
    column = bundle.feature_names.index(name)
    start = max(0, int(start))
    limit = max(1, min(int(limit), 2000))
    stop = min(bundle.total, start + limit)

    actual = bundle.actual[:, column]
    predicted = bundle.predictions[:, column]
    absolute = np.abs(actual - predicted)
    signed = actual - predicted
    state_mae = np.abs(bundle.actual - bundle.predictions).mean(axis=1)

    points = [
        {
            "index": position,
            "step": int(bundle.steps[position]),
            "timestamp": bundle.timestamps[position],
            "label": bundle.labels[position],
            "actual": float(actual[position]),
            "predicted": float(predicted[position]),
            "residual": float(signed[position]),
            "absolute_error": float(absolute[position]),
            "state_mae": float(state_mae[position]),
        }
        for position in range(start, stop)
    ]

    return {
        "source": {
            "mode": "dataset",
            "label": "Historical replay arrays",
            "detail": bundle.arrays_path.as_posix(),
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "run_id": run_id,
        "feature": name,
        "features": bundle.feature_names,
        "points": points,
        "stats": {
            "samples": int(bundle.total),
            "mae": float(absolute.mean()),
            "median_absolute_error": float(np.median(absolute)),
            "max_absolute_error": float(absolute.max()),
            "mean_signed_residual": float(signed.mean()),
        },
        "top_errors": [
            {
                "index": int(position),
                "step": int(bundle.steps[position]),
                "timestamp": bundle.timestamps[position],
                "label": bundle.labels[position],
                "actual": float(actual[position]),
                "predicted": float(predicted[position]),
                "residual": float(signed[position]),
                "absolute_error": float(absolute[position]),
            }
            for position in np.argsort(-absolute, kind="stable")[:10]
        ],
        "window": {"start": start, "end": stop, "total_states": bundle.total},
        "state_mae": float(state_mae.mean()),
        "limitations": LIMITATIONS,
    }
