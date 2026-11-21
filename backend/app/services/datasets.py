"""Dataset registry, validation and feature-compatibility checks.

Uploads are written to ``var/uploads`` and never replace the research datasets
under ``Datasets/``. Validation reports exactly what was checked and on how many
rows, so a sampled check cannot be mistaken for a full audit.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactStore

VALIDATION_SAMPLE_ROWS = 50_000
REQUIRED_COLUMNS = ("Timestamp", "Label")
METADATA_COLUMNS = {"Timestamp", "Label", "CaptureDay", "Segment ID"}


def _sanitize(filename: str) -> str:
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned or cleaned.lower().endswith((".py", ".sh", ".exe")):
        cleaned = "dataset.csv"
    if not cleaned.lower().endswith(".csv"):
        cleaned = f"{cleaned}.csv"
    return cleaned[:120]


def save_upload(uploads_dir: Path, filename: str, content: bytes) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = uploads_dir / f"{stamp}__{_sanitize(filename)}"
    destination.write_bytes(content)
    return destination


def analyse_file(path: Path, model_features: list[str] | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    limitations: list[str] = [
        "Validation reads at most 50,000 rows and reports the sampled size; it is not a full audit.",
        "Uploaded datasets stay in var/uploads and never replace Datasets/ artifacts.",
    ]

    def add(check_id: str, label: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "label": label, "status": status, "detail": detail})

    if not path.is_file():
        add("exists", "File present", "failed", str(path))
        return {
            "valid": False,
            "dataset": str(path),
            "checks": checks,
            "row_count": None,
            "column_count": None,
            "missing_features": 0,
            "extra_features": 0,
            "label_counts": {},
            "limitations": limitations,
        }

    add("exists", "File present", "ok", f"{path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    try:
        frame = pd.read_csv(path, nrows=VALIDATION_SAMPLE_ROWS, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        add("read", "File readable as CSV", "failed", str(exc))
        return {
            "valid": False,
            "dataset": str(path),
            "checks": checks,
            "row_count": None,
            "column_count": None,
            "missing_features": 0,
            "extra_features": 0,
            "label_counts": {},
            "limitations": limitations,
        }
    frame.columns = [str(column).strip() for column in frame.columns]
    add("read", "File readable as CSV", "ok", f"Sampled {len(frame):,} rows, {len(frame.columns)} columns")

    unique_columns = bool(frame.columns.is_unique)
    add(
        "unique_columns",
        "Unique column names",
        "ok" if unique_columns else "failed",
        "All column names are unique." if unique_columns else "Duplicate column names found.",
    )

    missing_required = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    add(
        "required_columns",
        "Required columns (Timestamp, Label)",
        "ok" if not missing_required else "failed",
        "Present." if not missing_required else f"Missing: {missing_required}",
    )

    label_counts: dict[str, int] = {}
    if "Label" in frame.columns:
        labels = frame["Label"].astype(str).str.strip()
        label_counts = {str(key): int(value) for key, value in labels.value_counts().items()}
        empty_labels = int((labels == "").sum())
        add(
            "labels",
            "Label values",
            "ok" if empty_labels == 0 else "warning",
            f"{len(label_counts)} distinct labels; {empty_labels} empty values in the sample.",
        )

    timestamps_ok = False
    if "Timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["Timestamp"], errors="coerce")
        invalid = int(parsed.isna().sum())
        timestamps_ok = invalid == 0
        add(
            "timestamps",
            "Timestamp parsing",
            "ok" if invalid == 0 else "warning",
            f"{invalid} unparsable timestamps in the sample." if invalid else "All sampled timestamps parsed.",
        )
        if timestamps_ok and len(parsed) > 1:
            monotonic = bool(parsed.is_monotonic_increasing)
            add(
                "chronology",
                "Chronological ordering",
                "ok" if monotonic else "warning",
                "Sampled rows are in ascending timestamp order."
                if monotonic
                else "Sampled rows are not chronologically ordered; the pipeline sorts, but ordering is expected.",
            )

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    add(
        "numeric",
        "Numeric feature columns",
        "ok" if len(numeric_columns) > 4 else "warning",
        f"{len(numeric_columns)} numeric columns detected.",
    )

    infinite_count = 0
    for column in numeric_columns[:120]:
        values = pd.to_numeric(frame[column], errors="coerce")
        infinite_count += int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
    add(
        "infinite",
        "Infinite values",
        "ok" if infinite_count == 0 else "warning",
        f"{infinite_count} infinite values found; the pipeline maps them to missing values."
        if infinite_count
        else "No infinite values in sampled numeric columns.",
    )

    feature_columns = {column for column in frame.columns if column not in METADATA_COLUMNS}
    missing_features = 0
    extra_features = 0
    if model_features:
        missing = [name for name in model_features if name not in frame.columns]
        extra = sorted(feature_columns - set(model_features))
        missing_features = len(missing)
        extra_features = len(extra)
        add(
            "feature_contract",
            "Compatibility with the 209-feature model contract",
            "ok" if not missing else "failed",
            (
                f"All {len(model_features)} model features present"
                + (f"; {len(extra)} additional columns present." if extra else ".")
                if not missing
                else f"Missing {len(missing)} model features, e.g. {missing[:5]}"
            ),
        )
    else:
        add(
            "feature_contract",
            "Compatibility with the model contract",
            "warning",
            "The frozen feature definition (sentinelx_model_features_v3.json) is not available, "
            "so feature compatibility cannot be verified.",
        )

    # Structural validity is required; the feature contract must also be verified
    # when the frozen feature definition is available. Without it, compatibility
    # stays unconfirmed and the dataset is not reported as usable.
    structural = bool(unique_columns and not missing_required and timestamps_ok)
    contract_ok = (missing_features == 0) if model_features else False
    valid = structural and contract_ok
    return {
        "valid": valid,
        "dataset": str(path),
        "checks": checks,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_features": missing_features,
        "extra_features": extra_features,
        "label_counts": label_counts,
        "limitations": limitations,
    }


def overview(store: ArtifactStore, uploads_dir: Path, run_id: str) -> dict[str, Any]:
    artifacts = store.dataset_artifacts()
    model_features = store.model_feature_names()
    research_files = store.dataset_files()
    uploads = sorted(uploads_dir.glob("*.csv")) if uploads_dir.is_dir() else []

    datasets: list[dict[str, Any]] = []
    for path in research_files:
        datasets.append(
            {
                "id": f"research:{path.name}",
                "name": path.name,
                "kind": "research_artifact",
                "path": str(path),
                "present": True,
                "size_bytes": path.stat().st_size,
                "rows": None,
                "columns": None,
                "label_counts": None,
                "feature_compatible": None,
                "notes": ["Local dataset file. Read-only; the API never rewrites it."],
            }
        )
    for path in uploads:
        datasets.append(
            {
                "id": f"upload:{path.name}",
                "name": path.name,
                "kind": "uploaded",
                "path": str(path),
                "present": True,
                "size_bytes": path.stat().st_size,
                "rows": None,
                "columns": None,
                "label_counts": None,
                "feature_compatible": None,
                "notes": ["Uploaded for validation only; not used by the active model."],
            }
        )

    limitations = [
        "Research datasets (CIC-IDS-2018) are not distributed with the repository; only their "
        "derived artifacts and reports are committed.",
        "Uploading a dataset never replaces training artifacts or activates a model.",
    ]
    if not research_files:
        limitations.insert(
            0,
            "No dataset CSV was found in the dataset directory, so flow-level validation and "
            "protocol/port analysis are unavailable in this deployment.",
        )

    return {
        "source": {
            "mode": "dataset",
            "label": "Dataset directory",
            "detail": str(store.datasets_dir),
        },
        "datasets": datasets,
        "artifacts": [
            {
                "key": artifact.key,
                "label": artifact.label,
                "path": artifact.display_path,
                "present": artifact.present,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in artifacts
        ],
        "label_coverage": store.label_coverage(),
        "active_run_id": run_id,
        "model_features": len(model_features) if model_features else None,
        "limitations": limitations,
    }
