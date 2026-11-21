"""Network monitoring service.

Two data planes exist and are reported separately:

* **state plane** – aggregated 1-second network states (always available once a
  replay/triage table exists): observation counts, label mix, residual activity.
* **flow plane** – raw CIC-IDS-2018 flow CSVs (only when ``Datasets/`` is present):
  protocol, destination-port and volume distributions.

Live capture is not implemented; the API never claims otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import calibration_for
from .artifacts import ArtifactStore
from .replay import load_replay

FLOW_COLUMN_CANDIDATES = {
    "protocol": ("protocol", "proto"),
    "dst_port": ("dst port", "destination port", "dst_port", "id.resp_p"),
    "src_port": ("src port", "source port", "src_port", "id.orig_p"),
    "flow_duration": ("flow duration", "duration"),
    "fwd_bytes": ("totlen fwd pkts", "fwd bytes", "orig_bytes"),
    "bwd_bytes": ("totlen bwd pkts", "bwd bytes", "resp_bytes"),
    "total_packets": ("tot fwd pkts", "tot bwd pkts", "packets"),
    "label": ("label",),
}

PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP", "1": "ICMP", "6": "TCP", "17": "UDP"}


def _find(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _flow_source(store: ArtifactStore) -> Path | None:
    files = store.dataset_files()
    for path in files:
        name = path.name.lower()
        if "state" in name or "sequence" in name or "metadata" in name:
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        columns = [str(column).lower().strip() for column in header.columns]
        if "label" in columns and any(candidate in columns for candidate in ("protocol", "dst port", "destination port")):
            return path
    return None


def _flow_plane(store: ArtifactStore) -> dict[str, Any]:
    path = _flow_source(store)
    if path is None:
        return {
            "available": False,
            "reason": (
                "No raw flow capture with protocol/port columns was found under the dataset directory."
            ),
            "path": str(store.datasets_dir),
            "protocol": [],
            "ports": [],
            "totals": {},
        }
    try:
        frame = pd.read_csv(path, nrows=200_000, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            "available": False,
            "reason": f"Unable to read flow capture: {exc}",
            "path": str(path),
            "protocol": [],
            "ports": [],
            "totals": {},
        }
    frame.columns = [str(column).strip() for column in frame.columns]
    columns = frame.columns.tolist()
    protocol_column = _find(columns, FLOW_COLUMN_CANDIDATES["protocol"])
    port_column = _find(columns, FLOW_COLUMN_CANDIDATES["dst_port"])
    label_column = _find(columns, FLOW_COLUMN_CANDIDATES["label"])

    protocol: list[dict[str, Any]] = []
    if protocol_column:
        values = (
            pd.to_numeric(frame[protocol_column], errors="coerce")
            .map(PROTOCOL_NAMES)
            .fillna(frame[protocol_column].astype(str))
        )
        counts = values.value_counts(dropna=False).head(12)
        total = int(counts.sum()) or 1
        protocol = [
            {"name": str(name), "count": int(count), "share": round(int(count) / total, 4)}
            for name, count in counts.items()
        ]

    ports: list[dict[str, Any]] = []
    if port_column:
        values = pd.to_numeric(frame[port_column], errors="coerce").dropna().astype(int)
        counts = values.value_counts().head(12)
        total = int(counts.sum()) or 1
        ports = [
            {"name": str(int(port)), "count": int(count), "share": round(int(count) / total, 4)}
            for port, count in counts.items()
        ]

    totals: dict[str, Any] = {"sampled_rows": int(len(frame)), "source": str(path)}
    if label_column:
        labels = frame[label_column].astype(str).str.strip()
        attack = ~labels.str.lower().isin(["benign", "normal", "no flows"])
        totals["attack_rows"] = int(attack.sum())
        totals["benign_rows"] = int((~attack).sum())
    return {
        "available": bool(protocol or ports),
        "reason": None if (protocol or ports) else "Flow capture has no usable protocol/port columns.",
        "path": str(path),
        "protocol": protocol,
        "ports": ports,
        "totals": totals,
    }


def overview(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    replay = load_replay(store, run_id)
    calibration = calibration_for(store, run_id)
    flow = _flow_plane(store)
    run_required = run.required([], ["replay"])

    if replay is None:
        return {
            "source": {
                "mode": "unavailable",
                "label": "Network states",
                "detail": "No aggregated state table is present for this run.",
            },
            "availability": {
                "available": False,
                "reason": "Network monitoring needs an aggregated state/replay table.",
                "required_artifacts": run_required,
            },
            "totals": {},
            "label_distribution": [],
            "protocol_distribution": flow["protocol"],
            "top_ports": flow["ports"],
            "activity_series": [],
            "segments": [],
            "flow_features": [],
            "limitations": [
                "Network monitoring needs an aggregated state or replay table for this run.",
                "No counts are shown because no telemetry artifact is present; this is not zero traffic.",
            ],
        }

    labels = replay.labels()
    residuals = replay.residuals()
    timestamps = replay.timestamps()
    segments = replay.segments()
    steps = replay.steps()

    label_counts = labels.value_counts()
    total = int(label_counts.sum())
    benign_like = {"benign", "no flows", "normal"}
    benign = int(
        sum(count for name, count in label_counts.items() if str(name).lower() in benign_like)
    )
    label_distribution = [
        {"name": str(name), "count": int(count), "share": round(int(count) / max(total, 1), 4)}
        for name, count in label_counts.items()
    ]

    parsed_timestamps = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    capture_days = sorted(
        {value.strftime("%Y-%m-%d") for value in parsed_timestamps.dropna()}
    )

    review_candidates = 0
    mean_residual = None
    if residuals is not None and residuals.notna().any():
        mean_residual = float(residuals.mean())
        if calibration.threshold is not None:
            review_candidates = int((residuals >= calibration.threshold).sum())

    series: list[dict[str, Any]] = []
    if residuals is not None and residuals.notna().any():
        frame = pd.DataFrame(
            {
                "step": steps.to_numpy(),
                "timestamp": parsed_timestamps,
                "residual": residuals.to_numpy(),
                "label": labels.to_numpy(),
            }
        )
        window = max(1, len(frame) // 180)
        frame["bucket"] = np.arange(len(frame)) // window
        grouped = frame.groupby("bucket", sort=True)
        rolling = grouped["residual"].mean()
        counts = grouped.size()
        first_step = grouped["step"].min()
        max_residual = grouped["residual"].max()
        timestamps_bucket = grouped["timestamp"].first()
        attack_flags = grouped["label"].apply(
            lambda values: int(
                sum(1 for value in values if str(value).lower() not in benign_like)
            )
        )
        for bucket in rolling.index:
            timestamp_value = timestamps_bucket.loc[bucket]
            series.append(
                {
                    "index": int(bucket),
                    "label": None,
                    "timestamp": timestamp_value.isoformat() if hasattr(timestamp_value, "isoformat") else None,
                    "value": float(rolling.loc[bucket]),
                    "secondary": float(attack_flags.loc[bucket]) / max(int(counts.loc[bucket]), 1),
                }
            )

    segment_rows: list[dict[str, Any]] = []
    for name, group in replay.frame.groupby(segments, sort=False):
        positions = group.index
        segment_labels = labels.loc[positions].value_counts().to_dict()
        segment_residual = (
            float(residuals.loc[positions].mean())
            if residuals is not None and len(positions)
            else None
        )
        segment_rows.append(
            {
                "segment_id": str(name),
                "observations": int(len(positions)),
                "labels": {str(key): int(value) for key, value in segment_labels.items()},
                "mean_residual": segment_residual,
                "first_step": int(steps.loc[positions].min()) if len(positions) else None,
                "last_step": int(steps.loc[positions].max()) if len(positions) else None,
            }
        )

    flow_features: list[dict[str, Any]] = []
    if replay.columns.pairs:
        for feature, (actual_column, predicted_column) in list(replay.columns.pairs.items())[:15]:
            actual = pd.to_numeric(replay.frame[actual_column], errors="coerce")
            predicted = pd.to_numeric(replay.frame[predicted_column], errors="coerce")
            flow_features.append(
                {
                    "feature": feature,
                    "mean_actual": float(actual.mean()) if actual.notna().any() else None,
                    "mean_predicted": float(predicted.mean()) if predicted.notna().any() else None,
                    "mean_abs_error": float((actual - predicted).abs().mean())
                    if actual.notna().any()
                    else None,
                }
            )

    limitations = [
        "Counts describe aggregated one-second network states, not individual packets.",
        "Dataset labels are reference annotations from the capture, not model predictions.",
        "Residual activity is a review signal; it is not an attack indicator.",
    ]
    if not flow["available"]:
        limitations.append(
            "Protocol and port analysis requires the raw CIC-IDS-2018 flow CSV in the dataset directory; "
            "aggregated states do not carry protocol or port fields."
        )

    return {
        "source": {
            "mode": "dataset",
            "label": "Aggregated network states",
            "detail": replay.path,
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "totals": {
            "observations": total,
            "benign_observations": benign,
            "attack_labelled_observations": total - benign,
            "review_candidates": review_candidates,
            "segments": int(segments.nunique()),
            "capture_days": capture_days,
            "mean_residual": mean_residual,
            "frozen_threshold": calibration.threshold,
            "flow_plane": flow["totals"],
        },
        "label_distribution": label_distribution,
        "protocol_distribution": flow["protocol"],
        "top_ports": flow["ports"],
        "activity_series": series,
        "segments": segment_rows,
        "flow_features": flow_features,
        "limitations": limitations,
    }
