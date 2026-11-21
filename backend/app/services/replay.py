"""Shared row-level replay access.

The research pipeline writes a replay table whose exact column names depend on
the script that produced it. Rather than hard-coding one schema, resolve columns
defensively and report which fields were recognised. Nothing is invented: a
missing column becomes ``None`` and the caller can degrade explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pandas as pd

from .artifacts import ArtifactStore


STEP_CANDIDATES = ("replay step", "step", "index", "observation")
TIMESTAMP_CANDIDATES = (
    "target timestamp",
    "timestamp",
    "predicted timestamp",
    "target_timestamp",
)
LABEL_CANDIDATES = ("target label", "label", "dataset label")
SEGMENT_CANDIDATES = ("segment id", "segment", "segment_id")
PRIORITY_CANDIDATES = ("triage status", "status", "review status", "priority")
ERROR_CANDIDATES = (
    "prediction mae",
    "mean absolute error",
    "mae",
    "residual score",
    "residual",
    "anomaly score",
    "mean_absolute_residual",
)
ACTUAL_PATTERN = re.compile(r"^(?:actual|target)[ _:]+(.+)$", re.IGNORECASE)
PREDICTED_PATTERN = re.compile(r"^(?:predicted|prediction)[ _:]+(.+)$", re.IGNORECASE)


def _find(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        lowered_column = column.lower()
        for candidate in candidates:
            if candidate in lowered_column:
                return column
    return None


@dataclass(frozen=True)
class ReplayColumnMap:
    step: str | None
    timestamp: str | None
    label: str | None
    segment: str | None
    priority: str | None
    error: str | None
    pairs: dict[str, tuple[str, str]]

    @property
    def features(self) -> list[str]:
        return sorted(self.pairs)


def resolve_columns(frame: pd.DataFrame) -> ReplayColumnMap:
    columns = [str(column) for column in frame.columns]
    actual = {}
    predicted = {}
    for column in columns:
        actual_match = ACTUAL_PATTERN.match(column)
        if actual_match:
            actual[actual_match.group(1).strip()] = column
            continue
        predicted_match = PREDICTED_PATTERN.match(column)
        if predicted_match:
            predicted[predicted_match.group(1).strip()] = column
    pairs = {
        feature: (actual[feature], predicted[feature])
        for feature in actual.keys() & predicted.keys()
    }
    return ReplayColumnMap(
        step=_find(columns, STEP_CANDIDATES),
        timestamp=_find(columns, TIMESTAMP_CANDIDATES),
        label=_find(columns, LABEL_CANDIDATES),
        segment=_find(columns, SEGMENT_CANDIDATES),
        priority=_find(columns, PRIORITY_CANDIDATES),
        error=_find(columns, ERROR_CANDIDATES) if not pairs else _find(columns, ERROR_CANDIDATES),
        pairs=pairs,
    )


@dataclass(frozen=True)
class ReplayFrame:
    frame: pd.DataFrame
    columns: ReplayColumnMap
    path: str

    @property
    def rows(self) -> int:
        return len(self.frame)

    def steps(self) -> pd.Series:
        if self.columns.step:
            values = pd.to_numeric(self.frame[self.columns.step], errors="coerce")
            if values.notna().all():
                return values.astype("int64")
        return pd.Series(range(1, len(self.frame) + 1), index=self.frame.index)

    def residuals(self) -> pd.Series | None:
        if self.columns.error:
            values = pd.to_numeric(self.frame[self.columns.error], errors="coerce")
            if values.notna().any():
                return values.astype(float)
        if self.columns.pairs:
            residuals = pd.Series(0.0, index=self.frame.index)
            counted = 0
            for actual_column, predicted_column in self.columns.pairs.values():
                actual = pd.to_numeric(self.frame[actual_column], errors="coerce")
                predicted = pd.to_numeric(self.frame[predicted_column], errors="coerce")
                residuals = residuals.add((actual - predicted).abs().fillna(0.0))
                counted += 1
            if counted:
                return residuals / counted
        return None

    def labels(self) -> pd.Series:
        if self.columns.label:
            values = (
                self.frame[self.columns.label]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
            )
            return values.mask(values.str.lower().isin({"", "nan", "none"}), "UNKNOWN")
        return pd.Series(["UNKNOWN"] * len(self.frame), index=self.frame.index)

    def timestamps(self) -> pd.Series:
        if not self.columns.timestamp:
            return pd.Series([None] * len(self.frame), index=self.frame.index)
        parsed = pd.to_datetime(self.frame[self.columns.timestamp], errors="coerce")
        return parsed.astype(object).where(parsed.notna(), None)

    def segments(self) -> pd.Series:
        if self.columns.segment:
            return self.frame[self.columns.segment].astype(str)
        return pd.Series(["unknown"] * len(self.frame), index=self.frame.index)


JOIN_KEYS = ("target state row", "replay step", "step", "observation")


def _join_key(columns) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in JOIN_KEYS:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _attach_labels(
    store: ArtifactStore,
    run_id: str,
    frame: pd.DataFrame,
    columns: ReplayColumnMap,
) -> tuple[pd.DataFrame, ReplayColumnMap]:
    """Supply labels when the preferred table does not carry them.

    The explainable-triage artifact is preferred for its residual and
    contributor columns, but some research pipeline versions wrote it without
    the dataset label. The replay artifact covers the same held-out rows and
    does carry labels, so join them by state row (or position) instead of
    reporting every row as unknown.
    """
    if columns.label:
        return frame, columns
    labelled = store.read_table(run_id, "replay")
    if labelled is None or labelled.empty:
        return frame, columns
    labelled_columns = resolve_columns(labelled)
    if not labelled_columns.label:
        return frame, columns

    values = (
        labelled[labelled_columns.label]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )
    joined: pd.Series | None = None
    left_key = _join_key(frame.columns)
    right_key = _join_key(labelled.columns)
    if left_key and right_key:
        mapping = pd.Series(values.to_numpy(), index=labelled[right_key].to_numpy())
        mapping = mapping[~mapping.index.duplicated(keep="first")]
        candidate = frame[left_key].map(mapping)
        if candidate.notna().all():
            joined = candidate
    if joined is None and len(frame) == len(labelled):
        joined = pd.Series(values.to_numpy(), index=frame.index)
    if joined is None:
        return frame, columns

    enriched = frame.copy()
    enriched["Dataset Label"] = joined.to_numpy()
    return enriched, replace(columns, label="Dataset Label")


def load_replay(store: ArtifactStore, run_id: str) -> ReplayFrame | None:
    """Load the preferred row-level table for a run.

    Preference: explainable triage (has residual + contributors), then test
    triage, then the replay table. Each fallback is reported to the caller.
    """
    for key in ("explainable_triage", "triage_test", "replay"):
        table = store.read_table(run_id, key)
        if table is not None and not table.empty:
            artifact = store.run(run_id).table(key)
            frame, columns = _attach_labels(
                store, run_id, table, resolve_columns(table)
            )
            return ReplayFrame(
                frame=frame,
                columns=columns,
                path=str(artifact.path) if artifact else key,
            )
    return None


def replay_source_key(store: ArtifactStore, run_id: str) -> str | None:
    for key in ("explainable_triage", "triage_test", "replay"):
        table = store.read_table(run_id, key)
        if table is not None and not table.empty:
            return key
    return None
