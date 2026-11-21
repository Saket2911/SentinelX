"""Threat review candidates, analyst workflow and attack timeline.

Naming is deliberate: these are **review candidates** derived from prediction
residuals, not confirmed incidents. Dataset reference labels are shown as
labels, never as model classifications, and MITRE mappings stay contextual.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..db import Database, utc_now
from .anomaly import Calibration, calibration_for
from .artifacts import ArtifactStore
from .replay import load_replay, replay_source_key

PRIORITY_ORDER = ["review_required", "high", "elevated", "within_range", "baseline", "unknown"]

ACTION_LIBRARY: dict[str, list[str]] = {
    "brute force": [
        "Correlate with authentication logs for the affected service and source address.",
        "Check for successful logins following repeated failures.",
        "Verify account lockout and rate-limiting policy coverage.",
    ],
    "ddos": [
        "Check service availability and upstream link utilisation for the affected window.",
        "Confirm whether the volume change is explained by known bulk transfer or backup jobs.",
        "Review DDoS mitigation and rate-limit telemetry before escalating.",
    ],
    "dos": [
        "Compare the interval against expected service traffic baselines.",
        "Check host resource metrics (CPU, connection table) for the target service.",
        "Verify whether the traffic is internally generated (load test, scan) before escalating.",
    ],
    "bot": [
        "Inspect outbound connection regularity and destination diversity for beaconing patterns.",
        "Pull DNS and proxy logs for repeated low-volume connections to the same endpoint.",
        "Check host telemetry for the internal host involved before drawing conclusions.",
    ],
    "infilteration": [
        "Review internal-to-internal flow records for unusual peer relationships.",
        "Check authentication and file-share auditing for the same interval.",
        "Compare against approved maintenance and administrative activity.",
    ],
    "sql": [
        "Inspect web server access logs for injection-shaped requests in the window.",
        "Check application error rates and database logs for query failures.",
        "Confirm with the application owner before classifying as exploitation.",
    ],
    "xss": [
        "Inspect web logs for script payloads in request parameters.",
        "Check whether responses reflect user input without encoding.",
        "Confirm application ownership of the affected endpoint.",
    ],
}

DEFAULT_ACTIONS = [
    "Compare the predicted state with the observed state for the contributing features.",
    "Corroborate with flow, DNS, proxy or authentication evidence for the same interval.",
    "Record the investigation outcome so the case history stays auditable.",
]

LABEL_LIMITATION = (
    "The label column is the dataset's reference annotation for the window. It is not a "
    "model output, and a high prediction residual does not make an observation malicious."
)


def _find_contributor_columns(columns: Iterable[str]) -> list[str]:
    found = []
    for column in columns:
        lowered = column.lower()
        if ("top" in lowered or "contributor" in lowered) and (
            "feature" in lowered or "contributor" in lowered
        ):
            found.append(column)
    return found[:5]


def _recommended_actions(label: str | None) -> list[str]:
    if not label:
        return list(DEFAULT_ACTIONS)
    lowered = label.lower()
    for key, actions in ACTION_LIBRARY.items():
        if key in lowered:
            return list(actions) + [DEFAULT_ACTIONS[-1]]
    return list(DEFAULT_ACTIONS)


def _mitre_for(mappings: list[dict[str, Any]], label: str | None) -> list[dict[str, Any]]:
    if not label:
        return []
    matches = [
        mapping
        for mapping in mappings
        if str(mapping.get("label", "")).strip().lower() == label.strip().lower()
    ]
    return [
        {
            "label": str(mapping.get("label")),
            "tactic": str(mapping.get("tactic")),
            "technique": str(mapping.get("technique")),
            "name": str(mapping.get("name")),
            "status": str(mapping.get("status")),
            "evidence": mapping.get("evidence"),
            "notes": mapping.get("notes"),
            "source": mapping.get("source"),
        }
        for mapping in matches
    ]


def _candidate_rows(
    store: ArtifactStore,
    run_id: str,
    include_top: bool,
    top_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay = load_replay(store, run_id)
    calibration = calibration_for(store, run_id)
    if replay is None:
        return [], {
            "available": False,
            "reason": "No row-level state table is available for this run.",
            "selection": "none",
        }
    residuals = replay.residuals()
    if residuals is None:
        return [], {
            "available": False,
            "reason": "The row-level table has no residual or actual/predicted columns.",
            "selection": "none",
        }

    steps = replay.steps()
    labels = replay.labels()
    timestamps = replay.timestamps()
    segments = replay.segments()
    priorities = calibrate_priorities(calibration, residuals)

    explicit_review = pd.Series(False, index=replay.frame.index)
    if replay.columns.priority:
        statuses = replay.frame[replay.columns.priority].astype(str).str.upper()
        explicit_review = statuses.str.contains("REVIEW", na=False)

    if calibration.threshold is not None:
        threshold_mask = residuals.fillna(0.0) >= calibration.threshold
    else:
        threshold_mask = pd.Series(False, index=replay.frame.index)

    selection_mask = threshold_mask | explicit_review
    selection = "threshold"
    if not selection_mask.any() and include_top and residuals.notna().any():
        top_positions = residuals.dropna().sort_values(ascending=False).head(top_count).index
        selection_mask = pd.Series(False, index=replay.frame.index)
        selection_mask.loc[top_positions] = True
        selection = "top_residuals"

    contributor_columns = _find_contributor_columns(replay.frame.columns.tolist())
    mappings, mapping_source = store.mitre_mappings(run_id)

    rows: list[dict[str, Any]] = []
    for position in replay.frame.index[selection_mask]:
        label = str(labels.loc[position])
        contributors = []
        for rank, column in enumerate(contributor_columns, start=1):
            value = replay.frame[column].iloc[replay.frame.index.get_loc(position)]
            if isinstance(value, str) and value.strip():
                contributors.append(
                    {
                        "feature": value.strip(),
                        "value": None,
                        "contribution": None,
                        "direction": "neutral",
                        "rank": rank,
                    }
                )
        residual_value = residuals.loc[position]
        rows.append(
            {
                "event_id": f"step-{int(steps.loc[position])}",
                "step": int(steps.loc[position]),
                "timestamp": _iso(timestamps.loc[position]),
                "capture_day": _day(timestamps.loc[position]),
                "segment_id": str(segments.loc[position]),
                "dataset_label": label,
                "residual_score": float(residual_value) if np.isfinite(residual_value) else None,
                "priority": str(priorities.loc[position]),
                "mitre": _mitre_for(mappings, label),
                "contributors": contributors,
                "recommended_actions": _recommended_actions(label),
            }
        )
    rows.sort(key=lambda row: (PRIORITY_ORDER.index(row["priority"]) if row["priority"] in PRIORITY_ORDER else 99, -(row["residual_score"] or 0)))
    context = {
        "available": True,
        "reason": None,
        "selection": selection,
        "threshold": calibration.threshold,
        "threshold_source": calibration.threshold_source,
        "source_table": replay_source_key(store, run_id),
        "mitre_source": mapping_source,
        "explicit_status_column": replay.columns.priority,
        "contributor_columns": contributor_columns,
    }
    return rows, context


def calibrate_priorities(calibration: Calibration, residuals: pd.Series) -> pd.Series:
    values = residuals.to_numpy(dtype=float)
    priorities = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        priorities[index] = calibration.priority_for(
            float(value) if np.isfinite(value) else None
        )
    return pd.Series(priorities, index=residuals.index)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _day(value: Any) -> str | None:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return None


def list_events(
    store: ArtifactStore,
    database: Database,
    run_id: str,
    status: str | None = None,
    priority: str | None = None,
    label: str | None = None,
    segment: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 25,
    include_top: bool = False,
) -> dict[str, Any]:
    rows, context = _candidate_rows(store, run_id, include_top, max(page_size, 25))
    states = database.event_states(run_id)
    merged: list[dict[str, Any]] = []
    for row in rows:
        state = states.get(row["event_id"]) or {}
        merged.append(
            {
                **row,
                "status": state.get("status", "new"),
                "assignee": state.get("assignee"),
                "notes": state.get("notes"),
                "updated_at": state.get("updated_at"),
            }
        )

    status_counts: dict[str, int] = {}
    for event in merged:
        status_counts[event["status"]] = status_counts.get(event["status"], 0) + 1
    priority_counts: dict[str, int] = {}
    for event in merged:
        priority_counts[event["priority"]] = priority_counts.get(event["priority"], 0) + 1

    filtered = merged
    if status:
        filtered = [event for event in filtered if event["status"] == status]
    if priority:
        filtered = [event for event in filtered if event["priority"] == priority]
    if label:
        filtered = [event for event in filtered if event["dataset_label"] == label]
    if segment:
        filtered = [event for event in filtered if event["segment_id"] == segment]
    if search:
        needle = search.lower()
        filtered = [
            event
            for event in filtered
            if needle
            in " ".join(
                str(value)
                for value in (
                    event.get("dataset_label"),
                    event.get("segment_id"),
                    event.get("step"),
                    event.get("assignee"),
                    event.get("notes"),
                )
                if value is not None
            ).lower()
        ]

    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    start = (page - 1) * page_size
    paged = filtered[start : start + page_size]

    limitations = [
        LABEL_LIMITATION,
        "Priority bands are residual quantiles; they rank observations for review and do not "
        "confirm an attack.",
        "The frozen residual threshold recalled 0 of the Bot observations in the recorded test "
        "evaluation, so an empty review queue does not mean the period was benign.",
    ]
    if context.get("selection") == "top_residuals":
        limitations.insert(
            0,
            "No observation crossed the frozen threshold. The rows shown are the highest-residual "
            "observations, provided for review context only; they are not incidents.",
        )
    if not context.get("available"):
        limitations.insert(0, str(context.get("reason")))

    return {
        "source": {
            "mode": "dataset" if context.get("available") else "unavailable",
            "label": "Residual review candidates",
            "detail": context.get("source_table"),
        },
        "availability": {
            "available": bool(context.get("available")),
            "reason": context.get("reason"),
            "required_artifacts": [] if context.get("available") else ["<run>/explainable_triage_v2/explainable_triage.csv"],
        },
        "events": paged,
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "filters": {
            "status": status,
            "priority": priority,
            "label": label,
            "segment": segment,
            "search": search,
            "include_top": include_top,
            "selection": context.get("selection"),
            "threshold": context.get("threshold"),
            "threshold_source": context.get("threshold_source"),
            "candidate_pool": len(merged),
            "contributor_columns": context.get("contributor_columns") or [],
        },
        "status_counts": status_counts,
        "priority_counts": priority_counts,
        "limitations": limitations,
    }


def get_event(
    store: ArtifactStore, database: Database, run_id: str, event_id: str, include_top: bool = True
) -> dict[str, Any] | None:
    rows, context = _candidate_rows(store, run_id, include_top, 50)
    row = next((item for item in rows if item["event_id"] == event_id), None)
    if row is None:
        return None
    state = database.event_state(run_id, event_id) or {}
    event = {
        **row,
        "status": state.get("status", "new"),
        "assignee": state.get("assignee"),
        "notes": state.get("notes"),
        "updated_at": state.get("updated_at"),
    }
    return {
        "source": {
            "mode": "dataset",
            "label": "Residual review candidate",
            "detail": context.get("source_table"),
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "event": event,
        "history": database.event_history(run_id, event_id),
        "limitations": [LABEL_LIMITATION, "Residuals do not confirm malicious activity."],
    }


def update_event(
    store: ArtifactStore,
    database: Database,
    run_id: str,
    event_id: str,
    status: str | None,
    assignee: str | None,
    notes: str | None,
    actor: str,
) -> dict[str, Any]:
    rows, _ = _candidate_rows(store, run_id, True, 50)
    if not any(row["event_id"] == event_id for row in rows):
        raise KeyError(event_id)
    previous = database.event_state(run_id, event_id) or {}
    next_status = status or previous.get("status") or "new"
    next_assignee = assignee if assignee is not None else previous.get("assignee")
    next_notes = notes if notes is not None else previous.get("notes")
    changes: list[tuple[str, str | None, str | None]] = []
    if previous.get("status") != next_status:
        changes.append(("status", previous.get("status"), next_status))
    if assignee is not None and previous.get("assignee") != next_assignee:
        changes.append(("assignee", previous.get("assignee"), next_assignee))
    if notes is not None and previous.get("notes") != next_notes:
        changes.append(("notes", previous.get("notes"), next_notes))
    database.save_event_state(
        run_id=run_id,
        event_id=event_id,
        status=str(next_status),
        assignee=next_assignee,
        notes=next_notes,
        actor=actor,
        changes=changes,
    )
    detail = get_event(store, database, run_id, event_id, include_top=True)
    if detail is None:  # pragma: no cover - defensive
        raise KeyError(event_id)
    return detail


def timeline(store: ArtifactStore, database: Database, run_id: str, include_top: bool = True) -> dict[str, Any]:
    replay = load_replay(store, run_id)
    if replay is None:
        return {
            "source": {"mode": "unavailable", "label": "Attack timeline", "detail": None},
            "availability": {
                "available": False,
                "reason": "Row-level state table is unavailable.",
                "required_artifacts": ["<run>/historical_replay_v1/test_replay.csv"],
            },
            "buckets": [],
            "phases": [],
            "events": [],
            "limitations": [LABEL_LIMITATION],
        }
    calibration = calibration_for(store, run_id)
    residuals = replay.residuals()
    labels = replay.labels()
    timestamps = replay.timestamps()
    steps = replay.steps()
    parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")

    frame = pd.DataFrame(
        {
            "step": steps.to_numpy(),
            "timestamp": parsed.to_numpy(),
            "label": labels.to_numpy(),
            "residual": residuals.to_numpy() if residuals is not None else np.nan,
        }
    )
    if frame["timestamp"].notna().any():
        frame["bucket"] = frame["timestamp"].dt.floor("5min")
    else:
        frame["bucket"] = (frame.index // 300)

    buckets: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("bucket", sort=True):
        label_counts = {
            str(key): int(value) for key, value in group["label"].value_counts().items()
        }
        review = 0
        if calibration.threshold is not None:
            review = int((group["residual"] >= calibration.threshold).sum())
        buckets.append(
            {
                "bucket": str(bucket),
                "first_step": int(group["step"].min()),
                "last_step": int(group["step"].max()),
                "observations": int(len(group)),
                "mean_residual": float(group["residual"].mean()) if group["residual"].notna().any() else None,
                "max_residual": float(group["residual"].max()) if group["residual"].notna().any() else None,
                "review_candidates": review,
                "labels": label_counts,
            }
        )

    phases: list[dict[str, Any]] = []
    current_label: str | None = None
    start_row: dict[str, Any] | None = None
    for position in range(len(frame)):
        label = str(frame["label"].iloc[position])
        if label != current_label:
            if current_label is not None and start_row is not None:
                phases.append(_phase(start_row, frame, position - 1, current_label))
            current_label = label
            start_row = {
                "step": int(frame["step"].iloc[position]),
                "timestamp": frame["timestamp"].iloc[position],
            }
            if start_row["timestamp"] is not None and hasattr(start_row["timestamp"], "isoformat"):
                start_row["timestamp"] = start_row["timestamp"].isoformat()
    if current_label is not None and start_row is not None:
        phases.append(_phase(start_row, frame, len(frame) - 1, current_label))
    phases = [phase for phase in phases if phase["observations"] >= 1][:200]

    events, context = _candidate_rows(store, run_id, include_top, 25)
    states = database.event_states(run_id)
    decorated = [
        {
            **row,
            "status": (states.get(row["event_id"]) or {}).get("status", "new"),
            "assignee": (states.get(row["event_id"]) or {}).get("assignee"),
            "notes": (states.get(row["event_id"]) or {}).get("notes"),
            "updated_at": (states.get(row["event_id"]) or {}).get("updated_at"),
        }
        for row in events
    ]
    return {
        "source": {
            "mode": "dataset",
            "label": "Observed label sequence with prediction residuals",
            "detail": replay.path,
        },
        "availability": {"available": True, "reason": None, "required_artifacts": []},
        "buckets": buckets,
        "phases": phases,
        "events": decorated,
        "limitations": [
            LABEL_LIMITATION,
            "Phases reflect contiguous runs of dataset reference labels in the replay window. They "
            "are not reconstructed attack stages and do not imply intent.",
            "Residual activity is shown alongside labels for context; the LSTM does not classify attacks.",
            f"Timeline selection: {context.get('selection')}.",
        ],
    }


def _phase(start_row: dict[str, Any], frame: pd.DataFrame, end_position: int, label: str) -> dict[str, Any]:
    end_timestamp = frame["timestamp"].iloc[end_position]
    return {
        "start_step": start_row["step"],
        "end_step": int(frame["step"].iloc[end_position]),
        "start_timestamp": start_row["timestamp"],
        "end_timestamp": end_timestamp.isoformat() if hasattr(end_timestamp, "isoformat") else None,
        "label": label,
        "observations": int(
            (frame["label"].iloc[: end_position + 1].astype(str) == label).sum()
        ),
    }


def summary_counts(store: ArtifactStore, database: Database, run_id: str) -> dict[str, Any]:
    rows, context = _candidate_rows(store, run_id, False, 25)
    states = database.event_states(run_id)
    priority_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        priority_counts[row["priority"]] = priority_counts.get(row["priority"], 0) + 1
        status = (states.get(row["event_id"]) or {}).get("status", "new")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "candidates": len(rows),
        "priority_counts": priority_counts,
        "status_counts": status_counts,
        "threshold": context.get("threshold"),
        "selection": context.get("selection"),
        "available": context.get("available"),
        "reason": context.get("reason"),
        "generated_at": utc_now(),
    }
