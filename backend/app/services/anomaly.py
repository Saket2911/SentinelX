"""Residual review-priority calibration.

SentinelX scores observations by the mean absolute error of the predicted next
network state. That score is a *review heuristic*. The bands below are quantiles
of the recorded test-partition residual distribution, reproduced from the
committed reliability audit, so an analyst can see how unusual an error is
relative to recorded behaviour.

A residual is not an attack. The API never labels an observation malicious.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore

REVIEW_PRIORITY_ORDER = ("baseline", "within_range", "elevated", "high", "review_required")

PRIORITY_SEMANTICS = (
    "Priority bands are residual quantiles from recorded evaluation data. They rank "
    "observations for analyst review; they do not confirm malicious activity and are "
    "not a validated intrusion-detection threshold."
)


@dataclass(frozen=True)
class Calibration:
    threshold: float | None
    threshold_source: str
    quantiles: dict[str, float]
    sample_count: int | None
    method: str
    notes: list[str] = field(default_factory=list)

    def priority_for(self, score: float | None) -> str:
        if score is None:
            return "unknown"
        if self.threshold is not None and score >= self.threshold:
            return "review_required"
        p99 = self.quantiles.get("0.99")
        p95 = self.quantiles.get("0.95")
        p90 = self.quantiles.get("0.9")
        p50 = self.quantiles.get("0.5")
        if p99 is not None and score >= p99:
            return "high"
        if p95 is not None and score >= p95:
            return "elevated"
        if p90 is not None and score >= p90:
            return "within_range"
        if p50 is not None and score >= p50:
            return "within_range"
        return "baseline"

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "quantiles": self.quantiles,
            "sample_count": self.sample_count,
            "method": self.method,
            "priority_semantics": PRIORITY_SEMANTICS,
            "notes": self.notes,
        }


def _report_json(reports_dir: Path, relative: str) -> dict[str, Any] | None:
    path = reports_dir / relative
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def calibration_for(store: ArtifactStore, run_id: str) -> Calibration:
    triage = store.read_json(run_id, "triage") or {}
    test_triage = triage.get("test") if isinstance(triage.get("test"), dict) else {}
    threshold = test_triage.get("frozen_threshold")
    if threshold is None:
        integrated = store.read_json(run_id, "integrated_triage") or {}
        decision = integrated.get("decision_method") or {}
        threshold = decision.get("threshold")
    source = (
        (test_triage.get("threshold_source") if test_triage else None)
        or "Validation benign 99th percentile (frozen)"
    )

    reliability = _report_json(
        store.reports_dir, "v3_reliability_audit_v1/reliability_summary.json"
    ) or {}
    sample_mae = ((reliability.get("overall_error_quantiles") or {}).get("sample_mae") or {})
    quantiles = {
        str(key): float(value)
        for key, value in sample_mae.items()
        if isinstance(value, (int, float))
    }
    notes: list[str] = []
    if not quantiles:
        notes.append(
            "Residual quantiles were not found; priority bands fall back to the "
            "frozen review threshold only."
        )
    test_anomaly = store.read_json(run_id, "test_anomaly") or {}
    if test_anomaly:
        metrics = test_anomaly.get("detection_metrics_benign_vs_bot") or {}
        notes.append(
            "Recorded test detection at the frozen threshold: "
            f"TP={metrics.get('TP')}, FN={metrics.get('FN')}, "
            f"precision={metrics.get('precision')}, recall={metrics.get('recall')}."
        )
    notes.append(
        "The frozen threshold was calibrated on benign validation observations and "
        "transferred poorly to the test partition; it is not a validated detector."
    )
    return Calibration(
        threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
        threshold_source=str(source),
        quantiles=quantiles,
        sample_count=reliability.get("samples") if isinstance(reliability.get("samples"), int) else None,
        method=str(
            test_anomaly.get("score_definition")
            or "Mean absolute residual across model features"
        ),
        notes=notes,
    )
