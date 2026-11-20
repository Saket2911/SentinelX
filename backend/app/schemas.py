"""Pydantic response schemas for the SentinelX API.

Only the shapes that benefit from validation are modelled explicitly; deeply
nested research payloads are passed through as ``dict`` because their structure
is owned by the research pipeline, not the API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProvenanceValue(BaseModel):
    value: Any = None
    provenance: Literal["recorded_artifact", "derived", "unavailable", "live_telemetry"]
    source: str | None = None
    note: str | None = None
    reason: str | None = None
    requires: list[str] | None = None


class ArtifactStatus(BaseModel):
    id: str
    label: str
    path: str
    present: bool
    required_for: str
    provenance_kind: Literal["json", "text", "binary"]


class ModelStatus(BaseModel):
    id: str
    name: str
    input_features: int
    sequence_length: int
    status: str
    checkpoint_present: bool
    checkpoint_path: str
    loadable: bool
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    data_mode: str
    inference_available: bool
    live_telemetry_available: bool
    artifacts: list[ArtifactStatus]
    models: list[ModelStatus]
    warnings: list[str] = Field(default_factory=list)


class MetricCard(BaseModel):
    key: str
    label: str
    value: float | int | str | None
    unit: str | None = None
    provenance: str
    source: str | None = None
    hint: str | None = None


class DashboardResponse(BaseModel):
    active_model: str
    security_posture: dict
    metrics: list[MetricCard]
    label_distribution: list[dict]
    detection_assessment: dict
    recent_signals: list[dict]
    system_health: dict
    limitations: list[str]


class MultiStepRow(BaseModel):
    horizon_seconds: int
    target_label: str
    valid_origins: int
    lstm_huber: float
    persistence_huber: float
    lstm_mae: float
    persistence_mae: float
    mae_improvement_percent: float
    lstm_rmse: float
    persistence_rmse: float
    rmse_improvement_percent: float


class SettingsPayload(BaseModel):
    anomaly_review_threshold: float = Field(default=1.881396770477295, gt=0, le=100)
    review_percentile: float = Field(default=99.0, gt=50, lt=100)
    default_split: Literal["train", "validation", "test"] = "test"
    table_page_size: int = Field(ge=5, le=200, default=25)
    show_provenance_badges: bool = True
    telemetry_mode: Literal["offline_dataset", "live"] = "offline_dataset"
