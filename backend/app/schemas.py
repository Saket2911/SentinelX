"""Pydantic response contracts for the SentinelX API.

Every analytical response carries the same three envelope fields:

``source``        where the bytes came from (recorded artifact, dataset, live)
``availability``  whether the data exists in this deployment, and if not why
``limitations``   interpretation constraints that must reach the analyst

This prevents "empty because nothing is loaded" from looking like "zero events"
and keeps every screen honest about what it is showing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceMode = Literal["recorded_artifact", "dataset", "live", "unavailable"]
ReviewPriority = Literal["baseline", "within_range", "elevated", "high", "review_required", "unknown"]
EventStatus = Literal["new", "investigating", "escalated", "benign", "closed"]


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SourceMode
    label: str
    detail: str | None = None


class Availability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None = None
    required_artifacts: list[str] = Field(default_factory=list)


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    value: float | int | None = None
    display: str
    unit: str | None = None
    status: Literal["good", "warning", "critical", "neutral", "unknown"] = "neutral"
    detail: str | None = None


class MitreMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    tactic: str
    technique: str
    name: str
    status: str
    evidence: str | None = None
    notes: str | None = None
    source: str | None = None


class FeatureContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float | None = None
    contribution: float | None = None
    direction: Literal["increase", "decrease", "neutral"] = "neutral"
    rank: int | None = None


# --------------------------------------------------------------------------- health


class HealthCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    status: Literal["ok", "warning", "failed", "unknown"]
    detail: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "failed"]
    version: str
    uptime_seconds: float
    checks: list[HealthCheck]
    runtime: dict[str, Any]


class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    state: Literal["implemented", "conditional", "planned"]
    detail: str


class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capabilities: list[Capability]
    limitations: list[str]


# --------------------------------------------------------------------------- system


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    path: str
    kind: Literal["json", "table", "checkpoint", "text", "directory"]
    present: bool
    size_bytes: int | None = None


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    active_run_id: str
    artifacts: list[ArtifactStatus]
    runtime: dict[str, Any]
    limitations: list[str]


# --------------------------------------------------------------------------- models


class CheckpointInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    size_bytes: int
    sha256: str | None = None


class ModelSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    name: str
    active: bool
    available: bool
    availability: Availability
    architecture: dict[str, Any]
    training: dict[str, Any]
    checkpoint: dict[str, Any]
    validation_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    generated_at: str
    active_run_id: str
    models: list[ModelSummary]
    limitations: list[str]


class ModelDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    model: ModelSummary
    artifacts: list[ArtifactStatus]
    training_curve: list[dict[str, Any]]
    evaluation: dict[str, Any]
    multi_step: list[dict[str, Any]]
    explanations: dict[str, Any]
    health: list[HealthCheck]
    limitations: list[str]


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    confirm: bool = False
    reason: str | None = Field(default=None, max_length=500)


class ActivationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_run_id: str
    previous_run_id: str | None
    activated_at: str
    reason: str | None
    warnings: list[str]


class ModelComparisonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    generated_at: str
    left: ModelSummary
    right: ModelSummary
    metrics: list[dict[str, Any]]
    warnings: list[str]


# --------------------------------------------------------------------------- datasets


class DatasetArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    path: str
    present: bool
    size_bytes: int | None = None


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal["research_artifact", "uploaded"]
    path: str
    present: bool
    size_bytes: int | None = None
    rows: int | None = None
    columns: int | None = None
    label_counts: dict[str, int] | None = None
    feature_compatible: bool | None = None
    notes: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    generated_at: str
    datasets: list[DatasetInfo]
    artifacts: list[DatasetArtifact]
    label_coverage: dict[str, dict[str, int]]
    active_run_id: str
    model_features: int | None
    limitations: list[str]


class DatasetValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    dataset: str
    checks: list[HealthCheck]
    row_count: int | None
    column_count: int | None
    missing_features: int
    extra_features: int
    label_counts: dict[str, int]
    limitations: list[str]


# --------------------------------------------------------------------------- network


class NetworkSeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    label: str | None = None
    timestamp: str | None = None
    value: float | None = None
    secondary: float | None = None


class ProtocolShare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int
    share: float


class NetworkOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    totals: dict[str, Any]
    label_distribution: list[ProtocolShare]
    protocol_distribution: list[ProtocolShare]
    top_ports: list[ProtocolShare]
    activity_series: list[NetworkSeriesPoint]
    segments: list[dict[str, Any]]
    flow_features: list[dict[str, Any]]
    limitations: list[str]


# --------------------------------------------------------------------------- predictions


class PredictionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    step: int | None = None
    timestamp: str | None = None
    actual: float | None = None
    predicted: float | None = None
    residual: float | None = None
    label: str | None = None
    priority: ReviewPriority = "unknown"


class PredictionSeriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    feature: str | None
    features: list[str]
    points: list[PredictionPoint]
    summary: dict[str, Any]
    calibration: dict[str, Any]
    horizon: dict[str, Any]
    limitations: list[str]


class PredictionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    metrics: list[Metric]
    error_distribution: list[dict[str, Any]]
    per_label: list[dict[str, Any]]
    multi_step: list[dict[str, Any]]
    calibration: dict[str, Any]
    limitations: list[str]


# --------------------------------------------------------------------------- events


class SecurityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    step: int | None = None
    timestamp: str | None = None
    capture_day: str | None = None
    segment_id: str | None = None
    dataset_label: str | None = None
    residual_score: float | None = None
    priority: ReviewPriority = "unknown"
    status: EventStatus = "new"
    assignee: str | None = None
    notes: str | None = None
    updated_at: str | None = None
    mitre: list[MitreMapping] = Field(default_factory=list)
    contributors: list[FeatureContribution] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class EventListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    events: list[SecurityEvent]
    total: int
    page: int
    page_size: int
    filters: dict[str, Any]
    status_counts: dict[str, int]
    priority_counts: dict[str, int]
    limitations: list[str]


class EventUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EventStatus | None = None
    assignee: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=4000)
    actor: str = Field(default="analyst", min_length=1, max_length=120)


class PlaybackEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, description="Replay step (1-based) whose next state is evaluated")
    run_id: str | None = None


class PlaybackAlertUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EventStatus | None = None
    assignee: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=4000)
    actor: str = Field(default="analyst", min_length=1, max_length=120)


class EventDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    event: SecurityEvent
    history: list[dict[str, Any]]
    limitations: list[str]


class TimelineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    buckets: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    events: list[SecurityEvent]
    limitations: list[str]


# --------------------------------------------------------------------------- explainability


class ExplainabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    method: dict[str, Any]
    global_importance: list[FeatureContribution]
    temporal_importance: list[dict[str, Any]]
    label_comparison: list[dict[str, Any]]
    stability: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    local: dict[str, Any] | None = None
    limitations: list[str]


# --------------------------------------------------------------------------- dashboard


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    availability: Availability
    generated_at: str
    status: dict[str, Any]
    metrics: list[Metric]
    activity_series: list[NetworkSeriesPoint]
    label_distribution: list[ProtocolShare]
    model_health: dict[str, Any]
    prediction_quality: dict[str, Any]
    dataset_status: dict[str, Any]
    system_health: dict[str, Any]
    recent_events: list[SecurityEvent]
    event_summary: dict[str, Any]
    alerts: dict[str, Any]
    limitations: list[str]


# --------------------------------------------------------------------------- reports


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    sections: list[Literal[
        "summary",
        "model_evaluation",
        "anomaly",
        "triage",
        "explainability",
        "dataset",
    ]] = Field(default_factory=lambda: ["summary", "model_evaluation"])
    run_id: str | None = None
    author: str = Field(default="analyst", min_length=1, max_length=120)


class ReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    title: str
    author: str
    created_at: str
    run_id: str
    sections: list[str]
    formats: list[str]
    size_bytes: int


class ReportListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: list[ReportSummary]
    generated_at: str


# --------------------------------------------------------------------------- settings


class SettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_run_id: str | None = None
    review_threshold: float | None = Field(default=None, gt=0, lt=100)
    page_size: int = Field(default=25, ge=5, le=200)
    timezone: str = Field(default="UTC", max_length=64)
    notify_on_review_required: bool = True
    analyst_name: str = Field(default="analyst", min_length=1, max_length=120)
    data_mode: Literal["dataset", "live"] = "dataset"


class SettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: SettingsPayload
    threshold_source: dict[str, Any]
    auth_enabled: bool
    editable_keys: list[str]


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
    context: dict[str, Any] | None = None
