export type SourceMode = "recorded_artifact" | "dataset" | "live" | "unavailable";

export interface Source {
  mode: SourceMode;
  label: string;
  detail?: string | null;
}

export interface Availability {
  available: boolean;
  reason?: string | null;
  required_artifacts: string[];
}

export interface Metric {
  id: string;
  label: string;
  value: number | null;
  display: string;
  unit?: string | null;
  status: "good" | "warning" | "critical" | "neutral" | "unknown";
  detail?: string | null;
}

export interface SeriesPoint {
  index: number;
  label: string | null;
  timestamp: string | null;
  value: number | null;
  secondary?: number | null;
}

export interface Share {
  name: string;
  count: number;
  share: number;
}

export interface FeatureContribution {
  feature: string;
  value: number | null;
  contribution: number | null;
  direction: "increase" | "decrease" | "neutral";
  rank?: number | null;
}

export interface MitreMapping {
  label: string;
  tactic: string;
  technique: string;
  name: string;
  status: string;
  evidence?: string | null;
  notes?: string | null;
  source?: string | null;
}

export type Priority =
  | "baseline"
  | "within_range"
  | "elevated"
  | "high"
  | "review_required"
  | "unknown";

export type EventStatus = "new" | "investigating" | "escalated" | "benign" | "closed";

export interface SecurityEvent {
  event_id: string;
  step: number | null;
  timestamp: string | null;
  capture_day: string | null;
  segment_id: string | null;
  dataset_label: string | null;
  residual_score: number | null;
  priority: Priority;
  status: EventStatus;
  assignee: string | null;
  notes: string | null;
  updated_at: string | null;
  mitre: MitreMapping[];
  contributors: FeatureContribution[];
  recommended_actions: string[];
}

export interface Dashboard {
  source: Source;
  availability: Availability;
  status: {
    posture: "nominal" | "review_required" | "degraded" | string;
    headline: string;
    detail: string;
    run_id: string;
    generated_at: string;
    threshold: number | null;
    threshold_source: string;
  };
  metrics: Metric[];
  activity_series: SeriesPoint[];
  label_distribution: Share[];
  model_health: ModelSummary & { calibration: Calibration };
  prediction_quality: {
    metrics: Metric[];
    per_label: Array<Record<string, unknown>>;
    space: string;
    test_partition_note?: string;
    detection: {
      test?: DetectionMetrics | null;
      validation?: DetectionMetrics | null;
      warning: string;
    };
  };
  dataset_status: {
    artifacts: Array<{ key: string; label: string; present: boolean; path: string }>;
    dataset_directory: string;
    label_coverage: Record<string, Record<string, number>>;
  };
  system_health: Record<string, unknown>;
  recent_events: SecurityEvent[];
  event_summary: {
    candidates: number;
    priority_counts: Record<string, number>;
    status_counts: Record<string, number>;
    threshold: number | null;
    selection: string | null;
    available: boolean;
    reason: string | null;
  };
  alerts: {
    available: boolean;
    reason: string | null;
    total: number;
    high_priority: number;
    unresolved: number;
    severity_counts: Record<string, number>;
    status_counts: Record<string, number>;
    recent: PlaybackAlert[];
    daily: Array<{ day: string; count: number }>;
    threshold: number | null;
    threshold_source: string;
    rules: PlaybackConditions["rules"];
    limitations: string[];
  };
  limitations: string[];
}

export interface DetectionMetrics {
  TP?: number;
  FN?: number;
  FP?: number;
  TN?: number;
  precision?: number;
  recall?: number;
  specificity?: number;
  false_positive_rate?: number;
  f1_score?: number;
  accuracy?: number;
}

export interface Calibration {
  threshold: number | null;
  threshold_source: string;
  quantiles: Record<string, number>;
  sample_count: number | null;
  method: string;
  priority_semantics: string;
  notes: string[];
}

export interface CheckpointFile {
  name: string;
  path: string;
  size_bytes: number;
  sha256: string | null;
}

export interface ModelSummary {
  run_id: string;
  name: string;
  active: boolean;
  available: boolean;
  availability: Availability;
  architecture: {
    model: string;
    input_features: number | null;
    hidden_size: number | null;
    num_layers: number | null;
    dropout: number | null;
    sequence_length: number | null;
    output_features: number | null;
    loss: string | null;
    optimizer: string | null;
  };
  training: {
    training_samples: number | null;
    validation_samples: number | null;
    batch_size: number | null;
    learning_rate: number | null;
    max_epochs: number | null;
    patience: number | null;
    seed: number | null;
    device: string | null;
    gpu_name: string | null;
    gpu_memory_gb: number | null;
    best_epoch: number | null;
    best_validation_huber: number | null;
    training_seconds: number | null;
    test_data_used_for_training: boolean;
  };
  checkpoint: {
    present: boolean;
    count: number;
    files: CheckpointFile[];
    primary: CheckpointFile | null;
  };
  validation_metrics: Record<string, unknown> | null;
  test_metrics: {
    samples: number | null;
    lstm: Record<string, number>;
    persistence: Record<string, number>;
    improvement_percent: Record<string, number>;
    best_epoch: number | null;
    evaluation_seconds: number | null;
    test_used_for_training: boolean | null;
    test_used_for_selection: boolean | null;
    space: string;
  } | null;
}

export interface ModelDetail {
  source: Source;
  availability: Availability;
  generated_at: string;
  model: ModelSummary;
  artifacts: ArtifactStatus[];
  training_curve: Array<Record<string, number>>;
  evaluation: {
    test: ModelSummary["test_metrics"];
    validation: Record<string, unknown> | null;
    test_anomaly: Record<string, unknown> | null;
    validation_anomaly: Record<string, unknown> | null;
    triage: Record<string, unknown> | null;
    feature_analysis: { horizons: Array<Record<string, unknown>>; verification_passed: boolean | null };
    training_facts: Record<string, unknown>;
    calibration: Calibration;
    limitations: string[];
  };
  multi_step: MultiStepRow[];
  explanations: Record<string, unknown>;
  health: HealthCheck[];
  limitations?: string[];
}

export interface ArtifactStatus {
  key: string;
  label: string;
  path: string;
  kind: string;
  present: boolean;
  size_bytes: number | null;
}

export interface HealthCheck {
  id: string;
  label: string;
  status: "ok" | "warning" | "failed" | "unknown";
  detail?: string | null;
}

export interface MultiStepRow {
  horizon_seconds: number;
  label: string;
  valid_origins?: number;
  lstm_huber?: number;
  persistence_huber?: number;
  lstm_mae?: number;
  persistence_mae?: number;
  mae_improvement_percent?: number;
  lstm_rmse?: number;
  persistence_rmse?: number;
  rmse_improvement_percent?: number;
}

export interface PredictionSummary {
  source: Source;
  availability: Availability;
  metrics: Metric[];
  features: string[];
  summary: {
    metrics: Metric[];
    per_label: Array<Record<string, unknown>>;
    multi_step: MultiStepRow[];
    samples: number | null;
    space: string;
    test_partition_note?: string;
  };
  calibration: Calibration;
  horizon: {
    one_step: boolean;
    multi_step_horizons?: number[];
    method?: string;
    prediction_horizon?: string;
    predictions_fed_back?: boolean;
  };
  error_distribution: Array<Record<string, number | string>>;
  per_label: Array<Record<string, unknown>>;
  multi_step: MultiStepRow[];
  limitations: string[];
}

export interface PredictionPoint {
  index: number;
  step: number | null;
  timestamp: string | null;
  actual: number | null;
  predicted: number | null;
  residual: number | null;
  label: string | null;
  priority: Priority;
}

export interface PredictionSeries {
  source: Source;
  availability: Availability;
  feature: string | null;
  features: string[];
  points: PredictionPoint[];
  summary: Record<string, number | string | null>;
  calibration: Calibration;
  horizon: Record<string, unknown>;
  limitations: string[];
}

export interface NetworkOverview {
  source: Source;
  availability: Availability;
  totals: {
    observations?: number;
    benign_observations?: number;
    attack_labelled_observations?: number;
    review_candidates?: number;
    segments?: number;
    capture_days?: string[];
    mean_residual?: number | null;
    frozen_threshold?: number | null;
    flow_plane?: Record<string, unknown>;
  };
  label_distribution: Share[];
  protocol_distribution: Share[];
  top_ports: Share[];
  activity_series: SeriesPoint[];
  segments: Array<{
    segment_id: string;
    observations: number;
    labels: Record<string, number>;
    mean_residual: number | null;
    first_step: number | null;
    last_step: number | null;
  }>;
  flow_features: Array<Record<string, number | string | null>>;
  limitations: string[];
}

export interface EventList {
  source: Source;
  availability: Availability;
  events: SecurityEvent[];
  total: number;
  page: number;
  page_size: number;
  filters: Record<string, unknown>;
  status_counts: Record<string, number>;
  priority_counts: Record<string, number>;
  limitations: string[];
}

export interface EventDetail {
  source: Source;
  availability: Availability;
  event: SecurityEvent;
  history: Array<{ actor: string; action: string; from_value: string | null; to_value: string | null; created_at: string }>;
  limitations: string[];
}

export type AlertSeverity = "medium" | "high" | "critical";

export interface PlaybackAlertFeature {
  feature: string;
  actual: number;
  predicted: number;
  signed_residual: number;
  absolute_error: number;
  share_percent: number;
}

export interface PlaybackAlert {
  alert_id: string;
  run_id: string;
  session_id: string;
  dataset_id: string;
  model_version: string;
  target_step: number;
  target_index: number;
  timestamp: string | null;
  segment_id: string | null;
  dataset_label: string | null;
  capture_day: string | null;
  anomaly_score: number;
  threshold: number | null;
  ratio: number | null;
  severity: AlertSeverity;
  trigger_reason: string;
  status: EventStatus;
  assignee: string | null;
  notes: string | null;
  features: PlaybackAlertFeature[];
  prediction_source: string;
  predicted_ref: string | null;
  actual_ref: string | null;
  context_ref: string | null;
  source: Source;
  created_at: string;
  updated_at: string;
  limitations: string[];
}

export interface PlaybackScore {
  method: string;
  space: string;
  anomaly_score: number;
  mse: number;
  rmse: number;
  mean_signed_residual: number;
  max_absolute_error: number;
  feature_count: number;
  feature_residuals: PlaybackAlertFeature[];
}

export interface PlaybackConditions {
  available: boolean;
  reason: string | null;
  triggered: boolean;
  single_threshold: boolean;
  persistence: boolean;
  same_segment?: boolean;
  severity: AlertSeverity | null;
  trigger: string | null;
  reasons: string[];
  recent_scores: number[];
  ratio?: number;
  elevated_threshold?: number;
  threshold?: number;
  score?: number;
  rules: {
    rule_version: string;
    persistence_window: number;
    elevated_ratio: number;
    high_ratio: number;
    critical_ratio: number;
  };
}

export interface ShapContext {
  available: boolean;
  reason?: string;
  note: string;
  source?: string;
  label?: string | null;
  matches?: Array<{ feature: string; mean_absolute_shap: number }>;
  top_features?: Array<{ feature: string; mean_absolute_shap: number }>;
}

export interface PlaybackEvaluation {
  source: Source;
  availability: Availability;
  run_id: string;
  model_version: string;
  dataset_id: string;
  step: number;
  state: {
    step: number;
    index: number;
    total_states: number;
    timestamp: string | null;
    capture_day: string | null;
    segment_id: string;
    dataset_label: string;
    threshold: number | null;
    feature_count: number;
  };
  prediction: {
    source: string;
    source_detail: string;
    model_version: string;
    sequence_length: number;
    features: number;
    target_step: number;
  };
  actual: { step: number; timestamp: string | null; reference: string | null };
  context: {
    states_used: number;
    first_input_step: number;
    last_input_step: number;
    target_step: number;
    segment_ids: string[];
    segment_consistent: boolean;
    reference: string | null;
  };
  score: PlaybackScore;
  conditions: PlaybackConditions;
  alert: PlaybackAlert | null;
  deduplicated: boolean;
  explainability: { shap: ShapContext };
  limitations: string[];
}

export interface PlaybackAlertList {
  run_id: string;
  alerts: PlaybackAlert[];
  total: number;
  rules: PlaybackConditions["rules"];
  limitations: string[];
}

export interface PlaybackAlertDetail {
  alert: PlaybackAlert;
  history: Array<{ actor: string; action: string; from_value: string | null; to_value: string | null; created_at: string }>;
  explainability?: { shap: ShapContext };
  limitations: string[];
}

export interface AnomalySeries {
  source: Source;
  availability: Availability;
  run_id: string;
  threshold: number | null;
  threshold_source: string;
  method: string;
  space: string;
  flagged_steps: number[];
  alert_steps: number[];
  points: Array<{
    index: number;
    step: number;
    timestamp: string | null;
    segment_id: string;
    label: string;
    anomaly_score: number;
    flagged: boolean;
    alert_id: string | null;
  }>;
  top_states: Array<{
    step: number;
    timestamp: string | null;
    segment_id: string;
    label: string;
    anomaly_score: number;
    ratio: number | null;
    flagged: boolean;
    alert_id: string | null;
  }>;
  label_summary: Record<string, number>;
  attack_labelled_states: number;
  benign_labelled_states: number;
  summary: {
    states: number;
    mean_score: number;
    max_score: number;
    flagged: number;
    persisted_alerts: number;
  };
  limitations: string[];
}

export interface PlaybackScan {
  run_id: string;
  threshold: number | null;
  scanned_states: number;
  candidate_states: number[];
  candidates: number;
  created: number;
  alerts: PlaybackAlert[];
  persisted: number;
  limited: boolean;
  limitations: string[];
}

export interface Timeline {
  source: Source;
  availability: Availability;
  buckets: Array<{
    bucket: string;
    first_step: number;
    last_step: number;
    observations: number;
    mean_residual: number | null;
    max_residual: number | null;
    review_candidates: number;
    labels: Record<string, number>;
  }>;
  phases: Array<{
    start_step: number;
    end_step: number;
    start_timestamp: string | null;
    end_timestamp: string | null;
    label: string;
    observations: number;
  }>;
  events: SecurityEvent[];
  limitations: string[];
}

export interface Explainability {
  source: Source;
  availability: Availability;
  method: {
    method: string | null;
    experiment: string | null;
    checkpoint_epoch: number | null;
    sequence_length: number | null;
    input_features: number | null;
    background_source: string | null;
    background_size: number | null;
    explanation_source: string | null;
    benign_samples: number | null;
    bot_samples: number | null;
    nsamples: number | null;
    outputs: string[];
    interpretation: string | null;
    local_available: boolean;
    selected_label: string | null;
    selected_output: string | null;
    available_labels: string[];
    available_outputs: string[];
  };
  global_importance: FeatureContribution[];
  temporal_importance: Array<{ position: number; mean_absolute: number; share: number; lag_seconds: number }>;
  label_comparison: Array<{ feature: string; benign: number; bot: number; difference: number }>;
  stability: Array<Record<string, string | number>>;
  samples: Array<{ sample_index: number; label: string; explanation_index: number }>;
  local: {
    sample_index: number;
    sample_count: number;
    explanation_index: number;
    label: string;
    output: string;
    outputs: string[];
    replay_step?: number;
    timestamp?: string | null;
    dataset_label?: string;
    contributions: FeatureContribution[];
    temporal: Array<{ position: number; contribution: number }>;
    feature_matrix: Array<{ feature: string; values: number[] }>;
    source: string;
  } | null;
  limitations: string[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  kind: "research_artifact" | "uploaded";
  path: string;
  present: boolean;
  size_bytes: number | null;
  rows: number | null;
  columns: number | null;
  label_counts: Record<string, number> | null;
  feature_compatible: boolean | null;
  notes: string[];
}

export interface DatasetOverview {
  source: Source;
  datasets: DatasetInfo[];
  artifacts: Array<{ key: string; label: string; path: string; present: boolean; size_bytes: number | null }>;
  label_coverage: Record<string, Record<string, number>>;
  active_run_id: string;
  model_features: number | null;
  limitations: string[];
}

export interface DatasetValidation {
  valid: boolean;
  dataset: string;
  checks: HealthCheck[];
  row_count: number | null;
  column_count: number | null;
  missing_features: number;
  extra_features: number;
  label_counts: Record<string, number>;
  limitations: string[];
}

export interface ReportSummary {
  report_id: string;
  title: string;
  author: string;
  created_at: string;
  run_id: string;
  sections: string[];
  formats: string[];
  size_bytes: number;
}

export interface SettingsResponse {
  settings: {
    default_run_id: string | null;
    review_threshold: number | null;
    page_size: number;
    timezone: string;
    notify_on_review_required: boolean;
    analyst_name: string;
    data_mode: "dataset" | "live";
  };
  threshold_source: {
    calibrated_threshold: number | null;
    source: string;
    quantiles: Record<string, number>;
    analyst_override_in_use: boolean;
    note: string;
  };
  auth_enabled: boolean;
  editable_keys: string[];
}

export interface ReplaySegment {
  segment_id: string;
  first_step: number;
  last_step: number;
  states: number;
  labels: Record<string, number>;
}

export interface ReplayMeta {
  source: Source;
  availability: Availability;
  run_id: string;
  total_states: number;
  step_min: number;
  step_max: number;
  feature_count: number;
  features: string[];
  sequence_length: number;
  threshold: number | null;
  segments: ReplaySegment[];
  capture_days: string[];
  manifest: Record<string, unknown>;
  table_path?: string;
  limitations: string[];
}

export interface ReplayFeatureState {
  feature: string;
  position: number;
  actual: number;
  predicted: number;
  residual: number;
  absolute_error: number;
  history: number[];
}

export interface ReplayContributor {
  rank: number;
  feature: string;
  signed_residual: number | null;
  absolute_error: number | null;
  contribution_percent: number | null;
}

export interface ReplayFeatureRow {
  feature: string;
  actual: number;
  predicted: number;
  residual: number;
  absolute_error: number;
}

export interface ReplayState {
  source: Source;
  availability: Availability;
  run_id: string;
  state: {
    step: number;
    index: number;
    total_states: number;
    timestamp: string | null;
    capture_day: string | null;
    segment_id: string;
    dataset_label: string;
    residual: number | null;
    priority: Priority;
    threshold: number | null;
    sequence_length: number;
    feature_count: number;
  } | null;
  features: ReplayFeatureState[];
  available_features: string[];
  contributors: ReplayContributor[];
  feature_table?: ReplayFeatureRow[];
  limitations: string[];
}

export interface ReplaySeriesPoint {
  index: number;
  step: number;
  timestamp: string | null;
  label: string;
  actual: number;
  predicted: number;
  residual: number;
  absolute_error: number;
  state_mae: number;
}

export interface ReplaySeries {
  source: Source;
  availability: Availability;
  run_id: string;
  feature: string;
  features: string[];
  points: ReplaySeriesPoint[];
  stats: {
    samples: number;
    mae: number;
    median_absolute_error: number;
    max_absolute_error: number;
    mean_signed_residual: number;
  };
  top_errors?: Array<{
    step: number;
    index: number;
    timestamp: string | null;
    label: string;
    actual: number;
    predicted: number;
    residual: number;
    absolute_error: number;
  }>;
  window: { start: number; end: number; total_states: number };
  limitations: string[];
}

export interface ReplayContributorSummary {
  feature: string;
  appearances: number;
  appearance_percent: number;
}

export interface Health {
  status: "ok" | "degraded" | "failed";
  version: string;
  uptime_seconds: number;
  checks: HealthCheck[];
  runtime: Record<string, unknown>;
}

export interface Capabilities {
  capabilities: Array<{
    id: string;
    label: string;
    state: "implemented" | "conditional" | "planned";
    detail: string;
  }>;
  limitations: string[];
}
