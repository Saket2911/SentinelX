/* Typed client for the SentinelX API.
   Only shapes the UI actually consumes are typed. */

export type Provenance = "recorded_artifact" | "derived" | "unavailable" | "live_telemetry";

export interface ArtifactStatus {
  id: string;
  label: string;
  path: string;
  present: boolean;
  required_for: string;
  provenance_kind: "json" | "text" | "binary";
}

export interface ModelStatus {
  id: string;
  name: string;
  input_features: number;
  sequence_length: number;
  status: string;
  checkpoint_present: boolean;
  checkpoint_path: string;
  loadable: boolean;
  note: string;
}

export interface Health {
  status: "ok" | "degraded";
  service: string;
  version: string;
  data_mode: string;
  inference_available: boolean;
  live_telemetry_available: boolean;
  artifacts: ArtifactStatus[];
  models: ModelStatus[];
  warnings: string[];
}

export interface MetricCard {
  key: string;
  label: string;
  value: number | string | null;
  unit?: string | null;
  provenance: string;
  source?: string | null;
  hint?: string | null;
}

export interface DetectionAssessment {
  validated_detector: boolean;
  headline: string;
  threshold: number;
  threshold_source: string;
  provenance: string;
  verdict: string;
  source: string;
  test_metrics?: Record<string, number>;
  validation_metrics?: Record<string, number>;
}

export interface Dashboard {
  active_model: string;
  security_posture: { level: string; label: string; detail: string; provenance: string };
  metrics: MetricCard[];
  label_distribution: { partition: string; label: string; count: number; provenance: string }[];
  detection_assessment: DetectionAssessment;
  recent_signals: Record<string, unknown>[];
  system_health: {
    inference_available: boolean;
    live_telemetry: boolean;
    data_mode: string;
    settings: Record<string, unknown>;
  };
  limitations: string[];
}

export interface PartitionStats {
  total: number;
  benign: number;
  no_flows: number;
  labelled_attack: number;
  attack_share_percent: number;
  distinct_attack_labels: number;
}

export interface NetworkOverview {
  available: boolean;
  provenance: string;
  source?: string;
  partitions: Record<string, PartitionStats>;
  labels: { partition: string; label: string; count: number }[];
  flow_level_detail: { provenance: string; reason: string };
}

export interface MultiStepRow {
  horizon_seconds: number;
  target_label: string;
  valid_origins: number;
  lstm_huber: number;
  persistence_huber: number;
  lstm_mae: number;
  persistence_mae: number;
  mae_improvement_percent: number;
  lstm_rmse: number;
  persistence_rmse: number;
  rmse_improvement_percent: number;
}

export interface Predictions {
  one_step: {
    model: Record<string, number> | null;
    persistence: Record<string, number> | null;
    improvement_percent: Record<string, number> | null;
    samples: number | null;
    provenance: string;
    source: string;
  };
  multi_step: {
    rows: MultiStepRow[];
    horizons: number[];
    provenance: string;
    source: string;
    note: string;
  };
  error_distribution: {
    quantiles: Record<string, Record<string, number>> | null;
    samples: number | null;
    features: number | null;
    provenance: string;
    source: string;
    note: string;
  };
  v2_reference: { metrics: Record<string, number> | null; provenance: string; note: string };
  live_forecast: { provenance: string; reason: string; requires: string[] };
}

export interface TriageAggregate {
  split: string;
  total_observations: number;
  review_required: number;
  within_calibrated_range: number;
  review_rate_percent: number;
  frozen_threshold: number;
  decision_input: string;
  labels_used: boolean;
}

export interface Triage {
  engine: string;
  status: string;
  aggregates: TriageAggregate[];
  provenance: string;
  source?: string;
  calibration: {
    threshold: number;
    percentile: number;
    source_description: string;
    recalibrated_on_test: boolean;
    user_threshold_override: number;
    note: string;
  };
  decision_policy: {
    input: string;
    ground_truth_labels_used: boolean;
    threshold_based_attack_decision: boolean;
  };
  integrity: Record<string, unknown> | null;
  per_event_rows: { provenance: string; reason: string; requires?: string[] };
  limitations: string[];
  recommended_actions: string[];
}

export interface FeatureFamily {
  horizon_seconds: number;
  feature_family: string;
  feature_count: number;
  lstm_better_count: number;
  persistence_better_count: number;
  equal_count: number;
  lstm_mean_mae: number;
  persistence_mean_mae: number;
  mae_improvement_percent: number;
  lstm_mean_rmse: number;
  persistence_mean_rmse: number;
  rmse_improvement_percent: number;
}

export interface Explainability {
  method: string | null;
  interpretation: string | null;
  experiments: { name: string; summary: Record<string, unknown>; provenance: string }[];
  output_features: string[];
  feature_families: FeatureFamily[];
  error_quantiles: Record<string, { observations: number; mean_mae_model_space: number }> | null;
  local_explanations: Record<string, unknown>;
  provenance: string;
  shap_values: { provenance: string; reason: string };
  caveats: string[];
}

export interface Timeline {
  available: boolean;
  provenance: string;
  source?: string;
  window: { start: string; end: string } | null;
  total_states: number | null;
  total_runs: number | null;
  runs: {
    label: string;
    run_count: number;
    mean_seconds: number;
    median_seconds: number;
    min_seconds: number;
    max_seconds: number;
  }[];
  histogram: {
    label: string;
    duration_category: string;
    run_count: number;
    states_in_runs: number;
  }[];
  transition_analysis_available: boolean;
  continuity_analysis_available: boolean;
  note: string;
  reason?: string;
}

export interface Datasets {
  sources: {
    id: string;
    name: string;
    role: string;
    description: string;
    files: string[];
    original_rows?: number;
    original_columns?: number;
    present_in_checkout: boolean;
  }[];
  partitions: Record<string, Record<string, number>>;
  feature_version: {
    v3_features: number | null;
    sequence_length: number | null;
    training_samples: number | null;
    validation_samples: number | null;
    provenance: string;
  };
  preprocessing_contract: string[];
  upload: { supported: boolean; reason: string };
  raw_files_present: boolean;
  provenance: string;
}

export interface ModelVersion {
  id: string;
  name: string;
  status: string;
  is_active: boolean;
  input_features: number;
  sequence_length: number;
  note: string;
  checkpoint_present: boolean;
  config: Record<string, unknown> | null;
  training: Record<string, unknown> | null;
  test_evaluation: { lstm_metrics?: Record<string, number>; _source?: string } | null;
}

export interface ModelsResponse {
  active_model: string;
  versions: ModelVersion[];
  comparison_policy: string;
  provenance: string;
}

export interface ReportArtifact {
  id: string;
  title: string;
  path: string;
  present: boolean;
  kind: "json" | "text";
}

export interface GeneratedReport {
  id: string;
  title: string;
  generated_at: string;
  markdown: string;
  sections: unknown[];
  provenance: string;
}

export interface Settings {
  anomaly_review_threshold: number;
  review_percentile: number;
  default_split: "train" | "validation" | "test";
  table_page_size: number;
  show_provenance_badges: boolean;
  telemetry_mode: "offline_dataset" | "live";
}

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => req<Health>("/health"),
  dashboard: () => req<Dashboard>("/dashboard"),
  network: () => req<NetworkOverview>("/network/overview"),
  predictions: () => req<Predictions>("/predictions"),
  multiStep: (horizon?: number, label?: string) => {
    const p = new URLSearchParams();
    if (horizon !== undefined) p.set("horizon", String(horizon));
    if (label) p.set("label", label);
    const q = p.toString();
    return req<Predictions["multi_step"] & { labels: string[]; filtered_count: number }>(
      `/predictions/multi-step${q ? `?${q}` : ""}`,
    );
  },
  triage: () => req<Triage>("/triage"),
  explainability: () => req<Explainability>("/explainability"),
  timeline: () => req<Timeline>("/timeline"),
  datasets: () => req<Datasets>("/datasets"),
  models: () => req<ModelsResponse>("/models"),
  activateModel: (id: string) =>
    req<{ active_model: string; status: string; inference_available: boolean; warning: string | null }>(
      `/models/${id}/activate`,
      { method: "POST" },
    ),
  reportCatalog: () => req<{ artifacts: ReportArtifact[] }>("/reports/catalog"),
  reportArtifact: (id: string) =>
    req<{ id: string; kind: string; content: unknown }>(`/reports/artifact/${id}`),
  generateReport: () => req<GeneratedReport>("/reports/generate", { method: "POST" }),
  generatedReports: () => req<{ reports: { id: string; bytes: number }[] }>("/reports/generated"),
  settings: () => req<{ settings: Settings; defaults: Settings; note: string }>("/settings"),
  saveSettings: (s: Settings) =>
    req<{ settings: Settings; status: string }>("/settings", {
      method: "PUT",
      body: JSON.stringify(s),
    }),
};
