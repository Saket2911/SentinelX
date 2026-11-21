import type {
  Capabilities,
  Dashboard,
  DatasetOverview,
  DatasetValidation,
  EventDetail,
  EventList,
  EventStatus,
  Explainability,
  Health,
  ModelDetail,
  ModelSummary,
  NetworkOverview,
  PredictionSeries,
  PredictionSummary,
  ReportSummary,
  SettingsResponse,
  Timeline,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, message: string, detail: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: {
      Accept: "application/json",
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as {
        detail?: unknown;
        error?: string;
      };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail && typeof payload.detail === "object") {
        const nested = payload.detail as { detail?: string; error?: string };
        detail = nested.detail ?? nested.error ?? detail;
      } else if (payload.error) {
        detail = payload.error;
      }
    } catch {
      /* keep the status message */
    }
    throw new ApiError(response.status, detail, detail);
  }
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    search.set(key, String(value));
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: () => request<Health>("/health"),
  capabilities: () => request<Capabilities>("/capabilities"),
  runs: () => request<{ active_run_id: string; runs: string[] }>("/runs"),
  dashboard: (runId?: string) => request<Dashboard>(`/dashboard${query({ run_id: runId })}`),

  network: (runId?: string) => request<NetworkOverview>(`/network${query({ run_id: runId })}`),

  predictionSummary: (runId?: string) =>
    request<PredictionSummary>(`/predictions/summary${query({ run_id: runId })}`),
  predictionSeries: (
    params: { feature?: string; start?: number; limit?: number; run_id?: string } = {},
  ) => request<PredictionSeries>(`/predictions/series${query(params)}`),
  topResiduals: (limit = 10, runId?: string) =>
    request<{
      run_id: string;
      residuals: Array<{
        step: number | null;
        timestamp: string | null;
        label: string | null;
        residual: number;
        priority: string;
      }>;
      note: string;
    }>(`/predictions/top-residuals${query({ limit, run_id: runId })}`),

  events: (
    params: {
      status?: string;
      priority?: string;
      label?: string;
      segment?: string;
      search?: string;
      page?: number;
      page_size?: number;
      include_top?: boolean;
      run_id?: string;
    } = {},
  ) => request<EventList>(`/events${query(params)}`),
  event: (eventId: string) => request<EventDetail>(`/events/${encodeURIComponent(eventId)}`),
  updateEvent: (
    eventId: string,
    payload: { status?: EventStatus; assignee?: string | null; notes?: string | null; actor?: string },
  ) =>
    request<EventDetail>(`/events/${encodeURIComponent(eventId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  timeline: (runId?: string) => request<Timeline>(`/timeline${query({ run_id: runId })}`),

  explainability: (
    params: { label?: string; output?: string; sample_index?: number; top_n?: number; run_id?: string } = {},
  ) => request<Explainability>(`/explainability${query(params)}`),

  models: () =>
    request<{
      active_run_id: string;
      models: ModelSummary[];
      limitations: string[];
      generated_at: string;
    }>("/models"),
  model: (runId: string) => request<ModelDetail>(`/models/${encodeURIComponent(runId)}`),
  compareModels: (left: string, right: string) =>
    request<{
      left: ModelSummary;
      right: ModelSummary;
      metrics: Array<{
        id: string;
        label: string;
        unit: string;
        left: number | null;
        right: number | null;
        delta: number | null;
        higher_is_better: boolean;
      }>;
      warnings: string[];
    }>(`/models/compare${query({ left, right })}`),
  activateModel: (runId: string, reason?: string) =>
    request<{
      active_run_id: string;
      previous_run_id: string | null;
      activated_at: string;
      warnings: string[];
    }>(`/models/${encodeURIComponent(runId)}/activate`, {
      method: "POST",
      body: JSON.stringify({ run_id: runId, confirm: true, reason }),
    }),
  inferenceStatus: (runId: string) =>
    request<{
      available: boolean;
      model_space_available: boolean;
      raw_state_available: boolean;
      checkpoint: string | null;
      torch_available: boolean;
      reasons: string[];
    }>(`/models/${encodeURIComponent(runId)}/inference-status`),

  datasets: () => request<DatasetOverview>("/datasets"),
  validateDataset: (datasetId: string) =>
    request<DatasetValidation>(`/datasets/validate${query({ dataset_id: datasetId })}`),
  uploadDataset: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ dataset_id: string; path: string; validation: DatasetValidation; note: string }>(
      "/datasets/upload",
      { method: "POST", body },
    );
  },

  reports: () => request<{ reports: ReportSummary[]; generated_at: string }>("/reports"),
  createReport: (payload: {
    title: string;
    sections: string[];
    run_id?: string;
    author: string;
  }) =>
    request<ReportSummary>("/reports", { method: "POST", body: JSON.stringify(payload) }),
  reportContent: (reportId: string) =>
    request<{ metadata: ReportSummary; markdown: string }>(`/reports/${encodeURIComponent(reportId)}`),
  reportDownloadUrl: (reportId: string, format: "md" | "json" = "md") =>
    `${BASE}/reports/${encodeURIComponent(reportId)}/download?format=${format}`,

  settings: () => request<SettingsResponse>("/settings"),
  updateSettings: (payload: Partial<SettingsResponse["settings"]>) =>
    request<SettingsResponse>("/settings", { method: "PUT", body: JSON.stringify(payload) }),
};
