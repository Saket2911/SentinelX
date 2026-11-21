import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import type { EventStatus, PlaybackAlert } from "../api/types";
import { Drawer } from "./Overlay";
import { CaseStatusPill, Pill } from "./Status";
import { EmptyState, ErrorState, Limitations, Loading } from "./States";
import { useToast } from "./Toast";
import { decimal, timestamp, titleCase } from "../lib/format";
import { useApi } from "../hooks/useApi";

const STATUSES: EventStatus[] = ["new", "investigating", "escalated", "benign", "closed"];

export function severityTone(severity: string): "critical" | "warning" | "info" {
  if (severity === "critical") return "critical";
  if (severity === "high") return "warning";
  return "info";
}

export function SeverityPill({ severity }: { severity: string }) {
  return (
    <Pill tone={severityTone(severity)} dot>
      {titleCase(severity)}
    </Pill>
  );
}

/** Evidence for one generated playback alert, including investigation workflow. */
export function PlaybackAlertDrawer({
  alertId,
  initial,
  onClose,
  onChanged,
}: {
  alertId: string;
  initial?: PlaybackAlert | null;
  onClose: () => void;
  onChanged?: () => void;
}) {
  const toast = useToast();
  const detail = useApi(() => api.playbackAlert(alertId), `playback-alert-${alertId}`);
  const alert = detail.data?.alert ?? initial ?? null;
  const [status, setStatus] = useState<EventStatus>(initial?.status ?? "new");
  const [assignee, setAssignee] = useState(initial?.assignee ?? "");
  const [notes, setNotes] = useState(initial?.notes ?? "");
  const [initialised, setInitialised] = useState(false);
  const [saving, setSaving] = useState(false);

  if (detail.data && !initialised) {
    setStatus(detail.data.alert.status);
    setAssignee(detail.data.alert.assignee ?? "");
    setNotes(detail.data.alert.notes ?? "");
    setInitialised(true);
  }

  async function save() {
    setSaving(true);
    try {
      await api.updatePlaybackAlert(alertId, {
        status,
        assignee: assignee || null,
        notes: notes || null,
        actor: "analyst",
      });
      detail.reload();
      onChanged?.();
      toast.push({ title: "Alert updated", tone: "success" });
    } catch (cause) {
      toast.push({
        title: "Alert update failed",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Drawer
      title={alert ? `Prediction alert ${alert.alert_id.slice(0, 18)}…` : "Loading alert"}
      subtitle={
        alert
          ? `${timestamp(alert.timestamp)} · step ${alert.target_step} · ${alert.model_version}`
          : undefined
      }
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn" onClick={onClose} disabled={saving}>
            Close
          </button>
          <button type="button" className="btn primary" onClick={save} disabled={saving || !alert}>
            {saving ? "Saving…" : "Save investigation"}
          </button>
        </>
      }
    >
      {detail.loading && !alert ? <Loading rows={5} /> : null}
      {detail.error ? <ErrorState message={detail.error} onRetry={detail.reload} /> : null}
      {alert ? (
        <>
          <div className="inline">
            <SeverityPill severity={alert.severity} />
            <CaseStatusPill status={alert.status} />
            <Pill tone="neutral">{alert.dataset_label ?? "unlabelled"}</Pill>
            <Pill tone="warning" title="Anomaly means unusual prediction error, not a confirmed attack">
              review signal, not an attack
            </Pill>
          </div>

          <dl className="kv">
            <dt>Anomaly score</dt>
            <dd>
              {decimal(alert.anomaly_score, 4)} / threshold {decimal(alert.threshold, 4)}
              {alert.ratio !== null ? ` (${alert.ratio.toFixed(2)}×)` : ""}
            </dd>
            <dt>Trigger</dt>
            <dd>{alert.trigger_reason}</dd>
            <dt>Target step</dt>
            <dd className="mono">
              {alert.target_step} (index {alert.target_index})
            </dd>
            <dt>Timestamp</dt>
            <dd>{timestamp(alert.timestamp)}</dd>
            <dt>Segment / capture day</dt>
            <dd>
              {alert.segment_id ?? "—"} · {alert.capture_day ?? "—"}
            </dd>
            <dt>Prediction source</dt>
            <dd>{alert.prediction_source.replace(/_/g, " ")}</dd>
            <dt>Created</dt>
            <dd>{timestamp(alert.created_at)}</dd>
            <dt>Last update</dt>
            <dd>{timestamp(alert.updated_at)}</dd>
          </dl>

          <div>
            <span className="section-label">Predicted vs actual state features</span>
            <p className="small muted" style={{ marginTop: 4 }}>
              Top deviations in standardized model feature space. Ranking for investigation, not
              attack evidence.
            </p>
            <div className="table-wrap" style={{ marginTop: 6 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th className="num">Predicted</th>
                    <th className="num">Actual</th>
                    <th className="num">Residual</th>
                    <th className="num">Share</th>
                  </tr>
                </thead>
                <tbody>
                  {alert.features.map((feature) => (
                    <tr key={feature.feature}>
                      <td className="mono">{feature.feature}</td>
                      <td className="num">{decimal(feature.predicted)}</td>
                      <td className="num">{decimal(feature.actual)}</td>
                      <td className="num">{decimal(feature.signed_residual)}</td>
                      <td className="num">{feature.share_percent.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <span className="section-label">Recorded references</span>
            <dl className="kv" style={{ marginTop: 6 }}>
              <dt>Actual state</dt>
              <dd className="mono small">{alert.actual_ref ?? "—"}</dd>
              <dt>Predicted state</dt>
              <dd className="mono small">{alert.predicted_ref ?? "—"}</dd>
              <dt>Input window</dt>
              <dd className="mono small">{alert.context_ref ?? "—"}</dd>
              <dt>Dataset / model</dt>
              <dd className="mono small">
                {alert.dataset_id} · {alert.model_version}
              </dd>
            </dl>
            <div className="inline" style={{ marginTop: 8 }}>
              <Link className="btn small" to={`/replay?step=${alert.target_step}`}>
                Open replay state
              </Link>
              <Link className="btn small" to="/explainability">
                Open SHAP explainability
              </Link>
            </div>
          </div>

          {detail.data?.explainability?.shap ? (
            <div>
              <span className="section-label">SHAP context (supporting evidence only)</span>
              <p className="small muted" style={{ marginTop: 4 }}>
                {detail.data.explainability.shap.note}
              </p>
              {detail.data.explainability.shap.matches?.length ? (
                <div className="tag-row" style={{ marginTop: 6 }}>
                  {detail.data.explainability.shap.matches.map((match) => (
                    <Pill key={match.feature} tone="accent">
                      {match.feature} · |SHAP| {match.mean_absolute_shap.toFixed(4)}
                    </Pill>
                  ))}
                </div>
              ) : (
                <p className="small muted" style={{ marginTop: 6 }}>
                  {detail.data.explainability.shap.reason ??
                    "None of this alert's residual features appear in the recorded SHAP table."}
                </p>
              )}
            </div>
          ) : null}

          <div className="divider" />

          <div className="stack">
            <span className="section-label">Investigation workflow</span>
            <div className="grid two" style={{ gap: 10 }}>
              <label className="field">
                <span className="field-label">Status</span>
                <select
                  className="select"
                  value={status}
                  onChange={(event) => setStatus(event.target.value as EventStatus)}
                >
                  {STATUSES.map((value) => (
                    <option key={value} value={value}>
                      {titleCase(value)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">Assignee</span>
                <input
                  className="input"
                  value={assignee}
                  onChange={(event) => setAssignee(event.target.value)}
                  placeholder="soc-tier-1"
                />
              </label>
            </div>
            <label className="field">
              <span className="field-label">Analyst notes</span>
              <textarea
                className="textarea"
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="Corroborating evidence, decision rationale…"
              />
            </label>
          </div>

          {detail.data?.history.length ? (
            <div>
              <span className="section-label">Audit trail</span>
              <div className="table-wrap" style={{ marginTop: 6 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Actor</th>
                      <th>Field</th>
                      <th>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.data.history.map((entry, index) => (
                      <tr key={`${entry.created_at}-${index}`}>
                        <td>{timestamp(entry.created_at)}</td>
                        <td>{entry.actor}</td>
                        <td>{entry.action}</td>
                        <td className="mono">
                          {entry.from_value ?? "—"} → {entry.to_value ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          <Limitations items={detail.data?.limitations ?? alert.limitations} />
        </>
      ) : null}
    </Drawer>
  );
}

export function PlaybackAlertList({
  alerts,
  onOpen,
  emptyMessage,
}: {
  alerts: PlaybackAlert[];
  onOpen: (alert: PlaybackAlert) => void;
  emptyMessage: string;
}) {
  if (!alerts.length) {
    return <EmptyState title={emptyMessage} detail="No alert has been generated for this selection." />;
  }
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Step</th>
            <th>Timestamp</th>
            <th className="num">Score</th>
            <th>Severity</th>
            <th>Trigger</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => (
            <tr key={alert.alert_id}>
              <td className="mono">{alert.target_step}</td>
              <td>{timestamp(alert.timestamp)}</td>
              <td className="num">{decimal(alert.anomaly_score, 4)}</td>
              <td>
                <SeverityPill severity={alert.severity} />
              </td>
              <td className="small">{alert.trigger_reason}</td>
              <td>
                <CaseStatusPill status={alert.status} />
              </td>
              <td>
                <button type="button" className="btn small" onClick={() => onOpen(alert)}>
                  Details
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

