import { Link } from "react-router-dom";

import { api } from "../api/client";
import { ActivityChart } from "../components/Charts";
import { MetricGrid } from "../components/Metric";
import { PageHeader, Panel } from "../components/Panel";
import { CaseStatusPill, Pill, PriorityPill } from "../components/Status";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { decimal, number, percent, timestamp } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function DashboardPage() {
  const { data, error, loading, reload } = useApi(() => api.dashboard(), "dashboard");

  if (loading && !data) {
    return (
      <Panel title="Loading SOC overview">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!data) return <EmptyState title="No dashboard data returned." />;

  const postureTone =
    data.status.posture === "review_required"
      ? "danger"
      : data.status.posture === "nominal"
        ? "info"
        : "warning";
  const detection = data.prediction_quality.detection;
  const testDetection = detection.test;

  return (
    <>
      <PageHeader
        title="SOC Overview"
        subtitle="Network-state forecasting posture, residual review queue and model/dataset health. Every figure is derived from recorded research artifacts."
        actions={
          <>
            <button type="button" className="btn" onClick={reload}>
              Refresh
            </button>
            <Link className="btn primary" to="/predictions">
              Open forecast
            </Link>
          </>
        }
      />

      <div className={`notice ${postureTone}`}>
        <span className="notice-icon">◉</span>
        <div>
          <strong>{data.status.headline}</strong>
          <div>{data.status.detail}</div>
          <div className="small muted" style={{ marginTop: 4 }}>
            Generated {timestamp(data.status.generated_at)} · frozen review threshold{" "}
            {decimal(data.status.threshold, 4)} ({data.status.threshold_source})
          </div>
        </div>
      </div>

      <AvailabilityNotice availability={data.availability} title="Row-level telemetry is unavailable" />

      <MetricGrid metrics={data.metrics} />

      <div className="grid sidebar-right">
        <Panel
          title="Residual activity"
          hint="Mean absolute next-state prediction error per time bucket, with the attack-labelled share of each bucket."
          actions={<SourceLine source={data.source} />}
        >
          {data.activity_series.length ? (
            <ActivityChart data={data.activity_series} />
          ) : (
            <EmptyState
              title="No activity series"
              detail="An aggregated state table is required to chart residual activity."
            />
          )}
        </Panel>

        <div className="stack">
          <Panel title="Model health" hint={data.model_health.name}>
            <dl className="kv">
              <dt>Run</dt>
              <dd className="mono">{data.model_health.run_id}</dd>
              <dt>Architecture</dt>
              <dd>
                {data.model_health.architecture.input_features} → hidden{" "}
                {data.model_health.architecture.hidden_size} ·{" "}
                {data.model_health.architecture.num_layers} LSTM layer
              </dd>
              <dt>Sequence</dt>
              <dd>{data.model_health.architecture.sequence_length} states (1 s)</dd>
              <dt>Best epoch</dt>
              <dd>{data.model_health.training.best_epoch ?? "n/a"}</dd>
              <dt>Validation Huber</dt>
              <dd>{decimal(data.model_health.training.best_validation_huber)}</dd>
              <dt>Checkpoint</dt>
              <dd>
                {data.model_health.checkpoint.present ? (
                  <Pill tone="good" dot>
                    present
                  </Pill>
                ) : (
                  <Pill tone="warning" dot>
                    not in checkout
                  </Pill>
                )}
              </dd>
            </dl>
            <div className="divider" />
            <div className="inline">
              <Link className="btn small" to="/models">
                Model registry
              </Link>
              <Link className="btn small" to="/explainability">
                Explanations
              </Link>
            </div>
          </Panel>

          <Panel title="Dataset readiness" hint={data.dataset_status.dataset_directory}>
            <div className="small muted" style={{ marginBottom: 8 }}>
              {data.dataset_status.artifacts.filter((artifact) => artifact.present).length} of{" "}
              {data.dataset_status.artifacts.length} frozen preprocessing artifacts present
            </div>
            <div className="tag-row">
              {data.dataset_status.artifacts.map((artifact) => (
                <Pill
                  key={artifact.key}
                  tone={artifact.present ? "good" : "neutral"}
                  title={artifact.path}
                >
                  {artifact.label}
                </Pill>
              ))}
            </div>
            <div className="divider" style={{ margin: "12px 0" }} />
            <Link className="btn small" to="/datasets">
              Dataset management
            </Link>
          </Panel>
        </div>
      </div>

      <div className="grid two">
        <Panel title="Label distribution" hint="Dataset reference annotations for the replay window.">
          {data.label_distribution.length ? (
            <div className="stack" style={{ gap: 8 }}>
              {data.label_distribution.map((entry) => (
                <div key={entry.name} className="bar-row">
                  <span className="bar-label">{entry.name}</span>
                  <span className="bar-track">
                    <span
                      className="bar-fill"
                      style={{ width: `${Math.max(1, entry.share * 100)}%` }}
                    />
                  </span>
                  <span className="bar-value">
                    {number(entry.count)} · {percent(entry.share * 100, 1)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              title="No label distribution"
              detail="Requires the replay or explainable triage table."
            />
          )}
        </Panel>

        <Panel
          title="Recorded detection behaviour"
          hint="Frozen residual threshold applied to the evaluation partitions."
        >
          {testDetection ? (
            <dl className="kv">
              <dt>Test true positives</dt>
              <dd>{number(testDetection.TP ?? null)}</dd>
              <dt>Test false negatives</dt>
              <dd>{number(testDetection.FN ?? null)}</dd>
              <dt>Precision / recall</dt>
              <dd>
                {decimal(testDetection.precision ?? null, 3)} /{" "}
                {decimal(testDetection.recall ?? null, 3)}
              </dd>
              <dt>Specificity</dt>
              <dd>{decimal(testDetection.specificity ?? null, 3)}</dd>
              <dt>Interpretation</dt>
              <dd>{detection.warning}</dd>
            </dl>
          ) : (
            <EmptyState
              title="No detection metrics recorded"
              detail="The run does not include a test anomaly evaluation artifact."
            />
          )}
        </Panel>
      </div>

      <Panel
        title="Recent review candidates"
        hint={
          data.event_summary.available
            ? `${data.event_summary.candidates} candidate(s) at threshold (${data.event_summary.selection})`
            : "Review queue unavailable"
        }
        actions={
          <Link className="btn small" to="/threats">
            Open triage
          </Link>
        }
        flush
      >
        {data.recent_events.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Timestamp</th>
                  <th>Dataset label</th>
                  <th className="num">Residual</th>
                  <th>Priority</th>
                  <th>Case status</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_events.map((event) => (
                  <tr key={event.event_id}>
                    <td className="mono">{event.step ?? "—"}</td>
                    <td>{timestamp(event.timestamp)}</td>
                    <td>{event.dataset_label ?? "—"}</td>
                    <td className="num">{decimal(event.residual_score)}</td>
                    <td>
                      <PriorityPill priority={event.priority} />
                    </td>
                    <td>
                      <CaseStatusPill status={event.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={
              data.event_summary.available
                ? "No observation crossed the frozen review threshold"
                : "No review queue in this deployment"
            }
            detail={
              data.event_summary.reason ??
              "A clear queue is not proof of benign activity: the recorded evaluation found 0 of the labelled Bot observations at this threshold."
            }
          />
        )}
      </Panel>

      <Panel title="Interpretation limits">
        <Limitations items={data.limitations} />
      </Panel>
    </>
  );
}
