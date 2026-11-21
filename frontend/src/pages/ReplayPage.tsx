import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api } from "../api/client";
import type { PlaybackAlert, PlaybackEvaluation, ReplayFeatureRow } from "../api/types";
import { PlaybackChart, ThresholdTrendChart } from "../components/Charts";
import { DataTable, type Column } from "../components/DataTable";
import { PageHeader, Panel } from "../components/Panel";
import { PlaybackAlertDrawer, PlaybackAlertList, SeverityPill } from "../components/PlaybackAlert";
import { Pill, PriorityPill } from "../components/Status";
import { useToast } from "../components/Toast";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { decimal, number, timestamp } from "../lib/format";
import { useApi } from "../hooks/useApi";

const WINDOW = 30;
const SPEEDS = [1, 2, 4, 8];

export function ReplayPage() {
  const toast = useToast();
  const [params] = useSearchParams();
  const initialStep = Math.max(1, Number(params.get("step")) || 1);
  const [step, setStep] = useState(initialStep);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [feature, setFeature] = useState("");
  const [loadAll, setLoadAll] = useState(false);
  const [tableQuery, setTableQuery] = useState("");
  const [evaluation, setEvaluation] = useState<PlaybackEvaluation | null>(null);
  const [evaluateError, setEvaluateError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [sessionAlerts, setSessionAlerts] = useState<PlaybackAlert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const seenAlerts = useRef<Set<string>>(new Set());
  const lastKey = useRef<string>("");

  const meta = useApi(() => api.replayMeta(), "replay-meta");
  const maxStep = meta.data?.step_max ?? 1;
  const minStep = meta.data?.step_min ?? 1;

  useEffect(() => {
    if (!meta.data) return;
    const key = `${meta.data.run_id}:${initialStep}`;
    if (lastKey.current !== key) {
      lastKey.current = key;
      setPlaying(false);
      setStep(Math.min(Math.max(initialStep, minStep), maxStep));
    }
  }, [meta.data, initialStep, minStep, maxStep]);

  const stateData = useApi(
    () => api.replayState({ step, features: feature ? [feature] : undefined, include_all: loadAll }),
    `replay-state-${step}-${feature}-${loadAll}`,
  );
  const activeFeature = feature || stateData.data?.features[0]?.feature || "Flow Count";
  const local = useApi(
    () =>
      api.replaySeries({
        feature: activeFeature,
        start: Math.max(0, step - 1 - WINDOW),
        limit: WINDOW * 2 + 1,
      }),
    `replay-local-${step}-${activeFeature}`,
  );
  const contributors = useApi(() => api.replayContributors(12), "replay-contributors");
  const persistedAlerts = useApi(
    () => api.playbackAlerts({ run_id: meta.data?.run_id, limit: 500 }),
    `replay-alerts-${meta.data?.run_id ?? "pending"}`,
  );

  // One debounced evaluation per settled playback position. The backend is
  // idempotent, so pausing, seeking and replaying cannot duplicate an alert.
  useEffect(() => {
    if (!meta.data?.availability.available) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setEvaluating(true);
      api
        .evaluatePlayback(step, meta.data?.run_id)
        .then((result) => {
          if (cancelled) return;
          setEvaluation(result);
          setEvaluateError(null);
          const alert = result.alert;
          if (alert) {
            if (!seenAlerts.current.has(alert.alert_id)) {
              seenAlerts.current.add(alert.alert_id);
              setSessionAlerts((current) =>
                [alert, ...current.filter((entry) => entry.alert_id !== alert.alert_id)].slice(
                  0,
                  50,
                ),
              );
              if (!result.deduplicated) {
                persistedAlerts.reload();
                toast.push({
                  title: `${alert.severity.toUpperCase()} playback alert · step ${alert.target_step}`,
                  message: `Anomaly score ${alert.anomaly_score.toFixed(3)} at ${timestamp(alert.timestamp)}. Review signal only.`,
                  tone: alert.severity === "critical" ? "error" : "info",
                });
              }
            }
          }
        })
        .catch((cause: unknown) => {
          if (cancelled) return;
          setEvaluateError(cause instanceof Error ? cause.message : "Evaluation failed");
        })
        .finally(() => {
          if (!cancelled) setEvaluating(false);
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, meta.data?.run_id, meta.data?.availability.available]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => Math.min(current + 1, maxStep));
    }, Math.round(900 / speed));
    return () => window.clearInterval(timer);
  }, [playing, speed, maxStep]);

  // Stop at the last state from an effect, never from inside a state updater.
  useEffect(() => {
    if (playing && step >= maxStep) setPlaying(false);
  }, [playing, step, maxStep]);

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (step >= maxStep) setStep(minStep);
    setPlaying(true);
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlay();
      } else if (event.key === "ArrowRight") {
        setPlaying(false);
        setStep((value) => Math.min(maxStep, value + 1));
      } else if (event.key === "ArrowLeft") {
        setPlaying(false);
        setStep((value) => Math.max(minStep, value - 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // togglePlay closes over the current step/playing values.
  }, [maxStep, minStep, step, playing]);

  const features = meta.data?.features ?? [];
  const state = stateData.data?.state ?? null;
  const selected = stateData.data?.features[0];

  const localPoints = useMemo(
    () =>
      (local.data?.points ?? []).map((point) => ({
        index: point.step,
        residual: point.state_mae,
      })),
    [local.data],
  );

  const markerAlerts = useMemo(() => {
    const map = new Map<number, PlaybackAlert>();
    (persistedAlerts.data?.alerts ?? []).forEach((alert) => map.set(alert.target_step, alert));
    sessionAlerts.forEach((alert) => map.set(alert.target_step, alert));
    return [...map.values()].sort((left, right) => left.target_step - right.target_step);
  }, [persistedAlerts.data, sessionAlerts]);

  const tableRows = useMemo(() => {
    const rows = stateData.data?.feature_table ?? [];
    const query = tableQuery.trim().toLowerCase();
    if (!query) return rows;
    return rows.filter((row) => row.feature.toLowerCase().includes(query));
  }, [stateData.data, tableQuery]);

  const columns: Column<ReplayFeatureRow>[] = [
    { key: "feature", label: "Feature", sortValue: (row) => row.feature },
    {
      key: "actual",
      label: "Actual",
      align: "right",
      render: (row) => decimal(row.actual),
      sortValue: (row) => row.actual,
    },
    {
      key: "predicted",
      label: "Predicted",
      align: "right",
      render: (row) => decimal(row.predicted),
      sortValue: (row) => row.predicted,
    },
    {
      key: "residual",
      label: "Signed residual",
      align: "right",
      render: (row) => decimal(row.residual),
      sortValue: (row) => row.residual,
    },
    {
      key: "absolute_error",
      label: "Absolute error",
      align: "right",
      render: (row) => decimal(row.absolute_error),
      sortValue: (row) => row.absolute_error,
    },
  ];

  if (meta.loading && !meta.data) {
    return (
      <Panel title="Loading replay timeline">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (meta.error) return <ErrorState message={meta.error} onRetry={meta.reload} />;
  if (!meta.data) return <EmptyState title="Replay metadata unavailable." />;

  const available = meta.data.availability.available;
  const progress = maxStep > minStep ? ((step - minStep) / (maxStep - minStep)) * 100 : 100;

  return (
    <>
      <PageHeader
        title="Network State Playback"
        subtitle="Step through held-out historical network states and inspect the world model's next-state forecast at each timestamp."
        actions={
          <div className="toolbar">
            <span className={`pill ${playing ? "accent" : "neutral"}`} title="Keyboard: Space play/pause, ← → step">
              {playing ? "Playing" : "Paused"}
            </span>
            <button type="button" className="btn" onClick={() => { meta.reload(); stateData.reload(); local.reload(); }}>
              Refresh
            </button>
          </div>
        }
      />

      <AvailabilityNotice availability={meta.data.availability} title="Replay arrays unavailable" />

      {available ? (
        <>
          <Panel flush>
            <div className="state-banner">
              <div className="state-banner-main">
                <span className="state-eyebrow">Current replay position</span>
                <span className="state-timestamp">{timestamp(state?.timestamp) || "Timestamp unavailable"}</span>
                <div className="state-tags">
                  <span className="tag">Step {number(step)} / {number(maxStep)}</span>
                  <span className="tag">Segment {state?.segment_id ?? "—"}</span>
                  <span className="tag">Capture day {state?.capture_day ?? "—"}</span>
                  {state ? <Pill tone={state.dataset_label === "Benign" ? "good" : "warning"}>{state.dataset_label}</Pill> : null}
                  {state ? <PriorityPill priority={state.priority} /> : null}
                </div>
              </div>
              <div className="state-banner-side">
                <span className="state-eyebrow">Mean absolute residual</span>
                <span className="state-residual">{decimal(state?.residual, 4)}</span>
                <span className="small muted">Frozen threshold {decimal(state?.threshold, 4)}</span>
              </div>
            </div>

            <div className="transport">
              <div className="transport-buttons">
                <button type="button" className="btn ghost icon" title="First state" onClick={() => { setPlaying(false); setStep(minStep); }}>
                  ⏮
                </button>
                <button type="button" className="btn ghost icon" title="Previous state" onClick={() => { setPlaying(false); setStep((value) => Math.max(minStep, value - 1)); }}>
                  ←
                </button>
                <button
                  type="button"
                  className="btn primary play"
                  onClick={togglePlay}
                  title={
                    playing
                      ? "Pause (Space)"
                      : step >= maxStep
                        ? "Replay from the first state (Space)"
                        : "Play (Space)"
                  }
                >
                  {playing ? "❚❚ Pause" : "▶ Play"}
                </button>
                <button type="button" className="btn ghost icon" title="Next state" onClick={() => { setPlaying(false); setStep((value) => Math.min(maxStep, value + 1)); }}>
                  →
                </button>
                <button type="button" className="btn ghost icon" title="Last state" onClick={() => { setPlaying(false); setStep(maxStep); }}>
                  ⏭
                </button>
                <button type="button" className="btn" title="Return to the first state" onClick={() => { setPlaying(false); setStep(minStep); }}>
                  ↺ Reset
                </button>
                <label className="field inline-field">
                  <span className="field-label">Speed</span>
                  <select className="select" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
                    {SPEEDS.map((value) => (
                      <option key={value} value={value}>{value}×</option>
                    ))}
                  </select>
                </label>
              </div>

              <input
                className="timeline"
                type="range"
                min={minStep}
                max={maxStep}
                value={step}
                aria-label="Replay timeline"
                onChange={(event) => {
                  setPlaying(false);
                  setStep(Number(event.target.value));
                }}
              />

              <div className="progress-track" role="progressbar" aria-valuemin={minStep} aria-valuemax={maxStep} aria-valuenow={step}>
                <div className="progress-fill" style={{ width: `${progress}%` }} />
                {markerAlerts.map((alert) => (
                  <button
                    key={alert.alert_id}
                    type="button"
                    className={`timeline-marker ${alert.severity}`}
                    style={{
                      left: `${maxStep > minStep ? ((alert.target_step - minStep) / (maxStep - minStep)) * 100 : 0}%`,
                    }}
                    title={`${alert.severity} alert · step ${alert.target_step} · score ${alert.anomaly_score.toFixed(3)}`}
                    aria-label={`Alert at step ${alert.target_step}`}
                    onClick={() => {
                      setPlaying(false);
                      setStep(alert.target_step);
                    }}
                  />
                ))}
              </div>
              <div className="transport-meta">
                <span>{number(step)} / {number(maxStep)} states</span>
                <span className="muted small">Space play/pause · ← → step · playback is historical replay, not live capture</span>
              </div>
            </div>
          </Panel>

          <Panel
            title="Prediction alerting"
            hint="Each settled playback position is scored against the LSTM forecast and the calibrated rule set."
            actions={
              <span className="small muted">
                {evaluating
                  ? "Evaluating…"
                  : evaluation
                    ? `${evaluation.prediction.source.replace(/_/g, " ")} · ${evaluation.context.states_used}-state window`
                    : "Awaiting evaluation"}
              </span>
            }
          >
            {evaluateError ? (
              <div className="notice warning">
                <span className="notice-icon">!</span>
                <div>
                  <strong>Playback evaluation failed</strong>
                  <div>{evaluateError}</div>
                </div>
              </div>
            ) : null}
            {evaluation ? (
              <>
                {evaluation.alert ? (
                  <div className={`notice ${evaluation.alert.severity === "critical" ? "danger" : "warning"}`}>
                    <span className="notice-icon">◉</span>
                    <div style={{ flex: 1 }}>
                      <strong>
                        {evaluation.alert.severity.toUpperCase()} alert at step{" "}
                        {evaluation.alert.target_step} · {timestamp(evaluation.alert.timestamp)}
                      </strong>
                      <div>
                        Anomaly score {decimal(evaluation.alert.anomaly_score, 4)} against threshold{" "}
                        {decimal(evaluation.alert.threshold, 4)}. {evaluation.alert.trigger_reason}.
                        {evaluation.deduplicated
                          ? " This alert was generated earlier; it was not created again."
                          : " New alert persisted."}
                      </div>
                      <div className="inline" style={{ marginTop: 8 }}>
                        <SeverityPill severity={evaluation.alert.severity} />
                        <button
                          type="button"
                          className="btn small"
                          onClick={() => setSelectedAlert(evaluation.alert!.alert_id)}
                        >
                          Open alert details
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="notice info">
                    <span className="notice-icon">i</span>
                    <div>
                      <strong>No alert at step {evaluation.state.step}</strong>
                      <div>
                        Anomaly score {decimal(evaluation.score.anomaly_score, 4)} is below the threshold{" "}
                        {decimal(evaluation.state.threshold, 4)}; {evaluation.conditions.reasons.join(" ")}
                      </div>
                    </div>
                  </div>
                )}
                <div className="small muted" style={{ marginTop: 8 }}>
                  Rules: single state ≥ threshold, or {evaluation.conditions.rules.persistence_window}{" "}
                  consecutive states ≥{" "}
                  {evaluation.conditions.elevated_threshold
                    ? decimal(evaluation.conditions.elevated_threshold, 4)
                    : "80% of threshold"}{" "}
                  within one segment. Scores are mean absolute residuals in standardized model
                  feature space; dataset labels are never used to alert. Anomalies are review
                  signals, not attack confirmations.
                </div>
              </>
            ) : null}

            <div style={{ marginTop: 14 }}>
              <div className="inline" style={{ justifyContent: "space-between" }}>
                <span className="section-label">Session alert history</span>
                <span className="small muted">
                  {sessionAlerts.length} this session · {markerAlerts.length} persisted on the timeline
                </span>
              </div>
              <PlaybackAlertList
                alerts={sessionAlerts}
                onOpen={(alert) => setSelectedAlert(alert.alert_id)}
                emptyMessage="No alert generated in this playback session yet"
              />
            </div>
          </Panel>

          <div className="grid two">
            <div className="metric-card">
              <span className="metric-label">Anomaly score</span>
              <span className="metric-value">
                {evaluation ? decimal(evaluation.score.anomaly_score, 4) : decimal(state?.residual, 4)}
              </span>
              <span className="metric-detail">
                Mean absolute residual vs threshold {decimal(evaluation?.state.threshold ?? state?.threshold, 4)}
              </span>
            </div>
            <div className={`metric-card ${evaluation?.alert ? "status-critical" : "status-good"}`}>
              <span className="metric-label">Alert status</span>
              <span className="metric-value small-value">
                {evaluation?.alert
                  ? `${evaluation.alert.severity} · ${evaluation.alert.status}`
                  : evaluation
                    ? "no alert"
                    : "evaluating"}
              </span>
              <span className="metric-detail">
                {evaluation?.alert
                  ? evaluation.alert.trigger_reason
                  : "Calibrated threshold and persistence rules"}
              </span>
            </div>
          </div>

          <div className="grid four state-kpis">
            <div className="metric-card">
              <span className="metric-label">Prediction MAE</span>
              <span className="metric-value">{decimal(state?.residual, 4)}</span>
              <span className="metric-detail">Across all {state?.feature_count ?? 209} model features</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Triage status</span>
              <span className="metric-value small-value">{state ? String(state.priority).replace(/_/g, " ") : "—"}</span>
              <span className="metric-detail">Frozen validation residual threshold</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Sequence window</span>
              <span className="metric-value">{state?.sequence_length ?? 10}s</span>
              <span className="metric-detail">Ten observed states predict the next</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Prediction horizon</span>
              <span className="metric-value">1 state</span>
              <span className="metric-detail">One-step-ahead forecast</span>
            </div>
          </div>

          <div className="grid sidebar-right">
            <Panel
              title="Observed history and predicted next state"
              hint="Ten input states, the observed target and the LSTM forecast for one feature."
              actions={
                <label className="field inline-field">
                  <span className="field-label">Feature</span>
                  <select
                    className="select"
                    value={feature || selected?.feature || ""}
                    onChange={(event) => setFeature(event.target.value)}
                    disabled={!features.length}
                  >
                    {features.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </label>
              }
            >
              {stateData.error ? (
                <ErrorState message={stateData.error} onRetry={stateData.reload} />
              ) : selected ? (
                <>
                  <PlaybackChart
                    history={selected.history}
                    actual={selected.actual}
                    predicted={selected.predicted}
                    feature={selected.feature}
                  />
                  <div className="legend" style={{ marginTop: 10 }}>
                    <span className="legend-item"><span className="legend-swatch" style={{ background: "#A78BFA" }} /> Observed input history</span>
                    <span className="legend-item"><span className="legend-swatch" style={{ background: "#3B82F6" }} /> Actual next state</span>
                    <span className="legend-item"><span className="legend-swatch" style={{ background: "#22D3EE" }} /> LSTM forecast</span>
                  </div>
                  <div className="grid three compact-stats">
                    <div><span className="section-label">Actual next state</span><b>{decimal(selected.actual)}</b></div>
                    <div><span className="section-label">Model prediction</span><b>{decimal(selected.predicted)}</b></div>
                    <div><span className="section-label">Absolute error</span><b>{decimal(selected.absolute_error)}</b></div>
                  </div>
                </>
              ) : stateData.loading ? (
                <Loading rows={3} />
              ) : (
                <EmptyState title="No feature values for this state." detail="The replay arrays did not return the selected feature." />
              )}
            </Panel>

            <Panel
              title="Local prediction error"
              hint={`Mean absolute residual across features, steps ${number(Math.max(minStep, step - WINDOW))}–${number(Math.min(maxStep, step + WINDOW))}.`}
            >
              {local.error ? (
                <ErrorState message={local.error} onRetry={local.reload} />
              ) : localPoints.length ? (
                <>
                  <ThresholdTrendChart data={localPoints} threshold={state?.threshold} />
                  <p className="small muted" style={{ marginTop: 8 }}>
                    The dashed line is the frozen validation p99 threshold. Bars above it are candidates for
                    analyst review, not confirmed attacks.
                  </p>
                </>
              ) : (
                <Loading rows={3} />
              )}
            </Panel>
          </div>

          <div className="grid two">
            <Panel title="Residual contributors at this state" hint="Largest feature contributions to this prediction's error." flush>
              {stateData.data?.contributors.length ? (
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th className="num">#</th>
                        <th>Feature</th>
                        <th className="num">Signed residual</th>
                        <th className="num">Absolute error</th>
                        <th className="num">Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stateData.data.contributors.map((item) => (
                        <tr key={`${item.rank}-${item.feature}`}>
                          <td className="num">{item.rank}</td>
                          <td className="strong">{item.feature}</td>
                          <td className="num">{decimal(item.signed_residual)}</td>
                          <td className="num">{decimal(item.absolute_error)}</td>
                          <td className="num">
                            {item.contribution_percent === null ? "—" : `${item.contribution_percent.toFixed(2)}%`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="Per-observation contributors unavailable"
                  detail="The replay table does not carry Top-N contributor columns for this run."
                />
              )}
            </Panel>

            <Panel title="State context" hint="Recorded metadata for the selected replay state.">
              <dl className="kv">
                <dt>Replay step</dt><dd>{number(step)} of {number(maxStep)}</dd>
                <dt>Timestamp</dt><dd>{timestamp(state?.timestamp) || "—"}</dd>
                <dt>Capture day</dt><dd>{state?.capture_day ?? "—"}</dd>
                <dt>Segment</dt><dd>{state?.segment_id ?? "—"}</dd>
                <dt>Dataset reference label</dt><dd>{state?.dataset_label ?? "—"}</dd>
                <dt>Frozen threshold</dt><dd>{decimal(state?.threshold, 4)}</dd>
                <dt>Features</dt><dd>{state?.feature_count ?? "—"}</dd>
              </dl>
              <div className="notice info" style={{ marginTop: 12 }}>
                <span className="notice-icon">i</span>
                <div>
                  The dataset label is retrospective annotation. The LSTM predicts values, not labels, and
                  the triage rule never reads it.{" "}
                  <Link to={`/threats?search=${encodeURIComponent(state?.timestamp ?? "")}`}>Open in threat triage →</Link>
                </div>
              </div>
            </Panel>
          </div>

          <Panel
            title="Feature-wise prediction analysis"
            hint="Every model feature for this state, ranked by absolute error."
            actions={
              <div className="toolbar">
                {loadAll ? (
                  <input
                    className="input"
                    placeholder="Filter features…"
                    value={tableQuery}
                    onChange={(event) => setTableQuery(event.target.value)}
                  />
                ) : null}
                <button type="button" className="btn small" onClick={() => setLoadAll((value) => !value)}>
                  {loadAll ? "Hide full table" : "Load all features"}
                </button>
              </div>
            }
          >
            {!loadAll ? (
              <p className="small muted">
                The full 209-feature table is loaded on demand to keep playback responsive while stepping.
              </p>
            ) : stateData.loading ? (
              <Loading rows={4} />
            ) : tableRows.length ? (
              <DataTable
                columns={columns}
                rows={tableRows}
                rowKey={(row) => row.feature}
                emptyMessage="No feature matches the filter."
                initialSort={{ key: "absolute_error", direction: "desc" }}
              />
            ) : (
              <EmptyState title="No feature rows returned." />
            )}
          </Panel>

          <Panel title="Most frequent error contributors" hint="Across the full replay, not just this state.">
            {contributors.data?.contributors.length ? (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>Feature</th><th className="num">Appearances</th><th className="num">Appearance %</th></tr>
                  </thead>
                  <tbody>
                    {contributors.data.contributors.map((row) => (
                      <tr key={row.feature}>
                        <td className="strong">{row.feature}</td>
                        <td className="num">{number(row.appearances)}</td>
                        <td className="num">{row.appearance_percent.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="Contributor summary unavailable." />
            )}
          </Panel>

          {selectedAlert ? (
            <PlaybackAlertDrawer
              alertId={selectedAlert}
              initial={markerAlerts.find((alert) => alert.alert_id === selectedAlert) ?? null}
              onClose={() => setSelectedAlert(null)}
              onChanged={() => persistedAlerts.reload()}
            />
          ) : null}

          <Panel title="Sources and interpretation limits">
            <SourceLine source={meta.data.source} />
            <div className="divider" style={{ margin: "10px 0" }} />
            <Limitations items={meta.data.limitations} />
          </Panel>
        </>
      ) : (
        <EmptyState
          title="Playback needs the replay arrays"
          detail="Generate historical_replay_v1/replay_arrays.npz and the frozen test sequences, then reload. Nothing is simulated in their place."
          action={<button type="button" className="btn" onClick={meta.reload}>Re-check artifacts</button>}
        />
      )}
    </>
  );
}
