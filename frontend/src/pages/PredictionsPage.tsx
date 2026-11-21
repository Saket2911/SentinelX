import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import {
  AnomalyScoreChart,
  CompareChart,
  DistributionChart,
  HorizonErrorChart,
  QuantileCurve,
} from "../components/Charts";
import { MetricGrid } from "../components/Metric";
import { PageHeader, Panel } from "../components/Panel";
import { Pill, PriorityPill } from "../components/Status";
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

const WINDOW = 180;

export function PredictionsPage() {
  const [feature, setFeature] = useState("");
  const [start, setStart] = useState(0);

  const summary = useApi(() => api.predictionSummary(), "prediction-summary");
  const topResiduals = useApi(() => api.topResiduals(10), "prediction-top-residuals");
  const anomalies = useApi(() => api.anomalySeries(), "prediction-anomalies");
  const series = useApi(
    () =>
      api.predictionSeries({
        feature: feature || undefined,
        start,
        limit: WINDOW,
      }),
    `prediction-series-${feature}-${start}`,
  );

  if (summary.loading && !summary.data) {
    return (
      <Panel title="Loading forecast evaluation">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (summary.error) return <ErrorState message={summary.error} onRetry={summary.reload} />;
  const data = summary.data;
  if (!data) return <EmptyState title="No prediction data returned." />;

  const calibration = data.calibration;
  const chartData =
    series.data?.points.map((point) => ({
      index: point.index,
      actual: point.actual,
      predicted: point.predicted,
    })) ?? [];

  const distribution = data.error_distribution
    .filter((entry) => entry.kind !== "recorded_quantile")
    .map((entry) => ({
      label: `${Number(entry.bucket_start).toFixed(2)}`,
      value: Number(entry.count),
    }));

  const features = series.data?.features ?? data.features;

  // Recorded quantiles are a distribution summary that needs no row-level table.
  const quantiles = data.error_distribution
    .filter((entry) => entry.kind === "recorded_quantile")
    .map((entry) => ({ label: `p${Number(entry.quantile)}`, value: Number(entry.value) }));

  const horizonCurve = data.multi_step
    .filter((row) => row.label === "ALL")
    .map((row) => ({
      horizon: Number(row.horizon_seconds),
      lstm: row.lstm_mae ?? null,
      persistence: row.persistence_mae ?? null,
    }));

  return (
    <>
      <PageHeader
        title="Predictive Intelligence"
        subtitle="P(t+1) forecast quality, forecast horizon, residual calibration and multi-step behaviour of the LSTM world model."
        actions={
          <button type="button" className="btn" onClick={() => { summary.reload(); series.reload(); }}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice availability={data.availability} title="Forecast evaluation unavailable" />

      <div className="notice info">
        <span className="notice-icon">i</span>
        <div>
          <strong>One-step-ahead network-state forecasting.</strong> The model consumes the previous
          ten one-second states and predicts the next state. It does not output attack classes.
          Values are in standardized model feature space.
        </div>
      </div>

      <MetricGrid metrics={data.metrics} />

      <div className="grid sidebar-right">
        <Panel
          title="Actual versus predicted state"
          hint={
            series.data?.feature
              ? `Feature: ${series.data.feature}`
              : "Mean absolute residual across all 209 features (no feature-level pair selected)."
          }
          actions={
            <div className="toolbar">
              <button
                type="button"
                className="btn small"
                disabled={start === 0}
                onClick={() => setStart(Math.max(0, start - WINDOW))}
              >
                ← Earlier
              </button>
              <button
                type="button"
                className="btn small"
                disabled={
                  series.data ? Number(series.data.summary.window_end) >= Number(series.data.summary.total_points) : false
                }
                onClick={() => setStart(start + WINDOW)}
              >
                Later →
              </button>
            </div>
          }
        >
          <div className="toolbar" style={{ marginBottom: 12 }}>
            <label className="field" style={{ gap: 3, minWidth: 240 }}>
              <span className="field-label">Model feature</span>
              <select
                className="select"
                value={feature}
                onChange={(event) => {
                  setFeature(event.target.value);
                  setStart(0);
                }}
                disabled={!features.length}
              >
                <option value="">All features (aggregate residual)</option>
                {features.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            {series.data?.summary.total_points ? (
              <span className="small muted" style={{ alignSelf: "flex-end", paddingBottom: 8 }}>
                steps {number(start)}–{number(Math.min(start + WINDOW, Number(series.data.summary.total_points)))}{" "}
                of {number(Number(series.data.summary.total_points))}
              </span>
            ) : null}
          </div>
          {series.error ? (
            <ErrorState message={series.error} onRetry={series.reload} />
          ) : chartData.length ? (
            <>
              <CompareChart data={chartData} />
              <div className="legend" style={{ marginTop: 8 }}>
                <span className="legend-item">
                  <span className="legend-swatch" style={{ background: "#3B82F6" }} /> Actual next
                  state
                </span>
                <span className="legend-item">
                  <span className="legend-swatch" style={{ background: "#22D3EE" }} /> LSTM forecast
                </span>
              </div>
            </>
          ) : (
            <EmptyState
              title="No row-level predictions"
              detail="Generate the replay table to compare actual and predicted states."
            />
          )}
        </Panel>

        <div className="stack">
          <Panel title="Residual calibration" hint={calibration.method}>
            <dl className="kv">
              <dt>Frozen threshold</dt>
              <dd>{decimal(calibration.threshold, 4)}</dd>
              <dt>Threshold source</dt>
              <dd>{calibration.threshold_source}</dd>
              {Object.entries(calibration.quantiles).map(([key, value]) => (
                <div key={key} style={{ display: "contents" }}>
                  <dt>p{key}</dt>
                  <dd>{decimal(value)}</dd>
                </div>
              ))}
            </dl>
            <div className="divider" style={{ margin: "10px 0" }} />
            <p className="small muted">{calibration.priority_semantics}</p>
          </Panel>

          <Panel title="Forecast horizon">
            <dl className="kv">
              <dt>One-step</dt>
              <dd>{data.horizon.one_step ? "evaluated" : "n/a"}</dd>
              <dt>Multi-step horizons</dt>
              <dd>
                {(data.horizon.multi_step_horizons ?? []).map((horizon) => `${horizon}s`).join(", ") ||
                  "n/a"}
              </dd>
              <dt>Method</dt>
              <dd>{data.horizon.method ?? "n/a"}</dd>
            </dl>
            <div className="divider" style={{ margin: "10px 0" }} />
            <SourceLine source={data.source} />
          </Panel>
        </div>
      </div>

      <Panel
        title="Anomaly score across the replay"
        hint="Mean absolute next-state residual per state, with the frozen calibrated threshold and the states that crossed it."
      >
        {anomalies.error ? (
          <ErrorState message={anomalies.error} onRetry={anomalies.reload} />
        ) : anomalies.data?.points.length ? (
          <>
            <AnomalyScoreChart
              data={anomalies.data.points}
              threshold={anomalies.data.threshold}
              alerts={(anomalies.data.points ?? [])
                .map((point) => point.alert_id)
                .filter((value): value is string => Boolean(value))}
            />
            <div className="inline" style={{ marginTop: 8, flexWrap: "wrap" }}>
              <Pill tone="neutral">
                {number(anomalies.data.summary.states)} states · mean score{" "}
                {decimal(anomalies.data.summary.mean_score, 4)} · max{" "}
                {decimal(anomalies.data.summary.max_score, 4)}
              </Pill>
              <Pill tone="warning">
                {number(anomalies.data.summary.flagged)} state(s) at/above threshold
              </Pill>
              <Pill tone="info">
                {number(anomalies.data.summary.persisted_alerts)} persisted playback alert(s)
              </Pill>
            </div>
            {anomalies.data.flagged_steps.length ? (
              <div className="tag-row" style={{ marginTop: 8 }}>
                {anomalies.data.flagged_steps.map((step) => (
                  <Link key={step} className="pill critical" to={`/replay?step=${step}`}>
                    Step {step} · open replay
                  </Link>
                ))}
              </div>
            ) : (
              <p className="small muted" style={{ marginTop: 8 }}>
                No state crossed the calibrated threshold in this replay. That is not evidence that
                the capture was benign: the recorded evaluation also found 0 labelled Bot states at
                this threshold.
              </p>
            )}
          </>
        ) : anomalies.loading ? (
          <Loading rows={4} />
        ) : (
          <EmptyState
            title="Anomaly history unavailable"
            detail={anomalies.data?.availability.reason ?? "No replay arrays are available for this run."}
          />
        )}
      </Panel>

      <div className="grid two">
        <Panel
          title="Residual distribution"
          hint={
            distribution.length
              ? "Histogram of per-observation mean absolute residuals."
              : "Recorded residual quantiles from the reliability audit; a per-row table is required for a histogram."
          }
        >
          {distribution.length ? (
            <DistributionChart data={distribution} />
          ) : quantiles.length ? (
            <QuantileCurve data={quantiles} />
          ) : (
            <EmptyState
              title="Distribution unavailable"
              detail="Neither a per-row residual table nor recorded error quantiles are present for this run."
            />
          )}
        </Panel>

        <Panel title="Per-label error at horizon 1" hint="Recorded multi-step report, horizon 1 second." flush>
          {data.per_label.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Label</th>
                    <th className="num">Sequences</th>
                    <th className="num">LSTM MAE</th>
                    <th className="num">Persistence MAE</th>
                    <th className="num">Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  {data.per_label.map((row) => (
                    <tr key={String(row.label)}>
                      <td className="strong">{String(row.label)}</td>
                      <td className="num">{number(Number(row.samples ?? 0))}</td>
                      <td className="num">{decimal(Number(row.lstm_mae ?? NaN))}</td>
                      <td className="num">{decimal(Number(row.persistence_mae ?? NaN))}</td>
                      <td className="num">{percent(Number(row.mae_improvement_percent ?? NaN), 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="No per-label evaluation recorded." />
          )}
        </Panel>
      </div>

      <Panel
        title="Forecast error by horizon"
        hint="Recursive forecasting re-feeds each prediction, so error grows with the horizon. Solid line: LSTM. Dashed: persistence baseline (repeat the last state)."
      >
        {horizonCurve.length ? (
          <>
            <HorizonErrorChart data={horizonCurve} />
            <div className="legend" style={{ marginTop: 8 }}>
              <span className="legend-item">
                <span className="legend-swatch" style={{ background: "#22D3EE" }} /> LSTM forecast MAE
              </span>
              <span className="legend-item">
                <span className="legend-swatch" style={{ background: "#A78BFA" }} /> Persistence MAE
              </span>
              <span className="small muted">All labels · common cohort of forecast origins</span>
            </div>
          </>
        ) : (
          <EmptyState title="No multi-step metrics recorded for this run." />
        )}
      </Panel>

      <Panel
        title="Highest-residual observations"
        hint="Top-ranked observations by mean absolute next-state residual. Ranking for review, not a classification."
        flush
      >
        {topResiduals.data?.residuals.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Timestamp</th>
                  <th>Dataset label</th>
                  <th className="num">Mean absolute residual</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                {topResiduals.data.residuals.map((row, index) => (
                  <tr key={`${row.step}-${index}`}>
                    <td className="mono">{row.step ?? "—"}</td>
                    <td>{row.timestamp ? timestamp(row.timestamp) : "—"}</td>
                    <td>{row.label ?? "—"}</td>
                    <td className="num">{decimal(row.residual)}</td>
                    <td>
                      <PriorityPill priority={row.priority} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No row-level residuals available"
            detail={
              topResiduals.error ??
              topResiduals.data?.note ??
              "Generate the replay table to rank individual observations."
            }
          />
        )}
      </Panel>

      <Panel
        title="Multi-step forecasting"
        hint="Recursive forecasting re-feeds predictions; longer horizons accumulate error."
        flush
      >
        {data.multi_step.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th className="num">Horizon</th>
                  <th>Label</th>
                  <th className="num">Origins</th>
                  <th className="num">LSTM MAE</th>
                  <th className="num">Persistence MAE</th>
                  <th className="num">MAE improvement</th>
                  <th className="num">LSTM RMSE</th>
                  <th className="num">RMSE improvement</th>
                </tr>
              </thead>
              <tbody>
                {data.multi_step.map((row) => (
                  <tr key={`${row.horizon_seconds}-${row.label}`}>
                    <td className="num">{row.horizon_seconds}s</td>
                    <td>
                      <Pill tone={row.label === "ALL" ? "accent" : "neutral"}>{row.label}</Pill>
                    </td>
                    <td className="num">{number(row.valid_origins ?? null)}</td>
                    <td className="num">{decimal(row.lstm_mae ?? null)}</td>
                    <td className="num">{decimal(row.persistence_mae ?? null)}</td>
                    <td className="num">{percent(row.mae_improvement_percent ?? null, 2)}</td>
                    <td className="num">{decimal(row.lstm_rmse ?? null)}</td>
                    <td className="num">{percent(row.rmse_improvement_percent ?? null, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No multi-step report recorded for this run." />
        )}
      </Panel>

      <Panel title="Interpretation limits">
        <Limitations items={data.limitations} />
      </Panel>
    </>
  );
}
