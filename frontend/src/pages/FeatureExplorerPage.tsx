import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import {
  ActualPredictedScatter,
  CompareChart,
  ContributorBars,
  SignedResidualChart,
} from "../components/Charts";
import { DataTable, type Column } from "../components/DataTable";
import { PageHeader, Panel } from "../components/Panel";
import { Pill } from "../components/Status";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { decimal, number } from "../lib/format";
import { useApi } from "../hooks/useApi";

const WINDOW = 120;

interface ErrorRow {
  step: number;
  index: number;
  timestamp: string | null;
  label: string;
  actual: number;
  predicted: number;
  residual: number;
  absolute_error: number;
}

export function FeatureExplorerPage() {
  const [feature, setFeature] = useState("");
  const [start, setStart] = useState(0);

  const meta = useApi(() => api.replayMeta(), "explorer-meta");
  const features = meta.data?.features ?? [];

  useEffect(() => {
    if (!feature && features.includes("Flow Count")) setFeature("Flow Count");
    else if (!feature && features.length) setFeature(features[0]);
  }, [feature, features]);

  const series = useApi(
    () => api.replaySeries({ feature: feature || undefined, start, limit: WINDOW }),
    `explorer-series-${feature}-${start}`,
  );
  const contributors = useApi(() => api.replayContributors(15), "explorer-contributors");

  // Unavailable payloads omit `window`; never dereference it unguarded.
  const totalStates = series.data?.window?.total_states ?? meta.data?.total_states ?? 0;

  const compareData = useMemo(
    () => (series.data?.points ?? []).map((point) => ({ index: point.step, actual: point.actual, predicted: point.predicted })),
    [series.data],
  );
  const residualData = useMemo(
    () =>
      (series.data?.points ?? []).map((point) => ({
        index: point.step,
        signed: point.residual,
        absolute: point.absolute_error,
      })),
    [series.data],
  );
  const contributorChart = (contributors.data?.contributors ?? []).map((row) => ({
    feature: row.feature,
    value: Number(row.appearance_percent.toFixed(2)),
  }));

  const columns: Column<ErrorRow>[] = [
    {
      key: "step",
      label: "State",
      render: (row) => <Link to={`/replay?step=${row.step}`}>{number(row.step)}</Link>,
      sortValue: (row) => row.step,
    },
    { key: "timestamp", label: "Timestamp", render: (row) => row.timestamp ?? "—", sortValue: (row) => row.timestamp },
    { key: "label", label: "Reference label", render: (row) => <Pill tone={row.label === "Benign" ? "good" : "warning"}>{row.label}</Pill> },
    { key: "actual", label: "Actual", align: "right", render: (row) => decimal(row.actual), sortValue: (row) => row.actual },
    { key: "predicted", label: "Predicted", align: "right", render: (row) => decimal(row.predicted), sortValue: (row) => row.predicted },
    { key: "residual", label: "Signed residual", align: "right", render: (row) => decimal(row.residual), sortValue: (row) => row.residual },
    { key: "absolute_error", label: "Absolute error", align: "right", render: (row) => decimal(row.absolute_error), sortValue: (row) => row.absolute_error },
  ];

  if (meta.loading && !meta.data) {
    return (
      <Panel title="Loading feature data">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (meta.error) return <ErrorState message={meta.error} onRetry={meta.reload} />;
  if (!meta.data) return <EmptyState title="Feature data unavailable." />;

  return (
    <>
      <PageHeader
        title="Feature Explorer"
        subtitle="Inspect how the world model tracks one network-state feature, where its error concentrates and which features contribute most across the replay."
        actions={
          <button type="button" className="btn" onClick={() => { meta.reload(); series.reload(); contributors.reload(); }}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice availability={meta.data.availability} title="Feature replay unavailable" />

      <Panel flush>
        <div className="toolbar feature-toolbar">
          <label className="field" style={{ minWidth: 280 }}>
            <span className="field-label">Model feature</span>
            <select className="select" value={feature} onChange={(event) => { setFeature(event.target.value); setStart(0); }} disabled={!features.length}>
              {features.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </label>
          <div className="field" style={{ flex: 1, minWidth: 260 }}>
            <span className="field-label">
              Replay window · states {number(start + 1)}–{number(Math.min(totalStates, start + WINDOW))} of {number(totalStates)}
            </span>
            <div className="field-row">
              <button type="button" className="btn small" disabled={start <= 0} onClick={() => setStart(Math.max(0, start - WINDOW))}>
                ← Earlier
              </button>
              <input
                className="timeline"
                type="range"
                min={0}
                max={Math.max(0, totalStates - WINDOW)}
                step={WINDOW}
                value={Math.min(start, Math.max(0, totalStates - WINDOW))}
                aria-label="Replay window start"
                onChange={(event) => setStart(Number(event.target.value))}
              />
              <button type="button" className="btn small" disabled={start + WINDOW >= totalStates} onClick={() => setStart(start + WINDOW)}>
                Later →
              </button>
            </div>
          </div>
        </div>
      </Panel>

      {series.error ? <ErrorState message={series.error} onRetry={series.reload} /> : null}

      <div className="grid four">
        <div className="metric-card">
          <span className="metric-label">Mean absolute error</span>
          <span className="metric-value">{decimal(series.data?.stats.mae)}</span>
          <span className="metric-detail">{feature || "Selected feature"} · all {number(series.data?.stats.samples)} states</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Median absolute error</span>
          <span className="metric-value">{decimal(series.data?.stats.median_absolute_error)}</span>
          <span className="metric-detail">Middle value across the held-out replay</span>
        </div>
        <div className="metric-card status-warning">
          <span className="metric-label">Maximum absolute error</span>
          <span className="metric-value">{decimal(series.data?.stats.max_absolute_error)}</span>
          <span className="metric-detail">Largest single-state gap for this feature</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Mean signed residual</span>
          <span className="metric-value">{decimal(series.data?.stats.mean_signed_residual)}</span>
          <span className="metric-detail">Positive means observed values ran above predictions</span>
        </div>
      </div>

      <div className="grid two">
        <Panel
          title="Actual versus predicted"
          hint={series.data ? `Feature: ${series.data.feature}` : "Select a feature to compare observed and forecast values."}
        >
          {series.loading && !series.data ? (
            <Loading rows={3} />
          ) : compareData.length ? (
            <>
              <CompareChart data={compareData} />
              <div className="legend" style={{ marginTop: 8 }}>
                <span className="legend-item"><span className="legend-swatch" style={{ background: "#3B82F6" }} /> Actual next state</span>
                <span className="legend-item"><span className="legend-swatch" style={{ background: "#22D3EE" }} /> LSTM prediction</span>
              </div>
            </>
          ) : (
            <EmptyState title="No states in this window." />
          )}
        </Panel>

        <Panel title="Feature residual analysis" hint="Signed residual gives direction; absolute residual gives magnitude.">
          {residualData.length ? (
            <SignedResidualChart data={residualData} />
          ) : (
            <EmptyState title="Residual series unavailable." />
          )}
        </Panel>
      </div>

      <Panel
        title="Observed versus predicted distribution"
        hint="Each point is one replay state. Close clustering along the diagonal indicates a well-tracked feature."
      >
        {compareData.length ? (
          <ActualPredictedScatter
            data={compareData.map((point) => ({
              actual: point.actual ?? 0,
              predicted: point.predicted ?? 0,
            }))}
          />
        ) : (
          <EmptyState title="No scatter data in this window." />
        )}
      </Panel>

      <Panel title="Largest errors for this feature" hint="The ten states where this feature's forecast was furthest from observation." flush>
        {series.data?.top_errors?.length ? (
          <DataTable
            columns={columns}
            rows={series.data.top_errors as ErrorRow[]}
            rowKey={(row) => String(row.step)}
            emptyMessage="No error rows returned."
            initialSort={{ key: "absolute_error", direction: "desc" }}
          />
        ) : (
          <div className="panel-body">
            <EmptyState title="No per-state errors available." />
          </div>
        )}
        <div className="panel-body small muted">
          Selecting a state opens it in network state playback. Reference labels are dataset annotations, not model output.
        </div>
      </Panel>

      <div className="grid sidebar-right">
        <Panel title="Most frequent error contributors" hint="Share of replay states where a feature was among the top five residual contributors.">
          {contributorChart.length ? (
            <ContributorBars data={contributorChart} height={Math.max(260, contributorChart.length * 30)} />
          ) : (
            <EmptyState title="Contributor report unavailable." />
          )}
        </Panel>

        <Panel title="Contributor table" flush>
          {contributors.data?.contributors.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>Feature</th><th className="num">Appearances</th><th className="num">Share</th></tr>
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
            <div className="panel-body">
              <EmptyState title="No contributor rows." />
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Sources and interpretation limits">
        <SourceLine source={meta.data.source} />
        <div className="divider" style={{ margin: "10px 0" }} />
        <Limitations items={meta.data.limitations} />
      </Panel>
    </>
  );
}
