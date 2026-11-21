import { api } from "../api/client";
import { ActivityChart } from "../components/Charts";
import { DataTable, type Column } from "../components/DataTable";
import { PageHeader, Panel } from "../components/Panel";
import { Pill, PriorityPill } from "../components/Status";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
} from "../components/States";
import { decimal, number, timestamp } from "../lib/format";
import { useApi } from "../hooks/useApi";

interface PhaseRow {
  start_step: number;
  end_step: number;
  start_timestamp: string | null;
  end_timestamp: string | null;
  label: string;
  observations: number;
}

export function TimelinePage() {
  const { data, error, loading, reload } = useApi(() => api.timeline(), "timeline");

  if (loading && !data) {
    return (
      <Panel title="Loading timeline">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!data) return <EmptyState title="No timeline data returned." />;

  const chart = data.buckets.map((bucket) => ({
    index: bucket.first_step,
    label: bucket.bucket,
    timestamp: bucket.bucket,
    value: bucket.mean_residual,
    secondary:
      bucket.observations > 0
        ? Object.entries(bucket.labels).reduce(
            (total, [label, count]) =>
              label.toLowerCase() === "benign" ? total : total + count,
            0,
          ) / bucket.observations
        : 0,
  }));

  const phaseColumns: Column<PhaseRow>[] = [
    {
      key: "label",
      label: "Dataset label",
      render: (row) => (
        <Pill tone={row.label.toLowerCase() === "benign" ? "good" : "warning"}>{row.label}</Pill>
      ),
    },
    {
      key: "start",
      label: "Start",
      sortValue: (row) => row.start_step,
      render: (row) => `${timestamp(row.start_timestamp)} · step ${number(row.start_step)}`,
    },
    {
      key: "end",
      label: "End",
      render: (row) => `${timestamp(row.end_timestamp)} · step ${number(row.end_step)}`,
    },
    {
      key: "observations",
      label: "Observations",
      align: "right",
      sortValue: (row) => row.observations,
      render: (row) => number(row.observations),
    },
  ];

  const bucketRows = data.buckets.map((bucket) => ({
    ...bucket,
    labelSummary: Object.entries(bucket.labels)
      .map(([label, count]) => `${label}: ${count}`)
      .join(" · "),
  }));

  const bucketColumns: Column<(typeof bucketRows)[number]>[] = [
    { key: "bucket", label: "Window", render: (row) => timestamp(row.bucket) },
    {
      key: "steps",
      label: "Steps",
      render: (row) => `${number(row.first_step)}–${number(row.last_step)}`,
    },
    {
      key: "observations",
      label: "Observations",
      align: "right",
      sortValue: (row) => row.observations,
      render: (row) => number(row.observations),
    },
    {
      key: "mean_residual",
      label: "Mean residual",
      align: "right",
      sortValue: (row) => row.mean_residual,
      render: (row) => decimal(row.mean_residual),
    },
    {
      key: "max_residual",
      label: "Max residual",
      align: "right",
      sortValue: (row) => row.max_residual,
      render: (row) => decimal(row.max_residual),
    },
    {
      key: "review",
      label: "Candidates",
      align: "right",
      sortValue: (row) => row.review_candidates,
      render: (row) =>
        row.review_candidates ? (
          <Pill tone="critical">{row.review_candidates}</Pill>
        ) : (
          <span className="muted">0</span>
        ),
    },
    { key: "labels", label: "Labels", render: (row) => row.labelSummary || "—" },
  ];

  return (
    <>
      <PageHeader
        title="Attack Timeline"
        subtitle="Chronological view of dataset reference labels alongside prediction residuals. Phases are contiguous label runs, not reconstructed attack stages."
        actions={
          <button type="button" className="btn" onClick={reload}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice availability={data.availability} title="Timeline unavailable" />

      <Panel
        title="Residual activity per five-minute window"
        hint="Mean prediction residual (teal) with attack-labelled share (blue)."
      >
        {chart.length ? (
          <ActivityChart data={chart} height={240} />
        ) : (
          <EmptyState title="No time buckets available." />
        )}
      </Panel>

      <Panel title="Observed phases" hint="Contiguous runs of dataset reference labels." flush>
        <DataTable
          columns={phaseColumns}
          rows={data.phases}
          rowKey={(row) => `${row.label}-${row.start_step}`}
          emptyMessage="No label phases are available in this deployment."
        />
      </Panel>

      <Panel title="Window detail" flush>
        <DataTable
          columns={bucketColumns}
          rows={bucketRows}
          rowKey={(row) => row.bucket}
          emptyMessage="No windows available."
        />
      </Panel>

      <Panel title="Review candidates in the timeline" flush>
        {data.events.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Timestamp</th>
                  <th>Dataset label</th>
                  <th className="num">Residual</th>
                  <th>Priority</th>
                  <th>Case</th>
                </tr>
              </thead>
              <tbody>
                {data.events.map((event) => (
                  <tr key={event.event_id}>
                    <td className="mono">{event.step ?? "—"}</td>
                    <td>{timestamp(event.timestamp)}</td>
                    <td>{event.dataset_label ?? "—"}</td>
                    <td className="num">{decimal(event.residual_score)}</td>
                    <td>
                      <PriorityPill priority={event.priority} />
                    </td>
                    <td>{event.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No candidates in the timeline"
            detail="A clear timeline is not evidence of benign activity; see the triage page for the detection caveats."
          />
        )}
      </Panel>

      <Panel title="Interpretation limits">
        <Limitations items={data.limitations} />
      </Panel>
    </>
  );
}
