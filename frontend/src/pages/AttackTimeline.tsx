import { GroupedBars, SERIES } from "../components/charts";
import { Icon } from "../components/icons";
import {
  DataTable,
  EmptyState,
  ErrorState,
  LoadingBlock,
  MetricCard,
  Note,
  PageHeader,
  Panel,
  ProvenanceTag,
  Tag,
  type Column,
} from "../components/ui";
import { api, type Timeline } from "../lib/api";
import { humanSeconds, int, num } from "../lib/format";
import { useApi } from "../lib/useApi";

const CATEGORIES = [
  "1 second",
  "2 seconds",
  "3-5 seconds",
  "6-10 seconds",
  "11-30 seconds",
  "31-60 seconds",
  "61-300 seconds",
  "Over 300 seconds",
];

export default function AttackTimeline() {
  const { data, error, loading, reload } = useApi<Timeline>(() => api.timeline(), []);

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Attack Timeline" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Attack Timeline" />
        <Panel title="Timeline">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  if (!data.available) {
    return (
      <div className="page">
        <PageHeader title="Attack Timeline" />
        <Panel title="Timeline">
          <EmptyState title="Timeline artifact unavailable" desc={data.reason} />
        </Panel>
      </div>
    );
  }

  const chart = CATEGORIES.map((cat) => {
    const benign = data.histogram.find((h) => h.duration_category === cat && h.label === "Benign");
    const bot = data.histogram.find((h) => h.duration_category === cat && h.label === "Bot");
    return {
      category: cat.replace(" seconds", "s").replace(" second", "s"),
      benign: benign?.states_in_runs ?? 0,
      bot: bot?.states_in_runs ?? 0,
    };
  });

  const cols: Column<Timeline["runs"][number]>[] = [
    { key: "label", header: "Label", render: (r) => <Tag tone={r.label === "Benign" ? "ok" : "crit"}>{r.label}</Tag> },
    { key: "run_count", header: "Contiguous runs", align: "right", sortable: true, render: (r) => int(r.run_count) },
    { key: "mean_seconds", header: "Mean duration", align: "right", sortable: true, render: (r) => humanSeconds(r.mean_seconds) },
    { key: "median_seconds", header: "Median", align: "right", sortable: true, render: (r) => humanSeconds(r.median_seconds) },
    { key: "min_seconds", header: "Min", align: "right", render: (r) => humanSeconds(r.min_seconds) },
    { key: "max_seconds", header: "Max", align: "right", sortable: true, render: (r) => humanSeconds(r.max_seconds) },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Attack Timeline"
        desc="Chronological structure of the test capture. A run is a contiguous stretch of one dataset label; it shows how long labelled activity persists, not a reconstructed attack chain."
        actions={
          <button className="btn sm" onClick={reload}>
            <Icon.Refresh /> Reload
          </button>
        }
      />

      <div className="stack">
        <Note tone="info" title="Label persistence, not attack progression.">
          {data.note}
        </Note>

        <div className="grid grid-4">
          <MetricCard label="Capture window start" value={data.window?.start ?? "—"} small provenance="recorded_artifact" source={data.source} />
          <MetricCard label="Capture window end" value={data.window?.end ?? "—"} small provenance="recorded_artifact" source={data.source} />
          <MetricCard label="Contiguous runs" value={int(data.total_runs)} provenance="recorded_artifact" source={data.source} hint="Runs of a single label in the test capture" />
          <MetricCard label="States covered" value={int(data.total_states)} provenance="recorded_artifact" source={data.source} hint="One-second states in the analysed window" />
        </div>

        <Panel
          title="Activity duration profile"
          sub="States belonging to runs of each duration bucket"
          actions={<ProvenanceTag provenance={data.provenance} source={data.source} compact />}
        >
          <GroupedBars
            data={chart}
            xKey="category"
            series={[
              { key: "benign", name: "Benign states", color: SERIES.accent },
              { key: "bot", name: "Bot states", color: SERIES.crit },
            ]}
            height={280}
            xLabel="Run duration"
          />
          <div className="chart-note">
            Both labels are dominated by a small number of very long runs. A long benign run and a long bot run
            look structurally similar at the state level, which is why residual size alone does not separate them.
          </div>
        </Panel>

        <div className="grid grid-main-side">
          <Panel title="Run statistics by label" flush>
            <DataTable columns={cols} rows={data.runs} rowKey={(r) => r.label} pageSize={10} />
          </Panel>

          <div className="stack">
            <Panel title="Available analyses">
              <div className="stack" style={{ gap: 9 }}>
                <div className="row-between">
                  <span className="muted">Label run structure</span>
                  <Tag tone="ok" dot>recorded</Tag>
                </div>
                <div className="row-between">
                  <span className="muted">Transition forecast analysis</span>
                  <Tag tone={data.transition_analysis_available ? "ok" : "unavailable"} dot>
                    {data.transition_analysis_available ? "recorded" : "absent"}
                  </Tag>
                </div>
                <div className="row-between">
                  <span className="muted">Label persistence baseline</span>
                  <Tag tone={data.continuity_analysis_available ? "ok" : "unavailable"} dot>
                    {data.continuity_analysis_available ? "recorded" : "absent"}
                  </Tag>
                </div>
              </div>
            </Panel>

            <Panel title="Duration extremes">
              <div className="kv">
                {data.runs.map((r) => (
                  <div key={r.label} style={{ display: "contents" }}>
                    <div>{r.label} longest run</div>
                    <div>{num(r.max_seconds, 0)}s</div>
                  </div>
                ))}
                <div>Histogram buckets</div>
                <div>{int(data.histogram.length)}</div>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
