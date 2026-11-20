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
import { api, type Triage, type TriageAggregate } from "../lib/api";
import { int, num, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

export default function ThreatTriage() {
  const { data, error, loading, reload } = useApi<Triage>(() => api.triage(), []);

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Threat Triage" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Threat Triage" />
        <Panel title="Triage">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const cols: Column<TriageAggregate>[] = [
    { key: "split", header: "Split", render: (r) => <span style={{ textTransform: "capitalize", color: "var(--text-0)" }}>{r.split}</span> },
    { key: "total_observations", header: "Observations", align: "right", sortable: true, render: (r) => int(r.total_observations) },
    { key: "review_required", header: "Review required", align: "right", sortable: true, render: (r) => int(r.review_required) },
    { key: "within_calibrated_range", header: "Within range", align: "right", sortable: true, render: (r) => int(r.within_calibrated_range) },
    {
      key: "review_rate_percent",
      header: "Review rate",
      align: "right",
      sortable: true,
      render: (r) => <span className="mono">{pct(r.review_rate_percent, 4)}</span>,
    },
    { key: "frozen_threshold", header: "Threshold", align: "right", render: (r) => num(r.frozen_threshold, 4) },
  ];

  const testAgg = data.aggregates.find((a) => a.split === "test");

  return (
    <div className="page">
      <PageHeader
        title="Threat Triage"
        desc="Triage ranks network states for analyst review by the size of the next-state prediction residual. It is a review aid, not a classifier: ground-truth labels are never used to make a decision."
        actions={
          <button className="btn sm" onClick={reload}>
            <Icon.Refresh /> Reload
          </button>
        }
      />

      <div className="stack">
        <Note tone="crit" title="This is a review heuristic, not a validated detector.">
          On the held-out test split the frozen threshold flagged only{" "}
          {int(testAgg?.review_required ?? 0)} of {int(testAgg?.total_observations ?? 0)} observations,
          yielding zero true positives against the dominant Bot-labelled class. Use it to prioritise
          investigation, never as proof of compromise.
        </Note>

        <div className="grid grid-4">
          <MetricCard label="Frozen threshold" value={num(data.calibration.threshold, 4)} provenance="recorded_artifact" source={data.source} hint={data.calibration.source_description} />
          <MetricCard label="Calibration percentile" value={`p${data.calibration.percentile}`} provenance="recorded_artifact" hint="Percentile of benign validation residual scores" />
          <MetricCard label="Validation review rate" value={pct(data.aggregates.find((a) => a.split === "validation")?.review_rate_percent ?? null, 2)} provenance="recorded_artifact" source={data.source} hint="Share of validation observations flagged for review" />
          <MetricCard label="Recalibrated on test" value={data.calibration.recalibrated_on_test ? "Yes" : "No"} provenance="recorded_artifact" hint="The test split was never used to set the threshold" />
        </div>

        <Panel
          title="Recorded triage decision aggregates"
          sub={`${data.engine} · ${data.status}`}
          actions={<ProvenanceTag provenance={data.provenance} source={data.source} compact />}
          flush
        >
          <DataTable columns={cols} rows={data.aggregates} rowKey={(r) => r.split} pageSize={10} />
        </Panel>

        <div className="grid grid-main-side">
          <Panel title="Per-observation records">
            <EmptyState
              icon={<Icon.Shield size={22} />}
              title="Event-level rows are not in this checkout"
              desc={data.per_event_rows.reason}
            />
            <div className="row wrap" style={{ marginTop: 12 }}>
              <ProvenanceTag provenance={data.per_event_rows.provenance} compact />
              {(data.per_event_rows.requires ?? []).map((r) => (
                <span className="code" key={r}>
                  {r}
                </span>
              ))}
            </div>
          </Panel>

          <div className="stack">
            <Panel title="Decision policy">
              <div className="kv">
                <div>Decision input</div>
                <div style={{ fontSize: 11 }}>{data.decision_policy.input}</div>
                <div>Ground-truth labels used</div>
                <div style={{ color: "var(--ok)" }}>No</div>
                <div>Attack decision by threshold</div>
                <div style={{ color: "var(--ok)" }}>No</div>
                <div>UI threshold override</div>
                <div>{num(data.calibration.user_threshold_override, 4)}</div>
              </div>
            </Panel>

            <Panel title="Recommended investigation steps">
              <div className="stack" style={{ gap: 9 }}>
                {data.recommended_actions.map((a, i) => (
                  <div className="checkline" key={a} style={{ alignItems: "flex-start" }}>
                    <span className="mono dim" style={{ flex: "0 0 16px", fontSize: 11 }}>
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span>{a}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        </div>

        <Panel title="Stated limitations" sub="Carried through from the research record">
          <div className="stack" style={{ gap: 8 }}>
            {data.limitations.map((l) => (
              <div className="checkline" key={l} style={{ alignItems: "flex-start" }}>
                <span style={{ color: "var(--high)", flex: "0 0 14px" }}>
                  <Icon.Alert size={13} />
                </span>
                <span>{l}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12 }}>
            <Tag tone="neutral">MITRE ATT&amp;CK mappings are contextual label references, never per-observation classifications.</Tag>
          </div>
        </Panel>
      </div>
    </div>
  );
}
