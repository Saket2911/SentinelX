import { Link } from "react-router-dom";
import { useHealth } from "../App";
import { SimpleBar, SERIES } from "../components/charts";
import { Icon } from "../components/icons";
import {
  DataTable,
  ErrorState,
  LoadingBlock,
  MetricCard,
  Note,
  PageHeader,
  Panel,
  ProvenanceTag,
  StatBar,
  Tag,
  type Column,
} from "../components/ui";
import { api, type Dashboard as DashboardData } from "../lib/api";
import { compact, int, num, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

interface LabelRow {
  partition: string;
  label: string;
  count: number;
}

export default function Dashboard() {
  const { data, error, loading, reload } = useApi<DashboardData>(() => api.dashboard(), []);
  const { health } = useHealth();

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="SOC Dashboard" desc="Loading recorded research artifacts…" />
        <div className="grid grid-4">
          {[0, 1, 2, 3].map((i) => (
            <div className="metric" key={i}>
              <LoadingBlock lines={3} />
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14 }}>
          <Panel title="Network composition">
            <LoadingBlock height={240} />
          </Panel>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="SOC Dashboard" />
        <Panel title="Dashboard">
          <ErrorState message={error ?? "No data returned"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const testLabels = data.label_distribution.filter((r) => r.partition === "test" && r.count > 0);
  const testTotal = testLabels.reduce((s, r) => s + r.count, 0);
  const partitionTotals: Record<string, number> = {};
  for (const r of data.label_distribution) {
    partitionTotals[r.partition] = (partitionTotals[r.partition] ?? 0) + r.count;
  }
  const chartData = testLabels
    .slice(0, 8)
    .map((r) => ({ label: r.label, count: r.count }));

  const det = data.detection_assessment;
  const tm = det.test_metrics ?? {};

  const labelCols: Column<LabelRow>[] = [
    { key: "partition", header: "Split", render: (r) => <span className="muted">{r.partition}</span> },
    { key: "label", header: "Label", render: (r) => <span style={{ color: "var(--text-0)" }}>{r.label}</span> },
    { key: "count", header: "Sequences", align: "right", sortable: true },
    {
      key: "share",
      header: "Share",
      align: "right",
      render: (r) => (
        <span className="mono">
          {pct((r.count / (partitionTotals[r.partition] || 1)) * 100, 2)}
        </span>
      ),
      sortValue: (r) => r.count,
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="SOC Dashboard"
        desc="Predictive network-state modelling for security operations. Every figure below is copied from a recorded evaluation artifact — no metric is simulated and no live telemetry is connected."
        actions={
          <>
            <Tag tone={health?.status === "ok" ? "ok" : "med"} dot>
              {health?.status === "ok" ? "Platform healthy" : "Platform degraded"}
            </Tag>
            <button className="btn sm" onClick={reload}>
              <Icon.Refresh /> Refresh
            </button>
          </>
        }
      />

      <div className="stack">
        <Note tone="warn" title={det.headline + "."}>
          {det.verdict}
        </Note>

        <div className="grid grid-4">
          {data.metrics.slice(0, 4).map((m) => (
            <MetricCard
              key={m.key}
              label={m.label}
              value={
                m.key === "test_samples" || m.key === "input_features"
                  ? int(m.value as number)
                  : num(m.value as number, 4)
              }
              unit={m.unit ?? undefined}
              hint={m.hint ?? undefined}
              provenance={m.provenance}
              source={m.source ?? undefined}
            />
          ))}
        </div>

        <div className="grid grid-main-side">
          <Panel
            title="Test-partition network composition"
            sub={`${int(testTotal)} labelled sequences · held-out test split`}
            actions={<ProvenanceTag provenance="recorded_artifact" source="reports/mitre_attack_contextual_mapping_v1/dataset_label_coverage_report.txt" compact />}
          >
            {chartData.length > 0 ? (
              <>
                <SimpleBar
                  data={chartData}
                  xKey="label"
                  yKey="count"
                  name="Sequences"
                  color={SERIES.accent}
                  layout="vertical"
                  height={230}
                  formatter={compact}
                />
                <div className="chart-note">
                  Label counts are recorded dataset annotations, not model predictions. The test split is
                  dominated by a single Bot-labelled capture, so aggregate detection rates are dominated by
                  that class.
                </div>
              </>
            ) : (
              <div className="dim">No test label distribution recorded.</div>
            )}
          </Panel>

          <div className="stack">
            <Panel title="Security posture" actions={<ProvenanceTag provenance="derived" compact />}>
              <div className="row" style={{ marginBottom: 10 }}>
                <Tag tone="med" dot>
                  {data.security_posture.label}
                </Tag>
                <span className="dim" style={{ fontSize: 11 }}>
                  model {data.active_model.toUpperCase()}
                </span>
              </div>
              <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.6 }}>
                {data.security_posture.detail}
              </p>
            </Panel>

            <Panel title="Detection assessment" actions={<ProvenanceTag provenance={det.provenance} compact />}>
              <div className="kv">
                <div>Validated detector</div>
                <div style={{ color: "var(--crit)" }}>No</div>
                <div>Frozen threshold</div>
                <div>{num(det.threshold, 4)}</div>
                <div>Test true positives</div>
                <div style={{ color: "var(--crit)" }}>{int((tm.TP ?? 0) as number)}</div>
                <div>Test false negatives</div>
                <div>{int((tm.FN ?? 0) as number)}</div>
                <div>Test recall</div>
                <div style={{ color: "var(--crit)" }}>{num(tm.recall ?? 0, 3)}</div>
              </div>
            </Panel>
          </div>
        </div>

        <div className="grid grid-main-side">
          <Panel title="Recent recorded signals" flush>
            {data.recent_signals.length === 0 ? (
              <div className="state">No recorded signals.</div>
            ) : (
              <div className="list">
                {data.recent_signals.map((s, i) => {
                  const rec = s as Record<string, unknown>;
                  const isTriage = rec.kind === "triage_aggregate";
                  return (
                    <div className="list-item" key={i}>
                      <span style={{ color: isTriage ? "var(--med)" : "var(--blue)", marginTop: 2 }}>
                        {isTriage ? <Icon.Shield size={15} /> : <Icon.Timeline size={15} />}
                      </span>
                      <div className="li-main">
                        <div className="li-title">
                          {isTriage
                            ? `${String(rec.split)} triage review window`
                            : "Test capture run structure"}
                        </div>
                        <div className="li-meta">
                          {isTriage
                            ? `${int(rec.review_required as number)} of ${int(
                                rec.total as number,
                              )} observations flagged · ${pct(rec.review_rate_percent as number, 4)} review rate`
                            : `${int(rec.total_runs as number)} contiguous label runs across ${int(
                                rec.total_states as number,
                              )} states`}
                        </div>
                      </div>
                      <ProvenanceTag provenance="recorded_artifact" compact />
                    </div>
                  );
                })}
              </div>
            )}
          </Panel>

          <Panel title="Model & system health">
            <div className="kv">
              <div>Active model</div>
              <div>{data.active_model.toUpperCase()}</div>
              <div>Inference available</div>
              <div style={{ color: data.system_health.inference_available ? "var(--ok)" : "var(--high)" }}>
                {data.system_health.inference_available ? "Yes" : "No — checkpoint absent"}
              </div>
              <div>Live telemetry</div>
              <div className="dim">Not implemented</div>
              <div>Data mode</div>
              <div>Offline dataset</div>
            </div>
            <div style={{ marginTop: 12 }}>
              <Link to="/predictive" className="btn sm">
                <Icon.Predict /> Open predictive view
              </Link>
            </div>
          </Panel>
        </div>

        <div className="grid grid-2">
          <Panel
            title="Recorded label inventory"
            sub="All partitions, from the dataset coverage report"
            flush
          >
            <DataTable
              columns={labelCols}
              rows={[...data.label_distribution].sort((a, b) => b.count - a.count)}
              rowKey={(r, i) => `${r.partition}-${r.label}-${i}`}
              pageSize={8}
              initialSort={{ key: "count", dir: "desc" }}
            />
          </Panel>

          <Panel title="Known limitations" sub="Stated plainly, because they bound every claim">
            <div className="stack" style={{ gap: 8 }}>
              {data.limitations.map((l) => (
                <div className="checkline" key={l}>
                  <span style={{ color: "var(--text-3)", flex: "0 0 14px" }}>
                    <Icon.Info size={13} />
                  </span>
                  <span>{l}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 14 }}>
              <div className="row-between" style={{ marginBottom: 6 }}>
                <span className="muted" style={{ fontSize: 11.5 }}>
                  Test-split class balance
                </span>
                <span className="dim" style={{ fontSize: 11 }}>
                  {int(testTotal)} sequences
                </span>
              </div>
              <StatBar
                segments={[
                  { label: "Bot", value: testLabels.find((l) => l.label === "Bot")?.count ?? 0, color: "var(--crit)" },
                  { label: "Benign", value: testLabels.find((l) => l.label === "Benign")?.count ?? 0, color: "var(--ok)" },
                  { label: "Other", value: testLabels.filter((l) => !["Bot", "Benign"].includes(l.label)).reduce((s, r) => s + r.count, 0), color: "var(--text-3)" },
                ]}
              />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
