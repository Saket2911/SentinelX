import { api } from "../api/client";
import { ActivityChart } from "../components/Charts";
import { MetricCard } from "../components/Metric";
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
import { decimal, number, percent } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function NetworkPage() {
  const { data, error, loading, reload } = useApi(() => api.network(), "network");

  if (loading && !data) {
    return (
      <Panel title="Loading network telemetry">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!data) return <EmptyState title="No network data returned." />;

  const totals = data.totals;

  return (
    <>
      <PageHeader
        title="Network Monitoring"
        subtitle="Aggregated one-second network states from the recorded capture, plus raw flow-plane analytics when the CIC-IDS-2018 flow CSV is available locally."
        actions={
          <button type="button" className="btn" onClick={reload}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice
        availability={data.availability}
        title="State-level network telemetry is unavailable"
      />

      {data.availability.available ? (
        <>
          <div className="grid metrics">
            <MetricCard
              metric={{
                id: "observations",
                label: "Observed states",
                value: totals.observations ?? null,
                display: number(totals.observations ?? null),
                unit: "1 s states",
                status: "neutral",
                detail: "Aggregated windows, not packet counts.",
              }}
            />
            <MetricCard
              metric={{
                id: "benign",
                label: "Labelled benign",
                value: totals.benign_observations ?? null,
                display: number(totals.benign_observations ?? null),
                unit: "states",
                status: "neutral",
                detail: "Dataset reference labels.",
              }}
            />
            <MetricCard
              metric={{
                id: "attack",
                label: "Labelled attack",
                value: totals.attack_labelled_observations ?? null,
                display: number(totals.attack_labelled_observations ?? null),
                unit: "states",
                status: (totals.attack_labelled_observations ?? 0) > 0 ? "warning" : "neutral",
                detail: "Dataset reference labels, not model output.",
              }}
            />
            <MetricCard
              metric={{
                id: "segments",
                label: "Capture segments",
                value: totals.segments ?? null,
                display: number(totals.segments ?? null),
                unit: "segments",
                status: "neutral",
                detail: "Independent contiguous captures.",
              }}
            />
            <MetricCard
              metric={{
                id: "review",
                label: "Review candidates",
                value: totals.review_candidates ?? null,
                display: number(totals.review_candidates ?? null),
                unit: "states",
                status: (totals.review_candidates ?? 0) > 0 ? "critical" : "good",
                detail: `Residual ≥ ${decimal(totals.frozen_threshold ?? null, 4)}`,
              }}
            />
            <MetricCard
              metric={{
                id: "mean_residual",
                label: "Mean residual",
                value: totals.mean_residual ?? null,
                display: decimal(totals.mean_residual ?? null),
                unit: "model space",
                status: "neutral",
                detail: "Mean absolute next-state prediction error.",
              }}
            />
          </div>

          <Panel
            title="Residual activity over the capture"
            hint="Bucket means of the prediction residual with the attack-labelled share."
            actions={<SourceLine source={data.source} />}
          >
            {data.activity_series.length ? (
              <ActivityChart data={data.activity_series} height={260} />
            ) : (
              <EmptyState title="No residual series available." />
            )}
          </Panel>

          <div className="grid two">
            <Panel title="Label mix" hint="Dataset reference annotations per state window.">
              {data.label_distribution.length ? (
                <div className="stack" style={{ gap: 8 }}>
                  {data.label_distribution.map((entry) => (
                    <div key={entry.name} className="bar-row">
                      <span className="bar-label">{entry.name}</span>
                      <span className="bar-track">
                        <span className="bar-fill" style={{ width: `${entry.share * 100}%` }} />
                      </span>
                      <span className="bar-value">
                        {number(entry.count)} · {percent(entry.share * 100, 1)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState title="No labels in the state table." />
              )}
            </Panel>

            <Panel
              title="Flow-plane analytics"
              hint="Protocol and destination-port distributions from the raw flow capture."
            >
              {data.protocol_distribution.length || data.top_ports.length ? (
                <div className="stack">
                  {data.protocol_distribution.length ? (
                    <div>
                      <span className="section-label">Protocols</span>
                      <div className="stack" style={{ gap: 7, marginTop: 6 }}>
                        {data.protocol_distribution.map((entry) => (
                          <div key={entry.name} className="bar-row">
                            <span className="bar-label">{entry.name}</span>
                            <span className="bar-track">
                              <span
                                className="bar-fill info"
                                style={{ width: `${entry.share * 100}%` }}
                              />
                            </span>
                            <span className="bar-value">{percent(entry.share * 100, 1)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {data.top_ports.length ? (
                    <div>
                      <span className="section-label">Top destination ports</span>
                      <div className="tag-row" style={{ marginTop: 6 }}>
                        {data.top_ports.map((entry) => (
                          <Pill key={entry.name} tone="neutral">
                            {entry.name} · {number(entry.count)}
                          </Pill>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <EmptyState
                  title="Raw flow capture not available"
                  detail="Aggregated states carry no protocol or port fields. Place the CIC-IDS-2018 flow CSV in the dataset directory to enable this panel."
                />
              )}
            </Panel>
          </div>

          <Panel title="Segments" hint="Gap-separated capture segments with per-segment label mix." flush>
            {data.segments.length ? (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Segment</th>
                      <th className="num">States</th>
                      <th className="num">First step</th>
                      <th className="num">Last step</th>
                      <th className="num">Mean residual</th>
                      <th>Labels</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.segments.map((segment) => (
                      <tr key={segment.segment_id}>
                        <td className="mono strong">{segment.segment_id}</td>
                        <td className="num">{number(segment.observations)}</td>
                        <td className="num">{number(segment.first_step)}</td>
                        <td className="num">{number(segment.last_step)}</td>
                        <td className="num">{decimal(segment.mean_residual)}</td>
                        <td>
                          {Object.entries(segment.labels).map(([label, count]) => (
                            <Pill key={label} tone="neutral">
                              {label}: {number(count)}
                            </Pill>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="No segment metadata." />
            )}
          </Panel>

          {data.flow_features.length ? (
            <Panel title="Feature-level comparison" hint="Actual versus predicted means for recognised feature pairs." flush>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th className="num">Mean actual</th>
                      <th className="num">Mean predicted</th>
                      <th className="num">Mean |error|</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.flow_features.map((row) => (
                      <tr key={String(row.feature)}>
                        <td className="mono">{String(row.feature)}</td>
                        <td className="num">{decimal(row.mean_actual as number | null)}</td>
                        <td className="num">{decimal(row.mean_predicted as number | null)}</td>
                        <td className="num">{decimal(row.mean_abs_error as number | null)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          ) : null}

          <Panel title="Interpretation limits">
            <Limitations items={data.limitations} />
          </Panel>
        </>
      ) : (
        <Panel title="Interpretation limits">
          <Limitations items={data.limitations} />
        </Panel>
      )}
    </>
  );
}
