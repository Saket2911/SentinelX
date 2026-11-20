import { useMemo, useState } from "react";
import { SERIES, SimpleBar } from "../components/charts";
import { Icon } from "../components/icons";
import {
  DataTable,
  ErrorState,
  LoadingBlock,
  Note,
  PageHeader,
  Panel,
  ProvenanceTag,
  Segmented,
  Tag,
  type Column,
} from "../components/ui";
import { api, type Explainability as ExpData, type FeatureFamily } from "../lib/api";
import { int, num, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

const FAMILY_LABEL: Record<string, string> = {
  "Packet and flow counts": "Packet & flow counts",
  "Packet lengths and headers": "Packet lengths & headers",
  "Protocol and network-state indicators": "Protocol & state",
  "Rates and throughput": "Rates & throughput",
  "TCP flags": "TCP flags",
  Timing: "Timing",
};

export default function Explainability() {
  const { data, error, loading, reload } = useApi<ExpData>(() => api.explainability(), []);
  const [horizon, setHorizon] = useState<number | null>(null);

  const horizons = useMemo(() => {
    if (!data) return [] as number[];
    return Array.from(new Set(data.feature_families.map((f) => f.horizon_seconds))).sort((a, b) => a - b);
  }, [data]);

  const activeHorizon = horizon ?? horizons[0] ?? 1;
  const families = useMemo(
    () =>
      (data?.feature_families ?? [])
        .filter((f) => f.horizon_seconds === activeHorizon)
        .sort((a, b) => b.mae_improvement_percent - a.mae_improvement_percent),
    [data, activeHorizon],
  );

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Explainability Centre" />
        <LoadingBlock height={300} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Explainability Centre" />
        <Panel title="Explainability">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const chartData = families.map((f) => ({
    family: FAMILY_LABEL[f.feature_family] ?? f.feature_family,
    gain: Number(f.mae_improvement_percent.toFixed(2)),
  }));

  const cols: Column<FeatureFamily>[] = [
    { key: "feature_family", header: "Feature family", render: (r) => <span style={{ color: "var(--text-0)" }}>{FAMILY_LABEL[r.feature_family] ?? r.feature_family}</span> },
    { key: "feature_count", header: "Features", align: "right", sortable: true },
    {
      key: "lstm_better_count",
      header: "LSTM better",
      align: "right",
      sortable: true,
      render: (r) => `${r.lstm_better_count}/${r.feature_count}`,
    },
    { key: "lstm_mean_mae", header: "LSTM MAE", align: "right", sortable: true, render: (r) => num(r.lstm_mean_mae, 4) },
    { key: "persistence_mean_mae", header: "Baseline MAE", align: "right", sortable: true, render: (r) => num(r.persistence_mean_mae, 4) },
    {
      key: "mae_improvement_percent",
      header: "MAE gain",
      align: "right",
      sortable: true,
      render: (r) => (
        <span style={{ color: r.mae_improvement_percent >= 0 ? "var(--ok)" : "var(--crit)" }}>
          {pct(r.mae_improvement_percent, 2)}
        </span>
      ),
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Explainability Centre"
        desc="Why the model predicts what it predicts. Feature attributions describe model behaviour in preprocessed feature space. They are not attack probabilities and do not establish causality."
        actions={
          <button className="btn sm" onClick={reload}>
            <Icon.Refresh /> Reload
          </button>
        }
      />

      <div className="stack">
        <div className="grid grid-3">
          <Panel title="Recorded method">
            <div className="kv">
              <div>Method</div>
              <div style={{ fontSize: 11 }}>{data.method ?? "—"}</div>
              <div>Experiments</div>
              <div>{int(data.experiments.length)}</div>
              <div>Output features</div>
              <div>{int(data.output_features.length)}</div>
            </div>
          </Panel>

          <Panel title="Local explanation contract">
            <div className="kv">
              <div>Top contributors / observation</div>
              <div>{String(data.local_explanations["feature_contributors_per_observation"] ?? "—")}</div>
              <div>Temporal history window</div>
              <div>{String(data.local_explanations["temporal_history_window"] ?? "—")}</div>
              <div>Labels used for explanations</div>
              <div style={{ color: "var(--ok)" }}>No</div>
            </div>
          </Panel>

          <Panel title="Attribution caveats">
            <div className="stack" style={{ gap: 8 }}>
              {data.caveats.map((c) => (
                <div className="checkline" key={c} style={{ alignItems: "flex-start" }}>
                  <span style={{ color: "var(--text-3)", flex: "0 0 13px" }}>
                    <Icon.Info size={12} />
                  </span>
                  <span style={{ fontSize: 12 }}>{c}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        <Panel
          title="Feature-family forecasting contribution"
          sub="Which feature groups the model predicts better than the persistence baseline"
          actions={
            <div className="row wrap">
              <Segmented
                value={String(activeHorizon)}
                onChange={(v) => setHorizon(Number(v))}
                options={horizons.map((h) => ({ value: String(h), label: `${h}s` }))}
              />
              <ProvenanceTag provenance="recorded_artifact" compact />
            </div>
          }
        >
          <SimpleBar
            data={chartData}
            xKey="family"
            yKey="gain"
            name="MAE gain vs persistence (%)"
            color={SERIES.accent}
            layout="vertical"
            height={Math.max(200, chartData.length * 42)}
          />
          <div className="chart-note">
            Negative values mean the persistence baseline performed better for that family. TCP flags shows a
            negative gain at short horizons, which is reported rather than hidden.
          </div>
        </Panel>

        <Panel title={`Feature-family detail at ${activeHorizon}s`} flush>
          <DataTable
            columns={cols}
            rows={families}
            rowKey={(r) => `${r.horizon_seconds}-${r.feature_family}`}
            pageSize={8}
            initialSort={{ key: "mae_improvement_percent", dir: "desc" }}
          />
        </Panel>

        <div className="grid grid-2">
          <Panel title="SHAP experiment summary" actions={<ProvenanceTag provenance={data.provenance} compact />}>
            {data.experiments.length ? (
              <div className="stack" style={{ gap: 10 }}>
                {data.experiments.map((e) => (
                  <div key={e.name} className="row-between" style={{ alignItems: "flex-start" }}>
                    <div>
                      <div style={{ color: "var(--text-0)", fontWeight: 550 }}>{e.name}</div>
                      <div className="dim" style={{ fontSize: 11.5 }}>
                        {String(e.summary["background_source"] ?? "")} · background{" "}
                        {String(e.summary["background_size"] ?? "—")} · nsamples{" "}
                        {String(e.summary["nsamples"] ?? "—")}
                      </div>
                    </div>
                    <Tag tone="recorded">recorded</Tag>
                  </div>
                ))}
              </div>
            ) : (
              <div className="dim">No SHAP experiments recorded.</div>
            )}
          </Panel>

          <Panel title="Attribution values">
            <Note tone="warn" title="SHAP tensors are not in this checkout.">
              {data.shap_values.reason}
            </Note>
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ fontSize: 11.5, marginBottom: 6 }}>
                Recorded explanation targets
              </div>
              <div className="row wrap">
                {data.output_features.map((f) => (
                  <span className="code" key={f}>
                    {f}
                  </span>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
