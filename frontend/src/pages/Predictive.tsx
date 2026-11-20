import { useMemo, useState } from "react";
import { GroupedBars, SERIES, TrendLine } from "../components/charts";
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
  Segmented,
  Tag,
  type Column,
} from "../components/ui";
import { api, type MultiStepRow, type Predictions } from "../lib/api";
import { int, num, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

const HORIZON_ALL = "ALL";

export default function Predictive() {
  const { data, error, loading, reload } = useApi<Predictions>(() => api.predictions(), []);
  const [label, setLabel] = useState<string>(HORIZON_ALL);

  const labels = useMemo(() => {
    if (!data) return [] as string[];
    return Array.from(new Set(data.multi_step.rows.map((r) => r.target_label)));
  }, [data]);

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Predictive Intelligence" />
        <LoadingBlock height={300} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Predictive Intelligence" />
        <Panel title="Predictions">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const one = data.one_step;
  const model = one.model ?? {};
  const persistence = one.persistence ?? {};
  const improvement = one.improvement_percent ?? {};

  const rows = data.multi_step.rows.filter((r) => r.target_label === label);
  const byHorizon = [...rows].sort((a, b) => a.horizon_seconds - b.horizon_seconds);

  const horizonChart = byHorizon.map((r) => ({
    horizon: `${r.horizon_seconds}s`,
    lstm: r.lstm_mae,
    persistence: r.persistence_mae,
  }));

  const quantiles = data.error_distribution.quantiles;
  const quantChart = quantiles
    ? Object.keys(quantiles["sample_mae"] ?? {}).map((q) => ({
        q: `p${(parseFloat(q) * 100).toFixed(1).replace(/\.0$/, "")}`,
        mae: quantiles["sample_mae"][q],
        rmse: quantiles["sample_rmse"][q],
      }))
    : [];

  const cols: Column<MultiStepRow>[] = [
    { key: "horizon_seconds", header: "Horizon", align: "right", sortable: true, render: (r) => `${r.horizon_seconds}s` },
    { key: "valid_origins", header: "Origins", align: "right", sortable: true, render: (r) => int(r.valid_origins) },
    { key: "lstm_mae", header: "LSTM MAE", align: "right", sortable: true, render: (r) => num(r.lstm_mae, 4) },
    { key: "persistence_mae", header: "Persistence MAE", align: "right", sortable: true, render: (r) => num(r.persistence_mae, 4) },
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
    {
      key: "rmse_improvement_percent",
      header: "RMSE gain",
      align: "right",
      sortable: true,
      render: (r) => (
        <span style={{ color: r.rmse_improvement_percent >= 0 ? "var(--ok)" : "var(--crit)" }}>
          {pct(r.rmse_improvement_percent, 2)}
        </span>
      ),
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Predictive Intelligence"
        desc="The LSTM World Model predicts the next numerical network state from the previous ten states. Performance is measured against a persistence baseline that simply repeats the last observed state."
        actions={
          <button className="btn sm" onClick={reload}>
            <Icon.Refresh /> Reload
          </button>
        }
      />

      <div className="stack">
        <div className="grid grid-4">
          <MetricCard label="One-step MAE" value={num(model.mae ?? null, 4)} provenance={one.provenance} source={one.source} hint="Held-out test split, 209-feature model space" />
          <MetricCard label="One-step RMSE" value={num(model.rmse ?? null, 4)} provenance={one.provenance} source={one.source} hint="Held-out test split" />
          <MetricCard label="MAE gain vs persistence" value={pct(improvement.mae ?? null, 2)} provenance={one.provenance} source={one.source} hint="Positive means the model beats repeating the last state" />
          <MetricCard label="Sequences evaluated" value={int(one.samples)} provenance={one.provenance} source={one.source} hint="Test sequences in the one-step evaluation" />
        </div>

        <div className="grid grid-main-side">
          <Panel
            title="One-step: model vs persistence"
            sub="Lower is better. Both rows use the same held-out test sequences."
            actions={<ProvenanceTag provenance={one.provenance} source={one.source} compact />}
          >
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="num">LSTM</th>
                    <th className="num">Persistence</th>
                    <th className="num">Gain</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Huber", "huber_loss"],
                    ["MAE", "mae"],
                    ["RMSE", "rmse"],
                  ].map(([labelText, key]) => (
                    <tr key={key}>
                      <td style={{ color: "var(--text-0)" }}>{labelText}</td>
                      <td className="num">{num(model[key], 5)}</td>
                      <td className="num">{num(persistence[key], 5)}</td>
                      <td className="num" style={{ color: "var(--ok)" }}>
                        {pct(improvement[key], 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="chart-note">
              Gains are the recorded improvement over the persistence baseline. They describe numerical
              forecasting accuracy, not attack-detection performance.
            </div>
          </Panel>

          <Panel
            title="Error distribution"
            sub="Per-sequence residual quantiles, test split"
            actions={<ProvenanceTag provenance={data.error_distribution.provenance} compact />}
          >
            {quantChart.length ? (
              <>
                <TrendLine
                  data={quantChart}
                  xKey="q"
                  series={[
                    { key: "mae", name: "Sample MAE", color: SERIES.accent },
                    { key: "rmse", name: "Sample RMSE", color: SERIES.blue },
                  ]}
                  height={196}
                />
                <div className="chart-note">
                  {data.error_distribution.note} {int(data.error_distribution.samples)} sequences,{" "}
                  {int(data.error_distribution.features)} features.
                </div>
              </>
            ) : (
              <div className="dim">No error quantiles recorded.</div>
            )}
          </Panel>
        </div>

        <Panel
          title="Recursive multi-step forecasting"
          sub="Each predicted state is fed back as the next input. Longer horizons accumulate error."
          actions={
            <div className="row wrap">
              <Segmented
                value={label}
                onChange={setLabel}
                options={labels.map((l) => ({ value: l, label: l }))}
              />
              <ProvenanceTag provenance={data.multi_step.provenance} source={data.multi_step.source} compact />
            </div>
          }
        >
          {horizonChart.length ? (
            <>
              <GroupedBars
                data={horizonChart}
                xKey="horizon"
                series={[
                  { key: "lstm", name: "LSTM MAE", color: SERIES.accent },
                  { key: "persistence", name: "Persistence MAE", color: SERIES.violet },
                ]}
                height={250}
                xLabel="Forecast horizon"
              />
              <div className="chart-note">{data.multi_step.note}</div>
            </>
          ) : (
            <div className="dim">No rows for this target label.</div>
          )}
        </Panel>

        <Panel title="Horizon detail" flush>
          <DataTable
            columns={cols}
            rows={rows}
            rowKey={(r) => `${r.target_label}-${r.horizon_seconds}`}
            pageSize={10}
            initialSort={{ key: "horizon_seconds", dir: "asc" }}
          />
        </Panel>

        <div className="grid grid-2">
          <Panel title="On-demand forecasting">
            <Note tone="warn" title="Not available in this checkout.">
              {data.live_forecast.reason}
            </Note>
            <div className="stack" style={{ gap: 6, marginTop: 12 }}>
              {data.live_forecast.requires.map((r) => (
                <div key={r} className="row" style={{ gap: 8 }}>
                  <Tag tone="unavailable">required</Tag>
                  <span className="code">{r}</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="V2 reference (212 features)" actions={<ProvenanceTag provenance={data.v2_reference.provenance} compact />}>
            <div className="kv">
              <div>Huber</div>
              <div>{num(data.v2_reference.metrics?.huber_loss, 5)}</div>
              <div>MAE</div>
              <div>{num(data.v2_reference.metrics?.mae, 5)}</div>
              <div>RMSE</div>
              <div>{num(data.v2_reference.metrics?.rmse, 5)}</div>
            </div>
            <div style={{ marginTop: 10 }}>
              <Note>{data.v2_reference.note}</Note>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
