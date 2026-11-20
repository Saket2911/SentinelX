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
import { api, type Datasets as DatasetsData } from "../lib/api";
import { int, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

interface PartitionRow {
  partition: string;
  total: number;
  benign: number;
  noFlows: number;
  attack: number;
}

export default function Datasets() {
  const { data, error, loading, reload } = useApi<DatasetsData>(() => api.datasets(), []);

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Dataset Management" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Dataset Management" />
        <Panel title="Datasets">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const rows: PartitionRow[] = Object.entries(data.partitions).map(([partition, counts]) => {
    const total = Object.values(counts).reduce((s, v) => s + v, 0);
    const benign = counts["Benign"] ?? 0;
    const noFlows = counts["NO_FLOWS"] ?? 0;
    return { partition, total, benign, noFlows, attack: total - benign - noFlows };
  });

  const cols: Column<PartitionRow>[] = [
    { key: "partition", header: "Partition", render: (r) => <span style={{ textTransform: "capitalize", color: "var(--text-0)" }}>{r.partition}</span> },
    { key: "total", header: "States", align: "right", sortable: true, render: (r) => int(r.total) },
    { key: "benign", header: "Benign", align: "right", sortable: true, render: (r) => int(r.benign) },
    { key: "attack", header: "Labelled attack", align: "right", sortable: true, render: (r) => int(r.attack) },
    { key: "noFlows", header: "No flows", align: "right", sortable: true, render: (r) => int(r.noFlows) },
    {
      key: "share",
      header: "Attack share",
      align: "right",
      sortable: true,
      sortValue: (r) => (r.total ? r.attack / r.total : 0),
      render: (r) => pct(r.total ? (r.attack / r.total) * 100 : 0, 2),
    },
    {
      key: "mix",
      header: "Composition",
      render: (r) => (
        <div style={{ width: 120 }}>
          <StatBar
            segments={[
              { label: "Benign", value: r.benign, color: "var(--ok)" },
              { label: "Attack", value: r.attack, color: "var(--crit)" },
              { label: "No flows", value: r.noFlows, color: "var(--text-3)" },
            ]}
          />
        </div>
      ),
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Dataset Management"
        desc="The datasets, partitions and preprocessing contract behind the active model. Uploads are disabled by design so a new dataset cannot silently replace the audited training artifacts."
        actions={
          <button className="btn sm" onClick={reload}>
            <Icon.Refresh /> Reload
          </button>
        }
      />

      <div className="stack">
        <div className="grid grid-4">
          <MetricCard label="Active feature width" value={int(data.feature_version.v3_features)} provenance={data.feature_version.provenance} hint="Numeric features per network state for V3" />
          <MetricCard label="Sequence length" value={int(data.feature_version.sequence_length)} provenance={data.feature_version.provenance} hint="Time steps fed to the LSTM" />
          <MetricCard label="Training sequences" value={int(data.feature_version.training_samples)} provenance={data.feature_version.provenance} />
          <MetricCard label="Validation sequences" value={int(data.feature_version.validation_samples)} provenance={data.feature_version.provenance} />
        </div>

        <div className="grid grid-2">
          {data.sources.map((s) => (
            <Panel
              key={s.id}
              title={s.name}
              sub={s.description}
              actions={<Tag tone={s.present_in_checkout ? "ok" : "unavailable"} dot>{s.present_in_checkout ? "present" : "not in checkout"}</Tag>}
            >
              <div className="kv">
                <div>Role</div>
                <div style={{ textTransform: "capitalize" }}>{s.role}</div>
                {s.original_rows && (
                  <>
                    <div>Original rows</div>
                    <div>{int(s.original_rows)}</div>
                  </>
                )}
                {s.original_columns && (
                  <>
                    <div>Original columns</div>
                    <div>{int(s.original_columns)}</div>
                  </>
                )}
                <div>Source files</div>
                <div style={{ fontSize: 11 }}>{s.files.length ? s.files.join(", ") : "—"}</div>
              </div>
            </Panel>
          ))}
        </div>

        <Panel
          title="Chronological partitions"
          sub="Counts recorded in the dataset coverage report"
          actions={<ProvenanceTag provenance={data.provenance} compact />}
          flush
        >
          <DataTable columns={cols} rows={rows} rowKey={(r) => r.partition} pageSize={10} />
        </Panel>

        <div className="grid grid-main-side">
          <Panel title="Preprocessing contract" sub="Frozen invariants the pipeline must preserve">
            <div className="stack" style={{ gap: 9 }}>
              {data.preprocessing_contract.map((c, i) => (
                <div className="checkline" key={c} style={{ alignItems: "flex-start" }}>
                  <span className="mono dim" style={{ flex: "0 0 16px", fontSize: 11 }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span>{c}</span>
                </div>
              ))}
            </div>
          </Panel>

          <div className="stack">
            <Panel title="Upload">
              <Note tone="warn" title="Disabled by design.">
                {data.upload.reason}
              </Note>
              <div style={{ marginTop: 12 }}>
                <button className="btn sm" disabled title="Upload is intentionally disabled in offline mode">
                  <Icon.Database size={13} /> Upload dataset
                </button>
                <div className="dim" style={{ fontSize: 11, marginTop: 6 }}>
                  {data.raw_files_present
                    ? "Raw CSVs detected on disk."
                    : "No raw CSVs present in this checkout."}
                </div>
              </div>
            </Panel>

            <Panel title="What a new dataset would require">
              <div className="stack" style={{ gap: 7 }}>
                {[
                  "CICFlowMeter-compatible flow schema",
                  "Timestamp and label columns preserved",
                  "Median imputation fitted on the new training split only",
                  "Scaler frozen before validation and test",
                  "Sequence regeneration within segment boundaries",
                ].map((x) => (
                  <div className="checkline" key={x} style={{ alignItems: "flex-start" }}>
                    <span style={{ color: "var(--accent)", flex: "0 0 13px", marginTop: 2 }}>
                      <Icon.Check size={12} />
                    </span>
                    <span style={{ fontSize: 12 }}>{x}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </div>
  );
}
