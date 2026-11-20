import { useMemo, useState } from "react";
import { SERIES, SimpleBar } from "../components/charts";
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
  SearchInput,
  Segmented,
  StatBar,
  Tag,
  type Column,
} from "../components/ui";
import { api, type NetworkOverview } from "../lib/api";
import { compact, int, pct } from "../lib/format";
import { useApi } from "../lib/useApi";

type Split = "train" | "validation" | "test";

interface Row {
  label: string;
  count: number;
  share: number;
}

export default function NetworkMonitoring() {
  const { data, error, loading, reload } = useApi<NetworkOverview>(() => api.network(), []);
  const [split, setSplit] = useState<Split>("test");
  const [q, setQ] = useState("");

  const rows: Row[] = useMemo(() => {
    if (!data) return [];
    const total = data.partitions[split]?.total ?? 0;
    return data.labels
      .filter((l) => l.partition === split)
      .map((l) => ({ label: l.label, count: l.count, share: total ? (l.count / total) * 100 : 0 }))
      .sort((a, b) => b.count - a.count);
  }, [data, split]);

  const filtered = useMemo(
    () => rows.filter((r) => r.label.toLowerCase().includes(q.toLowerCase())),
    [rows, q],
  );

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Network Monitoring" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Network Monitoring" />
        <Panel title="Network overview">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const p = data.partitions[split];
  const chartData = rows.slice(0, 9).map((r) => ({ label: r.label, count: r.count }));

  const cols: Column<Row>[] = [
    {
      key: "label",
      header: "Dataset label",
      render: (r) => <span style={{ color: "var(--text-0)", fontWeight: 520 }}>{r.label}</span>,
    },
    { key: "count", header: "Sequences", align: "right", sortable: true },
    {
      key: "share",
      header: "Share of split",
      align: "right",
      sortable: true,
      sortValue: (r) => r.share,
      render: (r) => <span className="mono">{pct(r.share, 2)}</span>,
    },
    {
      key: "bar",
      header: "Distribution",
      render: (r) => (
        <div style={{ width: 130 }}>
          <StatBar
            segments={[
              {
                label: r.label,
                value: r.count,
                color: r.label === "Benign" ? "var(--ok)" : r.label === "NO_FLOWS" ? "var(--text-3)" : "var(--crit)",
              },
            ]}
          />
        </div>
      ),
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Network Monitoring"
        desc="Network composition reconstructed from the CIC-IDS-2018 flow dataset as used by the SentinelX pipeline. Counts are recorded dataset annotations for one-second network states."
        actions={
          <>
            <Segmented
              value={split}
              onChange={(v) => setSplit(v)}
              options={[
                { value: "train", label: "Train" },
                { value: "validation", label: "Validation" },
                { value: "test", label: "Test" },
              ]}
            />
            <button className="btn sm" onClick={reload}>
              <Icon.Refresh /> Reload
            </button>
          </>
        }
      />

      <div className="stack">
        {p && (
          <div className="grid grid-4">
            <MetricCard
              label="Total states"
              value={int(p.total)}
              provenance="recorded_artifact"
              source={data.source}
              hint="One-second network states in this partition"
            />
            <MetricCard
              label="Benign"
              value={int(p.benign)}
              provenance="recorded_artifact"
              source={data.source}
              hint="States labelled Benign"
            />
            <MetricCard
              label="Labelled attack"
              value={int(p.labelled_attack)}
              provenance="recorded_artifact"
              source={data.source}
              hint="States carrying an explicit attack label in the dataset"
            />
            <MetricCard
              label="Attack share"
              value={pct(p.attack_share_percent, 2)}
              provenance="derived"
              hint="Derived from the recorded label counts above"
            />
          </div>
        )}

        <div className="grid grid-main-side">
          <Panel
            title="Label distribution"
            sub={`${split} partition · ${int(p?.total ?? 0)} states`}
            actions={<ProvenanceTag provenance="recorded_artifact" source={data.source} compact />}
          >
            <SimpleBar
              data={chartData}
              xKey="label"
              yKey="count"
              name="States"
              color={SERIES.blue}
              layout="vertical"
              height={270}
              formatter={compact}
            />
            <div className="chart-note">
              Attack labels present across the full corpus. The validation and test partitions each contain a
              single attack family, which constrains any detection claim made from them.
            </div>
          </Panel>

          <div className="stack">
            <Panel title="Flow-level detail">
              <EmptyState
                icon={<Icon.Network size={22} />}
                title="Per-flow records unavailable"
                desc={data.flow_level_detail.reason}
              />
              <div style={{ marginTop: 10 }}>
                <ProvenanceTag provenance={data.flow_level_detail.provenance} compact />
              </div>
            </Panel>

            <Panel title="Ingestion path">
              <div className="stack" style={{ gap: 9 }}>
                <div className="row-between">
                  <span className="muted">Offline dataset mode</span>
                  <Tag tone="ok" dot>
                    Implemented
                  </Tag>
                </div>
                <div className="row-between">
                  <span className="muted">PCAP / Zeek ingestion</span>
                  <Tag tone="unavailable">Not implemented</Tag>
                </div>
                <div className="row-between">
                  <span className="muted">Live flow capture</span>
                  <Tag tone="unavailable">Not implemented</Tag>
                </div>
              </div>
              <Note>
                A live path must reuse the same 71 source features aggregated into the 209/212-feature state
                contract. No live telemetry is claimed today.
              </Note>
            </Panel>
          </div>
        </div>

        <Panel
          title="All labels in this partition"
          sub={`${filtered.length} of ${rows.length} labels`}
          actions={
            <div style={{ width: 210 }}>
              <SearchInput value={q} onChange={setQ} placeholder="Filter labels" />
            </div>
          }
          flush
        >
          <DataTable
            columns={cols}
            rows={filtered}
            rowKey={(r) => r.label}
            pageSize={10}
            initialSort={{ key: "count", dir: "desc" }}
            emptyTitle="No labels match this filter"
            emptyDesc="Clear the search box to see the full recorded label set."
          />
        </Panel>
      </div>
    </div>
  );
}
