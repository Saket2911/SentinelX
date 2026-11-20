import { useState } from "react";
import { Icon } from "../components/icons";
import {
  DataTable,
  Drawer,
  ErrorState,
  LoadingBlock,
  Note,
  PageHeader,
  Panel,
  ProvenanceTag,
  Tag,
  useToast,
  type Column,
} from "../components/ui";
import { api, type GeneratedReport, type ReportArtifact } from "../lib/api";
import { useApi } from "../lib/useApi";

export default function Reports() {
  const catalog = useApi(() => api.reportCatalog(), []);
  const generated = useApi(() => api.generatedReports(), []);
  const toast = useToast();

  const [view, setView] = useState<{ title: string; content: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<GeneratedReport | null>(null);

  const openArtifact = async (a: ReportArtifact) => {
    try {
      const res = await api.reportArtifact(a.id);
      const content =
        typeof res.content === "string" ? res.content : JSON.stringify(res.content, null, 2);
      setView({ title: a.title, content });
    } catch (e) {
      toast({
        tone: "err",
        title: "Could not open artifact",
        desc: e instanceof Error ? e.message : String(e),
      });
    }
  };

  const generate = async () => {
    setBusy(true);
    try {
      const res = await api.generateReport();
      setResult(res);
      setView({ title: res.title, content: res.markdown });
      generated.reload();
      toast({
        tone: "ok",
        title: "Report generated",
        desc: `Built from recorded artifacts at ${new Date(res.generated_at).toLocaleTimeString()}.`,
      });
    } catch (e) {
      toast({
        tone: "err",
        title: "Report generation failed",
        desc: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  const cols: Column<ReportArtifact>[] = [
    { key: "title", header: "Artifact", render: (r) => <span style={{ color: "var(--text-0)", fontWeight: 520 }}>{r.title}</span> },
    { key: "path", header: "Path", render: (r) => <span className="code">{r.path}</span> },
    { key: "kind", header: "Format", render: (r) => <Tag tone="neutral">{r.kind.toUpperCase()}</Tag> },
    {
      key: "present",
      header: "Status",
      render: (r) => (
        <Tag tone={r.present ? "ok" : "unavailable"} dot>
          {r.present ? "present" : "absent"}
        </Tag>
      ),
    },
    {
      key: "actions",
      header: "",
      align: "right",
      render: (r) => (
        <button
          className="btn ghost sm"
          disabled={!r.present}
          onClick={() => openArtifact(r)}
          title={r.present ? "View artifact" : "Not present in this checkout"}
        >
          View
        </button>
      ),
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Reports & Analytics"
        desc="Inspect every recorded evaluation artifact, or generate a consolidated analytics summary. Generated reports are assembled from committed research outputs — no values are invented."
        actions={
          <>
            <button className="btn sm" onClick={() => { catalog.reload(); generated.reload(); }}>
              <Icon.Refresh /> Refresh
            </button>
            <button className="btn primary sm" onClick={generate} disabled={busy}>
              <Icon.Report size={13} /> {busy ? "Generating…" : "Generate summary report"}
            </button>
          </>
        }
      />

      <div className="stack">
        <Note tone="info" title="Provenance is preserved in every export.">
          Downloads reproduce the recorded metrics together with their source file, and restate the detection
          caveat so the report cannot be read as a validated detector result.
        </Note>

        <Panel
          title="Recorded evaluation artifacts"
          sub={catalog.data ? `${catalog.data.artifacts.filter((a) => a.present).length} of ${catalog.data.artifacts.length} present` : "Loading…"}
          actions={<ProvenanceTag provenance="recorded_artifact" compact />}
          flush
        >
          {catalog.loading ? (
            <div style={{ padding: 14 }}>
              <LoadingBlock lines={5} />
            </div>
          ) : catalog.error ? (
            <ErrorState message={catalog.error} onRetry={catalog.reload} />
          ) : (
            <DataTable
              columns={cols}
              rows={catalog.data?.artifacts ?? []}
              rowKey={(r) => r.id}
              pageSize={10}
              emptyTitle="No artifacts registered"
            />
          )}
        </Panel>

        <Panel title="Generated reports" sub="Server-side, downloadable as Markdown" flush>
          {generated.loading ? (
            <div style={{ padding: 14 }}>
              <LoadingBlock lines={3} />
            </div>
          ) : (generated.data?.reports.length ?? 0) === 0 ? (
            <div className="state">
              <div className="title">No reports generated yet</div>
              <div className="desc">
                Use “Generate summary report” to build one from the current recorded artifacts.
              </div>
            </div>
          ) : (
            <div className="list">
              {generated.data?.reports.map((r) => (
                <div className="list-item" key={r.id}>
                  <span style={{ color: "var(--accent)", marginTop: 2 }}>
                    <Icon.Report size={15} />
                  </span>
                  <div className="li-main">
                    <div className="li-title mono" style={{ fontSize: 12 }}>
                      {r.id}.md
                    </div>
                    <div className="li-meta">{(r.bytes / 1024).toFixed(1)} KB</div>
                  </div>
                  <a className="btn ghost sm" href={`/api/reports/generated/${r.id}`} download>
                    <Icon.Download /> Download
                  </a>
                </div>
              ))}
            </div>
          )}
        </Panel>

        {result && (
          <Panel title="Latest generated report" actions={<Tag tone="derived">derived</Tag>}>
            <div className="kv">
              <div>Title</div>
              <div style={{ fontSize: 12 }}>{result.title}</div>
              <div>Generated at</div>
              <div style={{ fontSize: 11 }}>{result.generated_at}</div>
              <div>Sections</div>
              <div>{result.sections.length}</div>
              <div>Identifier</div>
              <div style={{ fontSize: 11 }}>{result.id}</div>
            </div>
            <div style={{ marginTop: 12 }}>
              <a className="btn sm" href={`/api/reports/generated/${result.id}`} download>
                <Icon.Download /> Download Markdown
              </a>
            </div>
          </Panel>
        )}
      </div>

      {view && (
        <Drawer title={view.title} onClose={() => setView(null)}>
          <pre className="raw">{view.content}</pre>
        </Drawer>
      )}
    </div>
  );
}
