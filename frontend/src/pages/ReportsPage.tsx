import { useState } from "react";

import { api } from "../api/client";
import type { ReportSummary } from "../api/types";
import { DataTable, type Column } from "../components/DataTable";
import { Modal } from "../components/Overlay";
import { PageHeader, Panel } from "../components/Panel";
import { Pill } from "../components/Status";
import { EmptyState, ErrorState, Loading } from "../components/States";
import { useToast } from "../components/Toast";
import { bytes, timestamp, titleCase } from "../lib/format";
import { useApi } from "../hooks/useApi";

const SECTIONS = [
  { id: "summary", label: "Executive summary", hint: "Posture, metrics, interpretation limits" },
  { id: "model_evaluation", label: "Model evaluation", hint: "Architecture and held-out metrics" },
  { id: "anomaly", label: "Residual anomaly analysis", hint: "Calibration and recorded detection metrics" },
  { id: "triage", label: "Threat review candidates", hint: "Queue and analyst workflow state" },
  { id: "explainability", label: "Explainability", hint: "SHAP method and top attributions" },
  { id: "dataset", label: "Dataset status", hint: "Artifacts and label coverage" },
];

export function ReportsPage() {
  const toast = useToast();
  const listing = useApi(() => api.reports(), "reports");
  const [title, setTitle] = useState("SOC review");
  const [author, setAuthor] = useState("analyst");
  const [selected, setSelected] = useState<string[]>(["summary", "model_evaluation"]);
  const [generating, setGenerating] = useState(false);
  const [preview, setPreview] = useState<{ metadata: ReportSummary; markdown: string } | null>(null);

  async function generate() {
    if (!title.trim()) {
      toast.push({ title: "A report title is required", tone: "error" });
      return;
    }
    if (!selected.length) {
      toast.push({ title: "Select at least one section", tone: "error" });
      return;
    }
    setGenerating(true);
    try {
      const report = await api.createReport({ title: title.trim(), sections: selected, author });
      listing.reload();
      toast.push({ title: "Report generated", message: report.report_id, tone: "success" });
      const content = await api.reportContent(report.report_id);
      setPreview(content);
    } catch (cause) {
      toast.push({
        title: "Report generation failed",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setGenerating(false);
    }
  }

  async function open(reportId: string) {
    try {
      setPreview(await api.reportContent(reportId));
    } catch (cause) {
      toast.push({
        title: "Unable to open report",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    }
  }

  const columns: Column<ReportSummary>[] = [
    {
      key: "title",
      label: "Report",
      sortValue: (row) => row.title,
      render: (row) => <span className="strong">{row.title}</span>,
    },
    {
      key: "created_at",
      label: "Created",
      sortValue: (row) => row.created_at,
      render: (row) => timestamp(row.created_at),
    },
    { key: "author", label: "Author", render: (row) => row.author },
    { key: "run_id", label: "Run", render: (row) => <span className="mono small">{row.run_id}</span> },
    {
      key: "sections",
      label: "Sections",
      render: (row) => (
        <div className="tag-row">
          {row.sections.map((section) => (
            <Pill key={section} tone="neutral">
              {titleCase(section)}
            </Pill>
          ))}
        </div>
      ),
    },
    {
      key: "formats",
      label: "Size",
      align: "right",
      sortValue: (row) => row.size_bytes,
      render: (row) => bytes(row.size_bytes),
    },
    {
      key: "actions",
      label: "",
      render: (row) => (
        <div className="inline">
          <button type="button" className="btn small" onClick={() => open(row.report_id)}>
            View
          </button>
          <a className="btn small" href={api.reportDownloadUrl(row.report_id, "md")}>
            MD
          </a>
          <a className="btn small" href={api.reportDownloadUrl(row.report_id, "json")}>
            JSON
          </a>
        </div>
      ),
    },
  ];

  return (
    <>
      <PageHeader
        title="Reports & Analytics"
        subtitle="Reports are assembled from the same services that feed the console, so a report cannot contain numbers the dashboard would contradict."
        actions={
          <button type="button" className="btn" onClick={listing.reload}>
            Refresh
          </button>
        }
      />

      <div className="grid two">
        <Panel title="Generate report" hint="Sections are rendered from the active run's artifacts.">
          <div className="stack">
            <label className="field">
              <span className="field-label">Title</span>
              <input className="input" value={title} onChange={(event) => setTitle(event.target.value)} />
            </label>
            <label className="field">
              <span className="field-label">Author</span>
              <input className="input" value={author} onChange={(event) => setAuthor(event.target.value)} />
            </label>
            <div>
              <span className="field-label">Sections</span>
              <div className="stack" style={{ gap: 6, marginTop: 6 }}>
                {SECTIONS.map((section) => (
                  <label key={section.id} className="inline small" style={{ cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={selected.includes(section.id)}
                      style={{ accentColor: "var(--accent)" }}
                      onChange={(event) =>
                        setSelected((current) =>
                          event.target.checked
                            ? [...current, section.id]
                            : current.filter((value) => value !== section.id),
                        )
                      }
                    />
                    <span style={{ color: "var(--text-secondary)" }}>{section.label}</span>
                    <span className="muted">— {section.hint}</span>
                  </label>
                ))}
              </div>
            </div>
            <div>
              <button type="button" className="btn primary" onClick={generate} disabled={generating}>
                {generating ? "Generating…" : "Generate report"}
              </button>
            </div>
          </div>
        </Panel>

        <Panel title="Report policy" hint="What every generated report contains.">
          <ul className="limitations" style={{ color: "var(--text-secondary)" }}>
            <li>Provenance: active run, generation time, author and data source.</li>
            <li>Metrics are labelled with their measurement space and evaluation partition.</li>
            <li>Residual review candidates are explicitly not confirmed incidents.</li>
            <li>Dataset labels are described as reference annotations, not model output.</li>
            <li>Every report closes with the interpretation limits of its inputs.</li>
          </ul>
          <div className="divider" style={{ margin: "12px 0" }} />
          <p className="small muted">
            Reports are written to <code>var/reports</code> as Markdown with a JSON sidecar. They never
            modify research artifacts.
          </p>
        </Panel>
      </div>

      <Panel title="Generated reports" flush>
        {listing.loading && !listing.data ? (
          <Loading rows={3} />
        ) : listing.error ? (
          <ErrorState message={listing.error} onRetry={listing.reload} />
        ) : listing.data && listing.data.reports.length ? (
          <DataTable
            columns={columns}
            rows={listing.data.reports}
            rowKey={(row) => row.report_id}
            initialSort={{ key: "created_at", direction: "desc" }}
          />
        ) : (
          <EmptyState
            title="No reports generated yet"
            detail="Use the generator above; reports are stored locally for download and audit."
          />
        )}
      </Panel>

      {preview ? (
        <Modal
          title={preview.metadata.title}
          onClose={() => setPreview(null)}
          footer={
            <>
              <a className="btn" href={api.reportDownloadUrl(preview.metadata.report_id, "md")}>
                Download Markdown
              </a>
              <button type="button" className="btn primary" onClick={() => setPreview(null)}>
                Close
              </button>
            </>
          }
        >
          <div className="small muted">
            {timestamp(preview.metadata.created_at)} · {preview.metadata.author} ·{" "}
            {preview.metadata.report_id}
          </div>
          <pre
            className="mono"
            style={{
              whiteSpace: "pre-wrap",
              maxHeight: "52vh",
              overflow: "auto",
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: 12,
              fontSize: 11,
              lineHeight: 1.55,
              color: "var(--text-secondary)",
            }}
          >
            {preview.markdown}
          </pre>
        </Modal>
      ) : null}
    </>
  );
}
