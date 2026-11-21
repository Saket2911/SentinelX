import { useRef, useState } from "react";

import { api } from "../api/client";
import type { DatasetValidation } from "../api/types";
import { DataTable, type Column } from "../components/DataTable";
import { PageHeader, Panel } from "../components/Panel";
import { HealthPill, Pill } from "../components/Status";
import {
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { useToast } from "../components/Toast";
import { bytes, number } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function DatasetsPage() {
  const toast = useToast();
  const fileInput = useRef<HTMLInputElement>(null);
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [validating, setValidating] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const overview = useApi(() => api.datasets(), "datasets");

  if (overview.loading && !overview.data) {
    return (
      <Panel title="Loading dataset registry">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (overview.error) return <ErrorState message={overview.error} onRetry={overview.reload} />;
  const data = overview.data;
  if (!data) return <EmptyState title="No dataset data returned." />;

  const artifactRows: Column<(typeof data.artifacts)[number]>[] = [
    { key: "label", label: "Artifact", render: (row) => <span className="strong">{row.label}</span> },
    { key: "path", label: "Path", render: (row) => <span className="mono small">{row.path}</span> },
    {
      key: "present",
      label: "Status",
      render: (row) => (
        <Pill tone={row.present ? "good" : "warning"} dot>
          {row.present ? "present" : "missing"}
        </Pill>
      ),
    },
    {
      key: "size",
      label: "Size",
      align: "right",
      sortValue: (row) => row.size_bytes,
      render: (row) => (row.present ? bytes(row.size_bytes) : "—"),
    },
  ];

  async function validate(datasetId: string) {
    setValidating(datasetId);
    try {
      const result = await api.validateDataset(datasetId);
      setValidation(result);
      toast.push({
        title: result.valid ? "Dataset passed validation" : "Dataset failed validation",
        message: datasetId,
        tone: result.valid ? "success" : "error",
      });
    } catch (cause) {
      toast.push({
        title: "Validation failed",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setValidating(null);
    }
  }

  async function upload(file: File) {
    setUploading(true);
    try {
      const result = await api.uploadDataset(file);
      setValidation(result.validation);
      overview.reload();
      toast.push({
        title: "Upload stored for validation",
        message: "Research artifacts were not modified.",
        tone: "success",
      });
    } catch (cause) {
      toast.push({
        title: "Upload rejected",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const splits = Object.entries(data.label_coverage);

  return (
    <>
      <PageHeader
        title="Dataset Management"
        subtitle="Frozen preprocessing artifacts, recorded label coverage, and validation of new flow CSVs against the 209-feature model contract."
        actions={
          <button type="button" className="btn" onClick={overview.reload}>
            Refresh
          </button>
        }
      />

      <div className="notice warning">
        <span className="notice-icon">!</span>
        <div>
          <strong>Uploads never replace research data.</strong> Files are written to{" "}
          <code>var/uploads</code> for schema and compatibility checking only. Training arrays,
          scalers and checkpoints under the dataset and model directories are untouched.
        </div>
      </div>

      <div className="grid two">
        <Panel title="Preprocessing artifacts" hint="Required for inference with frozen training statistics." flush>
          <DataTable
            columns={artifactRows}
            rows={data.artifacts}
            rowKey={(row) => row.key}
            emptyMessage="No dataset artifact definitions available."
          />
        </Panel>

        <Panel
          title="Upload for validation"
          hint="CSV only. Validated against the frozen feature definition when it is present."
        >
          <div className="stack">
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="input"
              disabled={uploading}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
              }}
            />
            <p className="small muted">
              {data.model_features
                ? `Feature definition loaded: ${data.model_features} features. Compatibility is verified exactly.`
                : "Feature definition (sentinelx_model_features_v3.json) is unavailable, so compatibility cannot be verified and uploads cannot be marked valid."}
            </p>
            <SourceLine source={data.source} />
          </div>
        </Panel>
      </div>

      {validation ? (
        <Panel
          title="Validation result"
          hint={validation.dataset}
          actions={
            <Pill tone={validation.valid ? "good" : "warning"} dot>
              {validation.valid ? "valid" : "not valid"}
            </Pill>
          }
        >
          <div className="grid two">
            <div className="stack" style={{ gap: 6 }}>
              {validation.checks.map((check) => (
                <div key={check.id} className="inline" style={{ justifyContent: "space-between" }}>
                  <span className="small">{check.label}</span>
                  <span className="inline">
                    {check.detail ? <span className="small muted">{check.detail}</span> : null}
                    <HealthPill state={check.status} />
                  </span>
                </div>
              ))}
            </div>
            <dl className="kv" style={{ gridTemplateColumns: "150px 1fr" }}>
              <dt>Rows sampled</dt>
              <dd>{number(validation.row_count)}</dd>
              <dt>Columns</dt>
              <dd>{number(validation.column_count)}</dd>
              <dt>Missing model features</dt>
              <dd>{number(validation.missing_features)}</dd>
              <dt>Additional columns</dt>
              <dd>{number(validation.extra_features)}</dd>
              <dt>Labels sampled</dt>
              <dd>
                {Object.entries(validation.label_counts)
                  .slice(0, 6)
                  .map(([label, count]) => `${label}: ${count}`)
                  .join(" · ") || "—"}
              </dd>
            </dl>
          </div>
          <div className="divider" style={{ margin: "12px 0" }} />
          <Limitations items={validation.limitations} title="Validation limits" />
        </Panel>
      ) : null}

      <Panel title="Known datasets" hint="Research files and validated uploads." flush>
        <DataTable
          columns={[
            { key: "name", label: "Name", render: (row) => <span className="strong">{row.name}</span> },
            {
              key: "kind",
              label: "Kind",
              render: (row) => (
                <Pill tone={row.kind === "research_artifact" ? "accent" : "info"}>{row.kind}</Pill>
              ),
            },
            {
              key: "size",
              label: "Size",
              align: "right",
              sortValue: (row) => row.size_bytes,
              render: (row) => bytes(row.size_bytes),
            },
            { key: "notes", label: "Notes", render: (row) => row.notes.join(" ") },
            {
              key: "actions",
              label: "",
              render: (row) => (
                <button
                  type="button"
                  className="btn small"
                  disabled={validating === row.id}
                  onClick={() => validate(row.id)}
                >
                  {validating === row.id ? "Validating…" : "Validate"}
                </button>
              ),
            },
          ]}
          rows={data.datasets}
          rowKey={(row) => row.id}
          emptyMessage="No dataset CSV is present. The research datasets are not distributed with the repository."
        />
      </Panel>

      <Panel title="Recorded label coverage" hint="From the committed coverage report; partitions have different label sets." flush>
        {splits.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Partition</th>
                  <th>Labels and observations</th>
                </tr>
              </thead>
              <tbody>
                {splits.map(([split, labels]) => (
                  <tr key={split}>
                    <td className="strong" style={{ textTransform: "capitalize" }}>
                      {split}
                    </td>
                    <td>
                      <div className="tag-row">
                        {Object.entries(labels)
                          .sort((a, b) => b[1] - a[1])
                          .map(([label, count]) => (
                            <Pill key={label} tone={label === "Benign" ? "good" : "neutral"}>
                              {label}: {number(count)}
                            </Pill>
                          ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No label coverage report available." />
        )}
      </Panel>

      <Panel title="Dataset limitations">
        <Limitations items={data.limitations} />
      </Panel>
    </>
  );
}
