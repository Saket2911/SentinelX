import { useState } from "react";

import { api } from "../api/client";
import type { ModelSummary } from "../api/types";
import { HealthPill, Pill } from "../components/Status";
import { ConfirmDialog, Drawer } from "../components/Overlay";
import { PageHeader, Panel } from "../components/Panel";
import {
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { useToast } from "../components/Toast";
import { bytes, decimal, number } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function ModelsPage() {
  const toast = useToast();
  const [selected, setSelected] = useState<string | null>(null);
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [compareRequested, setCompareRequested] = useState<{ left: string; right: string } | null>(null);

  const listing = useApi(() => api.models(), "models");

  if (listing.loading && !listing.data) {
    return (
      <Panel title="Loading model registry">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (listing.error) return <ErrorState message={listing.error} onRetry={listing.reload} />;
  const data = listing.data;
  if (!data) return <EmptyState title="No model registry data returned." />;


  return (
    <>
      <PageHeader
        title="Model Management"
        subtitle="Registered LSTM world-model runs, integrity checks, evaluation metrics and guarded activation. Checkpoints are never overwritten."
        actions={
          <button type="button" className="btn" onClick={listing.reload}>
            Refresh
          </button>
        }
      />

      <div className="notice info">
        <span className="notice-icon">i</span>
        <div>
          <strong>Active run: <code>{data.active_run_id}</code></strong>. Activation requires a local
          checkpoint and the 209-feature V3 contract; recorded metrics alone cannot serve inference.
          V1/V2 (212-feature) experiments are quarantined and never mixed with V3.
        </div>
      </div>

      {data.models.length ? (
        <div className="grid metrics">
          {data.models.map((model) => (
            <ModelCard
              key={model.run_id}
              model={model}
              onOpen={() => setSelected(model.run_id)}
            />
          ))}
        </div>
      ) : (
        <Panel title="Registered runs">
          <EmptyState
            title="No model runs found"
            detail="A run is registered when Models/<run>/config.json exists."
          />
        </Panel>
      )}

      <Panel title="Controlled comparison" hint="Compare two runs without blending their feature contracts.">
        <div className="toolbar">
          <label className="field" style={{ minWidth: 220 }}>
            <span className="field-label">Baseline run</span>
            <select className="select" value={left} onChange={(event) => setLeft(event.target.value)}>
              <option value="">Select a run…</option>
              {data.models.map((model) => (
                <option key={model.run_id} value={model.run_id}>
                  {model.run_id}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ minWidth: 220 }}>
            <span className="field-label">Candidate run</span>
            <select className="select" value={right} onChange={(event) => setRight(event.target.value)}>
              <option value="">Select a run…</option>
              {data.models.map((model) => (
                <option key={model.run_id} value={model.run_id}>
                  {model.run_id}
                </option>
              ))}
            </select>
          </label>
          <div style={{ alignSelf: "flex-end", paddingBottom: 2 }}>
            <button
              type="button"
              className="btn primary"
              disabled={!left || !right || left === right}
              onClick={() => setCompareRequested({ left, right })}
            >
              Compare
            </button>
          </div>
        </div>
        {compareRequested ? (
          <CompareResult left={compareRequested.left} right={compareRequested.right} />
        ) : null}
      </Panel>

      <Panel title="Registry limitations">
        <Limitations items={data.limitations} />
      </Panel>

      {selected ? (
        <ModelDrawer
          runId={selected}
          activeRunId={data.active_run_id}
          onClose={() => setSelected(null)}
          onActivated={() => {
            listing.reload();
            toast.push({ title: "Model activated", message: selected, tone: "success" });
          }}
        />
      ) : null}
    </>
  );
}

function CompareResult({ left, right }: { left: string; right: string }) {
  const comparison = useApi(() => api.compareModels(left, right), `compare-${left}-${right}`);
  if (comparison.loading) return <Loading rows={3} />;
  if (comparison.error) return <ErrorState message={comparison.error} onRetry={comparison.reload} />;
  if (!comparison.data) return null;
  return (
    <div className="stack" style={{ marginTop: 14 }}>
      {comparison.data.warnings.map((warning) => (
        <div key={warning} className="notice warning">
          <span className="notice-icon">!</span>
          <div>{warning}</div>
        </div>
      ))}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Metric</th>
              <th className="num">{comparison.data.left.run_id}</th>
              <th className="num">{comparison.data.right.run_id}</th>
              <th className="num">Delta</th>
            </tr>
          </thead>
          <tbody>
            {comparison.data.metrics.map((metric) => (
              <tr key={metric.id}>
                <td>{metric.label}</td>
                <td className="num">{decimal(metric.left)}</td>
                <td className="num">{decimal(metric.right)}</td>
                <td className="num">{decimal(metric.delta)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModelCard({ model, onOpen }: { model: ModelSummary; onOpen: () => void }) {
  const testMetrics = model.test_metrics?.lstm ?? {};
  return (
    <article className="panel" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="inline" style={{ justifyContent: "space-between" }}>
        <span className="mono small strong" style={{ overflowWrap: "anywhere" }}>
          {model.run_id}
        </span>
        {model.active ? <Pill tone="accent" dot>active</Pill> : <Pill tone="neutral">standby</Pill>}
      </div>
      <div className="small muted">{model.name}</div>
      <dl className="kv" style={{ gridTemplateColumns: "90px 1fr", fontSize: 11 }}>
        <dt>Features</dt>
        <dd>{model.architecture.input_features ?? "—"}</dd>
        <dt>Hidden</dt>
        <dd>{model.architecture.hidden_size ?? "—"}</dd>
        <dt>Best epoch</dt>
        <dd>{model.training.best_epoch ?? "—"}</dd>
        <dt>Test MAE</dt>
        <dd>{decimal(testMetrics.mae)}</dd>
        <dt>Checkpoint</dt>
        <dd>
          {model.checkpoint.present ? (
            <Pill tone="good" dot>present</Pill>
          ) : (
            <Pill tone="warning" dot>absent</Pill>
          )}
        </dd>
      </dl>
      <button type="button" className="btn small" onClick={onOpen}>
        Open run detail
      </button>
    </article>
  );
}

function ModelDrawer({
  runId,
  activeRunId,
  onClose,
  onActivated,
}: {
  runId: string;
  activeRunId: string;
  onClose: () => void;
  onActivated: () => void;
}) {
  const toast = useToast();
  const detail = useApi(() => api.model(runId), `model-${runId}`);
  const inference = useApi(() => api.inferenceStatus(runId), `inference-${runId}`);
  const [confirming, setConfirming] = useState(false);
  const [activating, setActivating] = useState(false);

  async function activate() {
    setActivating(true);
    try {
      await api.activateModel(runId, "Activated from the console after review");
      setConfirming(false);
      onActivated();
    } catch (cause) {
      toast.push({
        title: "Activation blocked",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setActivating(false);
    }
  }

  const data = detail.data;
  const test = data?.model.test_metrics;
  const health = data?.health ?? [];

  return (
    <Drawer
      title={runId}
      subtitle={data ? `${data.model.name} · ${data.model.active ? "active" : "standby"}` : "Loading run…"}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={runId === activeRunId || !data?.model.checkpoint.present}
            onClick={() => setConfirming(true)}
            title={
              data?.model.checkpoint.present
                ? undefined
                : "A checkpoint is required before activation"
            }
          >
            {runId === activeRunId ? "Already active" : "Activate run"}
          </button>
        </>
      }
    >
      {detail.loading && !data ? <Loading rows={5} /> : null}
      {detail.error ? <ErrorState message={detail.error} onRetry={detail.reload} /> : null}
      {data ? (
        <>
          <div className="tag-row">
            <Pill tone={data.availability.available ? "good" : "warning"} dot>
              {data.availability.available ? "artifacts available" : "partial artifacts"}
            </Pill>
            <Pill tone="neutral">
              {data.model.architecture.input_features} features · seq {data.model.architecture.sequence_length}
            </Pill>
            <Pill tone="neutral">{data.model.architecture.optimizer}</Pill>
          </div>

          {data.availability.reason ? (
            <div className="notice warning">
              <span className="notice-icon">!</span>
              <div>{data.availability.reason}</div>
            </div>
          ) : null}

          <div>
            <span className="section-label">Integrity checks</span>
            <div className="stack" style={{ gap: 6, marginTop: 6 }}>
              {health.map((check) => (
                <div key={check.id} className="inline" style={{ justifyContent: "space-between" }}>
                  <span className="small">{check.label}</span>
                  <HealthPill state={check.status} />
                </div>
              ))}
            </div>
          </div>

          <dl className="kv">
            <dt>Training samples</dt>
            <dd>{number(data.model.training.training_samples)}</dd>
            <dt>Validation samples</dt>
            <dd>{number(data.model.training.validation_samples)}</dd>
            <dt>Batch / LR</dt>
            <dd>
              {data.model.training.batch_size} · {data.model.training.learning_rate}
            </dd>
            <dt>Loss</dt>
            <dd>{data.model.architecture.loss ?? "—"}</dd>
            <dt>Best epoch</dt>
            <dd>{data.model.training.best_epoch ?? "—"}</dd>
            <dt>Validation Huber</dt>
            <dd>{decimal(data.model.training.best_validation_huber)}</dd>
            <dt>Device recorded</dt>
            <dd>
              {data.model.training.device ?? "—"} {data.model.training.gpu_name ? `(${data.model.training.gpu_name})` : ""}
            </dd>
            <dt>Training time</dt>
            <dd>
              {data.model.training.training_seconds
                ? `${decimal(data.model.training.training_seconds, 1)} s`
                : "—"}
            </dd>
          </dl>

          {test ? (
            <div>
              <span className="section-label">Held-out test metrics</span>
              <div className="table-wrap" style={{ marginTop: 6 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th className="num">LSTM</th>
                      <th className="num">Persistence</th>
                      <th className="num">Improvement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["MAE", "mae"],
                      ["RMSE", "rmse"],
                      ["Huber", "huber_loss"],
                    ].map(([label, key]) => (
                      <tr key={key}>
                        <td>{label}</td>
                        <td className="num">{decimal(test.lstm[key] ?? null)}</td>
                        <td className="num">{decimal(test.persistence[key] ?? null)}</td>
                        <td className="num">{decimal(test.improvement_percent[key] ?? null, 2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="small muted" style={{ marginTop: 6 }}>
                {number(test.samples)} sequences · {test.space}. Test partition was previously used for
                exploratory analysis.
              </p>
            </div>
          ) : null}

          {data.evaluation.calibration ? (
            <div>
              <span className="section-label">Residual calibration</span>
              <dl className="kv" style={{ marginTop: 6 }}>
                <dt>Frozen threshold</dt>
                <dd>{decimal(data.evaluation.calibration.threshold)}</dd>
                <dt>Source</dt>
                <dd>{data.evaluation.calibration.threshold_source}</dd>
                <dt>Method</dt>
                <dd>{data.evaluation.calibration.method}</dd>
              </dl>
            </div>
          ) : null}

          {data.multi_step.length ? (
            <div>
              <span className="section-label">Multi-step error (ALL labels)</span>
              <div className="tag-row" style={{ marginTop: 6 }}>
                {data.multi_step
                  .filter((row) => row.label === "ALL")
                  .map((row) => (
                    <Pill key={row.horizon_seconds} tone="neutral">
                      {row.horizon_seconds}s MAE {decimal(row.lstm_mae ?? null)}
                    </Pill>
                  ))}
              </div>
            </div>
          ) : null}

          <div>
            <span className="section-label">Inference runtime</span>
            <div className="stack" style={{ gap: 6, marginTop: 6 }}>
              {inference.data ? (
                <>
                  <div className="inline">
                    <Pill tone={inference.data.available ? "good" : "warning"} dot>
                      {inference.data.available ? "inference ready" : "inference unavailable"}
                    </Pill>
                    <Pill tone={inference.data.torch_available ? "good" : "neutral"}>
                      torch {inference.data.torch_available ? "present" : "missing"}
                    </Pill>
                  </div>
                  {inference.data.reasons.map((reason) => (
                    <div key={reason} className="small muted">
                      {reason}
                    </div>
                  ))}
                  {inference.data.checkpoint ? (
                    <code className="small" style={{ overflowWrap: "anywhere" }}>
                      {inference.data.checkpoint}
                    </code>
                  ) : null}
                </>
              ) : (
                <span className="small muted">Checking…</span>
              )}
            </div>
          </div>

          <div>
            <span className="section-label">Checkpoint files</span>
            {data.model.checkpoint.files.length ? (
              <div className="table-wrap" style={{ marginTop: 6 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th className="num">Size</th>
                      <th>SHA-256</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.model.checkpoint.files.map((file) => (
                      <tr key={file.path}>
                        <td className="mono">{file.name}</td>
                        <td className="num">{bytes(file.size_bytes)}</td>
                        <td className="mono small">{file.sha256?.slice(0, 16) ?? "—"}…</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="small muted">
                No checkpoint in this checkout. Research metrics remain available; live inference does
                not.
              </p>
            )}
          </div>

          <div>
            <span className="section-label">Artifacts</span>
            <div className="tag-row" style={{ marginTop: 6 }}>
              {data.artifacts.map((artifact) => (
                <Pill key={artifact.key} tone={artifact.present ? "good" : "neutral"} title={artifact.path}>
                  {artifact.label}
                </Pill>
              ))}
            </div>
          </div>

          <SourceLine source={data.source} />
          <Limitations items={data.evaluation.limitations} />
        </>
      ) : null}

      {confirming ? (
        <ConfirmDialog
          title="Activate this run?"
          busy={activating}
          message={
            <>
              <p>
                This makes <code>{runId}</code> the run served by every analytics endpoint.
              </p>
              <p className="small muted">
                Research artifacts are not modified. The change is recorded in the activation audit
                trail.
              </p>
            </>
          }
          confirmLabel="Activate"
          onConfirm={activate}
          onCancel={() => setConfirming(false)}
        />
      ) : null}
    </Drawer>
  );
}
