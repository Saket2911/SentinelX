import { useState } from "react";
import { useHealth } from "../App";
import { Icon } from "../components/icons";
import {
  ErrorState,
  LoadingBlock,
  Note,
  PageHeader,
  Panel,
  ProvenanceTag,
  Tag,
  useToast,
} from "../components/ui";
import { api, type ModelsResponse, type ModelVersion } from "../lib/api";
import { humanSeconds, int, num } from "../lib/format";
import { useApi } from "../lib/useApi";

function Confirm({
  title,
  body,
  onCancel,
  onConfirm,
  busy,
}: {
  title: string;
  body: string;
  onCancel: () => void;
  onConfirm: () => void;
  busy: boolean;
}) {
  return (
    <>
      <div className="overlay" onClick={onCancel} />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        style={{
          position: "fixed",
          top: "22%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "min(460px, 92vw)",
          background: "var(--bg-1)",
          border: "1px solid var(--line-strong)",
          borderRadius: 10,
          boxShadow: "var(--shadow-2)",
          zIndex: 70,
        }}
      >
        <div className="panel-head">
          <h3 className="panel-title">{title}</h3>
        </div>
        <div style={{ padding: 16 }}>
          <p className="muted" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.6 }}>
            {body}
          </p>
          <div className="row" style={{ justifyContent: "flex-end", marginTop: 16 }}>
            <button className="btn sm" onClick={onCancel} disabled={busy}>
              Cancel
            </button>
            <button className="btn primary sm" onClick={onConfirm} disabled={busy}>
              {busy ? "Activating…" : "Activate model"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function VersionCard({
  v,
  onActivate,
}: {
  v: ModelVersion;
  onActivate: (v: ModelVersion) => void;
}) {
  const cfg = v.config ?? {};
  const train = (v.training ?? {}) as Record<string, unknown>;
  const metrics = v.test_evaluation?.lstm_metrics;
  const statusTone =
    v.status === "production_candidate" ? "med" : v.status === "experimental" ? "low" : "neutral";

  return (
    <Panel
      title={
        <span className="row" style={{ gap: 8 }}>
          {v.name}
          {v.is_active && <Tag tone="ok" dot>Active</Tag>}
        </span>
      }
      sub={v.note}
      actions={<Tag tone={statusTone}>{v.status.replace("_", " ")}</Tag>}
    >
      <div className="kv">
        <div>Input features</div>
        <div>{int(v.input_features)}</div>
        <div>Sequence length</div>
        <div>{int(v.sequence_length)}</div>
        <div>Hidden size</div>
        <div>{cfg.hidden_size ? int(cfg.hidden_size as number) : "—"}</div>
        <div>Best epoch</div>
        <div>{train.best_epoch ? int(train.best_epoch as number) : "—"}</div>
        <div>Best val. Huber</div>
        <div>{num((train.best_validation_huber as number) ?? null, 5)}</div>
        <div>Training time</div>
        <div>{train.training_seconds ? humanSeconds(train.training_seconds as number) : "—"}</div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div className="muted" style={{ fontSize: 11.5, marginBottom: 6 }}>
          Held-out test metrics
        </div>
        {metrics ? (
          <div className="row" style={{ gap: 18 }}>
            <div>
              <div className="dim" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.07em" }}>Huber</div>
              <div className="mono" style={{ fontSize: 15 }}>{num(metrics.huber_loss, 4)}</div>
            </div>
            <div>
              <div className="dim" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.07em" }}>MAE</div>
              <div className="mono" style={{ fontSize: 15 }}>{num(metrics.mae, 4)}</div>
            </div>
            <div>
              <div className="dim" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.07em" }}>RMSE</div>
              <div className="mono" style={{ fontSize: 15 }}>{num(metrics.rmse, 4)}</div>
            </div>
          </div>
        ) : (
          <div className="dim" style={{ fontSize: 12 }}>
            No structured test metrics recorded for this version.
          </div>
        )}
      </div>

      <div className="row-between" style={{ marginTop: 14 }}>
        <div className="row" style={{ gap: 6 }}>
          <Tag tone={v.checkpoint_present ? "ok" : "unavailable"} dot>
            {v.checkpoint_present ? "Checkpoint present" : "Checkpoint absent"}
          </Tag>
          <ProvenanceTag provenance="recorded_artifact" compact />
        </div>
        <button
          className="btn sm"
          disabled={v.is_active}
          onClick={() => onActivate(v)}
        >
          {v.is_active ? "Active" : "Activate"}
        </button>
      </div>
    </Panel>
  );
}

export default function Models() {
  const { data, error, loading, reload } = useApi<ModelsResponse>(() => api.models(), []);
  const { health, reload: reloadHealth } = useHealth();
  const toast = useToast();
  const [pending, setPending] = useState<ModelVersion | null>(null);
  const [busy, setBusy] = useState(false);

  if (loading) {
    return (
      <div className="page">
        <PageHeader title="Model Management" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <PageHeader title="Model Management" />
        <Panel title="Models">
          <ErrorState message={error ?? "No data"} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const activate = async () => {
    if (!pending) return;
    setBusy(true);
    try {
      const res = await api.activateModel(pending.id);
      toast({
        tone: "ok",
        title: `${pending.name} is now the active model`,
        desc: res.warning ?? "Activation changed the reporting context.",
      });
      setPending(null);
      reload();
      reloadHealth();
    } catch (e) {
      toast({
        tone: "err",
        title: "Activation failed",
        desc: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <PageHeader
        title="Model Management"
        desc="Every LSTM World Model version in the repository, with its own feature definition and recorded evaluation. Checkpoints are never overwritten from this interface."
        actions={
          <>
            <Tag tone="neutral">Active: {data.active_model.toUpperCase()}</Tag>
            <button className="btn sm" onClick={reload}>
              <Icon.Refresh /> Reload
            </button>
          </>
        }
      />

      <div className="stack">
        <Note tone="info" title="Versions are not interchangeable.">
          {data.comparison_policy}
        </Note>

        <div className="grid grid-3">
          {data.versions.map((v) => (
            <VersionCard key={v.id} v={v} onActivate={setPending} />
          ))}
        </div>

        <Panel title="Architecture" sub="Shared LSTM World Model design across versions">
          <div className="grid grid-3">
            <div>
              <div className="metric-label">Recurrent core</div>
              <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, margin: "6px 0 0" }}>
                Single-layer LSTM over ten consecutive one-second network states. The final hidden state is
                projected to a full next-state vector.
              </p>
            </div>
            <div>
              <div className="metric-label">Objective</div>
              <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, margin: "6px 0 0" }}>
                Learn P(S(t+1) | S(t-1..t-n)). The model predicts a numerical future state, not an attack
                category — a deliberate research choice, not an omission.
              </p>
            </div>
            <div>
              <div className="metric-label">Training regime</div>
              <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, margin: "6px 0 0" }}>
                AdamW with Huber loss, early stopping on validation loss, reproducible seed. The test split is
                never loaded during training.
              </p>
            </div>
          </div>
        </Panel>

        <Panel
          title="Inference readiness"
          sub="What the platform can and cannot do in this checkout"
          actions={<ProvenanceTag provenance="recorded_artifact" compact />}
        >
          <div className="stack" style={{ gap: 9 }}>
            {health?.models.map((m) => (
              <div className="row-between" key={m.id}>
                <div>
                  <span style={{ color: "var(--text-0)" }}>{m.name}</span>
                  <span className="dim" style={{ fontSize: 11.5, marginLeft: 8 }}>
                    {m.input_features} features · seq {m.sequence_length}
                  </span>
                </div>
                <div className="row" style={{ gap: 6 }}>
                  <Tag tone={m.loadable ? "ok" : "unavailable"} dot>
                    {m.loadable ? "loadable" : "checkpoint absent"}
                  </Tag>
                </div>
              </div>
            ))}
            {!health?.inference_available && (
              <Note tone="warn" title="On-demand inference is disabled.">
                No <span className="code">best_model.pt</span> is committed to the repository, so the API cannot
                load weights. All metrics shown across SentinelX come from recorded evaluation artifacts, which
                remain fully reproducible on a machine that has the checkpoint.
              </Note>
            )}
          </div>
        </Panel>
      </div>

      {pending && (
        <Confirm
          title={`Activate ${pending.name}?`}
          body={`This changes which version SentinelX reports as active. It does not retrain, modify or copy any checkpoint, and it does not mix feature definitions: ${pending.name} keeps its own ${pending.input_features}-feature contract. ${
            pending.checkpoint_present
              ? "Its checkpoint is present on disk."
              : "Its checkpoint is not present in this checkout, so activation changes reporting context only."
          }`}
          onCancel={() => setPending(null)}
          onConfirm={activate}
          busy={busy}
        />
      )}
    </div>
  );
}
