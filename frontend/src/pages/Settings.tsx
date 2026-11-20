import { useEffect, useState } from "react";
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
import { api, type Settings as SettingsType } from "../lib/api";
import { int } from "../lib/format";
import { useApi } from "../lib/useApi";

export default function Settings() {
  const { data, error, loading, reload } = useApi(() => api.settings(), []);
  const { health } = useHealth();
  const toast = useToast();
  const [form, setForm] = useState<SettingsType | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (data) setForm(data.settings);
  }, [data]);

  if (loading || !form) {
    return (
      <div className="page">
        <PageHeader title="System Settings" />
        <LoadingBlock height={280} />
      </div>
    );
  }
  if (error) {
    return (
      <div className="page">
        <PageHeader title="System Settings" />
        <Panel title="Settings">
          <ErrorState message={error} onRetry={reload} />
        </Panel>
      </div>
    );
  }

  const set = <K extends keyof SettingsType>(k: K, v: SettingsType[K]) =>
    setForm((f) => (f ? { ...f, [k]: v } : f));

  const save = async () => {
    if (!form) return;
    setBusy(true);
    try {
      await api.saveSettings(form);
      toast({ tone: "ok", title: "Settings saved", desc: "Stored server-side under backend/var." });
      reload();
    } catch (e) {
      toast({
        tone: "err",
        title: "Could not save settings",
        desc: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    if (!data) return;
    setForm(data.defaults);
    toast({ tone: "neutral", title: "Defaults restored in the form", desc: "Save to persist." });
  };

  const present = health?.artifacts.filter((a) => a.present).length ?? 0;
  const total = health?.artifacts.length ?? 0;

  return (
    <div className="page">
      <PageHeader
        title="System Settings"
        desc="Analyst preferences and platform configuration. Detection thresholds shown here are display preferences and never rewrite recorded research results."
        actions={
          <>
            <button className="btn sm" onClick={reset}>
              Reset to defaults
            </button>
            <button className="btn primary sm" onClick={save} disabled={busy}>
              <Icon.Check size={13} /> {busy ? "Saving…" : "Save settings"}
            </button>
          </>
        }
      />

      <div className="stack">
        <div className="grid grid-2">
          <Panel title="Detection & review" sub="Analyst-facing preferences">
            <div className="stack" style={{ gap: 14 }}>
              <div className="field">
                <label htmlFor="threshold">Anomaly review threshold</label>
                <input
                  id="threshold"
                  className="input mono"
                  type="number"
                  step="0.0001"
                  min="0.0001"
                  max="100"
                  value={form.anomaly_review_threshold}
                  onChange={(e) => set("anomaly_review_threshold", Number(e.target.value))}
                />
                <div className="help">
                  Visual reference only. Recorded triage results keep the frozen{" "}
                  <span className="code">1.8814</span> calibration.
                </div>
              </div>

              <div className="field">
                <label htmlFor="pct">Review percentile</label>
                <input
                  id="pct"
                  className="input mono"
                  type="number"
                  step="0.5"
                  min="50.5"
                  max="99.9"
                  value={form.review_percentile}
                  onChange={(e) => set("review_percentile", Number(e.target.value))}
                />
                <div className="help">Percentile of benign validation residuals used for calibration.</div>
              </div>

              <div className="field">
                <label htmlFor="split">Default dataset split</label>
                <select
                  id="split"
                  className="select"
                  value={form.default_split}
                  onChange={(e) => set("default_split", e.target.value as SettingsType["default_split"])}
                >
                  <option value="train">Train</option>
                  <option value="validation">Validation</option>
                  <option value="test">Test</option>
                </select>
              </div>
            </div>
          </Panel>

          <Panel title="Interface" sub="Presentation preferences">
            <div className="stack" style={{ gap: 14 }}>
              <div className="field">
                <label htmlFor="page">Table page size</label>
                <input
                  id="page"
                  className="input mono"
                  type="number"
                  min="5"
                  max="200"
                  step="5"
                  value={form.table_page_size}
                  onChange={(e) => set("table_page_size", Number(e.target.value))}
                />
                <div className="help">Default rows per page across data tables.</div>
              </div>

              <div className="field">
                <label>Data source mode</label>
                <select
                  className="select"
                  value={form.telemetry_mode}
                  onChange={(e) =>
                    set("telemetry_mode", e.target.value as SettingsType["telemetry_mode"])
                  }
                >
                  <option value="offline_dataset">Offline dataset</option>
                  <option value="live" disabled>
                    Live telemetry (not implemented)
                  </option>
                </select>
                <div className="help">
                  Live ingestion is not implemented. The option is visible so the gap is explicit rather than
                  hidden.
                </div>
              </div>

              <label className="checkline">
                <input
                  type="checkbox"
                  checked={form.show_provenance_badges}
                  onChange={(e) => set("show_provenance_badges", e.target.checked)}
                />
                Show provenance badges on metrics
              </label>
            </div>
          </Panel>
        </div>

        <div className="grid grid-3">
          <Panel title="Frozen calibration" actions={<ProvenanceTag provenance="recorded_artifact" compact />}>
            <div className="kv">
              <div>Threshold</div>
              <div>1.881396770477295</div>
              <div>Percentile</div>
              <div>99.0</div>
              <div>Source</div>
              <div style={{ fontSize: 11 }}>Benign validation residuals</div>
              <div>Recalibrated on test</div>
              <div style={{ color: "var(--ok)" }}>No</div>
            </div>
            <div style={{ marginTop: 10 }}>
              <Note tone="warn">
                Changing the preference above does not alter this frozen calibration or any recorded metric.
              </Note>
            </div>
          </Panel>

          <Panel title="Artifact availability" actions={<Tag tone={present === total ? "ok" : "med"}>{present}/{total}</Tag>}>
            <p className="muted" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.6 }}>
              {total - present > 0
                ? `${total - present} artifact(s) are absent, mostly git-ignored binaries such as checkpoints and per-observation CSVs.`
                : "All registered artifacts are present."}
            </p>
            <div className="stack" style={{ gap: 7, marginTop: 12 }}>
              {health?.warnings.map((w) => (
                <div className="checkline" key={w} style={{ alignItems: "flex-start" }}>
                  <span style={{ color: "var(--high)", flex: "0 0 13px", marginTop: 2 }}>
                    <Icon.Alert size={12} />
                  </span>
                  <span style={{ fontSize: 11.5 }}>{w}</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Platform">
            <div className="kv">
              <div>Service</div>
              <div style={{ fontSize: 11 }}>{health?.service ?? "sentinelx-api"}</div>
              <div>Version</div>
              <div>{health?.version ?? "4.0.0"}</div>
              <div>Status</div>
              <div style={{ color: health?.status === "ok" ? "var(--ok)" : "var(--med)" }}>
                {health?.status ?? "unknown"}
              </div>
              <div>Data mode</div>
              <div style={{ fontSize: 11 }}>offline_dataset</div>
              <div>Live telemetry</div>
              <div style={{ color: "var(--text-3)" }}>unavailable</div>
            </div>
          </Panel>
        </div>

        <Panel title="Registered artifacts" sub={`${int(total)} entries · presence checked at request time`} flush>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Artifact</th>
                  <th>Path</th>
                  <th>Kind</th>
                  <th>Required for</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {health?.artifacts.map((a) => (
                  <tr key={a.id}>
                    <td style={{ color: "var(--text-0)" }}>{a.label}</td>
                    <td>
                      <span className="code">{a.path}</span>
                    </td>
                    <td>
                      <Tag tone="neutral">{a.provenance_kind}</Tag>
                    </td>
                    <td className="muted">{a.required_for}</td>
                    <td>
                      <Tag tone={a.present ? "ok" : "unavailable"} dot>
                        {a.present ? "present" : "absent"}
                      </Tag>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}
