import { useEffect, useState } from "react";

import { api } from "../api/client";
import { PageHeader, Panel } from "../components/Panel";
import { HealthPill, Pill } from "../components/Status";
import { EmptyState, ErrorState, Loading } from "../components/States";
import { useToast } from "../components/Toast";
import { decimal } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function SettingsPage() {
  const toast = useToast();
  const settings = useApi(() => api.settings(), "settings");
  const health = useApi(() => api.health(), "health-detail");
  const capabilities = useApi(() => api.capabilities(), "capabilities");

  const [analystName, setAnalystName] = useState("analyst");
  const [pageSize, setPageSize] = useState(25);
  const [timezone, setTimezone] = useState("UTC");
  const [threshold, setThreshold] = useState<string>("");
  const [notify, setNotify] = useState(true);
  const [saving, setSaving] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  const loaded = settings.data;
  useEffect(() => {
    if (!loaded || hydrated) return;
    setAnalystName(loaded.settings.analyst_name);
    setPageSize(loaded.settings.page_size);
    setTimezone(loaded.settings.timezone);
    setThreshold(loaded.settings.review_threshold?.toString() ?? "");
    setNotify(loaded.settings.notify_on_review_required);
    setHydrated(true);
  }, [loaded, hydrated]);

  async function save() {
    setSaving(true);
    try {
      const parsedThreshold = threshold.trim() === "" ? null : Number(threshold);
      if (parsedThreshold !== null && (Number.isNaN(parsedThreshold) || parsedThreshold <= 0)) {
        toast.push({ title: "Review threshold must be a positive number", tone: "error" });
        return;
      }
      const updated = await api.updateSettings({
        analyst_name: analystName,
        page_size: pageSize,
        timezone,
        review_threshold: parsedThreshold,
        notify_on_review_required: notify,
      });
      settings.reload();
      toast.push({
        title: "Preferences saved",
        message: updated.threshold_source.analyst_override_in_use
          ? "An analyst display override is active; research calibration is unchanged."
          : "Using the run's calibrated review threshold.",
        tone: "success",
      });
    } catch (cause) {
      toast.push({
        title: "Save failed",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setSaving(false);
    }
  }

  if (settings.loading && !loaded) {
    return (
      <Panel title="Loading settings">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (settings.error) return <ErrorState message={settings.error} onRetry={settings.reload} />;
  if (!loaded) return <EmptyState title="No settings returned." />;

  return (
    <>
      <PageHeader
        title="System Settings"
        subtitle="Analyst preferences, review threshold policy, deployment health and an honest capability matrix."
        actions={
          <button type="button" className="btn primary" onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save preferences"}
          </button>
        }
      />

      <div className="grid two">
        <Panel title="Preferences" hint="Stored locally in var/settings.json.">
          <div className="stack">
            <label className="field">
              <span className="field-label">Analyst name</span>
              <input
                className="input"
                value={analystName}
                onChange={(event) => setAnalystName(event.target.value)}
              />
              <span className="field-help">Used as the actor for case updates and report authorship.</span>
            </label>
            <div className="grid two" style={{ gap: 12 }}>
              <label className="field">
                <span className="field-label">Page size</span>
                <input
                  className="input"
                  type="number"
                  min={5}
                  max={200}
                  value={pageSize}
                  onChange={(event) => setPageSize(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">Timezone label</span>
                <input
                  className="input"
                  value={timezone}
                  onChange={(event) => setTimezone(event.target.value)}
                />
              </label>
            </div>
            <label className="field">
              <span className="field-label">Review threshold override</span>
              <input
                className="input"
                type="number"
                step="0.0001"
                min="0"
                placeholder={decimal(loaded.threshold_source.calibrated_threshold)}
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
              />
              <span className="field-help">
                Display and filter preference only. Research artifacts keep their frozen calibration.
              </span>
            </label>
            <label className="inline small" style={{ cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={notify}
                style={{ accentColor: "var(--accent)" }}
                onChange={(event) => setNotify(event.target.checked)}
              />
              Notify when observations require review
            </label>
            <div className="inline">
              <Pill tone={loaded.auth_enabled ? "good" : "warning"} dot>
                {loaded.auth_enabled ? "API key required for writes" : "write access open (local mode)"}
              </Pill>
              <span className="small muted">
                Set <code>SENTINELX_API_KEY</code> to require an X-API-Key header for mutations.
              </span>
            </div>
          </div>
        </Panel>

        <Panel title="Calibration provenance" hint={loaded.threshold_source.source}>
          <dl className="kv">
            <dt>Calibrated threshold</dt>
            <dd>{decimal(loaded.threshold_source.calibrated_threshold)}</dd>
            <dt>Analyst override</dt>
            <dd>
              {loaded.threshold_source.analyst_override_in_use ? (
                <Pill tone="warning" dot>active</Pill>
              ) : (
                <Pill tone="good" dot>not used</Pill>
              )}
            </dd>
            {Object.entries(loaded.threshold_source.quantiles).map(([key, value]) => (
              <div key={key} style={{ display: "contents" }}>
                <dt>Residual p{key}</dt>
                <dd>{decimal(value)}</dd>
              </div>
            ))}
          </dl>
          <div className="divider" style={{ margin: "12px 0" }} />
          <p className="small muted">{loaded.threshold_source.note}</p>
        </Panel>
      </div>

      <Panel title="Deployment health" hint={health.data ? `uptime ${health.data.uptime_seconds.toFixed(0)} s` : "checking"} flush>
        {health.data ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Check</th>
                  <th>Status</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {health.data.checks.map((check) => (
                  <tr key={check.id}>
                    <td className="strong">{check.label}</td>
                    <td>
                      <HealthPill state={check.status} />
                    </td>
                    <td className="small">{check.detail ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : health.error ? (
          <ErrorState message={health.error} onRetry={health.reload} />
        ) : (
          <Loading rows={3} />
        )}
      </Panel>

      <Panel title="Capability matrix" hint="Implemented, conditional and planned — no feature is claimed beyond what ships.">
        {capabilities.data ? (
          <div className="stack" style={{ gap: 9 }}>
            {capabilities.data.capabilities.map((capability) => (
              <div key={capability.id} className="inline" style={{ justifyContent: "space-between", gap: 16 }}>
                <div>
                  <div className="small strong">{capability.label}</div>
                  <div className="small muted">{capability.detail}</div>
                </div>
                <Pill
                  tone={
                    capability.state === "implemented"
                      ? "good"
                      : capability.state === "conditional"
                        ? "warning"
                        : "neutral"
                  }
                  dot
                >
                  {capability.state}
                </Pill>
              </div>
            ))}
            <div className="divider" />
            <ul className="limitations">
              {capabilities.data.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </div>
        ) : (
          <Loading rows={4} />
        )}
      </Panel>
    </>
  );
}
