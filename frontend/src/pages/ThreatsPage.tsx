import { useState } from "react";

import { api } from "../api/client";
import type { EventStatus, PlaybackAlert, SecurityEvent } from "../api/types";
import { DataTable, type Column } from "../components/DataTable";
import { FilterBar, SearchInput, SelectField, Toggle } from "../components/Filters";
import { Drawer } from "../components/Overlay";
import { PageHeader, Panel } from "../components/Panel";
import { PlaybackAlertDrawer, PlaybackAlertList } from "../components/PlaybackAlert";
import { CaseStatusPill, Pill, PriorityPill } from "../components/Status";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
} from "../components/States";
import { useToast } from "../components/Toast";
import { decimal, number, timestamp, titleCase } from "../lib/format";
import { useApi } from "../hooks/useApi";

const STATUSES: EventStatus[] = ["new", "investigating", "escalated", "benign", "closed"];

export function ThreatsPage() {
  const toast = useToast();
  const [status, setStatus] = useState("");
  const [priority, setPriority] = useState("");
  const [label, setLabel] = useState("");
  const [search, setSearch] = useState("");
  const [includeTop, setIncludeTop] = useState(false);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<PlaybackAlert | null>(null);

  const playbackAlerts = useApi(
    () => api.playbackAlerts({ limit: 200 }),
    "threat-playback-alerts",
  );

  const listing = useApi(
    () =>
      api.events({
        status: status || undefined,
        priority: priority || undefined,
        label: label || undefined,
        search: search || undefined,
        include_top: includeTop,
        page,
        page_size: 25,
      }),
    `events-${status}-${priority}-${label}-${search}-${includeTop}-${page}`,
  );

  if (listing.loading && !listing.data) {
    return (
      <Panel title="Loading review queue">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (listing.error) return <ErrorState message={listing.error} onRetry={listing.reload} />;
  const data = listing.data;
  if (!data) return <EmptyState title="No triage data returned." />;

  const labels = Object.keys(
    (listing.data?.events ?? []).reduce<Record<string, true>>((accumulator, event) => {
      if (event.dataset_label) accumulator[event.dataset_label] = true;
      return accumulator;
    }, {}),
  ).sort();

  const columns: Column<SecurityEvent>[] = [
    {
      key: "step",
      label: "Step",
      sortValue: (row) => row.step,
      render: (row) => <span className="mono strong">{row.step ?? "—"}</span>,
    },
    {
      key: "timestamp",
      label: "Timestamp",
      sortValue: (row) => row.timestamp,
      render: (row) => timestamp(row.timestamp),
    },
    { key: "label", label: "Dataset label", render: (row) => row.dataset_label ?? "—" },
    {
      key: "residual",
      label: "Residual",
      align: "right",
      sortValue: (row) => row.residual_score,
      render: (row) => decimal(row.residual_score),
    },
    {
      key: "priority",
      label: "Priority",
      sortValue: (row) => row.priority,
      render: (row) => <PriorityPill priority={row.priority} />,
    },
    {
      key: "status",
      label: "Case",
      sortValue: (row) => row.status,
      render: (row) => <CaseStatusPill status={row.status} />,
    },
    { key: "assignee", label: "Assignee", render: (row) => row.assignee ?? "—" },
    {
      key: "actions",
      label: "Actions",
      render: (row) => (
        <button
          type="button"
          className="btn small"
          onClick={(event) => {
            event.stopPropagation();
            setSelected(row.event_id);
          }}
        >
          Inspect
        </button>
      ),
    },
  ];

  const selection = String(data.filters.selection ?? "none");
  const priorityEntries = Object.entries(data.priority_counts);
  const statusEntries = Object.entries(data.status_counts);

  return (
    <>
      <PageHeader
        title="Threat Triage"
        subtitle="Prediction-residual review candidates with dataset reference labels, contextual ATT&CK guidance and a persistent analyst workflow."
        actions={
          <button type="button" className="btn" onClick={listing.reload}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice availability={data.availability} title="Review queue unavailable" />

      <Panel
        title="Generated playback alerts"
        hint="Persisted automatically when a playback state crossed the calibrated residual alert rules. Each alert is a review signal, not a confirmed incident."
        actions={
          <button type="button" className="btn small" onClick={playbackAlerts.reload}>
            Refresh alerts
          </button>
        }
        flush
      >
        {playbackAlerts.error ? (
          <ErrorState message={playbackAlerts.error} onRetry={playbackAlerts.reload} />
        ) : (
          <PlaybackAlertList
            alerts={playbackAlerts.data?.alerts ?? []}
            onOpen={(alert) => setSelectedAlert(alert)}
            emptyMessage="No playback alert has been generated"
          />
        )}
      </Panel>

      <div className="notice warning">
        <span className="notice-icon">!</span>
        <div>
          <strong>Residual candidates are not confirmed incidents.</strong> Selection: {selection}{" "}
          at threshold {decimal(Number(data.filters.threshold ?? NaN))}. A high prediction error
          means the network state differed from the model's expectation — corroborate with independent
          evidence before escalating.
        </div>
      </div>

      <div className="grid two">
        <Panel title="Queue composition" hint="Counts across the candidate pool, before filters.">
          <div className="stack" style={{ gap: 10 }}>
            <div>
              <span className="section-label">Priority</span>
              <div className="tag-row" style={{ marginTop: 6 }}>
                {priorityEntries.length ? (
                  priorityEntries.map(([key, value]) => (
                    <Pill key={key} tone={key === "review_required" ? "critical" : "neutral"}>
                      {titleCase(key)}: {number(value)}
                    </Pill>
                  ))
                ) : (
                  <span className="small muted">No candidates.</span>
                )}
              </div>
            </div>
            <div>
              <span className="section-label">Case status</span>
              <div className="tag-row" style={{ marginTop: 6 }}>
                {statusEntries.length ? (
                  statusEntries.map(([key, value]) => (
                    <Pill key={key} tone={key === "escalated" ? "critical" : "info"}>
                      {titleCase(key)}: {number(value)}
                    </Pill>
                  ))
                ) : (
                  <span className="small muted">No cases yet.</span>
                )}
              </div>
            </div>
            <div className="small muted">
              Candidate pool: {number(Number(data.filters.candidate_pool ?? 0))} · filtered result:{" "}
              {number(data.total)}
            </div>
          </div>
        </Panel>

        <Panel title="Triage policy" hint="How this queue is produced.">
          <dl className="kv">
            <dt>Score</dt>
            <dd>Mean absolute next-state prediction residual</dd>
            <dt>Threshold</dt>
            <dd>{decimal(Number(data.filters.threshold ?? NaN))}</dd>
            <dt>Threshold source</dt>
            <dd>{String(data.filters.threshold_source ?? "—")}</dd>
            <dt>Ground-truth used?</dt>
            <dd>No — labels are never used for the decision</dd>
            <dt>Contributor columns</dt>
            <dd>
              {Array.isArray(data.filters.contributor_columns) &&
              (data.filters.contributor_columns as string[]).length
                ? (data.filters.contributor_columns as string[]).join(", ")
                : "Not present in this artifact"}
            </dd>
          </dl>
        </Panel>
      </div>

      <Panel title="Review candidates" flush>
        <FilterBar>
          <SearchInput
            value={search}
            onChange={(value) => {
              setSearch(value);
              setPage(1);
            }}
            placeholder="Search step, label, notes…"
          />
          <SelectField
            label="Status"
            value={status}
            onChange={(value) => {
              setStatus(value);
              setPage(1);
            }}
            options={STATUSES.map((value) => ({ value, label: titleCase(value) }))}
          />
          <SelectField
            label="Priority"
            value={priority}
            onChange={(value) => {
              setPriority(value);
              setPage(1);
            }}
            options={[
              "review_required",
              "high",
              "elevated",
              "within_range",
              "baseline",
            ].map((value) => ({ value, label: titleCase(value) }))}
          />
          <SelectField
            label="Dataset label"
            value={label}
            onChange={(value) => {
              setLabel(value);
              setPage(1);
            }}
            options={labels.map((value) => ({ value, label: value }))}
          />
          <div style={{ alignSelf: "flex-end", paddingBottom: 6 }}>
            <Toggle
              checked={includeTop}
              onChange={(value) => {
                setIncludeTop(value);
                setPage(1);
              }}
              label="Include highest-residual observations below threshold"
            />
          </div>
        </FilterBar>
        <DataTable
          columns={columns}
          rows={data.events}
          rowKey={(row) => row.event_id}
          onRowClick={(row) => setSelected(row.event_id)}
          emptyMessage={
            selection === "threshold"
              ? "No observation crossed the frozen threshold. This is not evidence that the period was benign."
              : "No candidates match the current filters."
          }
          pagination={{
            page: data.page,
            pageSize: data.page_size,
            total: data.total,
            onPage: setPage,
          }}
        />
      </Panel>

      <Panel title="Interpretation limits">
        <Limitations items={data.limitations} />
      </Panel>

      {selectedAlert ? (
        <PlaybackAlertDrawer
          alertId={selectedAlert.alert_id}
          initial={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onChanged={() => playbackAlerts.reload()}
        />
      ) : null}

      {selected ? (
        <EventDrawer
          eventId={selected}
          onClose={() => setSelected(null)}
          onSaved={() => {
            listing.reload();
            toast.push({ title: "Case updated", tone: "success" });
          }}
        />
      ) : null}
    </>
  );
}

function EventDrawer({
  eventId,
  onClose,
  onSaved,
}: {
  eventId: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const toast = useToast();
  const detail = useApi(() => api.event(eventId), `event-${eventId}`);
  const [status, setStatus] = useState<EventStatus>("new");
  const [assignee, setAssignee] = useState("");
  const [notes, setNotes] = useState("");
  const [initialised, setInitialised] = useState(false);
  const [saving, setSaving] = useState(false);

  if (detail.data && !initialised) {
    setStatus(detail.data.event.status);
    setAssignee(detail.data.event.assignee ?? "");
    setNotes(detail.data.event.notes ?? "");
    setInitialised(true);
  }

  async function save() {
    setSaving(true);
    try {
      await api.updateEvent(eventId, {
        status,
        assignee: assignee || null,
        notes: notes || null,
        actor: "analyst",
      });
      detail.reload();
      onSaved();
    } catch (cause) {
      toast.push({
        title: "Update failed",
        message: cause instanceof Error ? cause.message : "Unknown error",
        tone: "error",
      });
    } finally {
      setSaving(false);
    }
  }

  const event = detail.data?.event;

  return (
    <Drawer
      title={event ? `Candidate ${event.event_id}` : "Loading candidate"}
      subtitle={event ? `${timestamp(event.timestamp)} · segment ${event.segment_id ?? "—"}` : undefined}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn" onClick={onClose} disabled={saving}>
            Close
          </button>
          <button type="button" className="btn primary" onClick={save} disabled={saving || !event}>
            {saving ? "Saving…" : "Save case"}
          </button>
        </>
      }
    >
      {detail.loading && !event ? <Loading rows={4} /> : null}
      {detail.error ? <ErrorState message={detail.error} onRetry={detail.reload} /> : null}
      {event ? (
        <>
          <div className="inline">
            <PriorityPill priority={event.priority} />
            <CaseStatusPill status={event.status} />
            <Pill tone="neutral">{event.dataset_label ?? "unlabelled"}</Pill>
          </div>

          <dl className="kv">
            <dt>Residual</dt>
            <dd>{decimal(event.residual_score)} (model feature space)</dd>
            <dt>Step</dt>
            <dd className="mono">{event.step ?? "—"}</dd>
            <dt>Capture day</dt>
            <dd>{event.capture_day ?? "—"}</dd>
            <dt>Last update</dt>
            <dd>{timestamp(event.updated_at)}</dd>
          </dl>

          {event.contributors.length ? (
            <div>
              <span className="section-label">Contributing features</span>
              <div className="tag-row" style={{ marginTop: 6 }}>
                {event.contributors.map((contributor) => (
                  <Pill key={contributor.feature} tone="info">
                    {contributor.feature}
                  </Pill>
                ))}
              </div>
            </div>
          ) : null}

          {event.mitre.length ? (
            <div>
              <span className="section-label">Contextual ATT&amp;CK mapping</span>
              <div className="stack" style={{ gap: 8, marginTop: 6 }}>
                {event.mitre.map((mapping) => (
                  <div key={mapping.technique} className="panel" style={{ padding: 10 }}>
                    <div className="inline">
                      <Pill tone="accent">{mapping.technique}</Pill>
                      <span className="small">{mapping.name}</span>
                      <Pill tone="neutral">{mapping.status}</Pill>
                    </div>
                    <div className="small muted" style={{ marginTop: 5 }}>
                      Tactic: {mapping.tactic}
                    </div>
                    {mapping.evidence ? (
                      <div className="small muted" style={{ marginTop: 4 }}>
                        Evidence required: {mapping.evidence}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
              <p className="small muted" style={{ marginTop: 6 }}>
                Mappings are contextual candidates for the dataset label; they are not generated by
                the LSTM and do not assign a technique to this observation.
              </p>
            </div>
          ) : null}

          <div>
            <span className="section-label">Recommended defensive steps</span>
            <ul className="limitations" style={{ marginTop: 6, color: "var(--text-secondary)" }}>
              {event.recommended_actions.map((action) => (
                <li key={action}>{action}</li>
              ))}
            </ul>
          </div>

          <div className="divider" />

          <div className="stack">
            <span className="section-label">Case workflow</span>
            <div className="grid two" style={{ gap: 10 }}>
              <label className="field">
                <span className="field-label">Status</span>
                <select
                  className="select"
                  value={status}
                  onChange={(changeEvent) => setStatus(changeEvent.target.value as EventStatus)}
                >
                  {STATUSES.map((value) => (
                    <option key={value} value={value}>
                      {titleCase(value)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">Assignee</span>
                <input
                  className="input"
                  value={assignee}
                  onChange={(changeEvent) => setAssignee(changeEvent.target.value)}
                  placeholder="soc-tier-1"
                />
              </label>
            </div>
            <label className="field">
              <span className="field-label">Analyst notes</span>
              <textarea
                className="textarea"
                value={notes}
                onChange={(changeEvent) => setNotes(changeEvent.target.value)}
                placeholder="Evidence checked, corroboration, decision rationale…"
              />
            </label>
          </div>

          {detail.data?.history.length ? (
            <div>
              <span className="section-label">Audit trail</span>
              <div className="table-wrap" style={{ marginTop: 6 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Actor</th>
                      <th>Field</th>
                      <th>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.data.history.map((entry, index) => (
                      <tr key={`${entry.created_at}-${index}`}>
                        <td>{timestamp(entry.created_at)}</td>
                        <td>{entry.actor}</td>
                        <td>{entry.action}</td>
                        <td className="mono">
                          {entry.from_value ?? "—"} → {entry.to_value ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          <Limitations items={detail.data?.limitations ?? []} />
        </>
      ) : null}
    </Drawer>
  );
}
