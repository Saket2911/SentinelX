import type { ReactNode } from "react";

import type { Availability, Source } from "../api/types";

export function Loading({ rows = 3, label }: { rows?: number; label?: string }) {
  return (
    <div className="skeleton-stack" role="status" aria-live="polite" aria-label={label ?? "Loading"}>
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="skeleton"
          style={{ height: index === 0 ? 22 : 12, width: `${92 - index * 9}%` }}
        />
      ))}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state" role="alert">
      <span className="notice-icon" style={{ color: "var(--danger)" }}>
        ERR
      </span>
      <span className="state-title">Request failed</span>
      <span className="state-detail">{message}</span>
      {onRetry ? (
        <button type="button" className="btn small" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="state">
      <span className="state-title">{title}</span>
      {detail ? <div className="state-detail">{detail}</div> : null}
      {action}
    </div>
  );
}

export function AvailabilityNotice({
  availability,
  title = "Data unavailable in this deployment",
}: {
  availability: Availability;
  title?: string;
}) {
  if (availability.available) return null;
  return (
    <div className="notice warning">
      <span className="notice-icon">!</span>
      <div>
        <strong>{title}</strong>
        <div>{availability.reason ?? "The required artifact is not present."}</div>
        {availability.required_artifacts.length ? (
          <ul className="notice-list">
            {availability.required_artifacts.slice(0, 8).map((artifact) => (
              <li key={artifact}>
                <code>{artifact}</code>
              </li>
            ))}
          </ul>
        ) : null}
        <div className="small muted" style={{ marginTop: 6 }}>
          Nothing is simulated or substituted: the interface reports the gap instead of inventing data.
        </div>
      </div>
    </div>
  );
}

export function Limitations({ items, title = "Interpretation limits" }: { items: string[]; title?: string }) {
  if (!items.length) return null;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <span className="section-label">{title}</span>
      <ul className="limitations">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function SourceLine({ source }: { source: Source }) {
  return (
    <div className="small muted inline">
      <span className="pill neutral" title={source.detail ?? undefined}>
        {source.mode.replace(/_/g, " ")}
      </span>
      <span>{source.label}</span>
      {source.detail ? <code style={{ overflowWrap: "anywhere" }}>{source.detail}</code> : null}
    </div>
  );
}

export function SectionTitle({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="stack" style={{ gap: 2 }}>
      <span className="section-label">{title}</span>
      {hint ? <span className="small muted">{hint}</span> : null}
    </div>
  );
}
