import type { ReactNode } from "react";

import { priorityTone, statusTone, titleCase } from "../lib/format";

type Tone = "good" | "warning" | "critical" | "info" | "accent" | "neutral";

export function Pill({
  tone = "neutral",
  children,
  dot = false,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  title?: string;
}) {
  return (
    <span className={`pill ${tone}`} title={title}>
      {dot ? <span className="dot" /> : null}
      {children}
    </span>
  );
}

export function PriorityPill({ priority }: { priority: string }) {
  const tone = priorityTone(priority);
  return (
    <Pill tone={tone} dot title="Residual review priority — not an attack classification">
      {titleCase(priority)}
    </Pill>
  );
}

export function CaseStatusPill({ status }: { status: string }) {
  return <Pill tone={statusTone(status)}>{titleCase(status)}</Pill>;
}

export function HealthPill({
  state,
}: {
  state: "ok" | "warning" | "failed" | "unknown";
}) {
  const tone: Tone = state === "ok" ? "good" : state === "warning" ? "warning" : state === "failed" ? "critical" : "neutral";
  return <Pill tone={tone} dot>{titleCase(state)}</Pill>;
}

export function AvailabilityPill({ available }: { available: boolean }) {
  return (
    <Pill tone={available ? "good" : "warning"} dot>
      {available ? "Available" : "Unavailable"}
    </Pill>
  );
}
