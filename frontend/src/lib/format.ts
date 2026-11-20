export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US");
}

export function pct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(
    value,
  );
}

export function humanSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  if (s < 90) return `${s.toFixed(0)}s`;
  if (s < 5400) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(2)}h`;
}

export const PROVENANCE_LABEL: Record<string, string> = {
  recorded_artifact: "Recorded",
  derived: "Derived",
  unavailable: "Unavailable",
  live_telemetry: "Live",
};

export function provenanceClass(p: string | undefined): string {
  if (p === "recorded_artifact") return "recorded";
  if (p === "derived") return "derived";
  return "unavailable";
}
