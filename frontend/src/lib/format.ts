export function number(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function decimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(digits)}%`;
}

export function bytes(value: number | null | undefined): string {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function timestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().replace("T", " ").replace(".000Z", "Z").slice(0, 19);
}

export function titleCase(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function priorityTone(
  priority: string,
): "critical" | "warning" | "info" | "neutral" | "good" {
  switch (priority) {
    case "review_required":
      return "critical";
    case "high":
      return "warning";
    case "elevated":
      return "info";
    case "within_range":
      return "neutral";
    case "baseline":
      return "good";
    default:
      return "neutral";
  }
}

export function statusTone(status: string): "critical" | "warning" | "info" | "neutral" | "good" {
  switch (status) {
    case "new":
      return "info";
    case "investigating":
      return "warning";
    case "escalated":
      return "critical";
    case "benign":
    case "closed":
      return "good";
    default:
      return "neutral";
  }
}
