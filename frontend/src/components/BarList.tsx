import { decimal, number } from "../lib/format";

export interface BarItem {
  label: string;
  value: number;
  display?: string;
  tone?: "accent" | "info" | "warn" | "danger";
  title?: string;
}

export function BarList({
  items,
  max,
  unit,
}: {
  items: BarItem[];
  max?: number;
  unit?: string;
}) {
  if (!items.length) {
    return <div className="state" style={{ padding: 20 }}>No data available.</div>;
  }
  const ceiling = max ?? Math.max(...items.map((item) => item.value), 0.000001);
  return (
    <div className="bar-list">
      {items.map((item) => (
        <div key={`${item.label}-${item.value}`} className="bar-row" title={item.title}>
          <span className="bar-label">{item.label}</span>
          <span className="bar-track">
            <span
              className={`bar-fill ${item.tone ?? ""}`.trim()}
              style={{ width: `${Math.max(1, Math.min(100, (item.value / ceiling) * 100))}%` }}
            />
          </span>
          <span className="bar-value">
            {item.display ?? decimal(item.value, 3)}
            {unit ? ` ${unit}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ShareList({
  items,
  unit = "",
}: {
  items: Array<{ name: string; count: number; share: number }>;
  unit?: string;
}) {
  return (
    <BarList
      items={items.map((item) => ({
        label: item.name,
        value: item.count,
        display: `${number(item.count)}${unit ? ` ${unit}` : ""} · ${(item.share * 100).toFixed(1)}%`,
      }))}
    />
  );
}
