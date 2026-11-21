import type { ReactNode } from "react";

export function SearchInput({
  value,
  onChange,
  placeholder = "Search…",
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      className="input grow"
      type="search"
      value={value}
      placeholder={placeholder}
      aria-label={placeholder}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function SelectField({
  label,
  value,
  options,
  onChange,
  includeAll = true,
  allLabel = "All",
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  includeAll?: boolean;
  allLabel?: string;
}) {
  return (
    <label className="field" style={{ gap: 3 }}>
      <span className="field-label">{label}</span>
      <select className="select" value={value} onChange={(event) => onChange(event.target.value)}>
        {includeAll ? <option value="">{allLabel}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterBar({ children }: { children: ReactNode }) {
  return (
    <div className="toolbar" style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
      {children}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <label className="inline small" style={{ cursor: "pointer", color: "var(--text-secondary)" }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        style={{ accentColor: "var(--accent)" }}
      />
      {label}
    </label>
  );
}
