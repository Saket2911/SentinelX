import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { Icon } from "./icons";
import { PROVENANCE_LABEL, provenanceClass } from "../lib/format";

/* ------------------------------------------------------------------ Panel */

export function Panel({
  title,
  sub,
  actions,
  children,
  flush,
  className,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
}) {
  return (
    <section className={`panel${className ? ` ${className}` : ""}`}>
      {(title || actions || sub) && (
        <div className="panel-head">
          <div>
            {title && <h3 className="panel-title">{title}</h3>}
            {sub && <div className="panel-sub">{sub}</div>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      <div className={`panel-body${flush ? " flush" : ""}`}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------- MetricCard */

export function MetricCard({
  label,
  value,
  unit,
  hint,
  provenance,
  source,
  small,
  tone,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string;
  provenance?: string;
  source?: string;
  small?: boolean;
  tone?: "crit" | "high" | "med" | "ok" | "low";
}) {
  return (
    <div className="metric">
      <div className="metric-label">
        {label}
        {hint && <Tip text={hint} />}
      </div>
      <div
        className={`metric-value${small ? " sm" : ""}`}
        style={tone ? { color: `var(--${tone === "med" ? "med" : tone})` } : undefined}
      >
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className="metric-foot">
        {provenance ? <ProvenanceTag provenance={provenance} source={source} /> : <span />}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- tags */

export function Tag({
  tone = "neutral",
  children,
  dot,
}: {
  tone?: "neutral" | "crit" | "high" | "med" | "low" | "ok" | "recorded" | "derived" | "unavailable";
  children: ReactNode;
  dot?: boolean;
}) {
  return (
    <span className={`tag ${tone}`}>
      {dot && <span className="tag-dot" />}
      {children}
    </span>
  );
}

export function ProvenanceTag({
  provenance,
  source,
  compact,
}: {
  provenance: string;
  source?: string;
  compact?: boolean;
}) {
  const cls = provenanceClass(provenance);
  const label = PROVENANCE_LABEL[provenance] ?? provenance;
  const tag = (
    <span className={`tag ${cls}`}>
      <span className="tag-dot" />
      {compact ? label : `Source: ${label}`}
    </span>
  );
  if (!source) return tag;
  return <Tip text={source}>{tag}</Tip>;
}

export function Tip({ text, children }: { text: string; children?: ReactNode }) {
  return (
    <span className="tip" tabIndex={0} aria-label={text}>
      {children ?? <Icon.Info size={12} />}
      <span className="bubble" role="tooltip">
        {text}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------------- notes */

export function Note({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "crit";
  title?: ReactNode;
  children: ReactNode;
}) {
  const Ico = tone === "crit" ? Icon.Alert : tone === "warn" ? Icon.Alert : Icon.Info;
  return (
    <div className={`note ${tone}`}>
      <span className="ico">
        <Ico size={15} />
      </span>
      <div>
        {title && <strong>{title} </strong>}
        {children}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ states */

export function EmptyState({
  title,
  desc,
  icon,
  action,
}: {
  title: string;
  desc?: string;
  icon?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="state">
      <div className="ico">{icon ?? <Icon.Database size={22} />}</div>
      <div className="title">{title}</div>
      {desc && <div className="desc">{desc}</div>}
      {action}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state">
      <div className="ico" style={{ color: "var(--crit)" }}>
        <Icon.Alert size={22} />
      </div>
      <div className="title">Could not load this view</div>
      <div className="desc">{message}</div>
      {onRetry && (
        <button className="btn sm" onClick={onRetry} style={{ marginTop: 6 }}>
          <Icon.Refresh /> Retry
        </button>
      )}
    </div>
  );
}

export function LoadingBlock({ lines = 4, height }: { lines?: number; height?: number }) {
  if (height) return <div className="skeleton sk-block" style={{ height }} />;
  return (
    <div style={{ padding: 4 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="skeleton sk-line"
          style={{ width: `${100 - (i % 3) * 14}%` }}
        />
      ))}
    </div>
  );
}

/* -------------------------------------------------------------- controls */

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          className={value === o.value ? "on" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SearchInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="search">
      <Icon.Search />
      <input
        className="input"
        value={value}
        placeholder={placeholder ?? "Search"}
        onChange={(e) => onChange(e.target.value)}
        aria-label={placeholder ?? "Search"}
      />
    </div>
  );
}

/* --------------------------------------------------------------- DataTable */

export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => ReactNode;
  align?: "left" | "right";
  sortable?: boolean;
  sortValue?: (row: T) => number | string;
  width?: string;
}

export function DataTable<T>({
  columns,
  rows,
  pageSize = 12,
  initialSort,
  emptyTitle = "No rows",
  emptyDesc,
  rowKey,
  onRowClick,
}: {
  columns: Column<T>[];
  rows: T[];
  pageSize?: number;
  initialSort?: { key: string; dir: "asc" | "desc" };
  emptyTitle?: string;
  emptyDesc?: string;
  rowKey: (row: T, i: number) => string;
  onRowClick?: (row: T) => void;
}) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(
    initialSort ?? null,
  );
  const [page, setPage] = useState(0);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    const get = col.sortValue ?? ((r: T) => (r as Record<string, unknown>)[col.key] as number | string);
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (typeof va === "number" && typeof vb === "number") {
        return sort.dir === "asc" ? va - vb : vb - va;
      }
      const sa = String(va);
      const sb = String(vb);
      return sort.dir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [rows, sort, columns]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  const current = Math.min(page, pageCount - 1);
  const view = sorted.slice(current * pageSize, current * pageSize + pageSize);

  const toggleSort = (col: Column<T>) => {
    if (!col.sortable) return;
    setPage(0);
    setSort((s) =>
      s && s.key === col.key
        ? { key: col.key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key: col.key, dir: "desc" },
    );
  };

  if (rows.length === 0) {
    return <EmptyState title={emptyTitle} desc={emptyDesc} />;
  }

  return (
    <>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  className={`${c.align === "right" ? "num" : ""}${c.sortable ? " sortable" : ""}`}
                  style={c.width ? { width: c.width } : undefined}
                  onClick={() => toggleSort(c)}
                  aria-sort={
                    sort?.key === c.key
                      ? sort.dir === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  {c.header}
                  {sort?.key === c.key && (sort.dir === "asc" ? " ↑" : " ↓")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((row, i) => (
              <tr
                key={rowKey(row, i)}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                style={onRowClick ? { cursor: "pointer" } : undefined}
              >
                {columns.map((c) => (
                  <td key={c.key} className={c.align === "right" ? "num" : ""}>
                    {c.render
                      ? c.render(row)
                      : String((row as Record<string, unknown>)[c.key] ?? "—")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-foot">
        <span>
          {sorted.length === 0
            ? "No matches"
            : `${current * pageSize + 1}–${Math.min(
                (current + 1) * pageSize,
                sorted.length,
              )} of ${sorted.length}`}
        </span>
        <div className="pager">
          <button
            className="btn ghost sm"
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
            aria-label="Previous page"
          >
            <Icon.Chevron size={12} className="" />
            <span style={{ transform: "rotate(180deg)", display: "inline-flex" }}>
              <Icon.Chevron size={12} />
            </span>
          </button>
          <span className="mono" style={{ fontSize: 11 }}>
            {current + 1} / {pageCount}
          </span>
          <button
            className="btn ghost sm"
            disabled={current >= pageCount - 1}
            onClick={() => setPage(current + 1)}
            aria-label="Next page"
          >
            <Icon.Chevron size={12} />
          </button>
        </div>
      </div>
    </>
  );
}

/* ----------------------------------------------------------------- drawer */

export function Drawer({
  title,
  onClose,
  children,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Details">
        <div className="drawer-head">
          <h3 className="panel-title" style={{ fontSize: 13.5 }}>
            {title}
          </h3>
          <button className="btn ghost sm" onClick={onClose} style={{ marginLeft: "auto" }}>
            <Icon.X size={14} />
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </>
  );
}

/* ----------------------------------------------------------------- toasts */

interface Toast {
  id: number;
  title: string;
  desc?: string;
  tone: "ok" | "err" | "neutral";
}

const ToastCtx = createContext<(t: Omit<Toast, "id">) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = useCallback((t: Omit<Toast, "id">) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { ...t, id }].slice(-4));
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((x) => x.id !== id));
    }, 5200);
  }, []);

  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.tone}`}>
            <span
              style={{
                color: t.tone === "ok" ? "var(--ok)" : t.tone === "err" ? "var(--crit)" : "var(--text-2)",
                marginTop: 1,
              }}
            >
              {t.tone === "ok" ? <Icon.Check size={15} /> : t.tone === "err" ? <Icon.Alert size={15} /> : <Icon.Info size={15} />}
            </span>
            <div className="t-body">
              <div className="t-title">{t.title}</div>
              {t.desc && <div className="t-desc">{t.desc}</div>}
            </div>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast() {
  return useContext(ToastCtx);
}

/* ------------------------------------------------------------ page header */

export function PageHeader({
  title,
  desc,
  actions,
}: {
  title: string;
  desc?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-head">
      <div style={{ minWidth: 0 }}>
        <h1 className="page-title">{title}</h1>
        {desc && <p className="page-desc">{desc}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}

export function StatBar({
  segments,
}: {
  segments: { value: number; color: string; label: string }[];
}) {
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  return (
    <div className="bar-track" style={{ height: 8 }}>
      {segments.map((s) => (
        <div
          key={s.label}
          className="bar-fill"
          style={{ width: `${(s.value / total) * 100}%`, background: s.color }}
          title={`${s.label}: ${s.value.toLocaleString()}`}
        />
      ))}
    </div>
  );
}
