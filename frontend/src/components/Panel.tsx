import type { ReactNode } from "react";

export function Panel({
  title,
  hint,
  actions,
  children,
  flush = false,
  className = "",
}: {
  title?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`.trim()}>
      {title || actions ? (
        <header className="panel-head">
          <div className="panel-head-copy">
            {title ? <h2 className="panel-title">{title}</h2> : null}
            {hint ? <span className="panel-hint">{hint}</span> : null}
          </div>
          {actions ? <div className="inline">{actions}</div> : null}
        </header>
      ) : null}
      <div className={`panel-body ${flush ? "flush" : ""}`.trim()}>{children}</div>
    </section>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div className="page-title">
        <h1>{title}</h1>
        <div className="page-subtitle">{subtitle}</div>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}
