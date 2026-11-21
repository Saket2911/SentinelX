import type { Metric as MetricType } from "../api/types";

export function MetricCard({ metric }: { metric: MetricType }) {
  return (
    <article className={`metric-card status-${metric.status}`}>
      <span className="metric-label">{metric.label}</span>
      <div>
        <span className="metric-value">{metric.display}</span>
        {metric.unit ? <span className="metric-unit">{metric.unit}</span> : null}
      </div>
      {metric.detail ? <span className="metric-detail">{metric.detail}</span> : null}
    </article>
  );
}

export function MetricGrid({ metrics }: { metrics: MetricType[] }) {
  return (
    <div className="grid metrics">
      {metrics.map((metric) => (
        <MetricCard key={metric.id} metric={metric} />
      ))}
    </div>
  );
}
