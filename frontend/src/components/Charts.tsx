import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SeriesPoint } from "../api/types";

const AXIS = { stroke: "#65748B", fontSize: 10, tickLine: false, axisLine: false } as const;
const GRID = { stroke: "#253249", strokeDasharray: "3 3" } as const;
const LEGEND = { fontSize: 11, color: "#91A0B7" } as const;

export const SERIES_COLOURS = {
  actual: "#3B82F6",
  predicted: "#22D3EE",
  history: "#A78BFA",
  residual: "#F59E0B",
  critical: "#EF4444",
  grid: "#253249",
} as const;

function TooltipBox({
  active,
  payload,
  label,
  labelPrefix,
  format,
}: {
  active?: boolean;
  payload?: Array<{
    dataKey?: string | number;
    name?: string;
    value?: number | string;
    color?: string;
  }>;
  label?: string | number;
  labelPrefix?: string;
  format?: Record<string, (value: number) => string>;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="label">
        {labelPrefix ?? "Index"} {label}
      </div>
      {payload.map((entry) => {
        const formatter = entry.dataKey ? format?.[String(entry.dataKey)] : undefined;
        return (
          <div className="row" key={`${entry.name}-${entry.value}`}>
            <span style={{ color: entry.color }}>{entry.name}</span>
            <span>
              {typeof entry.value === "number"
                ? (formatter ? formatter(entry.value) : entry.value.toFixed(4))
                : entry.value}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function ActivityChart({
  data,
  height = 220,
  valueLabel = "Mean residual",
  secondaryLabel = "Attack-labelled share",
}: {
  data: SeriesPoint[];
  height?: number;
  valueLabel?: string;
  secondaryLabel?: string;
}) {
  // Residuals are small (model space) while the label share is 0–100%. They must
  // never share an axis: a 100% share would flatten the residual line onto zero.
  const shaped = data.map((point) => ({
    index: point.index,
    value: typeof point.value === "number" && Number.isFinite(point.value) ? point.value : null,
    secondary:
      point.secondary === null || point.secondary === undefined
        ? null
        : point.secondary * 100,
  }));
  const hasShare = shaped.some((point) => point.secondary !== null);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={shaped} margin={{ top: 8, right: 4, bottom: 0, left: -10 }}>
        <defs>
          <linearGradient id="residualFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={SERIES_COLOURS.predicted} stopOpacity={0.3} />
            <stop offset="100%" stopColor={SERIES_COLOURS.predicted} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis
          yAxisId="residual"
          {...AXIS}
          width={50}
          domain={["auto", "auto"]}
          allowDecimals
          tickFormatter={(value: number) =>
            Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2)
          }
        />
        {hasShare ? (
          <YAxis
            yAxisId="share"
            orientation="right"
            {...AXIS}
            width={42}
            domain={[0, 100]}
            ticks={[0, 25, 50, 75, 100]}
            tickFormatter={(value: number) => `${value}%`}
          />
        ) : null}
        <Tooltip
          content={
            <TooltipBox
              labelPrefix="Bucket"
              format={{
                value: (value) => value.toFixed(4),
                secondary: (value) => `${value.toFixed(0)}%`,
              }}
            />
          }
        />
        <Legend wrapperStyle={LEGEND} />
        <Area
          yAxisId="residual"
          type="monotone"
          dataKey="value"
          name={valueLabel}
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={1.6}
          fill="url(#residualFill)"
          dot={shaped.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
        {hasShare ? (
          <Line
            yAxisId="share"
            type="monotone"
            dataKey="secondary"
            name={secondaryLabel}
            stroke={SERIES_COLOURS.history}
            strokeWidth={1.2}
            strokeDasharray="4 3"
            dot={shaped.length <= 2 ? { r: 3 } : false}
            isAnimationActive={false}
          />
        ) : null}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function AnomalyScoreChart({
  data,
  threshold,
  height = 260,
  alerts = [],
}: {
  data: Array<{
    index: number;
    step: number;
    timestamp?: string | null;
    anomaly_score: number;
    flagged: boolean;
    alert_id?: string | null;
  }>;
  threshold: number | null;
  height?: number;
  alerts?: string[];
}) {
  const shaped = data.map((point) => ({
    index: point.step,
    score: point.anomaly_score,
    flaggedScore: point.flagged ? point.anomaly_score : null,
    alertScore: point.alert_id && alerts.includes(point.alert_id) ? point.anomaly_score : null,
    timestamp: point.timestamp,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={shaped} margin={{ top: 8, right: 10, bottom: 0, left: -12 }}>
        <defs>
          <linearGradient id="anomalyFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={SERIES_COLOURS.residual} stopOpacity={0.28} />
            <stop offset="100%" stopColor={SERIES_COLOURS.residual} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={52} domain={["auto", "auto"]} />
        <Tooltip content={<TooltipBox labelPrefix="Step" />} />
        <Legend wrapperStyle={LEGEND} />
        <Area
          type="monotone"
          dataKey="score"
          name="Anomaly score (mean |residual|)"
          stroke={SERIES_COLOURS.residual}
          strokeWidth={1.6}
          fill="url(#anomalyFill)"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="flaggedScore"
          name="Flagged state"
          legendType="none"
          stroke="transparent"
          dot={{ r: 3, fill: SERIES_COLOURS.critical, stroke: SERIES_COLOURS.critical }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="alertScore"
          name="Persisted alert"
          legendType="none"
          stroke="transparent"
          dot={{ r: 5, fill: SERIES_COLOURS.actual, stroke: "#FFFFFF", strokeWidth: 1 }}
          isAnimationActive={false}
        />
        {typeof threshold === "number" ? (
          <ReferenceLine
            y={threshold}
            stroke={SERIES_COLOURS.critical}
            strokeDasharray="5 4"
            label={{
              value: `Threshold ${threshold.toFixed(3)}`,
              position: "insideTopRight",
              fill: SERIES_COLOURS.critical,
              fontSize: 10,
            }}
          />
        ) : null}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Persisted playback alerts per day. */
export function AlertTrendChart({
  data,
  height = 180,
}: {
  data: Array<{ day: string; count: number }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="day" {...AXIS} />
        <YAxis {...AXIS} width={46} allowDecimals={false} />
        <Tooltip content={<TooltipBox labelPrefix="Day" />} />
        <Bar
          dataKey="count"
          name="Alerts"
          fill={SERIES_COLOURS.critical}
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CompareChart({
  data,
  height = 260,
  actualLabel = "Actual",
  predictedLabel = "Predicted",
  xLabel = "Step",
}: {
  data: Array<{ index: number; actual: number | null; predicted: number | null }>;
  height?: number;
  actualLabel?: string;
  predictedLabel?: string;
  xLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={52} />
        <Tooltip content={<TooltipBox labelPrefix={xLabel} />} />
        <Legend wrapperStyle={LEGEND} />
        <Line
          type="monotone"
          dataKey="actual"
          name={actualLabel}
          stroke={SERIES_COLOURS.actual}
          strokeWidth={1.5}
          dot={data.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name={predictedLabel}
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={1.5}
          strokeDasharray="4 3"
          dot={data.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Ten observed states followed by the observed and predicted next state. */
export function PlaybackChart({
  history,
  actual,
  predicted,
  feature,
  height = 300,
}: {
  history: number[];
  actual: number;
  predicted: number;
  feature: string;
  height?: number;
}) {
  const length = history.length || 10;
  const data = history.map((value, index) => ({
    position: index === length - 1 ? "t-1" : `t-${length - index}`,
    observed: value,
    actual: null as number | null,
    predicted: null as number | null,
  }));
  data.push({
    position: "t",
    observed: null as unknown as number,
    actual,
    predicted,
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="position" {...AXIS} />
        <YAxis {...AXIS} width={58} />
        <Tooltip content={<TooltipBox labelPrefix="State" />} />
        <Legend wrapperStyle={LEGEND} />
        <Line
          type="monotone"
          dataKey="observed"
          name={`Observed history · ${feature}`}
          stroke={SERIES_COLOURS.history}
          strokeWidth={1.6}
          dot={{ r: 2 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="actual"
          name="Actual next state"
          stroke={SERIES_COLOURS.actual}
          strokeWidth={2}
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name="LSTM prediction"
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Residual trend for the local replay window with the frozen threshold. */
export function ThresholdTrendChart({
  data,
  threshold,
  height = 220,
}: {
  data: Array<{ index: number; residual: number | null }>;
  threshold?: number | null;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <defs>
          <linearGradient id="windowFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={SERIES_COLOURS.residual} stopOpacity={0.28} />
            <stop offset="100%" stopColor={SERIES_COLOURS.residual} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={50} />
        <Tooltip content={<TooltipBox labelPrefix="Step" />} />
        <Area
          type="monotone"
          dataKey="residual"
          name="Mean absolute residual"
          stroke={SERIES_COLOURS.residual}
          strokeWidth={1.5}
          fill="url(#windowFill)"
          dot={data.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
        {typeof threshold === "number" ? (
          <ReferenceLine
            y={threshold}
            stroke={SERIES_COLOURS.critical}
            strokeDasharray="5 4"
            label={{
              value: "Frozen threshold",
              position: "insideTopRight",
              fill: SERIES_COLOURS.critical,
              fontSize: 10,
            }}
          />
        ) : null}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function SignedResidualChart({
  data,
  height = 220,
}: {
  data: Array<{ index: number; signed: number | null; absolute: number | null }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={50} />
        <Tooltip content={<TooltipBox labelPrefix="Step" />} />
        <Legend wrapperStyle={LEGEND} />
        <ReferenceLine y={0} stroke="#33445F" />
        <Line
          type="monotone"
          dataKey="signed"
          name="Signed residual (actual − predicted)"
          stroke={SERIES_COLOURS.actual}
          strokeWidth={1.4}
          dot={data.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="absolute"
          name="Absolute residual"
          stroke={SERIES_COLOURS.residual}
          strokeWidth={1.2}
          dot={data.length <= 2 ? { r: 3 } : false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ResidualBarChart({
  data,
  height = 220,
  color = SERIES_COLOURS.predicted,
}: {
  data: Array<{ label: string; value: number }>;
  height?: number;
  color?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} interval={0} angle={-18} textAnchor="end" height={44} />
        <YAxis {...AXIS} width={46} />
        <Tooltip content={<TooltipBox labelPrefix="Position" />} />
        <Bar dataKey="value" name="Value" radius={[2, 2, 0, 0]} isAnimationActive={false}>
          {data.map((entry) => (
            <Cell key={entry.label} fill={color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Horizontal ranking bars for feature contributors. */
export function ContributorBars({
  data,
  height = 320,
  unitLabel = "Appearance %",
}: {
  data: Array<{ feature: string; value: number }>;
  height?: number;
  unitLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 24, bottom: 4, left: 12 }}
      >
        <CartesianGrid {...GRID} horizontal={false} />
        <XAxis type="number" {...AXIS} />
        <YAxis type="category" dataKey="feature" {...AXIS} width={190} />
        <Tooltip content={<TooltipBox labelPrefix="Rank" />} />
        <Bar dataKey="value" name={unitLabel} fill={SERIES_COLOURS.predicted} radius={[0, 2, 2, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Actual versus predicted scatter for one feature across the replay. */
export function ActualPredictedScatter({
  data,
  height = 300,
}: {
  data: Array<{ actual: number; predicted: number }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="actual"
          name="Actual"
          {...AXIS}
          label={{ value: "Actual next state", position: "insideBottom", offset: -2, fill: "#65748B", fontSize: 10 }}
        />
        <YAxis
          type="number"
          dataKey="predicted"
          name="Predicted"
          {...AXIS}
          width={54}
          label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#65748B", fontSize: 10 }}
        />
        <Tooltip content={<TooltipBox labelPrefix="State" />} cursor={{ strokeDasharray: "3 3" }} />
        <Scatter data={data} name="States" fill={SERIES_COLOURS.predicted} fillOpacity={0.65} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function DistributionChart({
  data,
  height = 200,
}: {
  data: Array<{ label: string; value: number }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} />
        <YAxis {...AXIS} width={46} />
        <Tooltip content={<TooltipBox labelPrefix="Residual" />} />
        <Bar
          dataKey="value"
          name="Observations"
          fill={SERIES_COLOURS.actual}
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Recorded residual quantiles (p50 … p99.9) when row-level data is unavailable. */
export function QuantileCurve({
  data,
  height = 200,
}: {
  data: Array<{ label: string; value: number }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} />
        <YAxis {...AXIS} width={50} />
        <Tooltip content={<TooltipBox labelPrefix="Quantile" />} />
        <Line
          type="monotone"
          dataKey="value"
          name="Mean absolute residual"
          stroke={SERIES_COLOURS.residual}
          strokeWidth={1.8}
          dot={{ r: 2.5 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Measured forecast error across recursive horizons from the recorded multi-step report. */
export function HorizonErrorChart({
  data,
  height = 260,
}: {
  data: Array<{ horizon: number; lstm: number | null; persistence: number | null }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="horizon" {...AXIS} tickFormatter={(value: number) => `${value}s`} />
        <YAxis
          {...AXIS}
          width={54}
          label={{
            value: "Mean absolute error",
            angle: -90,
            position: "insideLeft",
            fill: "#65748B",
            fontSize: 10,
          }}
        />
        <Tooltip content={<TooltipBox labelPrefix="Horizon" />} />
        <Legend wrapperStyle={LEGEND} />
        <Line
          type="monotone"
          dataKey="lstm"
          name="LSTM forecast"
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={2}
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="persistence"
          name="Persistence baseline"
          stroke={SERIES_COLOURS.history}
          strokeWidth={1.6}
          strokeDasharray="4 3"
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Grouped bars comparing the world model with the persistence baseline. */
export function MetricComparisonBars({
  data,
  height = 240,
}: {
  data: Array<{ metric: string; lstm: number | null; persistence: number | null }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="metric" {...AXIS} />
        <YAxis {...AXIS} width={50} />
        <Tooltip content={<TooltipBox labelPrefix="Metric" />} />
        <Legend wrapperStyle={LEGEND} />
        <Bar dataKey="lstm" name="LSTM world model" fill={SERIES_COLOURS.actual} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        <Bar dataKey="persistence" name="Persistence baseline" fill={SERIES_COLOURS.history} radius={[2, 2, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
