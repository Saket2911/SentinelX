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
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string; color?: string }>;
  label?: string | number;
  labelPrefix?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="label">
        {labelPrefix ?? "Index"} {label}
      </div>
      {payload.map((entry) => (
        <div className="row" key={`${entry.name}-${entry.value}`}>
          <span style={{ color: entry.color }}>{entry.name}</span>
          <span>
            {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
          </span>
        </div>
      ))}
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
  const shaped = data.map((point) => ({
    index: point.index,
    value: point.value,
    secondary:
      point.secondary === null || point.secondary === undefined ? null : point.secondary * 100,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={shaped} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <defs>
          <linearGradient id="residualFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={SERIES_COLOURS.predicted} stopOpacity={0.3} />
            <stop offset="100%" stopColor={SERIES_COLOURS.predicted} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={46} />
        <Tooltip content={<TooltipBox labelPrefix="Bucket" />} />
        <Area
          type="monotone"
          dataKey="value"
          name={valueLabel}
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={1.6}
          fill="url(#residualFill)"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="secondary"
          name={secondaryLabel}
          stroke={SERIES_COLOURS.history}
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
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
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name={predictedLabel}
          stroke={SERIES_COLOURS.predicted}
          strokeWidth={1.5}
          strokeDasharray="4 3"
          dot={false}
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
          dot={false}
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
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="absolute"
          name="Absolute residual"
          stroke={SERIES_COLOURS.residual}
          strokeWidth={1.2}
          dot={false}
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
