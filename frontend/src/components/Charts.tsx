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
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SeriesPoint } from "../api/types";

const AXIS = { stroke: "#5d708c", fontSize: 10, tickLine: false, axisLine: false } as const;
const GRID = { stroke: "#1d2938", strokeDasharray: "3 3" } as const;

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
    secondary: point.secondary === null || point.secondary === undefined ? null : point.secondary * 100,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={shaped} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <defs>
          <linearGradient id="residualFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3ed3b6" stopOpacity={0.28} />
            <stop offset="100%" stopColor="#3ed3b6" stopOpacity={0.02} />
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
          stroke="#3ed3b6"
          strokeWidth={1.6}
          fill="url(#residualFill)"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="secondary"
          name={secondaryLabel}
          stroke="#5aa9f0"
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
}: {
  data: Array<{ index: number; actual: number | null; predicted: number | null }>;
  height?: number;
  actualLabel?: string;
  predictedLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -14 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="index" {...AXIS} />
        <YAxis {...AXIS} width={52} />
        <Tooltip content={<TooltipBox labelPrefix="Step" />} />
        <Legend wrapperStyle={{ fontSize: 11, color: "#8a99ac" }} />
        <Line
          type="monotone"
          dataKey="actual"
          name={actualLabel}
          stroke="#5aa9f0"
          strokeWidth={1.4}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name={predictedLabel}
          stroke="#3ed3b6"
          strokeWidth={1.4}
          strokeDasharray="4 3"
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
  color = "#3ed3b6",
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
        <Bar dataKey="value" name="Observations" fill="#5aa9f0" radius={[2, 2, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
