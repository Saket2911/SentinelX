import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import type { TooltipProps } from "recharts";

const AXIS = "#54677a";
const GRID = "#1d2937";

export const SERIES = {
  accent: "#34d3bd",
  blue: "#5b9cff",
  orange: "#ff9f45",
  crit: "#ff5f6d",
  violet: "#a78bfa",
  med: "#e8c547",
};

function ChartTip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div
      style={{
        background: "#0a0f15",
        border: "1px solid #2a3a4d",
        borderRadius: 6,
        padding: "8px 10px",
        fontSize: 11.5,
        boxShadow: "0 8px 24px -8px rgba(0,0,0,.7)",
      }}
    >
      <div style={{ color: "#e8eef5", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div
          key={String(p.dataKey)}
          style={{ display: "flex", justifyContent: "space-between", gap: 14, color: "#a9b8c8" }}
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <i
              style={{
                width: 7,
                height: 7,
                borderRadius: 2,
                background: p.color,
                display: "inline-block",
              }}
            />
            {p.name}
          </span>
          <span style={{ fontFamily: "var(--mono)", color: "#e8eef5" }}>
            {typeof p.value === "number" ? p.value.toFixed(4) : String(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function SimpleBar({
  data,
  xKey,
  yKey,
  name,
  color = SERIES.accent,
  height = 260,
  layout = "vertical",
  formatter,
}: {
  data: Record<string, string | number>[];
  xKey: string;
  yKey: string;
  name: string;
  color?: string;
  height?: number;
  layout?: "vertical" | "horizontal";
  formatter?: (v: number) => string;
}) {
  const isVertical = layout === "vertical";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={isVertical ? "vertical" : "horizontal"}
        margin={{ top: 6, right: 18, bottom: 4, left: isVertical ? 8 : 0 }}
      >
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" horizontal={!isVertical} vertical={isVertical} />
        {isVertical ? (
          <>
            <XAxis type="number" stroke={AXIS} tickLine={false} axisLine={false}
              tickFormatter={formatter} />
            <YAxis
              type="category"
              dataKey={xKey}
              stroke={AXIS}
              tickLine={false}
              axisLine={false}
              width={150}
              tick={{ fontSize: 10.5 }}
            />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} stroke={AXIS} tickLine={false} axisLine={false} />
            <YAxis stroke={AXIS} tickLine={false} axisLine={false} tickFormatter={formatter} />
          </>
        )}
        <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
        <Bar dataKey={yKey} name={name} radius={[2, 2, 2, 2]} maxBarSize={22}>
          {data.map((_, i) => (
            <Cell key={i} fill={color} fillOpacity={0.82} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function GroupedBars({
  data,
  xKey,
  series,
  height = 280,
  xLabel,
}: {
  data: Record<string, string | number>[];
  xKey: string;
  series: { key: string; name: string; color: string }[];
  height?: number;
  xLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          label={
            xLabel
              ? { value: xLabel, position: "insideBottom", offset: -4, fill: AXIS, fontSize: 10 }
              : undefined
          }
        />
        <YAxis stroke={AXIS} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
        <Legend
          iconType="square"
          iconSize={8}
          wrapperStyle={{ fontSize: 11.5, color: "#74869a", paddingTop: 6 }}
        />
        {series.map((s) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.name}
            fill={s.color}
            fillOpacity={0.85}
            radius={[2, 2, 0, 0]}
            maxBarSize={26}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TrendLine({
  data,
  xKey,
  series,
  height = 260,
  xLabel,
}: {
  data: Record<string, string | number>[];
  xKey: string;
  series: { key: string; name: string; color: string }[];
  height?: number;
  xLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 18, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          label={
            xLabel
              ? { value: xLabel, position: "insideBottom", offset: -5, fill: AXIS, fontSize: 10 }
              : undefined
          }
        />
        <YAxis stroke={AXIS} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTip />} />
        <Legend
          iconType="plainline"
          iconSize={12}
          wrapperStyle={{ fontSize: 11.5, color: "#74869a", paddingTop: 6 }}
        />
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.name}
            stroke={s.color}
            strokeWidth={1.8}
            dot={{ r: 2.4, fill: s.color, strokeWidth: 0 }}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
