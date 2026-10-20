import { useRef } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistogramBin, ThresholdPoint } from "../api";
import { SERIES_COLORS } from "../lib";

export const AXIS = { stroke: "#94a3b8", fontSize: 12, tickLine: false } as const;
export const GRID = "#e2e8f0";

export function ConfusionMatrix({
  tp,
  fp,
  fn,
  tn,
  positive,
  unparseable,
}: {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  positive: string;
  unparseable: number;
}) {
  const negative = positive === "1" ? "0" : "1";
  const cell = (value: number, ok: boolean) => (
    <td
      className={`h-16 w-28 border border-slate-200 text-center text-xl font-semibold tabular-nums ${
        ok ? "bg-accent-soft text-slate-900" : "bg-slate-50 text-slate-700"
      }`}
    >
      {value}
    </td>
  );
  const head = "px-2 py-1 text-center text-xs font-medium uppercase tracking-wide text-slate-500";
  return (
    <div>
      <table className="border-collapse">
        <thead>
          <tr>
            <th />
            <th className={head} colSpan={2}>
              Predicted
            </th>
          </tr>
          <tr>
            <th />
            <th className={head}>{positive} (positive)</th>
            <th className={head}>{negative}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className={`${head} text-right`}>
              Actual {positive}
              <br />
              (positive)
            </th>
            {cell(tp, true)}
            {cell(fn, false)}
          </tr>
          <tr>
            <th className={`${head} text-right`}>Actual {negative}</th>
            {cell(fp, false)}
            {cell(tn, true)}
          </tr>
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">
        TP {tp} · FN {fn} · FP {fp} · TN {tn}
        {unparseable > 0 && ` · includes ${unparseable} unparseable counted as the wrong class`}
      </p>
    </div>
  );
}

export function LatencyHistogram({ bins }: { bins: HistogramBin[] }) {
  const data = bins.map((b) => ({ label: `${Math.round((b.lo + b.hi) / 2)}`, range: `${b.lo.toFixed(0)}–${b.hi.toFixed(0)} ms`, count: b.count }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 16, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} label={{ value: "latency (ms, bin centre)", position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 12 }} />
        <YAxis {...AXIS} allowDecimals={false} width={32} />
        <Tooltip formatter={(v) => [String(v), "samples"]} labelFormatter={(_, p) => p?.[0]?.payload?.range ?? ""} cursor={{ fill: "#f1f5f9" }} />
        <Bar dataKey="count" fill={SERIES_COLORS[0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ThresholdChart({
  points,
  threshold,
  onChange,
}: {
  points: ThresholdPoint[];
  threshold: number;
  onChange: (t: number) => void;
}) {
  // Dragging on the chart sets the threshold from the x position under the cursor.
  const dragging = useRef(false);
  const setFrom = (state: { activeLabel?: string | number }) => {
    if (state.activeLabel === undefined) return;
    const t = Number(state.activeLabel);
    if (!Number.isNaN(t)) onChange(Math.min(1, Math.max(0, Math.round(t * 100) / 100)));
  };

  return (
    <div>
      <div className="cursor-ew-resize select-none">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart
            data={points}
            margin={{ top: 22, right: 16, bottom: 8, left: 0 }}
            onMouseDown={(s) => {
              dragging.current = true;
              setFrom(s);
            }}
            onMouseMove={(s) => dragging.current && setFrom(s)}
            onMouseUp={() => (dragging.current = false)}
            onMouseLeave={() => (dragging.current = false)}
          >
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="threshold" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} />
            <YAxis domain={[0, 1]} {...AXIS} width={36} />
            <Legend verticalAlign="bottom" height={28} iconType="plainline" />
            <Line type="monotone" dataKey="precision" name="Precision" stroke={SERIES_COLORS[0]} strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="recall" name="Recall" stroke={SERIES_COLORS[1]} strokeWidth={2} strokeDasharray="6 3" dot={false} activeDot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="f1" name="F1" stroke={SERIES_COLORS[2]} strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
            <ReferenceLine x={threshold} stroke="#0f172a" strokeWidth={2} label={{ value: threshold.toFixed(2), position: "top", fill: "#0f172a", fontSize: 12 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex items-center gap-3 text-sm text-slate-600">
        <label htmlFor="threshold" className="whitespace-nowrap">
          Threshold
        </label>
        <input
          id="threshold"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-[#1d4ed8]"
        />
        <span className="w-10 text-right tabular-nums text-slate-900">{threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}
