// Core charts: confusion matrix, latency histogram, threshold curve, interval bars.
import { useRef } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
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
import { useReducedMotion } from "../lib";
import { AccentDefs, ACCENT_TO, AXIS, GRID, INK, LEGEND_STYLE, MODEL_PALETTE, TOOLTIP } from "../theme";

export function useChartMotion() {
  const reduced = useReducedMotion();
  return { isAnimationActive: !reduced, animationDuration: 700 } as const;
}

/**
 * Rows are true classes, columns are predictions. Every cell is shaded by its count:
 * the diagonal (correct) in the accent, everything else in the error colour.
 */
export function ConfusionMatrix({
  rowLabels,
  columnLabels,
  matrix,
  caption,
}: {
  rowLabels: string[];
  columnLabels: string[];
  matrix: number[][];
  caption?: string;
}) {
  const max = Math.max(1, ...matrix.flat());
  const many = columnLabels.length > 4;
  const cell = many ? "h-12 min-w-14 text-sm" : "h-20 min-w-28 text-2xl";
  return (
    <div className="flex flex-col items-center">
      <div className="max-w-full overflow-x-auto">
        <table className="border-separate border-spacing-1.5">
          <thead>
            <tr>
              <th />
              <th colSpan={columnLabels.length} className="label pb-1 text-center">
                Predicted
              </th>
            </tr>
            <tr>
              <th />
              {columnLabels.map((c) => (
                <th key={c} className="max-w-28 truncate px-1 pb-1 text-center text-xs font-medium text-sub" title={c}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rowLabels.map((r, i) => (
              <tr key={r}>
                <th className="max-w-36 truncate pr-2 text-right text-xs font-medium text-sub" title={`Actual ${r}`}>
                  {i === 0 && <span className="label mr-2 hidden sm:inline">Actual</span>}
                  {r}
                </th>
                {columnLabels.map((c, j) => {
                  const v = matrix[i]?.[j] ?? 0;
                  const diagonal = c.toLowerCase() === r.toLowerCase();
                  const a = v === 0 ? 0.04 : 0.14 + 0.62 * (v / max);
                  const bg = diagonal ? `rgba(99,102,241,${a})` : `rgba(251,113,133,${a * 0.85})`;
                  return (
                    <td
                      key={c}
                      className={`num rounded-[10px] border border-white/10 text-center font-semibold text-ink ${cell}`}
                      style={{ background: bg }}
                      title={`actual ${r}, predicted ${c}: ${v}`}
                    >
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && <p className="mt-3 text-center text-xs text-faint">{caption}</p>}
    </div>
  );
}

export function LatencyHistogram({ bins }: { bins: HistogramBin[] }) {
  const motion = useChartMotion();
  const data = bins.map((b) => ({ label: `${Math.round((b.lo + b.hi) / 2)}`, range: `${b.lo.toFixed(0)}–${b.hi.toFixed(0)} ms`, count: b.count }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 18, left: 0 }}>
        <AccentDefs id="lat" vertical />
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} label={{ value: "latency (ms, bin centre)", position: "insideBottom", offset: -12, fill: "#bcc4d2", fontSize: 12 }} />
        <YAxis {...AXIS} allowDecimals={false} width={32} />
        <Tooltip {...TOOLTIP} formatter={(v) => [String(v), "samples"]} labelFormatter={(_, p) => p?.[0]?.payload?.range ?? ""} />
        <Bar dataKey="count" fill="url(#lat)" radius={[4, 4, 0, 0]} {...motion} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ThresholdChart({ points, threshold, onChange }: { points: ThresholdPoint[]; threshold: number; onChange: (t: number) => void }) {
  // Dragging on the chart sets the threshold from the x position under the cursor.
  const dragging = useRef(false);
  // Interactive: redrawing from scratch on every drag would look broken, so no draw-in here.
  const motion = { isAnimationActive: false } as const;
  const setFrom = (state: { activeLabel?: string | number }) => {
    if (state.activeLabel === undefined) return;
    const t = Number(state.activeLabel);
    if (!Number.isNaN(t)) onChange(Math.min(1, Math.max(0, Math.round(t * 100) / 100)));
  };

  return (
    <div>
      <div className="cursor-ew-resize select-none">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={points}
            margin={{ top: 24, right: 16, bottom: 8, left: 0 }}
            onMouseDown={(s) => {
              dragging.current = true;
              setFrom(s);
            }}
            onMouseMove={(s) => dragging.current && setFrom(s)}
            onMouseUp={() => (dragging.current = false)}
            onMouseLeave={() => (dragging.current = false)}
          >
            <AccentDefs id="thr" />
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="threshold" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} />
            <YAxis domain={[0, 1]} {...AXIS} width={36} />
            <Legend verticalAlign="bottom" height={28} iconType="plainline" wrapperStyle={LEGEND_STYLE} />
            <Line type="monotone" dataKey="f1" name="F1" stroke="url(#thr)" strokeWidth={3} dot={false} activeDot={false} {...motion} />
            <Line type="monotone" dataKey="precision" name="Precision" stroke={MODEL_PALETTE[1]} strokeWidth={2} dot={false} activeDot={false} {...motion} />
            <Line type="monotone" dataKey="recall" name="Recall" stroke={MODEL_PALETTE[2]} strokeWidth={2} strokeDasharray="6 3" dot={false} activeDot={false} {...motion} />
            <ReferenceLine x={threshold} stroke={INK} strokeWidth={2} label={{ value: threshold.toFixed(2), position: "top", fill: INK, fontSize: 12 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex items-center gap-3 text-sm text-sub">
        <label htmlFor="threshold" className="whitespace-nowrap">
          Threshold
        </label>
        <input id="threshold" type="range" min={0} max={1} step={0.01} value={threshold} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
        <span className="num w-10 text-right text-ink">{threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}

export interface CiRow {
  label: string;
  value: number;
  lo: number | null;
  hi: number | null;
  color: string;
  sub?: string;
}

/** Horizontal bars with 95% interval whiskers; one bar per row, coloured by model. */
export function CiBars({ rows, domain = [0, 1], height }: { rows: CiRow[]; domain?: [number, number]; height?: number }) {
  const motion = useChartMotion();
  const data = rows.map((r) => ({
    ...r,
    err: [r.lo === null ? 0 : r.value - r.lo, r.hi === null ? 0 : r.hi - r.value] as [number, number],
  }));
  const digits = domain[1] > 1 ? 2 : 3;
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(120, rows.length * 40 + 40)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" domain={domain} {...AXIS} />
        <YAxis type="category" dataKey="label" {...AXIS} width={230} tick={{ fill: "#cdd5e1", fontSize: 12 }} tickFormatter={(v: string) => (v.length > 34 ? `${v.slice(0, 33)}…` : v)} />
        <Tooltip
          {...TOOLTIP}
          formatter={(_v, _n, p) => {
            const r = p.payload as CiRow;
            const ci = r.lo !== null && r.hi !== null ? `  (95% CI ${r.lo.toFixed(digits)}–${r.hi.toFixed(digits)})` : "";
            return [`${r.value.toFixed(digits)}${ci}`, r.sub ?? "value"];
          }}
        />
        <Bar dataKey="value" barSize={18} radius={[0, 5, 5, 0]} {...motion}>
          {data.map((r) => (
            <Cell key={r.label} fill={r.color} fillOpacity={0.85} />
          ))}
          <ErrorBar dataKey="err" direction="x" width={6} stroke={INK} strokeWidth={1.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export { ACCENT_TO };
