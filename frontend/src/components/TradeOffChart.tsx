// Accuracy against median latency: one point per run, coloured by model.
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { RunSummary } from "../api";
import { ms } from "../lib";
import { AXIS, FAINT, GRID, TOOLTIP, useModelColor } from "../theme";
import { useChartMotion } from "./charts";

export default function TradeOffChart({ runs, height = 300 }: { runs: RunSummary[]; height?: number }) {
  const colorOf = useModelColor();
  const motion = useChartMotion();
  const byModel = new Map<string, { latency: number; accuracy: number; label: string }[]>();
  for (const r of runs) {
    if (r.metrics.accuracy === undefined || !r.metrics.latency_median_ms) continue;
    const list = byModel.get(r.model) ?? [];
    list.push({ latency: r.metrics.latency_median_ms, accuracy: r.metrics.accuracy, label: `${r.model} · ${r.task}` });
    byModel.set(r.model, list);
  }
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis
            type="number"
            dataKey="latency"
            name="median latency"
            scale="log"
            domain={["auto", "auto"]}
            {...AXIS}
            tickFormatter={(v) => ms(v)}
            label={{ value: "median latency (log scale)", position: "insideBottom", offset: -12, fill: FAINT, fontSize: 12 }}
          />
          <YAxis type="number" dataKey="accuracy" name="accuracy" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} width={36} />
          <ZAxis range={[90, 90]} />
          <Tooltip
            {...TOOLTIP}
            cursor={{ strokeDasharray: "3 3", stroke: "rgba(58,50,38,0.4)" }}
            content={({ payload }) => {
              const p = payload?.[0]?.payload as { label: string; latency: number; accuracy: number } | undefined;
              return p ? (
                <div style={TOOLTIP.contentStyle} className="px-3 py-2">
                  <div className="text-xs text-sub">{p.label}</div>
                  <div className="num text-sm text-ink">
                    accuracy {p.accuracy.toFixed(3)} · {ms(p.latency)}
                  </div>
                </div>
              ) : null;
            }}
          />
          {[...byModel.entries()].map(([model, points]) => (
            <Scatter key={model} name={model} data={points} fill={colorOf(model)} fillOpacity={0.9} stroke="#ffffff" {...motion} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-sub">
        {[...byModel.keys()].map((m) => (
          <span key={m} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: colorOf(m) }} />
            {m}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-faint">Up and to the left is better. Latency is measured on this machine and is not like-for-like between CPU and GPU models.</p>
    </div>
  );
}
