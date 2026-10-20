// Analysis charts for a single run or a comparison: ROC, precision/recall, score distribution,
// calibration, accuracy by input length and per-sample latency. All inputs come from /api/.../analysis.
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Calibration, LengthSlice, PrCurve, RocCurve, ScoreBin } from "../api";
import { SERIES_COLORS } from "../lib";
import { AXIS, GRID } from "./charts";

const INK = "#0f172a";
const NEUTRAL = "#64748b";
const DANGER = "#b42318";
const UNIT_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const DIAGONAL = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
] as const;

const bottomLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -10, fill: NEUTRAL, fontSize: 12 });
const leftLabel = (value: string) => ({ value, position: "insideLeft" as const, angle: -90, offset: 14, fill: NEUTRAL, fontSize: 12 });
const fixed3 = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v));

/** A point the user is currently inspecting, such as the chosen threshold. */
export interface CurveMarker {
  x: number;
  y: number;
}

export function RocChart({
  curves,
  marker,
  height = 300,
}: {
  curves: { name: string; color: string; curve: RocCurve }[];
  marker?: CurveMarker;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="fpr" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("false positive rate")} />
        <YAxis type="number" dataKey="tpr" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("true positive rate (recall)")} />
        <ReferenceLine segment={DIAGONAL} stroke="#cbd5e1" strokeDasharray="4 4" />
        <Tooltip formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        {curves.length > 1 && <Legend verticalAlign="bottom" height={24} />}
        {curves.map((c) => (
          <Line
            key={c.name}
            name={`${c.name} (AUC ${c.curve.auc.toFixed(3)})`}
            data={c.curve.points}
            dataKey="tpr"
            stroke={c.color}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
          />
        ))}
        {marker && <ReferenceDot x={marker.x} y={marker.y} r={5} fill={INK} stroke="#fff" strokeWidth={1.5} ifOverflow="visible" />}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrChart({ curve, marker, height = 300 }: { curve: PrCurve; marker?: CurveMarker; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={curve.points} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="recall" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("recall")} />
        <YAxis type="number" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("precision")} />
        <ReferenceLine
          y={curve.baseline}
          stroke="#cbd5e1"
          strokeDasharray="4 4"
          label={{ value: `prevalence ${curve.baseline.toFixed(2)}`, position: "insideTopLeft", fill: NEUTRAL, fontSize: 11 }}
        />
        <Tooltip formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        <Line name="precision" dataKey="precision" stroke={SERIES_COLORS[0]} strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
        {marker && <ReferenceDot x={marker.x} y={marker.y} r={5} fill={INK} stroke="#fff" strokeWidth={1.5} ifOverflow="visible" />}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Histogram of P(positive), stacked by the true class. */
export function ScoreDistributionChart({ bins }: { bins: ScoreBin[] }) {
  const data = bins.map((b) => ({ ...b, label: ((b.lo + b.hi) / 2).toFixed(3), range: `${b.lo.toFixed(2)}–${b.hi.toFixed(2)}` }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 0 }} barCategoryGap={1}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" interval={3} tickFormatter={(v) => Number(v).toFixed(1)} {...AXIS} label={bottomLabel("P(positive)")} />
        <YAxis {...AXIS} allowDecimals={false} width={32} />
        <Tooltip labelFormatter={(_, p) => `P = ${p?.[0]?.payload?.range ?? ""}`} cursor={{ fill: "#f1f5f9" }} />
        <Legend verticalAlign="top" height={24} />
        <Bar dataKey="positive" name="true positive class" stackId="a" fill={SERIES_COLORS[0]} isAnimationActive={false} />
        <Bar dataKey="negative" name="true negative class" stackId="a" fill={SERIES_COLORS[2]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Reliability diagram: observed positive rate against mean predicted probability, per non-empty bin. */
export function CalibrationChart({ calibration }: { calibration: Calibration }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={calibration.bins} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="mean_probability" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("mean predicted probability")} />
        <YAxis type="number" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("observed positive rate")} />
        <ReferenceLine segment={DIAGONAL} stroke="#cbd5e1" strokeDasharray="4 4" />
        <Tooltip formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        <Line
          name="observed rate"
          dataKey="fraction_positive"
          stroke={SERIES_COLORS[0]}
          strokeWidth={2}
          dot={{ r: 4, fill: SERIES_COLORS[0], stroke: "#fff", strokeWidth: 1 }}
          activeDot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Accuracy per input-length bucket with 95% intervals. */
export function LengthSliceChart({ slices }: { slices: LengthSlice[] }) {
  const data = slices.map((s) => ({
    label: `${s.min_chars}–${s.max_chars} chars`,
    n: s.n,
    accuracy: s.accuracy,
    err: [s.accuracy - s.ci_low, s.ci_high - s.accuracy] as [number, number],
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} />
        <YAxis domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={36} />
        <Tooltip formatter={fixed3} labelFormatter={(l, p) => `${l} (n = ${p?.[0]?.payload?.n ?? ""})`} cursor={{ fill: "#f1f5f9" }} />
        <Bar dataKey="accuracy" name="accuracy" fill={SERIES_COLORS[0]} isAnimationActive={false}>
          <ErrorBar dataKey="err" width={5} stroke={INK} strokeWidth={1.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Latency of each sample in dataset order; incorrect answers are drawn in red. */
export function LatencyScatter({ points }: { points: { index: number; latency_ms: number; correct: boolean }[] }) {
  const right = points.filter((p) => p.correct);
  const wrong = points.filter((p) => !p.correct);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="index" name="sample" {...AXIS} label={bottomLabel("sample (dataset order)")} />
        <YAxis type="number" dataKey="latency_ms" name="latency" unit=" ms" {...AXIS} width={60} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
        <Legend verticalAlign="bottom" height={24} />
        <Scatter name="correct" data={right} fill={SERIES_COLORS[0]} fillOpacity={0.75} isAnimationActive={false} />
        <Scatter name="incorrect" data={wrong} fill={DANGER} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
