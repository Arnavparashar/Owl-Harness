// Analysis charts: ROC, precision/recall, score distribution, calibration, accuracy by
// input length, per-sample latency and judge score distribution.
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { AccentDefs, AXIS, BAD, FAINT, GRID, INK, LEGEND_STYLE, MODEL_PALETTE, OK, TOOLTIP } from "../theme";
import { useChartMotion } from "./charts";

const UNIT_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const DIAGONAL = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
] as const;
const CHANCE = "rgba(255,255,255,0.22)";

const bottomLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -12, fill: FAINT, fontSize: 12 });
const leftLabel = (value: string) => ({ value, position: "insideLeft" as const, angle: -90, offset: 14, fill: FAINT, fontSize: 12 });
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
  const motion = useChartMotion();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <AccentDefs id="roc" />
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="fpr" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("false positive rate")} />
        <YAxis type="number" dataKey="tpr" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("true positive rate")} />
        <ReferenceLine segment={DIAGONAL} stroke={CHANCE} strokeDasharray="4 4" />
        <Tooltip {...TOOLTIP} formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        {curves.length > 1 && <Legend verticalAlign="bottom" height={24} wrapperStyle={LEGEND_STYLE} />}
        {curves.map((c) => (
          <Line
            key={c.name}
            name={`${c.name} (AUC ${c.curve.auc.toFixed(3)})`}
            data={c.curve.points}
            dataKey="tpr"
            stroke={curves.length === 1 ? "url(#roc)" : c.color}
            strokeWidth={curves.length === 1 ? 3 : 2}
            dot={false}
            activeDot={false}
            {...motion}
          />
        ))}
        {marker && <ReferenceDot x={marker.x} y={marker.y} r={6} fill={INK} stroke="#0b1020" strokeWidth={2} ifOverflow="visible" />}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrChart({ curve, marker, height = 300 }: { curve: PrCurve; marker?: CurveMarker; height?: number }) {
  const motion = useChartMotion();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={curve.points} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <AccentDefs id="pr" />
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="recall" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("recall")} />
        <YAxis type="number" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("precision")} />
        <ReferenceLine
          y={curve.baseline}
          stroke={CHANCE}
          strokeDasharray="4 4"
          label={{ value: `prevalence ${curve.baseline.toFixed(2)}`, position: "insideTopLeft", fill: FAINT, fontSize: 11 }}
        />
        <Tooltip {...TOOLTIP} formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        <Line name="precision" dataKey="precision" stroke="url(#pr)" strokeWidth={3} dot={false} activeDot={false} {...motion} />
        {marker && <ReferenceDot x={marker.x} y={marker.y} r={6} fill={INK} stroke="#0b1020" strokeWidth={2} ifOverflow="visible" />}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Histogram of P(positive), stacked by the true class. */
export function ScoreDistributionChart({ bins }: { bins: ScoreBin[] }) {
  const motion = useChartMotion();
  const data = bins.map((b) => ({ ...b, label: ((b.lo + b.hi) / 2).toFixed(3), range: `${b.lo.toFixed(2)}–${b.hi.toFixed(2)}` }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 0 }} barCategoryGap={1}>
        <AccentDefs id="sd" vertical />
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" interval={3} tickFormatter={(v) => Number(v).toFixed(1)} {...AXIS} label={bottomLabel("P(positive)")} />
        <YAxis {...AXIS} allowDecimals={false} width={32} />
        <Tooltip {...TOOLTIP} labelFormatter={(_, p) => `P = ${p?.[0]?.payload?.range ?? ""}`} />
        <Legend verticalAlign="top" height={24} wrapperStyle={LEGEND_STYLE} />
        <Bar dataKey="positive" name="true positive class" stackId="a" fill="url(#sd)" {...motion} />
        <Bar dataKey="negative" name="true negative class" stackId="a" fill="#7c8aa3" {...motion} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Reliability diagram: observed positive rate against mean predicted probability, per non-empty bin. */
export function CalibrationChart({ calibration }: { calibration: Calibration }) {
  const motion = useChartMotion();
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={calibration.bins} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <AccentDefs id="cal" />
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="mean_probability" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} label={bottomLabel("mean predicted probability")} />
        <YAxis type="number" domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={44} label={leftLabel("observed positive rate")} />
        <ReferenceLine segment={DIAGONAL} stroke={CHANCE} strokeDasharray="4 4" />
        <Tooltip {...TOOLTIP} formatter={fixed3} labelFormatter={() => ""} cursor={false} />
        <Line
          name="observed rate"
          dataKey="fraction_positive"
          stroke="url(#cal)"
          strokeWidth={3}
          dot={{ r: 4, fill: "#22d3ee", stroke: "#0b1020", strokeWidth: 1.5 }}
          activeDot={false}
          {...motion}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Accuracy per input-length bucket with 95% intervals. */
export function LengthSliceChart({ slices }: { slices: LengthSlice[] }) {
  const motion = useChartMotion();
  const data = slices.map((s) => ({
    label: `${s.min_chars}–${s.max_chars} chars`,
    n: s.n,
    accuracy: s.accuracy,
    err: [s.accuracy - s.ci_low, s.ci_high - s.accuracy] as [number, number],
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <AccentDefs id="len" vertical />
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" {...AXIS} />
        <YAxis domain={[0, 1]} ticks={UNIT_TICKS} {...AXIS} width={36} />
        <Tooltip {...TOOLTIP} formatter={fixed3} labelFormatter={(l, p) => `${l} (n = ${p?.[0]?.payload?.n ?? ""})`} />
        <Bar dataKey="accuracy" name="accuracy" fill="url(#len)" radius={[5, 5, 0, 0]} {...motion}>
          <ErrorBar dataKey="err" width={6} stroke={INK} strokeWidth={1.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Latency of each sample in dataset order; incorrect answers are drawn in the error colour. */
export function LatencyScatter({ points }: { points: { index: number; latency_ms: number; correct: boolean }[] }) {
  const motion = useChartMotion();
  const right = points.filter((p) => p.correct);
  const wrong = points.filter((p) => !p.correct);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="index" name="sample" {...AXIS} label={bottomLabel("sample (dataset order)")} />
        <YAxis type="number" dataKey="latency_ms" name="latency" unit=" ms" {...AXIS} width={64} />
        <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.3)" }} />
        <Legend verticalAlign="bottom" height={24} wrapperStyle={LEGEND_STYLE} />
        <Scatter name="correct" data={right} fill={MODEL_PALETTE[0]} fillOpacity={0.8} {...motion} />
        <Scatter name="incorrect" data={wrong} fill={BAD} {...motion} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** How often the judge gave each verdict or score. */
export function JudgeDistributionChart({ distribution }: { distribution: { score: string; count: number }[] }) {
  const motion = useChartMotion();
  const colour = (s: string) => (s === "unparseable" ? "#7c8aa3" : s === "pass" || s === "5" || s === "4" ? OK : s === "fail" || s === "1" || s === "2" ? BAD : "#fbbf24");
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={distribution} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="score" {...AXIS} />
        <YAxis {...AXIS} allowDecimals={false} width={32} />
        <Tooltip {...TOOLTIP} formatter={(v) => [String(v), "samples"]} />
        <Bar dataKey="count" radius={[5, 5, 0, 0]} {...motion}>
          {distribution.map((d) => (
            <Cell key={d.score} fill={colour(d.score)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
