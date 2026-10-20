import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Comparison, type RunSummary } from "../api";
import { RocChart } from "../components/analysis";
import { useChartMotion } from "../components/charts";
import { OutcomeGrid } from "../components/OutcomeGrid";
import { Card, Chip, Dense, EmptyLine, ErrorNote, Loading, MetricTile } from "../components/ui";
import { MODE_LABEL, ms, num, pct, pValue, statText, useAsync } from "../lib";
import { AXIS, GRID, INK, LEGEND_STYLE, TOOLTIP, useModelColor } from "../theme";

const PAGE = 15;

/** A distinct label per run: the model, plus a short run id when the same model appears twice. */
function runLabels(runs: RunSummary[]): Record<string, string> {
  const counts: Record<string, number> = {};
  for (const r of runs) counts[r.model] = (counts[r.model] ?? 0) + 1;
  return Object.fromEntries(runs.map((r) => [r.id, counts[r.model] > 1 ? `${r.model} (${r.id.slice(-4)})` : r.model]));
}

export default function ComparePage() {
  const [params] = useSearchParams();
  const ids = useMemo(() => (params.get("runs") ?? "").split(",").filter(Boolean), [params]);
  const { data, error, loading } = useAsync(() => (ids.length ? api.compare(ids) : Promise.resolve(undefined)), [ids.join(",")]);

  return (
    <div className="space-y-5">
      {ids.length === 0 && (
        <Card>
          <EmptyLine>
            Select two or more runs on the{" "}
            <Link to="/runs" className="text-accent hover:underline">
              Runs
            </Link>{" "}
            page, then choose Compare selected.
          </EmptyLine>
        </Card>
      )}
      {loading && ids.length > 0 && !data && <Loading />}
      {error && <ErrorNote message={error} />}
      {data && <CompareView data={data} />}
    </div>
  );
}

function CompareView({ data }: { data: Comparison }) {
  const labels = runLabels(data.runs);
  const colorOf = useModelColor();
  const motion = useChartMotion();
  const color = (r: RunSummary) => colorOf(r.model);
  const tasks = new Set(data.runs.map((r) => r.task));

  // One group per metric, one bar per run; whiskers are 95% intervals where the run has one.
  const groups: { name: string; get: (r: RunSummary) => { v?: number; lo?: number | null; hi?: number | null } }[] = [
    {
      name: "Accuracy",
      get: (r) => (r.headline.main && r.headline.main.key !== "mean_score" ? { v: r.headline.main.value, lo: r.headline.main.ci_low, hi: r.headline.main.ci_high } : { v: r.metrics.accuracy }),
    },
    { name: "Precision", get: (r) => ({ v: r.metrics.precision }) },
    { name: "Recall", get: (r) => ({ v: r.metrics.recall }) },
    { name: "F1 / macro-F1", get: (r) => ({ v: r.headline.secondary?.value, lo: r.headline.secondary?.ci_low, hi: r.headline.secondary?.ci_high }) },
  ];
  const metricRows = groups
    .map((g) => {
      const row: Record<string, string | number | [number, number]> = { metric: g.name };
      let any = false;
      for (const r of data.runs) {
        const { v, lo, hi } = g.get(r);
        if (v === undefined) continue;
        any = true;
        row[r.id] = v;
        row[`${r.id}__err`] = [lo === null || lo === undefined ? 0 : v - lo, hi === null || hi === undefined ? 0 : hi - v];
      }
      return any ? row : null;
    })
    .filter((r): r is Record<string, string | number | [number, number]> => r !== null);

  const latencyRows = data.runs.map((r) => ({ run: labels[r.id], median: r.metrics.latency_median_ms ?? 0, p95: r.metrics.latency_p95_ms ?? 0 }));
  const rocCurves = data.runs.flatMap((r) => (data.roc[r.id] ? [{ name: labels[r.id], color: color(r), curve: data.roc[r.id] }] : []));

  return (
    <>
      {tasks.size > 1 && (
        <div className="rounded-[12px] border border-warn/40 bg-warn/10 px-4 py-2.5 text-sm text-warn">
          These runs are on different tests ({[...tasks].join(", ")}). Per-sample comparisons only use samples with the same id in every run.
        </div>
      )}

      <Card title="Metrics" pad={false}>
        <div className="overflow-x-auto p-4">
          <Dense>
            <table className="w-full border-collapse">
              <thead className="border-b border-white/10">
                <tr>
                  {["Model", "Test", "n", "Main metric", "F1 / macro-F1", "Precision", "Recall", "Unparseable", "Median", "p95"].map((h, i) => (
                    <th key={h} className={`label px-4 py-2.5 ${i < 2 ? "text-left" : "text-right"}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.runs.map((r) => (
                  <tr key={r.id} className="border-b border-white/[0.06] last:border-0">
                    <td className="px-4 py-2.5 text-[13px]">
                      <Link to={`/runs/${r.id}`} className="flex items-center gap-2 font-medium text-ink hover:text-accent">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: color(r) }} />
                        {labels[r.id]}
                      </Link>
                    </td>
                    <td className="px-4 py-2.5 text-[13px] text-sub">
                      {r.task} <span className="text-faint">· {MODE_LABEL[r.scoring_mode]}</span>
                    </td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{r.metrics.n?.toFixed(0) ?? "–"}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px]">
                      <span className="font-semibold text-ink">{statText(r.headline.main)}</span>
                      {r.headline.main?.ci_low !== null && r.headline.main?.ci_low !== undefined && (
                        <span className="block text-xs text-faint">
                          95% CI {num(r.headline.main.ci_low, 2)}–{num(r.headline.main.ci_high, 2)}
                        </span>
                      )}
                    </td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{num(r.headline.secondary?.value)}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{num(r.metrics.precision)}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{num(r.metrics.recall)}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{r.metrics.unparseable?.toFixed(0) ?? "–"}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{ms(r.metrics.latency_median_ms)}</td>
                    <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{ms(r.metrics.latency_p95_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Dense>
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Main metrics" right={<span className="text-xs text-faint">whiskers: 95% CI where available</span>}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={metricRows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="metric" {...AXIS} />
              <YAxis domain={[0, 1]} {...AXIS} width={36} />
              <Tooltip {...TOOLTIP} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
              <Legend verticalAlign="top" height={30} wrapperStyle={LEGEND_STYLE} />
              {data.runs.map((r) => (
                <Bar key={r.id} dataKey={r.id} name={labels[r.id]} fill={color(r)} fillOpacity={0.88} radius={[4, 4, 0, 0]} {...motion}>
                  <ErrorBar dataKey={`${r.id}__err`} width={4} stroke={INK} strokeWidth={1.4} />
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Latency" right={<span className="text-xs text-faint">median and p95</span>}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={latencyRows} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="run" {...AXIS} />
              <YAxis {...AXIS} width={56} tickFormatter={(v) => ms(v)} />
              <Tooltip {...TOOLTIP} formatter={(v) => (typeof v === "number" ? ms(v) : String(v))} />
              <Legend verticalAlign="top" height={30} wrapperStyle={LEGEND_STYLE} />
              <Bar dataKey="median" name="Median" radius={[4, 4, 0, 0]} {...motion} fill="#8b9bff" />
              <Bar dataKey="p95" name="p95" radius={[4, 4, 0, 0]} {...motion} fill="#5b6a86" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Agreement data={data} labels={labels} />

      {rocCurves.length > 0 && (
        <Card title="ROC curves" right={<span className="text-xs text-faint">models that output a probability</span>}>
          <RocChart curves={rocCurves} height={340} />
        </Card>
      )}

      {data.sample_ids.length > 0 && (
        <Card title="Per-sample outcome">
          <OutcomeGrid runs={data.runs.map((r) => ({ id: r.id, label: labels[r.id], color: color(r) }))} sampleIds={data.sample_ids} outcomes={data.outcomes} />
        </Card>
      )}

      <Disagreements data={data} labels={labels} />
    </>
  );
}

function Agreement({ data, labels }: { data: Comparison; labels: Record<string, string> }) {
  const shared = data.shared_samples;
  const c = data.coverage;
  return (
    <Card title="Agreement and significance">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricTile label="Shared samples" text={String(shared)} />
        <MetricTile label="All correct" text={String(c.all_correct)} sub={pct(shared ? c.all_correct / shared : 0)} />
        <MetricTile label="All wrong" text={String(c.all_wrong)} sub={pct(shared ? c.all_wrong / shared : 0)} />
        <MetricTile label="At least one correct" text={String(c.any_correct)} sub={`${pct(shared ? c.any_correct / shared : 0)} · best case for picking per sample`} />
      </div>
      {data.pairwise.length > 0 && (
        <Dense className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse">
            <thead className="border-b border-white/10">
              <tr>
                <th className="label px-4 py-2.5 text-left">Pair</th>
                <th className="label px-4 py-2.5 text-right">Same answer</th>
                <th className="label px-4 py-2.5 text-right">Only first correct</th>
                <th className="label px-4 py-2.5 text-right">Only second correct</th>
                <th className="label px-4 py-2.5 text-right">McNemar p</th>
                <th className="label px-4 py-2.5 text-left">At 5%</th>
              </tr>
            </thead>
            <tbody>
              {data.pairwise.map((p) => (
                <tr key={`${p.a}-${p.b}`} className="border-b border-white/[0.06] last:border-0">
                  <td className="px-4 py-2.5 text-[13px] text-ink">
                    {labels[p.a]} <span className="text-faint">vs</span> {labels[p.b]}
                  </td>
                  <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{pct(p.agreement)}</td>
                  <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{p.a_only_correct}</td>
                  <td className="num px-4 py-2.5 text-right text-[13px] text-sub">{p.b_only_correct}</td>
                  <td className="num px-4 py-2.5 text-right text-[13px] text-ink">{pValue(p.mcnemar_p)}</td>
                  <td className="px-4 py-2.5 text-[13px]">{p.mcnemar_p < 0.05 ? <Chip tone="accent">accuracy differs</Chip> : <span className="text-sub">no detectable difference</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Dense>
      )}
      <p className="mt-3 text-xs text-faint">
        Exact McNemar test on the samples where exactly one of the two models is right. P-values are not corrected for several comparisons, and a small sample can only detect large differences.
      </p>
    </Card>
  );
}

function Disagreements({ data, labels }: { data: Comparison; labels: Record<string, string> }) {
  const [page, setPage] = useState(1);
  const rows = data.disagreements;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const slice = rows.slice((page - 1) * PAGE, page * PAGE);
  return (
    <Card title={`Samples where the models disagree (${data.total_disagreements} of ${data.shared_samples})`}>
      {rows.length === 0 ? (
        <EmptyLine>The models gave the same answer on every shared sample.</EmptyLine>
      ) : (
        <>
          <Dense className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-white/10">
                <tr>
                  <th className="label px-4 py-2.5 text-left">ID</th>
                  <th className="label px-4 py-2.5 text-left">Input</th>
                  <th className="label px-4 py-2.5 text-left">Target</th>
                  {data.runs.map((r) => (
                    <th key={r.id} className="label px-4 py-2.5 text-left">
                      {labels[r.id]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {slice.map((d) => (
                  <tr key={d.sample_id} className="border-b border-white/[0.06] last:border-0">
                    <td className="num px-4 py-2 text-[13px] text-sub">{d.sample_id}</td>
                    <td className="max-w-[440px] truncate px-4 py-2 text-[13px] text-ink" title={d.input}>
                      {d.input}
                    </td>
                    <td className="max-w-[140px] truncate px-4 py-2 text-[13px] text-sub">{d.target}</td>
                    {data.runs.map((r) => (
                      <td key={r.id} className={`max-w-[160px] truncate px-4 py-2 text-[13px] font-medium ${d.correct[r.id] ? "text-ok" : "text-bad"}`}>
                        {d.preds[r.id] ?? "unparseable"}
                        <span className="ml-1.5 text-xs font-normal">{d.correct[r.id] ? "✓" : "✗"}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Dense>
          <div className="mt-3 flex items-center justify-between text-sm text-sub">
            <span className="num">
              Showing {(page - 1) * PAGE + 1}–{Math.min(page * PAGE, rows.length)} of {rows.length}
              {data.total_disagreements > rows.length && ` (first ${rows.length} loaded)`}
            </span>
            <div className="flex items-center gap-2">
              <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="btn btn-secondary h-8">
                Prev
              </button>
              <span className="num">
                {page} / {pages}
              </span>
              <button disabled={page >= pages} onClick={() => setPage(page + 1)} className="btn btn-secondary h-8">
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}
