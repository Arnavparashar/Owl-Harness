import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Comparison } from "../api";
import { RocChart } from "../components/analysis";
import { Card, ErrorNote, Loading, MetricCard, th, td } from "../components/ui";
import { ms, num, pct, pValue, runLabels, SERIES_COLORS, useAsync } from "../lib";

const AXIS = { stroke: "#94a3b8", fontSize: 12, tickLine: false } as const;
const PAGE = 15;

export default function ComparePage() {
  const [params] = useSearchParams();
  const ids = useMemo(() => (params.get("runs") ?? "").split(",").filter(Boolean), [params]);
  const { data, error, loading } = useAsync(() => (ids.length ? api.compare(ids) : Promise.resolve(undefined)), [ids.join(",")]);

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <h1 className="text-lg font-semibold text-slate-900">Compare</h1>
        <Link to="/" className="text-sm text-accent hover:underline">
          ← Select runs
        </Link>
      </div>
      {ids.length === 0 && <p className="text-sm text-slate-600">Select two or more runs on the Runs page, then choose Compare selected.</p>}
      {loading && ids.length > 0 && <Loading />}
      {error && <ErrorNote message={error} />}
      {data && <CompareView data={data} />}
    </div>
  );
}

function CompareView({ data }: { data: Comparison }) {
  const labels = runLabels(data.runs);

  // One group per metric; one bar per run. Accuracy carries its stderr as an error bar.
  const metricRows = (
    [
      ["Accuracy", "accuracy"],
      ["Precision", "precision"],
      ["Recall", "recall"],
      ["F1", "f1"],
    ] as const
  ).map(([name, key]) => {
    const row: Record<string, string | number> = { metric: name };
    for (const r of data.runs) {
      row[r.id] = r.metrics[key] ?? 0;
      row[`${r.id}__err`] = key === "accuracy" ? (r.metrics.stderr ?? 0) : 0;
    }
    return row;
  });

  const rocCurves = data.runs.flatMap((r, i) =>
    data.roc[r.id] ? [{ name: labels[r.id], color: SERIES_COLORS[i % SERIES_COLORS.length], curve: data.roc[r.id] }] : [],
  );

  const latencyRows = data.runs.map((r) => ({
    run: labels[r.id],
    median: r.metrics.latency_median_ms ?? 0,
    p95: r.metrics.latency_p95_ms ?? 0,
  }));

  return (
    <>
      <Card title="Metrics">
        <div className="-m-4 mb-3 overflow-x-auto">
          <table className="w-full border-collapse">
            <thead className="border-b border-slate-200 bg-slate-50">
              <tr>
                <th className={th}>Model</th>
                <th className={`${th} text-right`}>n</th>
                <th className={`${th} text-right`}>Accuracy</th>
                <th className={`${th} text-right`}>Precision</th>
                <th className={`${th} text-right`}>Recall</th>
                <th className={`${th} text-right`}>F1</th>
                <th className={`${th} text-right`}>Unparseable</th>
                <th className={`${th} text-right`}>Median latency</th>
                <th className={`${th} text-right`}>p95 latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.runs.map((r, i) => (
                <tr key={r.id}>
                  <td className={td}>
                    <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }} />
                    <Link to={`/runs/${r.id}`} className="font-medium text-accent hover:underline">
                      {labels[r.id]}
                    </Link>
                  </td>
                  <td className={`${td} text-right tabular-nums`}>{r.metrics.n?.toFixed(0) ?? "–"}</td>
                  <td className={`${td} text-right tabular-nums`}>
                    {num(r.metrics.accuracy)} <span className="text-slate-500">± {num(r.metrics.stderr)}</span>
                    {r.accuracy_ci && (
                      <span className="block text-xs text-slate-500">
                        95% CI {r.accuracy_ci[0].toFixed(2)}–{r.accuracy_ci[1].toFixed(2)}
                      </span>
                    )}
                  </td>
                  <td className={`${td} text-right tabular-nums`}>{num(r.metrics.precision)}</td>
                  <td className={`${td} text-right tabular-nums`}>{num(r.metrics.recall)}</td>
                  <td className={`${td} text-right tabular-nums`}>{num(r.metrics.f1)}</td>
                  <td className={`${td} text-right tabular-nums`}>{r.metrics.unparseable?.toFixed(0) ?? "–"}</td>
                  <td className={`${td} text-right tabular-nums`}>{ms(r.metrics.latency_median_ms)}</td>
                  <td className={`${td} text-right tabular-nums`}>{ms(r.metrics.latency_p95_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Accuracy, precision, recall, F1">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricRows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="metric" {...AXIS} />
              <YAxis domain={[0, 1]} {...AXIS} width={36} />
              <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} cursor={{ fill: "#f1f5f9" }} />
              <Legend verticalAlign="top" height={28} />
              {data.runs.map((r, i) => (
                <Bar key={r.id} dataKey={r.id} name={labels[r.id]} fill={SERIES_COLORS[i % SERIES_COLORS.length]} isAnimationActive={false}>
                  <ErrorBar dataKey={(row: Record<string, number>) => row[`${r.id}__err`]} width={4} stroke="#0f172a" strokeWidth={1.5} />
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-1 text-xs text-slate-500">Error bars on accuracy show ± 1 standard error.</p>
        </Card>

        <Card title="Latency">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={latencyRows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="run" {...AXIS} />
              <YAxis {...AXIS} width={48} unit=" ms" />
              <Tooltip formatter={(v) => (typeof v === "number" ? `${v.toFixed(0)} ms` : String(v))} cursor={{ fill: "#f1f5f9" }} />
              <Legend verticalAlign="top" height={28} />
              <Bar dataKey="median" name="Median" fill={SERIES_COLORS[0]} isAnimationActive={false} />
              <Bar dataKey="p95" name="p95" fill={SERIES_COLORS[2]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Agreement data={data} labels={labels} />

      {rocCurves.length > 0 && (
        <Card title="ROC curves">
          <RocChart curves={rocCurves} height={340} />
          <p className="mt-1 text-xs text-slate-500">
            Only models that output a probability appear here. AUC measures ranking quality independent of any threshold.
          </p>
        </Card>
      )}

      <CorrectnessGrid data={data} labels={labels} />

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
        <MetricCard label="Shared samples" value={String(shared)} />
        <MetricCard label="All correct" value={String(c.all_correct)} sub={pct(shared ? c.all_correct / shared : 0)} />
        <MetricCard label="All wrong" value={String(c.all_wrong)} sub={pct(shared ? c.all_wrong / shared : 0)} />
        <MetricCard label="At least one correct" value={String(c.any_correct)} sub={`${pct(shared ? c.any_correct / shared : 0)} (best case for picking per sample)`} />
      </div>
      {data.pairwise.length > 0 && (
        <div className="-mx-4 mt-4 overflow-x-auto border-t border-slate-200">
          <table className="w-full border-collapse">
            <thead className="border-b border-slate-200 bg-slate-50">
              <tr>
                <th className={th}>Pair</th>
                <th className={`${th} text-right`}>Same answer</th>
                <th className={`${th} text-right`}>Only first correct</th>
                <th className={`${th} text-right`}>Only second correct</th>
                <th className={`${th} text-right`}>McNemar p</th>
                <th className={th}>Verdict at 5%</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.pairwise.map((p) => (
                <tr key={`${p.a}-${p.b}`}>
                  <td className={td}>
                    {labels[p.a]} <span className="text-slate-400">vs</span> {labels[p.b]}
                  </td>
                  <td className={`${td} text-right tabular-nums`}>{pct(p.agreement)}</td>
                  <td className={`${td} text-right tabular-nums`}>{p.a_only_correct}</td>
                  <td className={`${td} text-right tabular-nums`}>{p.b_only_correct}</td>
                  <td className={`${td} text-right tabular-nums`}>{pValue(p.mcnemar_p)}</td>
                  <td className={td}>{p.mcnemar_p < 0.05 ? "accuracy differs" : "no detectable difference"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-3 text-xs text-slate-500">
        Exact McNemar test on the samples where exactly one of the two models is right. P-values are not corrected for several comparisons, and
        a small sample can only detect large differences.
      </p>
    </Card>
  );
}

function CorrectnessGrid({ data, labels }: { data: Comparison; labels: Record<string, string> }) {
  const n = data.sample_ids.length;
  if (n === 0) return null;
  return (
    <Card title="Per-sample outcome">
      <div className="space-y-2">
        {data.runs.map((r) => (
          <div key={r.id} className="flex items-center gap-3">
            <div className="w-40 shrink-0 truncate text-sm text-slate-700" title={labels[r.id]}>
              {labels[r.id]}
            </div>
            <div className="grid flex-1 gap-px" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
              {data.outcomes[r.id].map((ok, k) => (
                <div
                  key={data.sample_ids[k]}
                  title={`sample ${data.sample_ids[k]}: ${ok ? "correct" : "incorrect"}`}
                  className={`h-5 ${ok ? "bg-slate-300" : "bg-danger"}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 bg-slate-300" /> correct
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 bg-danger" /> incorrect
        </span>
        <span>One cell per sample in dataset order. Columns that are red in every row are samples no model got right.</span>
      </div>
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
      <div className="-m-4 overflow-x-auto">
        <table className="w-full border-collapse">
          <thead className="border-b border-slate-200 bg-slate-50">
            <tr>
              <th className={th}>ID</th>
              <th className={th}>Input</th>
              <th className={th}>Target</th>
              {data.runs.map((r) => (
                <th key={r.id} className={th}>
                  {labels[r.id]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {slice.map((d) => (
              <tr key={d.sample_id}>
                <td className={`${td} tabular-nums`}>{d.sample_id}</td>
                <td className={`${td} max-w-[480px] truncate`} title={d.input}>
                  {d.input}
                </td>
                <td className={td}>{d.target}</td>
                {data.runs.map((r) => (
                  <td key={r.id} className={`${td} font-medium ${d.correct[r.id] ? "text-slate-900" : "text-danger"}`}>
                    {d.preds[r.id] ?? "unparseable"}
                    <span className="ml-1.5 text-xs font-normal">{d.correct[r.id] ? "✓" : "✗"}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-6 flex items-center justify-between text-sm text-slate-600">
        <span>
          Showing {rows.length === 0 ? 0 : (page - 1) * PAGE + 1}–{Math.min(page * PAGE, rows.length)} of {rows.length}
        </span>
        <div className="flex items-center gap-2">
          <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="rounded border border-slate-300 px-2 py-0.5 disabled:text-slate-300">
            Prev
          </button>
          <span className="tabular-nums">
            {page} / {pages}
          </span>
          <button disabled={page >= pages} onClick={() => setPage(page + 1)} className="rounded border border-slate-300 px-2 py-0.5 disabled:text-slate-300">
            Next
          </button>
        </div>
      </div>
    </Card>
  );
}
