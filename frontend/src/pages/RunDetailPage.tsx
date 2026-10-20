import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, type RunDetail } from "../api";
import { CalibrationChart, LatencyScatter, LengthSliceChart, PrChart, RocChart, ScoreDistributionChart } from "../components/analysis";
import { ConfusionMatrix, LatencyHistogram, ThresholdChart } from "../components/charts";
import SampleExplorer from "../components/SampleExplorer";
import { btnSecondary, Card, ErrorNote, Loading, MetricCard, StatusBadge, td, th } from "../components/ui";
import { dateTime, derivedMetrics, ms, num, SERIES_COLORS, seconds, useAsync } from "../lib";

type Tab = "overview" | "scores" | "latency" | "samples";

export default function RunDetailPage() {
  const { runId = "" } = useParams();
  const { data: run, error, loading } = useAsync(() => api.run(runId), [runId]);

  return (
    <div className="space-y-5">
      <Link to="/" className="text-sm text-accent hover:underline">
        ← All runs
      </Link>
      {loading && <Loading />}
      {error && <ErrorNote message={error} />}
      {run && <RunView run={run} />}
    </div>
  );
}

function RunView({ run }: { run: RunDetail }) {
  const m = run.metrics;
  const positive = run.positive_class;
  const curve = useAsync(() => (run.has_probabilities ? api.thresholdCurve(run.id) : Promise.resolve(undefined)), [run.id]);
  const analysis = useAsync(() => api.analysis(run.id), [run.id]);
  const byOutcome = analysis.data?.latency_by_outcome;
  const [threshold, setThreshold] = useState<number>();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string>();

  const deleteRun = async () => {
    try {
      await api.deleteRun(run.id);
      navigate("/");
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    }
  };

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    ...(run.has_probabilities ? [{ id: "scores" as const, label: "Score analysis" }] : []),
    { id: "latency", label: "Latency" },
    { id: "samples", label: `Samples (${run.sample_counts.total})` },
  ];
  const requested = params.get("tab") as Tab | null;
  const tab = tabs.some((t) => t.id === requested) ? (requested as Tab) : "overview";

  useEffect(() => {
    if (curve.data) setThreshold(curve.data.run_threshold);
  }, [curve.data]);

  // When the marker is moved off the threshold the run used, recompute from the curve.
  const runThreshold = curve.data?.run_threshold;
  const point =
    curve.data && threshold !== undefined && threshold !== runThreshold
      ? curve.data.points[Math.round(threshold * 100)]
      : undefined;

  const n = m.n ?? 0;
  const view = point
    ? {
        accuracy: point.accuracy,
        stderr: n > 1 ? Math.sqrt((point.accuracy * (1 - point.accuracy)) / (n - 1)) : 0,
        precision: point.precision,
        recall: point.recall,
        f1: point.f1,
        tp: point.tp,
        fp: point.fp,
        fn: point.fn,
        tn: point.tn,
        unparseable: 0,
      }
    : {
        accuracy: m.accuracy,
        stderr: m.stderr,
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        tp: m.tp,
        fp: m.fp,
        fn: m.fn,
        tn: m.tn,
        unparseable: m.unparseable ?? 0,
      };
  const hasBinary = positive !== null && m.tp !== undefined;
  const derived = hasBinary ? derivedMetrics(view.tp ?? 0, view.fp ?? 0, view.fn ?? 0, view.tn ?? 0) : undefined;
  const auc = analysis.data?.roc?.auc;
  const ci = !point && run.accuracy_ci ? `95% CI ${run.accuracy_ci[0].toFixed(2)}–${run.accuracy_ci[1].toFixed(2)}` : undefined;

  const selectTab = (id: Tab) => {
    const next = new URLSearchParams(params);
    if (id === "overview") next.delete("tab");
    else next.set("tab", id);
    setParams(next, { replace: true });
  };

  return (
    <>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <h1 className="text-lg font-semibold text-slate-900">{run.model}</h1>
            <span className="text-sm text-slate-600">{run.task}</span>
            <StatusBadge status={run.status} />
          </div>
          <p className="mt-1 text-xs text-slate-500">
            run {run.id} · started {dateTime(run.started_at)} · {run.sample_counts.total} samples
            {run.sample_counts.errors > 0 && ` (${run.sample_counts.errors} errored)`} · dataset {run.dataset_hash.slice(0, 12)} · owl {run.owl_version}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <a href={api.exportUrl(run.id)} download className={btnSecondary}>
            Export samples (CSV)
          </a>
          {!confirmDelete ? (
            <button className={btnSecondary} onClick={() => setConfirmDelete(true)}>
              Delete run
            </button>
          ) : (
            <span className="flex items-center gap-2 text-sm">
              <span className="text-slate-700">Delete this run and its samples?</span>
              <button className="font-medium text-danger hover:underline" onClick={deleteRun}>
                Delete
              </button>
              <button className="text-slate-600 hover:underline" onClick={() => setConfirmDelete(false)}>
                Keep
              </button>
            </span>
          )}
        </div>
      </div>
      {deleteError && <ErrorNote message={deleteError} />}

      {point && threshold !== undefined && (
        <div className="flex items-center justify-between rounded border border-accent/30 bg-accent-soft px-4 py-2 text-sm text-slate-800">
          <span>
            Metrics recomputed at threshold {threshold.toFixed(2)} (the run used {runThreshold?.toFixed(2)}).
          </span>
          <button onClick={() => setThreshold(runThreshold)} className="font-medium text-accent hover:underline">
            Reset
          </button>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard
          label="Accuracy"
          value={num(view.accuracy)}
          sub={view.stderr !== undefined ? `± ${view.stderr.toFixed(3)} stderr${ci ? ` · ${ci}` : ""}` : undefined}
        />
        <MetricCard label="Precision" value={num(view.precision)} />
        <MetricCard label="Recall" value={num(view.recall)} sub="true positive rate" />
        <MetricCard label="F1" value={num(view.f1)} />
        {derived && (
          <>
            <MetricCard label="Specificity" value={num(derived.specificity)} sub="true negative rate" />
            <MetricCard label="Balanced accuracy" value={num(derived.balancedAccuracy)} sub="mean of recall and specificity" />
            <MetricCard label="MCC" value={num(derived.mcc)} sub="Matthews correlation" />
          </>
        )}
        {auc !== undefined && <MetricCard label="ROC AUC" value={num(auc)} sub="independent of threshold" />}
        <MetricCard label="Unparseable" value={String(m.unparseable ?? 0)} />
        <MetricCard label="Median latency" value={ms(m.latency_median_ms)} />
        <MetricCard label="p95 latency" value={ms(m.latency_p95_ms)} />
        <MetricCard label="Wall time" value={seconds(m.wall_time_s)} />
      </div>

      <nav className="flex gap-6 border-b border-slate-200" aria-label="Run sections">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => selectTab(t.id)}
            aria-current={tab === t.id ? "page" : undefined}
            className={`-mb-px border-b-2 px-0.5 pb-2 text-sm font-medium ${
              tab === t.id ? "border-accent text-accent" : "border-transparent text-slate-600 hover:text-slate-900"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <div className="space-y-5">
          <div className="grid gap-5 lg:grid-cols-2">
            {hasBinary && positive !== null && (
              <Card title="Confusion matrix">
                <ConfusionMatrix
                  tp={view.tp ?? 0}
                  fp={view.fp ?? 0}
                  fn={view.fn ?? 0}
                  tn={view.tn ?? 0}
                  positive={positive}
                  unparseable={view.unparseable}
                />
                {derived && (
                  <p className="mt-1 text-xs text-slate-500">
                    False positive rate {num(derived.fpr)} · negative predictive value {num(derived.npv)}
                  </p>
                )}
              </Card>
            )}
            <Card title="Accuracy by input length">
              {analysis.error && <ErrorNote message={analysis.error} />}
              {analysis.data &&
                (analysis.data.length_slices.length > 1 ? (
                  <>
                    <LengthSliceChart slices={analysis.data.length_slices} />
                    <p className="mt-1 text-xs text-slate-500">
                      Samples split into equal-sized groups by character count; error bars are 95% Wilson intervals. Each group is small, so
                      differences between groups are suggestive only.
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-slate-600">Too few samples to split by length (at least 16 are needed for two groups).</p>
                ))}
            </Card>
          </div>
          <RunConfigCard run={run} />
        </div>
      )}

      {tab === "scores" && run.has_probabilities && (
        <div className="space-y-5">
          <Card title="Threshold analysis">
            {curve.error && <ErrorNote message={curve.error} />}
            {curve.data && threshold !== undefined && (
              <>
                <p className="mb-2 text-sm text-slate-600">
                  Precision, recall and F1 if the model answered positive whenever P(positive) ≥ threshold. Drag on the chart or use the
                  slider; the cards and the operating point on the curves below update.
                </p>
                <ThresholdChart points={curve.data.points} threshold={threshold} onChange={setThreshold} />
              </>
            )}
          </Card>
          {analysis.error && <ErrorNote message={analysis.error} />}
          {analysis.data && (
            <>
              <div className="grid gap-5 lg:grid-cols-2">
                {analysis.data.roc && (
                  <Card title={`ROC curve (AUC ${analysis.data.roc.auc.toFixed(3)})`}>
                    <RocChart
                      curves={[{ name: run.model, color: SERIES_COLORS[0], curve: analysis.data.roc }]}
                      marker={derived ? { x: derived.fpr, y: view.recall ?? 0 } : undefined}
                    />
                    <p className="mt-1 text-xs text-slate-500">The dot is the operating point at the current threshold. The dashed line is chance.</p>
                  </Card>
                )}
                {analysis.data.pr && (
                  <Card title={`Precision–recall curve (AP ${analysis.data.pr.average_precision.toFixed(3)})`}>
                    <PrChart curve={analysis.data.pr} marker={hasBinary ? { x: view.recall ?? 0, y: view.precision ?? 0 } : undefined} />
                    <p className="mt-1 text-xs text-slate-500">AP is average precision. The dashed line is the share of positives, the score of a model with no skill.</p>
                  </Card>
                )}
              </div>
              <div className="grid gap-5 lg:grid-cols-2">
                {analysis.data.score_distribution && (
                  <Card title="Score distribution">
                    <ScoreDistributionChart bins={analysis.data.score_distribution} />
                    <p className="mt-1 text-xs text-slate-500">How the model's P(positive) is spread, split by the true class. Less overlap means easier separation.</p>
                  </Card>
                )}
                {analysis.data.calibration && (
                  <Card title="Calibration">
                    <CalibrationChart calibration={analysis.data.calibration} />
                    <p className="mt-1 text-xs text-slate-500">
                      Brier score {analysis.data.calibration.brier.toFixed(3)} · expected calibration error {analysis.data.calibration.ece.toFixed(3)}.
                      Points on the dashed line mean the probabilities match observed frequencies; only non-empty bins are drawn.
                    </p>
                  </Card>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {tab === "latency" && (
        <div className="space-y-5">
          <div className="grid gap-5 lg:grid-cols-2">
            <Card title="Latency distribution">
              <LatencyHistogram bins={run.latency_histogram} />
            </Card>
            <Card title="Latency per sample">
              {analysis.error && <ErrorNote message={analysis.error} />}
              {analysis.data && <LatencyScatter points={analysis.data.latency_points} />}
            </Card>
          </div>
          {byOutcome && (
            <Card title="Median latency by outcome">
              <div className="-m-4 overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead className="border-b border-slate-200 bg-slate-50">
                    <tr>
                      <th className={th}>Outcome</th>
                      <th className={`${th} text-right`}>Samples</th>
                      <th className={`${th} text-right`}>Median latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(["correct", "incorrect"] as const).map((k) => {
                      const row = byOutcome[k];
                      return (
                        <tr key={k}>
                          <td className={td}>{k}</td>
                          <td className={`${td} text-right tabular-nums`}>{row?.n ?? 0}</td>
                          <td className={`${td} text-right tabular-nums`}>{row ? ms(row.median_ms) : "–"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === "samples" && <SampleExplorer run={run} />}
    </>
  );
}

function RunConfigCard({ run }: { run: RunDetail }) {
  const c = run.config;
  const args = Object.entries(c.model_args).filter(([k]) => k !== "instruction");
  const rows: [string, string][] = [
    ["Provider", run.provider],
    ["Dataset hash", run.dataset_hash.slice(0, 16)],
    ...(!run.has_probabilities && c.generate.temperature !== null ? [["Temperature", String(c.generate.temperature)] as [string, string]] : []),
    ...(!run.has_probabilities && c.generate.max_tokens !== null ? [["Max tokens", String(c.generate.max_tokens)] as [string, string]] : []),
    ["Concurrency", String(c.max_concurrency)],
    ["Timeout", `${c.timeout_s} s`],
    ["Retries", String(c.retries)],
    ["Sample limit", c.limit === null ? "none" : String(c.limit)],
    ...args.map(([k, v]) => [k, JSON.stringify(v)] as [string, string]),
  ];
  const instruction = typeof c.model_args.instruction === "string" ? c.model_args.instruction : undefined;
  return (
    <Card title="Run configuration">
      <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm sm:grid-cols-[max-content_1fr_max-content_1fr]">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-slate-500">{k}</dt>
            <dd className="font-mono text-xs leading-5 text-slate-900">{v}</dd>
          </div>
        ))}
      </dl>
      {(c.system_prompt || instruction) && (
        <div className="mt-4">
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
            {c.system_prompt ? "System prompt" : "Instruction given to the model"}
          </div>
          <pre className="whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-3 font-mono text-xs leading-5 text-slate-800">
            {c.system_prompt ?? instruction}
          </pre>
        </div>
      )}
    </Card>
  );
}
