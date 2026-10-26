import { useEffect, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, type Analysis, type RunDetail, type Stat } from "../api";
import { CalibrationChart, JudgeDistributionChart, LatencyScatter, LengthSliceChart, PrChart, RocChart, ScoreDistributionChart } from "../components/analysis";
import { ConfusionMatrix, LatencyHistogram, ThresholdChart } from "../components/charts";
import SampleExplorer from "../components/SampleExplorer";
import { useBottomTabs } from "../components/nav";
import { Card, Chip, ConfirmAction, CopyChip, EmptyLine, ErrorNote, HeroMetric, Loading, MetricTile, PageHeader, StatusPill } from "../components/ui";
import { dateTime, derivedMetrics, MODE_LABEL, ms, num, seconds, shortHash, useAsync, wilson } from "../lib";
import { useModelColor } from "../theme";

type Tab = "overview" | "latency" | "threshold" | "samples" | "config";

export default function RunDetailPage() {
  const { runId = "" } = useParams();
  const { data: run, error, loading } = useAsync(() => api.run(runId), [runId]);
  return (
    <div className="space-y-6">
      <PageHeader
        title="Run detail"
        subtitle="One evaluation; every headline number with its interval."
        right={
          <Link to="/runs" className="btn btn-secondary">
            ← All runs
          </Link>
        }
      />
      {loading && !run && <Loading />}
      {error && <ErrorNote message={error} />}
      {run && <RunView key={run.id} run={run} />}
    </div>
  );
}

function RunView({ run }: { run: RunDetail }) {
  const m = run.metrics;
  const mode = run.scoring_mode;
  const binary = mode === "binary" && run.positive_class !== null && m.tp !== undefined;
  const thresholdable = binary && run.has_probabilities;
  const colorOf = useModelColor();
  const navigate = useNavigate();
  const curve = useAsync(() => (thresholdable ? api.thresholdCurve(run.id) : Promise.resolve(undefined)), [run.id]);
  const analysis = useAsync(() => api.analysis(run.id), [run.id]);
  const [threshold, setThreshold] = useState<number>();
  const [params, setParams] = useSearchParams();
  const [deleteError, setDeleteError] = useState<string>();

  useEffect(() => {
    if (curve.data) setThreshold(curve.data.run_threshold);
  }, [curve.data]);

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "latency", label: "Latency" },
    ...(thresholdable ? [{ id: "threshold" as const, label: "Threshold" }] : []),
    { id: "samples", label: "Samples" },
    { id: "config", label: "Config" },
  ];
  const requested = params.get("tab") as Tab | null;
  const tab = tabs.some((t) => t.id === requested) ? (requested as Tab) : "overview";
  const selectTab = (id: Tab) => {
    const next = new URLSearchParams(params);
    if (id === "overview") next.delete("tab");
    else next.set("tab", id);
    setParams(next, { replace: true });
  };
  // the tabs sit in the floating bottom pill
  useBottomTabs(tabs, tab, selectTab);

  // Moving the threshold off the one the run used recomputes everything from the stored probabilities.
  const runThreshold = curve.data?.run_threshold;
  const point = curve.data && threshold !== undefined && threshold !== runThreshold ? curve.data.points[Math.round(threshold * 100)] : undefined;
  const n = m.n ?? 0;
  const counts = point
    ? { tp: point.tp, fp: point.fp, fn: point.fn, tn: point.tn, precision: point.precision, recall: point.recall }
    : { tp: m.tp ?? 0, fp: m.fp ?? 0, fn: m.fn ?? 0, tn: m.tn ?? 0, precision: m.precision, recall: m.recall };
  const derived = binary ? derivedMetrics(counts.tp, counts.fp, counts.fn, counts.tn) : undefined;

  let main: Stat | null = run.headline.main;
  let secondary: Stat | null = run.headline.secondary;
  if (point) {
    const [lo, hi] = wilson(point.tp + point.tn, n);
    main = { key: "accuracy", label: "Accuracy", value: point.accuracy, stderr: n > 1 ? Math.sqrt((point.accuracy * (1 - point.accuracy)) / (n - 1)) : 0, ci_low: lo, ci_high: hi };
    secondary = { key: "f1", label: "F1", value: point.f1, stderr: null, ci_low: null, ci_high: null };
  }

  const scorer = (run.config.scorer ?? {}) as Record<string, unknown>;
  const spec = (run.config.task_spec ?? {}) as Record<string, unknown>;
  const positiveName = String(scorer.positive_label ?? spec.positive_label ?? run.positive_class ?? "1");

  const remove = async () => {
    try {
      await api.deleteRun(run.id);
      navigate("/runs");
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <>
      {/* header */}
      <div className="glass rise flex flex-wrap items-start justify-between gap-4 px-6 py-5">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: colorOf(run.model) }} />
            <h2 className="truncate text-3xl font-semibold tracking-tight text-ink">{run.model}</h2>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Chip>
              <span className="text-faint">test</span> {run.task}
            </Chip>
            {run.example && <Chip tone="accent">Example</Chip>}
            <Chip>{MODE_LABEL[mode]}</Chip>
            <StatusPill status={run.status} />
            <Chip>
              <span className="num">{run.sample_counts.total}</span> samples
              {run.sample_counts.errors > 0 && <span className="text-bad">({run.sample_counts.errors} errored)</span>}
            </Chip>
            <CopyChip label="dataset" text={run.dataset_hash} display={shortHash(run.dataset_hash)} />
            <Chip>owl {run.owl_version}</Chip>
          </div>
          <p className="num mt-2 text-xs text-faint">
            run {run.id} · started {dateTime(run.started_at)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <a href={api.exportUrl(run.id)} download className="btn btn-secondary">
            Export CSV
          </a>
          <ConfirmAction label="Delete run" question="Delete this run and its samples?" onConfirm={remove} />
        </div>
      </div>
      {deleteError && <ErrorNote message={deleteError} />}

      {point && threshold !== undefined && (
        <div className="rise flex items-center justify-between rounded-[16px] border border-emerald/40 bg-emerald/12 px-4 py-2.5 text-sm text-ink">
          <span>
            Metrics recomputed at threshold <span className="num">{threshold.toFixed(2)}</span> (the run used <span className="num">{runThreshold?.toFixed(2)}</span>).
          </span>
          <button onClick={() => setThreshold(runThreshold)} className="btn btn-quiet h-8">
            Reset
          </button>
        </div>
      )}

      {/* hero row */}
      <div className="grid gap-5 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Card hero to={`/runs/${run.id}?tab=samples`} toLabel="Open the samples">
          <HeroMetric
            onHero
            stat={main}
            note={mode === "judge_scale" ? "Unparseable verdicts count as 1." : mode === "judge_passfail" ? "Unparseable verdicts count as fail." : "Unparseable answers count as wrong."}
          />
        </Card>
        {secondary && (
          <Card>
            <HeroMetric stat={secondary} size="md" note={point ? "Recomputed at the chosen threshold; no interval." : "95% interval by bootstrap (1,000 resamples)."} />
          </Card>
        )}
      </div>

      {/* secondary row: what applies to the scoring mode */}
      {binary && derived && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
          <MetricTile label="Precision" value={counts.precision} />
          <MetricTile label="Recall" value={counts.recall} sub="true positive rate" />
          <MetricTile label="Specificity" value={derived.specificity} sub="true negative rate" />
          <MetricTile label="Balanced acc." value={derived.balancedAccuracy} />
          <MetricTile label="MCC" value={derived.mcc} />
          <MetricTile label="ROC AUC" value={analysis.data?.roc?.auc ?? null} sub={analysis.data?.roc ? "any threshold" : "no probabilities"} />
          <MetricTile label="Unparseable" text={String(m.unparseable ?? 0)} />
        </div>
      )}
      {mode === "multiclass" && analysis.data?.multiclass && <PerClass analysis={analysis.data} unparseable={m.unparseable ?? 0} />}
      {mode.startsWith("judge") && analysis.data?.judge && (
        <Card title="Score distribution" right={<span className="text-xs text-faint">the judge's verdicts</span>}>
          <JudgeDistributionChart distribution={analysis.data.judge.distribution} />
        </Card>
      )}
      {mode === "exact_match" && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricTile label="Correct" text={String(Math.round((m.accuracy ?? 0) * n))} />
          <MetricTile label="Incorrect" text={String(n - Math.round((m.accuracy ?? 0) * n))} />
          <MetricTile label="Unparseable" text={String(m.unparseable ?? 0)} />
          <MetricTile label="Errors" text={String(run.sample_counts.errors)} />
        </div>
      )}

      {/* latency strip */}
      <div className="glass-dense rise flex flex-wrap items-center gap-x-10 gap-y-2 px-6 py-3.5">
        <span className="label">Latency</span>
        <Strip label="median" value={ms(m.latency_median_ms)} />
        <Strip label="p95" value={ms(m.latency_p95_ms)} />
        <Strip label="wall time" value={seconds(m.wall_time_s)} />
        <span className="text-xs text-faint">measured around the model call only, after warm-up</span>
      </div>

      {tab === "overview" && (
        <div className="space-y-5">
          {binary && (
            <Card title="Confusion matrix">
              <ConfusionMatrix
                rowLabels={[`${positiveName} (positive)`, "negative"]}
                columnLabels={[`${positiveName} (positive)`, "negative"]}
                matrix={[
                  [counts.tp, counts.fn],
                  [counts.fp, counts.tn],
                ]}
                caption={`TP ${counts.tp} · FN ${counts.fn} · FP ${counts.fp} · TN ${counts.tn}${!point && (m.unparseable ?? 0) > 0 ? ` · includes ${m.unparseable} unparseable counted as the wrong class` : ""}`}
              />
            </Card>
          )}
          {mode === "multiclass" && analysis.data?.multiclass && (
            <Card title="Confusion matrix">
              <ConfusionMatrix
                rowLabels={analysis.data.multiclass.labels}
                columnLabels={analysis.data.multiclass.matrix_columns}
                matrix={analysis.data.multiclass.matrix}
                caption="Rows are the true label, columns the prediction. The last column counts answers that named no label, or several."
              />
            </Card>
          )}
          <Card title="Accuracy by input length">
            {analysis.error && <ErrorNote message={analysis.error} />}
            {analysis.data &&
              (analysis.data.length_slices.length > 1 ? (
                <>
                  <LengthSliceChart slices={analysis.data.length_slices} />
                  <p className="mt-2 text-xs text-faint">Samples split into equal-sized groups by character count; whiskers are 95% Wilson intervals.</p>
                </>
              ) : (
                <EmptyLine>Too few samples to split by length (at least 16 are needed for two groups).</EmptyLine>
              ))}
          </Card>
        </div>
      )}

      {tab === "latency" && <LatencyTab run={run} analysis={analysis.data} />}

      {tab === "threshold" && thresholdable && (
        <div className="space-y-5">
          <Card title="Threshold analysis">
            {curve.error && <ErrorNote message={curve.error} />}
            {curve.data && threshold !== undefined && (
              <>
                <p className="mb-3 text-sm text-sub">
                  Precision, recall and F1 if the model answered positive whenever P(positive) ≥ threshold. Drag on the chart or use the slider; the numbers above update.
                </p>
                <ThresholdChart points={curve.data.points} threshold={threshold} onChange={setThreshold} />
              </>
            )}
          </Card>
          {analysis.data && (
            <>
              <div className="grid gap-5 xl:grid-cols-2">
                {analysis.data.roc && (
                  <Card title={`ROC curve · AUC ${analysis.data.roc.auc.toFixed(3)}`}>
                    <RocChart curves={[{ name: run.model, color: colorOf(run.model), curve: analysis.data.roc }]} marker={derived ? { x: derived.fpr, y: counts.recall ?? 0 } : undefined} />
                    <p className="mt-2 text-xs text-faint">The dot is the operating point at the current threshold; the dashed line is chance.</p>
                  </Card>
                )}
                {analysis.data.pr && (
                  <Card title={`Precision–recall · AP ${analysis.data.pr.average_precision.toFixed(3)}`}>
                    <PrChart curve={analysis.data.pr} marker={{ x: counts.recall ?? 0, y: counts.precision ?? 0 }} />
                    <p className="mt-2 text-xs text-faint">The dashed line is the share of positives: what a model with no skill scores.</p>
                  </Card>
                )}
              </div>
              <div className="grid gap-5 xl:grid-cols-2">
                {analysis.data.score_distribution && (
                  <Card title="Score distribution">
                    <ScoreDistributionChart bins={analysis.data.score_distribution} />
                  </Card>
                )}
                {analysis.data.calibration && (
                  <Card title="Calibration">
                    <CalibrationChart calibration={analysis.data.calibration} />
                    <p className="num mt-2 text-xs text-faint">
                      Brier {analysis.data.calibration.brier.toFixed(3)} · ECE {analysis.data.calibration.ece.toFixed(3)}. On the dashed line, probabilities match observed frequencies.
                    </p>
                  </Card>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {tab === "samples" && <SampleExplorer run={run} />}
      {tab === "config" && <ConfigTab run={run} />}
    </>
  );
}

function Strip({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-2">
      <span className="num text-lg font-semibold text-ink">{value}</span>
      <span className="text-xs text-faint">{label}</span>
    </span>
  );
}

function PerClass({ analysis, unparseable }: { analysis: Analysis; unparseable: number }) {
  const mc = analysis.multiclass!;
  return (
    <Card
      title="Per class"
      right={
        <span className="num text-xs text-faint">
          weighted-F1 {mc.weighted_f1.toFixed(3)} · {unparseable} unparseable
        </span>
      }
    >
      <div className="overflow-x-auto">
        <table className="strip-table">
          <thead>
            <tr>
              <th className="label px-4 py-2.5 text-left">Label</th>
              <th className="label px-4 py-2.5 text-right">Precision</th>
              <th className="label px-4 py-2.5 text-right">Recall</th>
              <th className="label px-4 py-2.5 text-right">F1</th>
              <th className="label px-4 py-2.5 text-right">Support</th>
            </tr>
          </thead>
          <tbody>
            {mc.per_class.map((c) => (
              <tr key={c.label} className="">
                <td className="px-4 py-2 text-[13px] text-ink">{c.label}</td>
                <td className="num px-4 py-2 text-right text-[13px] text-sub">{num(c.precision)}</td>
                <td className="num px-4 py-2 text-right text-[13px] text-sub">{num(c.recall)}</td>
                <td className="num px-4 py-2 text-right text-[13px] font-semibold text-ink">{num(c.f1)}</td>
                <td className="num px-4 py-2 text-right text-[13px] text-sub">{c.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function LatencyTab({ run, analysis }: { run: RunDetail; analysis?: Analysis }) {
  const byOutcome = analysis?.latency_by_outcome;
  return (
    <div className="space-y-5">
      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Latency distribution">
          <LatencyHistogram bins={run.latency_histogram} />
        </Card>
        <Card title="Latency per sample">{analysis && <LatencyScatter points={analysis.latency_points} />}</Card>
      </div>
      {byOutcome && (
        <Card title="Median latency by outcome">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricTile label="Correct" text={byOutcome.correct ? ms(byOutcome.correct.median_ms) : "–"} sub={`${byOutcome.correct?.n ?? 0} samples`} />
            <MetricTile label="Incorrect" text={byOutcome.incorrect ? ms(byOutcome.incorrect.median_ms) : "–"} sub={`${byOutcome.incorrect?.n ?? 0} samples`} />
          </div>
        </Card>
      )}
    </div>
  );
}

function ConfigTab({ run }: { run: RunDetail }) {
  const c = run.config;
  const prompt = (c.prompt ?? {}) as Record<string, unknown>;
  const system = (prompt.system as string | null | undefined) ?? c.system_prompt;
  const user = (prompt.user as string | undefined) ?? "{input}";
  const scorer = c.scorer ?? {};
  const judge = (scorer as Record<string, unknown>).judge as Record<string, unknown> | undefined;
  const rubric = (scorer as Record<string, unknown>).rubric_template as string | undefined;
  const scorerRest = Object.fromEntries(Object.entries(scorer).filter(([k]) => k !== "judge" && k !== "rubric_template"));
  const rows: [string, ReactNode][] = [
    ["Provider", run.provider],
    ["Model", run.model],
    ["Scoring mode", MODE_LABEL[run.scoring_mode]],
    ["Temperature", c.generate.temperature ?? "provider default"],
    ["Max tokens", c.generate.max_tokens ?? "provider default"],
    ["Seed", c.generate.seed ?? "not recorded (run made before seeds were stored)"],
    ["Sample limit", c.limit ?? "none"],
    ["Parallel requests", c.max_concurrency],
    ["Timeout", `${c.timeout_s} s`],
    ["Retries", c.retries],
    ["Dataset hash", run.dataset_hash],
    ["Owl version", run.owl_version],
  ];
  const json = (v: unknown) => JSON.stringify(v, null, 2);
  const block = "max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-[10px] border border-wash/10 bg-white/70 p-3 font-mono text-xs leading-relaxed text-sub";
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <Card title="Run">
        <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2 text-[13px]">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-faint">{k}</dt>
              <dd className="num break-all font-mono text-xs leading-5 text-ink">{String(v)}</dd>
            </div>
          ))}
        </dl>
      </Card>
      <Card title="Provider settings and model version">
        <div className="space-y-4">
          <div>
            <div className="label mb-1.5">Settings (no secrets are stored)</div>
            <pre className={block}>{json(c.model_args)}</pre>
          </div>
          <div>
            <div className="label mb-1.5">Version or digest reported by the provider</div>
            {c.model_info && Object.keys(c.model_info).length > 0 ? <pre className={block}>{json(c.model_info)}</pre> : <EmptyLine>Not recorded for this run.</EmptyLine>}
          </div>
        </div>
      </Card>
      <Card title="Prompt template">
        <div className="space-y-4">
          <div>
            <div className="label mb-1.5">System</div>
            {system ? <pre className={block}>{system}</pre> : <EmptyLine>No system message.</EmptyLine>}
          </div>
          <div>
            <div className="label mb-1.5">User</div>
            <pre className={block}>{user}</pre>
          </div>
        </div>
      </Card>
      <Card title="Scorer">
        <div className="space-y-4">
          <pre className={block}>{Object.keys(scorerRest).length ? json(scorerRest) : json({ positive_class: c.positive_class })}</pre>
          {judge && (
            <div>
              <div className="label mb-1.5">Judge</div>
              <pre className={block}>{json(judge)}</pre>
            </div>
          )}
          {rubric && (
            <div>
              <div className="label mb-1.5">Rubric template</div>
              <pre className={block}>{rubric}</pre>
            </div>
          )}
        </div>
      </Card>
      {c.task_spec && (
        <Card title="Test definition" className="xl:col-span-2">
          <pre className={block}>{json(c.task_spec)}</pre>
        </Card>
      )}
    </div>
  );
}
