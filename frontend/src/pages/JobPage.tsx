import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, isActive, type JobItem } from "../api";
import { Card, Chip, ErrorNote, Loading, PageHeader, ProgressBar, StatusPill } from "../components/ui";
import { dateTime, duration, MODE_LABEL, pct, usePolling, useAsync } from "../lib";
import { useModelColor } from "../theme";

export default function JobPage() {
  const { jobId = "" } = useParams();
  const [tick, setTick] = useState(0);
  const { data: job, error, loading } = useAsync(() => api.job(jobId), [jobId, tick]);
  const [cancelError, setCancelError] = useState<string>();
  const [now, setNow] = useState(Date.now());

  usePolling(job ? isActive(job) : false, () => setTick((n) => n + 1), 1000);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const cancel = async () => {
    setCancelError(undefined);
    try {
      await api.cancelJob(jobId);
      setTick((n) => n + 1);
    } catch (e) {
      setCancelError(e instanceof Error ? e.message : String(e));
    }
  };

  const finished = job?.items.filter((i) => i.run_id && i.status.startsWith("completed")) ?? [];

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader title="Progress" subtitle="Models run one after another; numbers update live." />
      {loading && !job && <Loading />}
      {error && !job && <ErrorNote message={error} />}
      {job && (
        <>
          <div className="glass rise flex flex-wrap items-start justify-between gap-4 px-6 py-5">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-2xl font-semibold tracking-tight text-ink">{job.task}</h2>
                <StatusPill status={job.status} />
                <Chip>{MODE_LABEL[job.scoring_mode]}</Chip>
              </div>
              <p className="mt-2 text-sm text-sub">
                started {dateTime(job.created_at)}
                {job.limit ? ` · first ${job.limit} samples` : ""} · {job.items.length} model{job.items.length === 1 ? "" : "s"}, one after another
                {job.cancel_requested && isActive(job) ? " · cancelling…" : ""}
              </p>
              {job.description && <p className="mt-1 text-xs text-faint">{job.description}</p>}
            </div>
            {isActive(job) && (
              <button className="btn btn-secondary" onClick={cancel} disabled={job.cancel_requested}>
                {job.cancel_requested ? "Cancelling…" : "Cancel"}
              </button>
            )}
          </div>
          {cancelError && <ErrorNote message={cancelError} />}
          {job.warnings.map((w) => (
            <div key={w} className="rounded-[12px] border border-warn/40 bg-warn/10 px-4 py-2.5 text-sm text-warn">
              {w}
            </div>
          ))}

          <div className="space-y-3">
            {job.items.map((item, i) => (
              <ItemCard key={`${item.model}-${i}`} item={item} now={now} judged={job.scoring_mode.startsWith("judge")} />
            ))}
          </div>

          {!isActive(job) && (
            <div className="flex flex-wrap items-center gap-3">
              {finished.length >= 2 && (
                <Link to={`/compare?runs=${finished.map((i) => i.run_id).join(",")}`} className="btn btn-primary">
                  Compare these {finished.length} runs
                </Link>
              )}
              <Link to="/new" className="btn btn-secondary">
                Start another run
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ItemCard({ item, now, judged }: { item: JobItem; now: number; judged: boolean }) {
  const colorOf = useModelColor();
  const running = item.status === "running";
  const answered = item.done - item.errors;
  // ETA from the pace so far; the warm-up before the first sample is not part of it.
  const started = item.started_at ? new Date(item.started_at).getTime() : undefined;
  const eta = running && started && item.done > 0 && item.total > item.done ? ((now - started) / 1000 / item.done) * (item.total - item.done) : undefined;

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="h-3 w-3 rounded-full" style={{ background: colorOf(item.model) }} />
          <h3 className="text-[15px] font-semibold text-ink">{item.model}</h3>
          <StatusPill status={item.status} />
        </div>
        {item.run_id && (
          <Link to={`/runs/${item.run_id}`} className="text-sm font-medium text-accent hover:underline">
            {item.status.startsWith("completed") ? "View results →" : "View partial results →"}
          </Link>
        )}
      </div>

      {item.status !== "queued" && (
        <div className="mt-4 space-y-2.5">
          {item.total > 0 ? (
            <>
              <ProgressBar value={item.done} max={item.total} running={running} />
              <div className="num flex flex-wrap justify-between gap-2 text-[13px] text-sub">
                <span>
                  <span className="text-ink">{item.done}</span> of {item.total} samples
                  {eta !== undefined && <span className="text-faint"> · about {duration(eta)} left</span>}
                </span>
                <span>
                  {answered > 0 ? (
                    <>
                      <span className="text-ink">{item.correct}</span> {judged ? "passed" : "correct"} of {answered} · current {judged ? "pass rate" : "accuracy"}{" "}
                      <span className="font-semibold text-ink">{pct(item.correct / answered)}</span>
                    </>
                  ) : (
                    "no answers yet"
                  )}
                  {item.errors > 0 && <span className="text-bad"> · {item.errors} failed</span>}
                </span>
              </div>
            </>
          ) : (
            running && <p className="text-[13px] text-faint">Loading the model and warming up. This is not counted in latency.</p>
          )}
          {item.main_metric && (
            <p className="text-sm text-sub">
              Final {item.main_metric.name}{" "}
              <span className="num text-lg font-bold text-accent">{item.main_metric.name === "mean score" ? item.main_metric.value.toFixed(2) : item.main_metric.value.toFixed(3)}</span>
            </p>
          )}
        </div>
      )}
      {item.status === "queued" && <p className="mt-3 text-[13px] text-faint">Waiting for the models before it.</p>}
      {item.error && <p className="mt-3 break-words rounded-[10px] border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{item.error}</p>}
    </Card>
  );
}
