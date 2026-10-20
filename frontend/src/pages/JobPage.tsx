import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, isActive, type Job, type JobItem } from "../api";
import { btnSecondary, Card, ErrorNote, Loading, ProgressBar } from "../components/ui";
import { dateTime, num, pct, usePolling, useAsync } from "../lib";

const ITEM_LABEL: Record<JobItem["status"], string> = {
  queued: "waiting",
  running: "running",
  completed: "done",
  completed_with_errors: "done, some samples failed",
  failed: "failed",
  cancelled: "cancelled",
};

const JOB_LABEL: Record<Job["status"], string> = {
  queued: "Waiting for an earlier job to finish",
  running: "Running",
  completed: "Finished",
  failed: "Finished with a failure",
  cancelled: "Cancelled",
};

export default function JobPage() {
  const { jobId = "" } = useParams();
  const [tick, setTick] = useState(0);
  const { data: job, error, loading } = useAsync(() => api.job(jobId), [jobId, tick]);
  const [cancelError, setCancelError] = useState<string>();

  usePolling(job ? isActive(job) : false, () => setTick((n) => n + 1), 1000);

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
    <div className="mx-auto max-w-4xl space-y-5">
      <Link to="/" className="text-sm text-accent hover:underline">
        ← All runs
      </Link>
      {loading && !job && <Loading />}
      {error && !job && <ErrorNote message={error} />}
      {job && (
        <>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="text-lg font-semibold text-slate-900">{job.task}</h1>
              <p className="mt-1 text-sm text-slate-600">
                {JOB_LABEL[job.status]}
                {job.cancel_requested && isActive(job) ? ", cancelling…" : ""} · started {dateTime(job.created_at)}
                {job.limit ? ` · first ${job.limit} samples` : ""}
              </p>
              {job.description && <p className="mt-0.5 text-xs text-slate-500">{job.description}</p>}
            </div>
            {isActive(job) && (
              <button className={btnSecondary} onClick={cancel} disabled={job.cancel_requested}>
                {job.cancel_requested ? "Cancelling…" : "Cancel"}
              </button>
            )}
          </div>
          {cancelError && <ErrorNote message={cancelError} />}

          <div className="space-y-3">
            {job.items.map((item, i) => (
              <ItemCard key={`${item.model}-${i}`} item={item} />
            ))}
          </div>

          {!isActive(job) && (
            <div className="flex flex-wrap items-center gap-3">
              {finished.length >= 2 && (
                <Link
                  to={`/compare?runs=${finished.map((i) => i.run_id).join(",")}`}
                  className="rounded bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-[#1e40af]"
                >
                  Compare these {finished.length} runs
                </Link>
              )}
              <Link to="/new" className={btnSecondary}>
                Start another run
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ItemCard({ item }: { item: JobItem }) {
  const pending = item.status === "queued";
  const answered = item.done - item.errors;
  return (
    <Card>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-3">
          <h2 className="text-sm font-semibold text-slate-900">{item.model}</h2>
          <span className={`text-sm ${item.status === "failed" ? "text-danger" : item.status === "running" ? "text-accent" : "text-slate-600"}`}>
            {ITEM_LABEL[item.status]}
          </span>
        </div>
        {item.run_id && (
          <Link to={`/runs/${item.run_id}`} className="text-sm font-medium text-accent hover:underline">
            {item.status.startsWith("completed") ? "View results →" : "View partial results →"}
          </Link>
        )}
      </div>

      {!pending && (
        <div className="mt-3 space-y-1.5">
          {item.total > 0 ? (
            <>
              <ProgressBar value={item.done} max={item.total} />
              <div className="flex flex-wrap justify-between gap-2 text-xs tabular-nums text-slate-600">
                <span>
                  {item.done} of {item.total} samples
                </span>
                <span>
                  {answered > 0 ? `${item.correct} of ${answered} correct so far (${pct(item.correct / answered, 0)})` : "no answers yet"}
                  {item.errors > 0 && <span className="text-danger"> · {item.errors} failed</span>}
                </span>
              </div>
            </>
          ) : (
            item.status === "running" && <p className="text-xs text-slate-500">Loading the model and warming up…</p>
          )}
          {item.accuracy !== null && (
            <p className="text-sm text-slate-800">
              Final accuracy <span className="font-semibold tabular-nums">{num(item.accuracy)}</span>
            </p>
          )}
        </div>
      )}
      {item.error && <p className="mt-2 break-words rounded border border-danger/30 px-3 py-2 text-sm text-danger">{item.error}</p>}
    </Card>
  );
}
