// Jobs that are running right now, shown live at the top of Overview and Runs.
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, isActive } from "../api";
import { usePolling, useAsync } from "../lib";
import { ProgressBar } from "./ui";

export default function JobsBanner({ onChange }: { onChange?: () => void }) {
  const [tick, setTick] = useState(0);
  const jobs = useAsync(() => api.jobs(), [tick]);
  const active = (jobs.data ?? []).filter(isActive);
  usePolling(active.length > 0, () => {
    setTick((n) => n + 1);
    onChange?.();
  }, 1500);
  if (active.length === 0) return null;

  return (
    <div className="space-y-2">
      {active.map((job) => {
        const item = job.items.find((i) => i.status === "running");
        const done = job.items.filter((i) => i.status.startsWith("completed")).length;
        return (
          <Link key={job.id} to={`/jobs/${job.id}`} className="glass rise block px-5 py-3 transition-colors hover:bg-white/[0.08]">
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <span className="text-ink">
                <span className="font-semibold">{job.task}</span>
                <span className="text-sub">
                  {" "}
                  · {job.status === "queued" ? "waiting to start" : item ? `running ${item.model}` : "starting"} · model {Math.min(done + 1, job.items.length)} of{" "}
                  {job.items.length}
                </span>
              </span>
              <span className="font-medium text-accent">Watch progress →</span>
            </div>
            {item && item.total > 0 && (
              <div className="mt-2 flex items-center gap-3">
                <ProgressBar value={item.done} max={item.total} running />
                <span className="num w-24 shrink-0 text-right text-xs text-sub">
                  {item.done} / {item.total}
                </span>
              </div>
            )}
          </Link>
        );
      })}
    </div>
  );
}
