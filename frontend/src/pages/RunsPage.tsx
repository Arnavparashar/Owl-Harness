import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api, isActive, type RunSummary } from "../api";
import { AXIS, GRID } from "../components/charts";
import { Card, ErrorNote, Loading, MetricCard, StatusBadge, td, th } from "../components/ui";
import { dateTime, ms, num, runLabels, SERIES_COLORS, useAsync, usePolling } from "../lib";

type SortKey = "task" | "model" | "n" | "accuracy" | "f1" | "latency" | "status" | "started_at";

const sortValue = (r: RunSummary, key: SortKey): string | number => {
  switch (key) {
    case "task":
      return r.task;
    case "model":
      return r.model;
    case "n":
      return r.metrics.n ?? -1;
    case "accuracy":
      return r.metrics.accuracy ?? -1;
    case "f1":
      return r.metrics.f1 ?? -1;
    case "latency":
      return r.metrics.latency_median_ms ?? -1;
    case "status":
      return r.status;
    case "started_at":
      return r.started_at;
  }
};

export default function RunsPage() {
  const [refresh, setRefresh] = useState(0);
  const { data, error, loading } = useAsync(() => api.runs(), [refresh]);
  const jobs = useAsync(() => api.jobs(), [refresh]);
  const activeJobs = (jobs.data ?? []).filter(isActive);
  const [sortKey, setSortKey] = useState<SortKey>("started_at");
  const [desc, setDesc] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [task, setTask] = useState("");
  const [search, setSearch] = useState("");
  const navigate = useNavigate();

  // Runs that are still in progress change under us, so refresh the list while any exist.
  usePolling((data?.some((r) => r.status === "running") ?? false) || activeJobs.length > 0, () => setRefresh((n) => n + 1));

  const tasks = useMemo(() => [...new Set((data ?? []).map((r) => r.task))].sort(), [data]);

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = (data ?? []).filter((r) => (!task || r.task === task) && (!q || r.model.toLowerCase().includes(q)));
    list.sort((a, b) => {
      const x = sortValue(a, sortKey);
      const y = sortValue(b, sortKey);
      const c = x < y ? -1 : x > y ? 1 : 0;
      return desc ? -c : c;
    });
    return list;
  }, [data, sortKey, desc, task, search]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setDesc(!desc);
    else {
      setSortKey(key);
      setDesc(key !== "task" && key !== "model" && key !== "status");
    }
  };

  const toggleSelected = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const header = (key: SortKey, label: string, align = "text-left") => (
    <th className={`${th} ${align} cursor-pointer select-none hover:text-slate-900`} onClick={() => toggleSort(key)}>
      {label}
      {sortKey === key && <span className="ml-1 text-accent">{desc ? "↓" : "↑"}</span>}
    </th>
  );

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <h1 className="text-lg font-semibold text-slate-900">Runs</h1>
        <button
          disabled={selected.length < 2}
          onClick={() => navigate(`/compare?runs=${selected.join(",")}`)}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Compare selected{selected.length > 0 ? ` (${selected.length})` : ""}
        </button>
      </div>

      {activeJobs.map((job) => (
        <Link
          key={job.id}
          to={`/jobs/${job.id}`}
          className="flex items-center justify-between rounded border border-accent/30 bg-accent-soft px-4 py-2 text-sm text-slate-800 hover:bg-[#d5e1fa]"
        >
          <span>
            <span className="font-medium">{job.task}</span> is {job.status === "queued" ? "waiting to start" : "running"} on{" "}
            {job.items.length} model{job.items.length === 1 ? "" : "s"}
            {job.items.some((i) => i.status === "running" && i.total > 0) &&
              ` · ${job.items.find((i) => i.status === "running")!.done} of ${job.items.find((i) => i.status === "running")!.total} samples`}
          </span>
          <span className="font-medium text-accent">Watch progress →</span>
        </Link>
      ))}

      {loading && !data && <Loading />}
      {error && <ErrorNote message={error} />}
      {data && data.length === 0 && (
        <Card>
          <div className="space-y-2 py-4 text-sm text-slate-700">
            <p className="font-medium text-slate-900">No runs yet.</p>
            <p>
              <Link to="/new" className="text-accent hover:underline">
                Start a new run
              </Link>{" "}
              to test a model on a built-in test or on your own dataset.
            </p>
          </div>
        </Card>
      )}
      {data && data.length > 0 && (
        <>
          <Overview runs={rows} />
          {rows.length > 0 && <OverviewCharts runs={rows} />}

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-slate-600">
              Task
              <select
                value={task}
                onChange={(e) => setTask(e.target.value)}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-sm text-slate-900"
              >
                <option value="">All tasks</option>
                {tasks.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter by model"
              aria-label="Filter by model"
              className="w-56 rounded border border-slate-300 bg-white px-2 py-1 text-sm"
            />
            <span className="text-sm text-slate-500">
              {rows.length} of {data.length} runs
            </span>
            {selected.length > 0 && (
              <button onClick={() => setSelected([])} className="text-sm text-accent hover:underline">
                Clear selection
              </button>
            )}
          </div>

          <Card>
            <div className="-m-4 overflow-x-auto">
              <table className="w-full border-collapse">
                <thead className="border-b border-slate-200 bg-slate-50">
                  <tr>
                    <th className={`${th} w-8`} />
                    {header("task", "Task")}
                    {header("model", "Model")}
                    {header("n", "n", "text-right")}
                    {header("accuracy", "Accuracy", "text-right")}
                    {header("f1", "F1", "text-right")}
                    {header("latency", "Median latency", "text-right")}
                    {header("status", "Status")}
                    {header("started_at", "Started")}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {rows.map((r) => (
                    <tr key={r.id} className={selected.includes(r.id) ? "bg-accent-soft/50" : "hover:bg-slate-50"}>
                      <td className={td}>
                        <input
                          type="checkbox"
                          aria-label={`Select ${r.model} run ${r.id}`}
                          checked={selected.includes(r.id)}
                          onChange={() => toggleSelected(r.id)}
                          className="accent-[#1d4ed8]"
                        />
                      </td>
                      <td className={td}>{r.task}</td>
                      <td className={td}>
                        <Link to={`/runs/${r.id}`} className="font-medium text-accent hover:underline">
                          {r.model}
                        </Link>
                      </td>
                      <td className={`${td} text-right tabular-nums`}>{r.metrics.n?.toFixed(0) ?? "–"}</td>
                      <td className={`${td} text-right tabular-nums`}>{num(r.metrics.accuracy)}</td>
                      <td className={`${td} text-right tabular-nums`}>{num(r.metrics.f1)}</td>
                      <td className={`${td} text-right tabular-nums`}>{ms(r.metrics.latency_median_ms)}</td>
                      <td className={td}>
                        <StatusBadge status={r.status} />
                      </td>
                      <td className={`${td} whitespace-nowrap text-slate-600`}>{dateTime(r.started_at)}</td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={9} className="px-3 py-6 text-center text-sm text-slate-500">
                        No runs match the current filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Overview({ runs }: { runs: RunSummary[] }) {
  const scored = runs.filter((r) => r.metrics.accuracy !== undefined);
  const best = scored.reduce<RunSummary | undefined>((a, r) => (a === undefined || r.metrics.accuracy > a.metrics.accuracy ? r : a), undefined);
  const samples = runs.reduce((sum, r) => sum + (r.metrics.n ?? 0), 0);
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <MetricCard label="Runs" value={String(runs.length)} sub={`${runs.filter((r) => r.status === "running").length} in progress`} />
      <MetricCard label="Models" value={String(new Set(runs.map((r) => r.model)).size)} sub={`${new Set(runs.map((r) => r.task)).size} task(s)`} />
      <MetricCard label="Samples scored" value={samples.toLocaleString()} />
      <MetricCard label="Highest accuracy" value={best ? num(best.metrics.accuracy) : "–"} sub={best?.model} />
    </div>
  );
}

function OverviewCharts({ runs }: { runs: RunSummary[] }) {
  const labels = runLabels(runs);
  const scored = runs.filter((r) => r.metrics.accuracy !== undefined).sort((a, b) => b.metrics.accuracy - a.metrics.accuracy);
  if (scored.length === 0) return null;

  const bars = scored.map((r) => {
    const [lo, hi] = r.accuracy_ci ?? [r.metrics.accuracy, r.metrics.accuracy];
    return { label: labels[r.id], accuracy: r.metrics.accuracy, err: [r.metrics.accuracy - lo, hi - r.metrics.accuracy] as [number, number] };
  });
  const points = scored
    .filter((r) => r.metrics.latency_median_ms !== undefined)
    .map((r) => ({ label: labels[r.id], latency: r.metrics.latency_median_ms, accuracy: r.metrics.accuracy }));
  const mixedTasks = new Set(runs.map((r) => r.task)).size > 1;

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card title="Accuracy by run">
        <ResponsiveContainer width="100%" height={Math.max(240, bars.length * 44 + 40)}>
          <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} />
            <YAxis type="category" dataKey="label" {...AXIS} width={150} />
            <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} cursor={{ fill: "#f1f5f9" }} />
            <Bar dataKey="accuracy" fill={SERIES_COLORS[0]} isAnimationActive={false} barSize={20}>
              <ErrorBar dataKey="err" direction="x" width={5} stroke="#0f172a" strokeWidth={1.5} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-1 text-xs text-slate-500">
          Error bars are 95% Wilson intervals. {mixedTasks ? "Runs span several tasks, so filter to one task before comparing them." : ""}
        </p>
      </Card>

      <Card title="Accuracy against median latency">
        <ResponsiveContainer width="100%" height={Math.max(240, bars.length * 44 + 40)}>
          <ScatterChart margin={{ top: 12, right: 40, bottom: 22, left: 4 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis
              type="number"
              dataKey="latency"
              name="median latency"
              unit=" ms"
              domain={[0, "auto"]}
              {...AXIS}
              label={{ value: "median latency (ms)", position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 12 }}
            />
            <YAxis type="number" dataKey="accuracy" name="accuracy" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} width={36} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => (typeof v === "number" ? v.toFixed(v < 5 ? 3 : 0) : String(v))} />
            <Scatter data={points} fill={SERIES_COLORS[0]} isAnimationActive={false}>
              <LabelList dataKey="label" position="top" fontSize={11} fill="#334155" />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <p className="mt-1 text-xs text-slate-500">
          Up and to the left is better. Latency is measured on this machine and is not like-for-like across CPU and GPU models.
        </p>
      </Card>
    </div>
  );
}
