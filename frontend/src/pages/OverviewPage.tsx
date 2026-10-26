import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, isActive, type Job, type RunSummary } from "../api";
import { useChartMotion } from "../components/charts";
import {
  Card,
  CornerLink,
  CountUp,
  DeltaPill,
  EmptyLine,
  ErrorNote,
  FilterPill,
  KpiCard,
  Loading,
  PageHeader,
  PERIODS,
  PeriodSelector,
  ProgressBar,
  SearchField,
  StatusPill,
  type Period,
} from "../components/ui";
import { VirtualTable, type Column } from "../components/VirtualTable";
import { MODE_LABEL, statText, usePolling, useAsync } from "../lib";
import { AXIS, GRID, MODEL_PALETTE, TOOLTIP, useModelColor } from "../theme";

const DAY = 24 * 60 * 60 * 1000;
const done = (r: RunSummary) => r.status.startsWith("completed") && r.headline.main !== null;

/** The items in [from, to), by their start time. */
const within = <T,>(items: T[], at: (item: T) => string, from: number, to: number) =>
  items.filter((x) => {
    const t = new Date(at(x)).getTime();
    return t >= from && t < to;
  });

/** Relative change, or null when there is nothing to compare with. */
const change = (now: number | null, before: number | null) => (now === null || before === null || before === 0 ? null : (now - before) / before);

/** The model name without its provider prefix, cut to fit under a bar. */
const shortName = (model: string) => {
  const name = model.includes("/") ? model.slice(model.indexOf("/") + 1) : model;
  return name.length > 11 ? `${name.slice(0, 10)}…` : name;
};

const greeting = () => {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
};

const rowDate = (iso: string) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export default function OverviewPage() {
  const [period, setPeriod] = useState<Period>("30d");
  const [tick, setTick] = useState(0);
  const { data, error, loading } = useAsync(() => api.runs(), [tick]);
  const jobs = useAsync(() => api.jobs(), [tick]);
  const running = (jobs.data ?? []).filter(isActive);
  usePolling(running.length > 0, () => setTick((n) => n + 1), 1500);

  const now = Date.now();
  const days = PERIODS.find((p) => p.id === period)!.days;
  const all = data ?? [];
  const runAt = (r: RunSummary) => r.started_at;
  const current = days === null ? all : within(all, runAt, now - days * DAY, now + DAY);
  const previous = days === null ? null : within(all, runAt, now - 2 * days * DAY, now - days * DAY);
  const allJobs = jobs.data ?? [];
  const jobAt = (j: Job) => j.created_at;
  const jobsCurrent = days === null ? allJobs : within(allJobs, jobAt, now - days * DAY, now + DAY);
  const jobsPrevious = days === null ? null : within(allJobs, jobAt, now - 2 * days * DAY, now - days * DAY);

  const best = (runs: RunSummary[]) =>
    runs.filter((r) => done(r) && r.headline.main!.key !== "mean_score").reduce<RunSummary | null>((a, r) => (!a || r.headline.main!.value > a.headline.main!.value ? r : a), null);
  const samples = (runs: RunSummary[]) => runs.reduce((s, r) => s + (r.metrics.n ?? 0), 0);
  const models = (runs: RunSummary[]) => new Set(runs.map((r) => r.model)).size;
  const top = best(current);
  const topBefore = previous ? best(previous) : null;
  const topChange = previous ? change(top?.headline.main!.value ?? null, topBefore?.headline.main!.value ?? null) : null;

  return (
    <div className="space-y-6">
      <PageHeader title={greeting()} subtitle="Here's what's happening across your evaluations." right={<PeriodSelector value={period} onChange={setPeriod} />} />
      {loading && !data && <Loading />}
      {error && <ErrorNote message={error} />}
      {data && (
        <>
          {/* row 1: the best result, four small cards, and the models chart */}
          <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_minmax(0,1.05fr)]">
            <motion.section className="glass hoverable rise flex min-h-[300px] flex-col justify-between p-7" whileHover={{ y: -4 }} transition={{ type: "spring", stiffness: 380, damping: 30 }}>
              <div className="flex items-start justify-between gap-2">
                <span className="text-[13px] text-faint">{top ? `Best ${top.headline.main!.label.toLowerCase()}` : "Best result"}</span>
                {top && <CornerLink to={`/runs/${top.id}`} label="Open this run" />}
              </div>
              <div>
                <div className="num text-[76px] font-bold leading-none tracking-tight text-ink">
                  {top ? <CountUp value={top.headline.main!.value} format={(v) => (top.headline.main!.key === "mean_score" ? v.toFixed(2) : v.toFixed(3))} /> : "–"}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
                  <span className="min-w-0 truncate text-[13px] text-sub">
                    {top ? `${top.model} · ${top.task} · n=${top.metrics.n?.toFixed(0) ?? "?"}` : "no finished runs in this period"}
                  </span>
                  {topChange !== null && Number.isFinite(topChange) && <DeltaPill change={topChange} />}
                </div>
              </div>
              <div className="flex flex-wrap gap-2.5">
                <Link to="/new" className="btn btn-secondary">
                  New run
                </Link>
                <Link to="/compare" className="btn btn-secondary">
                  Compare
                </Link>
              </div>
            </motion.section>

            <div className="grid grid-cols-2 gap-5">
              <KpiCard
                hero
                label="Runs"
                value={current.length}
                format={(v) => Math.round(v).toLocaleString()}
                caption="started in this period"
                change={previous ? change(current.length, previous.length) : undefined}
                to="/runs"
                toLabel="All runs"
              />
              <KpiCard
                label="Samples scored"
                value={samples(current)}
                format={(v) => Math.round(v).toLocaleString()}
                caption="across all runs"
                change={previous ? change(samples(current), samples(previous)) : undefined}
                to="/runs"
                toLabel="All runs"
              />
              <KpiCard
                label="Models evaluated"
                value={models(current)}
                format={(v) => Math.round(v).toLocaleString()}
                caption="distinct models"
                change={previous ? change(models(current), models(previous)) : undefined}
                to="/models"
                toLabel="Models"
              />
              <KpiCard
                label="Running jobs"
                value={running.length}
                format={(v) => Math.round(v).toLocaleString()}
                caption={running.length === 0 ? "nothing is running" : "in progress now"}
                change={jobsPrevious ? change(jobsCurrent.length, jobsPrevious.length) : undefined}
                to={running[0] ? `/jobs/${running[0].id}` : undefined}
                toLabel="Watch progress"
              />
            </div>

            <div className="lg:col-span-2 xl:col-span-1">
              <ModelsOnTest runs={current.filter(done)} />
            </div>
          </div>

          {/* row 2: current job, recent runs and the scoring-mode donut */}
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,3.4fr)]">
            <CurrentJob jobs={running} />
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1.9fr)_minmax(0,1fr)]">
              <RecentRuns runs={current} />
              <ByMode runs={current} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** Each model's main metric on one test (its best among its largest runs), with 95% intervals. */
function ModelsOnTest({ runs }: { runs: RunSummary[] }) {
  const colorOf = useModelColor();
  const motion = useChartMotion();
  const tests = useMemo(() => {
    const latest = new Map<string, string>();
    for (const r of runs) if (!latest.has(r.task) || r.started_at > latest.get(r.task)!) latest.set(r.task, r.started_at);
    return [...latest.entries()].sort((a, b) => b[1].localeCompare(a[1])).map(([t]) => t);
  }, [runs]);
  const [chosen, setChosen] = useState<string>();
  const test = chosen && tests.includes(chosen) ? chosen : tests[0];

  const rows = useMemo(() => {
    const best = new Map<string, RunSummary>();
    for (const r of runs.filter((x) => x.task === test)) {
      const cur = best.get(r.model);
      const n = r.metrics.n ?? 0;
      const curN = cur?.metrics.n ?? 0;
      if (!cur || n > curN || (n === curN && r.headline.main!.value > cur.headline.main!.value)) best.set(r.model, r);
    }
    return [...best.values()].sort((a, b) => b.headline.main!.value - a.headline.main!.value);
  }, [runs, test]);

  const main = rows[0]?.headline.main;
  const scale = main?.key === "mean_score";
  const data = rows.map((r) => {
    const s = r.headline.main!;
    return {
      model: r.model,
      short: shortName(r.model),
      value: s.value,
      n: r.metrics.n ?? 0,
      err: [s.ci_low === null ? 0 : s.value - s.ci_low, s.ci_high === null ? 0 : s.ci_high - s.value] as [number, number],
      color: colorOf(r.model),
    };
  });

  return (
    <Card
      className="h-full"
      title={main ? `${main.label} by model` : "Models on a test"}
      right={
        tests.length > 0 && (
          <select className="input h-9 w-44 rounded-full text-xs" value={test} onChange={(e) => setChosen(e.target.value)} aria-label="Test">
            {tests.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        )
      }
      to={rows.length >= 2 ? `/compare?runs=${rows.map((r) => r.id).join(",")}` : undefined}
      toLabel="Compare these runs"
    >
      {rows.length === 0 ? (
        <EmptyLine>No finished runs in this period.</EmptyLine>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={data} margin={{ top: 12, right: 8, bottom: 4, left: -8 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="short" {...AXIS} interval={0} />
              <YAxis domain={scale ? [1, 5] : [0, 1]} {...AXIS} width={44} />
              <Tooltip
                {...TOOLTIP}
                formatter={(v, _n, p) => [`${Number(v).toFixed(scale ? 2 : 3)}  (n = ${(p.payload as { n: number }).n})`, (p.payload as { model: string }).model]}
                labelFormatter={() => ""}
              />
              <Bar dataKey="value" radius={[12, 12, 0, 0]} maxBarSize={48} {...motion}>
                {data.map((d) => (
                  <Cell key={d.model} fill={d.color} />
                ))}
                <ErrorBar dataKey="err" width={8} stroke="#7c7366" strokeWidth={2} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-1 text-xs text-faint">Each model's best run among its largest runs on this test. Whiskers are 95% intervals.</p>
        </>
      )}
    </Card>
  );
}

/** The job in progress as a slim emerald bar, or one muted line when nothing runs. */
function CurrentJob({ jobs }: { jobs: Job[] }) {
  if (jobs.length === 0) {
    return (
      <section className="glass rise flex items-center self-start p-6">
        <p className="text-sm text-faint">No job is running.</p>
      </section>
    );
  }
  return (
    <Card title="Current job" to={`/jobs/${jobs[0].id}`} toLabel="Watch progress" className="self-start">
      <div className="space-y-5">
        {jobs.map((job) => {
          const item = job.items.find((i) => i.status === "running");
          return (
            <Link key={job.id} to={`/jobs/${job.id}`} className="block">
              <div className="flex justify-between gap-2 text-[13px]">
                <span className="truncate font-semibold text-ink">{job.task}</span>
                <span className="num shrink-0 text-faint">{item ? `${item.done} / ${item.total || "…"}` : "waiting"}</span>
              </div>
              <div className="mb-2.5 mt-0.5 truncate text-xs text-faint">{item ? item.model : `${job.items.length} models queued`}</div>
              <div className="[&>div]:h-1.5">
                <ProgressBar value={item?.done ?? 0} max={item?.total ?? 1} running={!!item} />
              </div>
            </Link>
          );
        })}
      </div>
    </Card>
  );
}

const STATUS_FILTERS = [
  { value: "", label: "All statuses" },
  { value: "completed", label: "Completed" },
  { value: "completed_with_errors", label: "Completed, some errors" },
  { value: "pending", label: "Running or waiting" },
  { value: "failed", label: "Failed or interrupted" },
];

const statusMatches = (status: string, filter: string) =>
  filter === "" ||
  (filter === "pending" ? status === "running" || status === "queued" : filter === "failed" ? status === "failed" || status === "interrupted" : status === filter);

function RecentRuns({ runs }: { runs: RunSummary[] }) {
  const colorOf = useModelColor();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return runs.filter((r) => statusMatches(r.status, status) && (!q || `${r.model} ${r.task}`.toLowerCase().includes(q)));
  }, [runs, search, status]);

  const columns: Column<RunSummary>[] = [
    {
      key: "model",
      header: "Model",
      render: (r) => (
        <Link to={`/runs/${r.id}`} onClick={(e) => e.stopPropagation()} className="flex min-w-0 items-center gap-2 font-medium text-ink hover:text-accent">
          <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colorOf(r.model) }} />
          <span className="truncate">{r.model}</span>
        </Link>
      ),
    },
    { key: "task", header: "Test", render: (r) => <span className="truncate text-sub">{r.task}</span> },
    { key: "main", header: "Metric", align: "right", render: (r) => <span className="num font-bold">{statText(r.headline.main)}</span> },
    { key: "status", header: "Status", render: (r) => <StatusPill status={r.status} /> },
    { key: "date", header: "Date", render: (r) => <span className="num whitespace-nowrap text-sub">{rowDate(r.started_at)}</span> },
  ];

  return (
    <Card title="Recent runs" to="/runs" toLabel="All runs" className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-2.5">
        <SearchField value={search} onChange={setSearch} placeholder="Search runs" />
        <FilterPill value={status} options={STATUS_FILTERS} onChange={setStatus} />
      </div>
      <VirtualTable
        rows={rows}
        columns={columns}
        template="minmax(150px,1.5fr) minmax(110px,1.1fr) 76px minmax(110px,1fr) 128px"
        rowKey={(r) => r.id}
        maxHeight={380}
        onRowClick={(r) => navigate(`/runs/${r.id}`)}
        empty={runs.length === 0 ? "No runs in this period." : "No runs match."}
      />
    </Card>
  );
}

const MODE_GROUPS: { key: string; label: string; modes: string[] }[] = [
  { key: "binary", label: MODE_LABEL.binary, modes: ["binary"] },
  { key: "multiclass", label: MODE_LABEL.multiclass, modes: ["multiclass"] },
  { key: "exact", label: MODE_LABEL.exact_match, modes: ["exact_match"] },
  { key: "judge", label: "Judge-graded", modes: ["judge_passfail", "judge_scale"] },
];

function ByMode({ runs }: { runs: RunSummary[] }) {
  const motion = useChartMotion();
  const data = MODE_GROUPS.map((g, i) => ({ name: g.label, value: runs.filter((r) => g.modes.includes(r.scoring_mode)).length, color: MODEL_PALETTE[i] }));
  const total = runs.length;
  return (
    <Card title="Runs by scoring mode" to="/runs" toLabel="All runs" className="min-w-0">
      {total === 0 ? (
        <EmptyLine>No runs in this period.</EmptyLine>
      ) : (
        <div className="flex flex-col items-center gap-5">
          <div className="relative h-[170px] w-[170px] shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.filter((d) => d.value > 0)} dataKey="value" nameKey="name" innerRadius={54} outerRadius={80} paddingAngle={3} cornerRadius={6} stroke="none" {...motion}>
                  {data
                    .filter((d) => d.value > 0)
                    .map((d) => (
                      <Cell key={d.name} fill={d.color} />
                    ))}
                </Pie>
                <Tooltip {...TOOLTIP} formatter={(v, n) => [`${v} runs`, String(n)]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
              <div>
                <div className="num text-2xl font-bold text-ink">{total}</div>
                <div className="text-[11px] text-faint">runs</div>
              </div>
            </div>
          </div>
          <ul className="w-full min-w-0 space-y-2.5">
            {data.map((d) => (
              <li key={d.name} className="flex items-center gap-2 text-[13px]">
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: d.color }} />
                <span className="truncate text-sub">{d.name}</span>
                <span className="num ml-auto font-semibold text-ink">{d.value}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
