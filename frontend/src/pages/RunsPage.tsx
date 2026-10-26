import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type RunSummary } from "../api";
import { CiBars } from "../components/charts";
import JobsBanner from "../components/JobsBanner";
import TradeOffChart from "../components/TradeOffChart";
import { Card, Chip, ErrorNote, Loading, PageHeader, PERIODS, PeriodSelector, StatusPill, type Period } from "../components/ui";
import { VirtualTable, type Column } from "../components/VirtualTable";
import { dateTime, MODE_LABEL, ms, num, statText, useAsync } from "../lib";
import { useModelColor } from "../theme";

type SortKey = "task" | "model" | "n" | "main" | "f1" | "latency" | "status" | "started_at";

const sortValue = (r: RunSummary, key: SortKey): string | number => {
  switch (key) {
    case "task":
      return r.task;
    case "model":
      return r.model;
    case "n":
      return r.metrics.n ?? -1;
    case "main":
      return r.headline.main?.value ?? -1;
    case "f1":
      return r.headline.secondary?.value ?? -1;
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
  const [sortKey, setSortKey] = useState<SortKey>("started_at");
  const [desc, setDesc] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [task, setTask] = useState("");
  const [mode, setMode] = useState("");
  const [search, setSearch] = useState("");
  const [period, setPeriod] = useState<Period>("all");
  const [deleting, setDeleting] = useState<string>();
  const [actionError, setActionError] = useState<string>();
  const navigate = useNavigate();
  const colorOf = useModelColor();

  const tasks = useMemo(() => [...new Set((data ?? []).map((r) => r.task))].sort(), [data]);
  const modes = useMemo(() => [...new Set((data ?? []).map((r) => r.scoring_mode))], [data]);

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    const days = PERIODS.find((p) => p.id === period)!.days;
    const since = days === null ? 0 : Date.now() - days * 24 * 60 * 60 * 1000;
    const list = (data ?? []).filter(
      (r) =>
        new Date(r.started_at).getTime() >= since &&
        (!task || r.task === task) &&
        (!mode || r.scoring_mode === mode) &&
        (!q || r.model.toLowerCase().includes(q) || r.id.includes(q)),
    );
    list.sort((a, b) => {
      const x = sortValue(a, sortKey);
      const y = sortValue(b, sortKey);
      const c = x < y ? -1 : x > y ? 1 : 0;
      return desc ? -c : c;
    });
    return list;
  }, [data, sortKey, desc, task, mode, search, period]);

  const sortBy = (key: SortKey) => () => {
    if (key === sortKey) setDesc(!desc);
    else {
      setSortKey(key);
      setDesc(!["task", "model", "status"].includes(key));
    }
  };
  const sorted = (key: SortKey) => (sortKey === key ? (desc ? "desc" : "asc") : null);
  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const remove = async (id: string) => {
    setActionError(undefined);
    try {
      await api.deleteRun(id);
      setSelected((s) => s.filter((x) => x !== id));
      setRefresh((n) => n + 1);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(undefined);
    }
  };

  const columns: Column<RunSummary>[] = [
    {
      key: "select",
      header: "",
      render: (r) => (
        <input
          type="checkbox"
          aria-label={`Select ${r.model} run ${r.id}`}
          checked={selected.includes(r.id)}
          onChange={() => toggle(r.id)}
          onClick={(e) => e.stopPropagation()}
        />
      ),
    },
    {
      key: "model",
      header: "Model",
      onSort: sortBy("model"),
      sorted: sorted("model"),
      render: (r) => (
        <Link to={`/runs/${r.id}`} className="flex min-w-0 items-center gap-2 font-medium text-ink hover:text-accent">
          <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colorOf(r.model) }} />
          <span className="truncate">{r.model}</span>
        </Link>
      ),
    },
    {
      key: "task",
      header: "Test",
      onSort: sortBy("task"),
      sorted: sorted("task"),
      render: (r) => (
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sub">{r.task}</span>
          {r.example && <Chip tone="accent">Example</Chip>}
        </span>
      ),
    },
    { key: "mode", header: "Scoring", render: (r) => <span className="truncate text-sub">{MODE_LABEL[r.scoring_mode]}</span> },
    { key: "n", header: "n", align: "right", onSort: sortBy("n"), sorted: sorted("n"), render: (r) => <span className="num text-sub">{r.metrics.n?.toFixed(0) ?? "–"}</span> },
    {
      key: "main",
      header: "Main metric",
      align: "right",
      onSort: sortBy("main"),
      sorted: sorted("main"),
      render: (r) => (
        <span className="num" title={r.headline.main ? `${r.headline.main.label}, 95% CI ${num(r.headline.main.ci_low)}–${num(r.headline.main.ci_high)}` : ""}>
          <span className="font-semibold">{statText(r.headline.main)}</span>
          <span className="ml-1.5 text-xs text-faint">{r.headline.main?.label.toLowerCase().replace("accuracy", "acc.")}</span>
        </span>
      ),
    },
    {
      key: "f1",
      header: "F1 / macro-F1",
      align: "right",
      onSort: sortBy("f1"),
      sorted: sorted("f1"),
      render: (r) => <span className="num text-sub">{num(r.headline.secondary?.value)}</span>,
    },
    { key: "latency", header: "Median latency", align: "right", onSort: sortBy("latency"), sorted: sorted("latency"), render: (r) => <span className="num text-sub">{ms(r.metrics.latency_median_ms)}</span> },
    { key: "status", header: "Status", onSort: sortBy("status"), sorted: sorted("status"), render: (r) => <StatusPill status={r.status} /> },
    { key: "started_at", header: "Started", onSort: sortBy("started_at"), sorted: sorted("started_at"), render: (r) => <span className="num whitespace-nowrap text-sub">{dateTime(r.started_at)}</span> },
    {
      key: "actions",
      header: "",
      align: "right",
      render: (r) =>
        deleting === r.id ? (
          <span className="flex items-center gap-1 text-xs">
            <button className="btn btn-danger h-7 px-2" onClick={() => remove(r.id)}>
              Delete
            </button>
            <button className="btn btn-quiet h-7 px-2" onClick={() => setDeleting(undefined)}>
              Keep
            </button>
          </span>
        ) : (
          <span className="flex items-center gap-1">
            <a href={api.exportUrl(r.id)} download className="btn btn-quiet h-7 px-2 text-xs" title="Export samples as CSV">
              CSV
            </a>
            <button className="btn btn-danger h-7 px-2 text-xs" onClick={() => setDeleting(r.id)} title="Delete this run">
              Delete
            </button>
          </span>
        ),
    },
  ];

  const ciRows = rows
    .filter((r) => r.headline.main && r.headline.main.key !== "mean_score")
    .sort((a, b) => b.headline.main!.value - a.headline.main!.value)
    .slice(0, 12);

  return (
    <div className="space-y-6">
      <PageHeader title="Runs" subtitle="Every evaluation, with its main metric and interval." right={<PeriodSelector value={period} onChange={setPeriod} />} />
      <JobsBanner onChange={() => setRefresh((n) => n + 1)} />

      <div className="glass rise flex flex-wrap items-center gap-3 px-5 py-3.5">
        <select value={task} onChange={(e) => setTask(e.target.value)} className="input w-52 rounded-full" aria-label="Filter by test">
          <option value="">All tests</option>
          {tasks.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value)} className="input w-52 rounded-full" aria-label="Filter by scoring mode">
          <option value="">All scoring modes</option>
          {modes.map((m) => (
            <option key={m} value={m}>
              {MODE_LABEL[m]}
            </option>
          ))}
        </select>
        <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Filter by model or run id" className="input w-60 rounded-full" aria-label="Filter by model" />
        <span className="num text-sm text-faint">
          {rows.length} of {data?.length ?? 0} runs
        </span>
        <div className="ml-auto flex items-center gap-2">
          {selected.length > 0 && (
            <button className="btn btn-quiet" onClick={() => setSelected([])}>
              Clear selection
            </button>
          )}
          <button className="btn btn-primary" disabled={selected.length < 2} onClick={() => navigate(`/compare?runs=${selected.join(",")}`)}>
            Compare selected{selected.length > 0 ? ` (${selected.length})` : ""}
          </button>
        </div>
      </div>

      {loading && !data && <Loading />}
      {error && <ErrorNote message={error} />}
      {actionError && <ErrorNote message={actionError} />}
      {data && (
        <VirtualTable
          rows={rows}
          columns={columns}
          template="40px minmax(190px,1.6fr) minmax(160px,1.3fr) 110px 56px 140px 100px 100px 160px 140px 130px"
          rowKey={(r) => r.id}
          rowHeight={52}
          maxHeight={620}
          rowClassName={(r) => (selected.includes(r.id) ? "!bg-emerald/15" : "hover:bg-white")}
          empty={data.length === 0 ? "No runs yet. Start one from New run." : "No runs match the filters."}
        />
      )}

      {rows.length > 0 && (
        <div className="grid gap-5 xl:grid-cols-2">
          <Card title="Main metric by run" right={<span className="text-xs text-faint">95% CI · top 12 shown</span>}>
            <CiBars
              rows={ciRows.map((r) => ({
                label: `${r.model} · ${r.task}`.slice(0, 40),
                value: r.headline.main!.value,
                lo: r.headline.main!.ci_low,
                hi: r.headline.main!.ci_high,
                color: colorOf(r.model),
                sub: r.headline.main!.label,
              }))}
            />
            {new Set(ciRows.map((r) => r.task)).size > 1 && <p className="mt-2 text-xs text-faint">These runs span several tests. Filter to one test before comparing them.</p>}
          </Card>
          <Card title="Accuracy against median latency">
            <TradeOffChart runs={rows.filter((r) => r.status.startsWith("completed"))} />
          </Card>
        </div>
      )}
    </div>
  );
}
