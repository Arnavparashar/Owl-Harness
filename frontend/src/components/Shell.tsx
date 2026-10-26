// The app shell: floating icon rail, top bar with search, job state and notifications, and the bottom tab pill.
import { AnimatePresence, motion } from "framer-motion";
import { Bell, Boxes, ChevronLeft, ChevronRight, CircleCheck, CircleX, Database, Gauge, GitCompareArrows, LayoutDashboard, List, Plus, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { api, isActive, type DatasetInfo, type Job, type RunSummary, type SavedModel } from "../api";
import { dateTime, statText, usePolling, useAsync } from "../lib";
import { useBottomNav } from "./nav";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/new", label: "New run", icon: Plus },
  { to: "/runs", label: "Runs", icon: List },
  { to: "/compare", label: "Compare", icon: GitCompareArrows },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/datasets", label: "Datasets", icon: Database },
];

/** localStorage can be unavailable (private windows, blocked storage): never let that break the UI. */
function readStore<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}
function writeStore(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* not persisted; fine */
  }
}

export function Rail() {
  // the expanded rail floats over the page, so it starts collapsed on every load
  const [open, setOpen] = useState(false);

  return (
    <aside
      className={`glass fixed left-4 top-1/2 z-40 flex -translate-y-1/2 flex-col gap-1.5 p-2 transition-[width] duration-200 ${open ? "w-[196px] !rounded-[32px]" : "w-[60px] !rounded-full"}`}
      aria-label="Main navigation"
    >
      <nav className="flex flex-col gap-1.5">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} aria-label={label} className={`group relative block ${open ? "" : "tip tip-right"}`}>
            {({ isActive }) => (
              <>
                <span
                  className={`flex h-11 items-center gap-3 rounded-full transition-colors ${open ? "px-3.5" : "w-11 justify-center"} ${
                    isActive ? "active-pill shadow-md" : "text-sub group-hover:bg-wash/[0.08] group-hover:text-ink"
                  }`}
                >
                  <Icon size={19} strokeWidth={isActive ? 2.2 : 1.9} className="shrink-0" />
                  {open && <span className="text-[13px] font-medium">{label}</span>}
                </span>
                {!open && (
                  <span role="tooltip" className="tip-body">
                    {label}
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={`mt-1 grid h-9 place-items-center rounded-full text-faint hover:bg-wash/[0.08] hover:text-ink ${open ? "w-full" : "w-11"}`}
        aria-label={open ? "Collapse navigation" : "Expand navigation"}
        aria-expanded={open}
      >
        {open ? <ChevronLeft size={17} /> : <ChevronRight size={17} />}
      </button>
    </aside>
  );
}

// --- search -----------------------------------------------------------------------

type Hit = { kind: "Run" | "Model" | "Dataset"; label: string; detail: string; to: string };

function SearchPill() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [active, setActive] = useState(0);
  const [stamp, setStamp] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  // Load the three lists when the field is focused, so results are fresh but typing stays instant.
  const data = useAsync(
    () => (stamp === 0 ? Promise.resolve(undefined) : Promise.all([api.runs(), api.models(), api.datasets()])),
    [stamp],
  );

  const hits = useMemo<Hit[]>(() => {
    const q = query.trim().toLowerCase();
    if (!q || !data.data) return [];
    const [runs, models, datasets] = data.data as [RunSummary[], SavedModel[], DatasetInfo[]];
    const out: Hit[] = [];
    for (const r of runs) {
      if (`${r.model} ${r.task} ${r.id}`.toLowerCase().includes(q)) {
        out.push({ kind: "Run", label: r.model, detail: `${r.task} · ${statText(r.headline.main)} · ${dateTime(r.started_at)}`, to: `/runs/${r.id}` });
      }
      if (out.filter((h) => h.kind === "Run").length >= 6) break;
    }
    for (const m of models.filter((m) => `${m.name} ${m.provider}/${m.model}`.toLowerCase().includes(q)).slice(0, 3)) {
      out.push({ kind: "Model", label: m.name, detail: `${m.provider}/${m.model}`, to: `/models?focus=${m.id}` });
    }
    for (const d of datasets.filter((d) => `${d.name} ${d.filename} ${d.source?.hub ?? ""}`.toLowerCase().includes(q)).slice(0, 3)) {
      out.push({ kind: "Dataset", label: d.name, detail: `${d.rows.toLocaleString()} rows · ${d.format}`, to: `/datasets?id=${d.id}` });
    }
    return out;
  }, [query, data.data]);

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    const close = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setFocused(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const go = (h: Hit) => {
    navigate(h.to);
    setQuery("");
    setFocused(false);
  };

  return (
    <div ref={box} className="relative w-full max-w-[420px]">
      <label className="pill w-full">
        <Search size={16} className="shrink-0 text-faint" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            setFocused(true);
            setStamp(Date.now());
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => Math.min(a + 1, hits.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(a - 1, 0));
            } else if (e.key === "Enter" && hits[active]) go(hits[active]);
            else if (e.key === "Escape") setFocused(false);
          }}
          placeholder="Search runs, models, datasets…"
          aria-label="Search runs, models and datasets"
          className="w-full bg-transparent text-[13px] text-ink outline-none placeholder:text-faint"
        />
      </label>
      {focused && query.trim() && (
        <div className="popover absolute inset-x-0 top-12 z-50 max-h-[420px] overflow-y-auto p-1.5 shadow-2xl" role="listbox">
          {data.loading && !data.data && <p className="px-3 py-2 text-sm text-faint">Searching…</p>}
          {data.data && hits.length === 0 && <p className="px-3 py-2 text-sm text-faint">Nothing matches “{query.trim()}”.</p>}
          {hits.map((h, i) => (
            <button
              key={`${h.kind}-${h.to}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onClick={() => go(h)}
              className={`flex w-full items-center gap-3 rounded-[10px] px-3 py-2 text-left ${i === active ? "bg-wash/[0.08]" : ""}`}
            >
              <span className="w-16 shrink-0 text-[11px] font-semibold uppercase tracking-wide text-faint">{h.kind}</span>
              <span className="min-w-0">
                <span className="block truncate text-[13px] font-medium text-ink">{h.label}</span>
                <span className="block truncate text-xs text-faint">{h.detail}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// --- notifications --------------------------------------------------------------------

function NotificationBell() {
  const [open, setOpen] = useState(false);
  const [tick, setTick] = useState(0);
  const [seen, setSeen] = useState<string[]>(() => readStore("owl.jobs.seen", []));
  const jobs = useAsync(() => api.jobs(), [tick]);
  usePolling(true, () => setTick((n) => n + 1), 5000);
  const box = useRef<HTMLDivElement>(null);

  const finished = (jobs.data ?? [])
    .filter((j: Job) => j.status === "completed" || j.status === "failed")
    .sort((a, b) => (b.finished_at ?? "").localeCompare(a.finished_at ?? ""));
  const unread = finished.filter((j) => !seen.includes(j.id));

  useEffect(() => {
    const close = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && unread.length) {
      const all = [...new Set([...seen, ...finished.map((j) => j.id)])].slice(-200);
      setSeen(all);
      writeStore("owl.jobs.seen", all);
    }
  };

  return (
    <div ref={box} className="relative">
      <button type="button" onClick={toggle} className="icon-btn relative h-10 w-10" aria-label={`Notifications${unread.length ? `, ${unread.length} unread` : ""}`} title="Notifications">
        <Bell size={17} />
        {unread.length > 0 && <span className="absolute right-2 top-2 h-2.5 w-2.5 rounded-full border-2 border-white bg-emerald" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            className="popover absolute right-0 top-12 z-50 w-[340px] p-2 shadow-2xl"
          >
            <div className="px-2 pb-2 pt-1 text-[13px] font-semibold text-ink">Notifications</div>
            {finished.length === 0 && <p className="px-2 pb-2 text-sm text-faint">No finished or failed jobs since the server started.</p>}
            {finished.slice(0, 8).map((j) => {
              const ok = j.status === "completed";
              const Icon = ok ? CircleCheck : CircleX;
              return (
                <Link key={j.id} to={`/jobs/${j.id}`} onClick={() => setOpen(false)} className="flex items-start gap-3 rounded-[10px] px-2 py-2 hover:bg-wash/[0.07]">
                  <Icon size={17} className={`mt-0.5 shrink-0 ${ok ? "text-ok" : "text-bad"}`} />
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] text-ink">
                      {j.task} {ok ? "finished" : "failed"}
                    </span>
                    <span className="block truncate text-xs text-faint">
                      {j.items.length} model{j.items.length === 1 ? "" : "s"} · {dateTime(j.finished_at)}
                    </span>
                  </span>
                </Link>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** "Running 1 of 3" while a job runs (the model being run, out of the models in the job), otherwise "No jobs running". */
function JobState() {
  const [tick, setTick] = useState(0);
  const jobs = useAsync(() => api.jobs(), [tick]);
  const active = (jobs.data ?? []).filter(isActive);
  usePolling(true, () => setTick((n) => n + 1), active.length > 0 ? 2000 : 5000);
  const job = active[0];
  if (!job) {
    return (
      <span className="pill hidden h-11 text-[13px] font-medium text-sub sm:inline-flex">
        <span className="h-2 w-2 rounded-full bg-[#8a7f70]" />
        No jobs running
      </span>
    );
  }
  const total = job.items.length;
  const doneCount = job.items.filter((i) => i.status !== "queued" && i.status !== "running").length;
  const position = Math.min(total, doneCount + 1);
  return (
    <Link
      to={`/jobs/${job.id}`}
      className="pill h-11 text-[13px] font-medium text-ink"
      title={active.length > 1 ? `${active.length} jobs are active; showing the first` : "Watch progress"}
    >
      <span className="h-2 w-2 animate-pulse rounded-full bg-emerald motion-reduce:animate-none" />
      <span className="num">
        Running {position} of {total}
      </span>
    </Link>
  );
}

export function TopBar() {
  return (
    <header className="flex min-h-[76px] items-center gap-3 py-4">
      <Link to="/" className="mr-1 hidden items-center gap-2 text-[15px] font-bold tracking-tight text-ink xl:flex" aria-label="Owl Harness home">
        <span className="grid h-9 w-9 place-items-center rounded-full bg-emerald text-white">
          <Gauge size={18} strokeWidth={2.2} />
        </span>
        Owl Harness
      </Link>
      <SearchPill />
      <div className="ml-auto flex items-center gap-3">
        <JobState />
        <NotificationBell />
      </div>
    </header>
  );
}

// --- bottom pill ------------------------------------------------------------------------

/** The floating bottom pill: the current page's tabs (when it has any) and a round "+" that starts a new run. */
export function BottomNav() {
  const nav = useBottomNav();
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-40 flex items-center justify-center gap-3 px-4">
      {nav && nav.tabs.length > 0 && (
        <nav className="pill pointer-events-auto h-[52px] max-w-full gap-1 overflow-x-auto p-1.5" role="tablist" aria-label="Page sections">
          {nav.tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={nav.active === t.id}
              onClick={() => nav.onChange(t.id)}
              className={`h-10 shrink-0 rounded-full px-4 text-[13px] font-medium transition-colors ${nav.active === t.id ? "active-pill" : "text-sub hover:text-ink"}`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      )}
      <span className="tip pointer-events-auto">
        <Link to="/new" aria-label="New run" className="grid h-[52px] w-[52px] place-items-center rounded-full bg-[#17150f] text-white shadow-xl shadow-black/25 transition-transform hover:scale-105 motion-reduce:transition-none">
          <Plus size={22} strokeWidth={2.4} />
        </Link>
        <span role="tooltip" className="tip-body">
          New run
        </span>
      </span>
    </div>
  );
}
