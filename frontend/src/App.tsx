import { useState, type ReactNode } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import { usePolling, useAsync } from "./lib";
import ComparePage from "./pages/ComparePage";
import DatasetsPage from "./pages/DatasetsPage";
import JobPage from "./pages/JobPage";
import ModelsPage from "./pages/ModelsPage";
import NewRunPage from "./pages/NewRunPage";
import OverviewPage from "./pages/OverviewPage";
import RunDetailPage from "./pages/RunDetailPage";
import RunsPage from "./pages/RunsPage";
import { ModelColorProvider } from "./theme";

const NAV: { to: string; label: string; icon: ReactNode; end?: boolean }[] = [
  { to: "/", label: "Overview", end: true, icon: <path d="M3 13h8V3H3v10Zm0 8h8v-6H3v6Zm10 0h8V11h-8v10Zm0-18v6h8V3h-8Z" /> },
  { to: "/new", label: "New run", icon: <path d="M12 5v14M5 12h14" strokeWidth={2.2} stroke="currentColor" fill="none" strokeLinecap="round" /> },
  { to: "/runs", label: "Runs", icon: <path d="M4 6h16M4 12h16M4 18h10" strokeWidth={2} stroke="currentColor" fill="none" strokeLinecap="round" /> },
  { to: "/compare", label: "Compare", icon: <path d="M5 20V10m7 10V4m7 16v-7" strokeWidth={2.2} stroke="currentColor" fill="none" strokeLinecap="round" /> },
  {
    to: "/models",
    label: "Models",
    icon: <path d="M12 3 4 7.5v9L12 21l8-4.5v-9L12 3Zm0 0v18M4 7.5l8 4.5 8-4.5" strokeWidth={1.8} stroke="currentColor" fill="none" strokeLinejoin="round" />,
  },
  {
    to: "/datasets",
    label: "Datasets",
    icon: (
      <path
        d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"
        strokeWidth={1.8}
        stroke="currentColor"
        fill="none"
      />
    ),
  },
];

const TITLES: [RegExp, string][] = [
  [/^\/$/, "Overview"],
  [/^\/new/, "New run"],
  [/^\/runs\/.+/, "Run"],
  [/^\/runs/, "Runs"],
  [/^\/jobs\//, "Progress"],
  [/^\/compare/, "Compare"],
  [/^\/models/, "Models"],
  [/^\/datasets/, "Datasets"],
];

export default function App() {
  const location = useLocation();
  const title = TITLES.find(([re]) => re.test(location.pathname))?.[1] ?? "Owl";
  // Runs feed the model colours; refresh them now and then so new models get a colour.
  const [tick, setTick] = useState(0);
  const runs = useAsync(() => api.runs(), [tick, location.pathname]);
  usePolling(true, () => setTick((n) => n + 1), 30000);

  return (
    <ModelColorProvider runs={runs.data}>
      <div className="app-fields" aria-hidden />
      <div className="flex min-h-screen">
        <aside className="glass sticky top-3 m-3 flex h-[calc(100vh-24px)] w-[212px] shrink-0 flex-col rounded-[16px] px-3 py-4">
          <Link to="/" className="mb-6 flex items-center gap-2.5 px-2">
            <span className="accent-bar grid h-8 w-8 place-items-center rounded-[10px] text-[15px] font-black text-[#061019]">O</span>
            <span className="text-[15px] font-semibold tracking-tight text-ink">Owl Harness</span>
          </Link>
          <nav className="flex flex-col gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  `relative flex h-10 items-center gap-3 rounded-[10px] px-3 text-[13px] font-medium transition-colors ${
                    isActive ? "bg-white/[0.1] text-ink" : "text-sub hover:bg-white/[0.05] hover:text-ink"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && <span className="accent-bar absolute left-0 top-2 bottom-2 w-[3px] rounded-full" />}
                    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="currentColor" aria-hidden>
                      {n.icon}
                    </svg>
                    {n.label}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto px-2 text-xs leading-relaxed text-faint">
            Local only · results stay on this machine
          </div>
        </aside>

        <div className="min-w-0 flex-1 pr-3">
          <header className="glass sticky top-3 z-20 mt-3 flex h-14 items-center justify-between px-5">
            <h1 className="text-[17px] font-semibold tracking-tight text-ink">{title}</h1>
            {!location.pathname.startsWith("/new") && (
              <Link to="/new" className="btn btn-primary">
                New run
              </Link>
            )}
          </header>
          <main className="py-5">
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/new" element={<NewRunPage />} />
              <Route path="/jobs/:jobId" element={<JobPage />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/runs/:runId" element={<RunDetailPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="*" element={<p className="text-sm text-sub">Page not found.</p>} />
            </Routes>
          </main>
        </div>
      </div>
    </ModelColorProvider>
  );
}
