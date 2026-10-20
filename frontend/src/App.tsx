import { Link, NavLink, Route, Routes } from "react-router-dom";
import ComparePage from "./pages/ComparePage";
import JobPage from "./pages/JobPage";
import NewRunPage from "./pages/NewRunPage";
import RunDetailPage from "./pages/RunDetailPage";
import RunsPage from "./pages/RunsPage";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `border-b-2 px-1 py-4 text-sm font-medium ${
    isActive ? "border-accent text-slate-900" : "border-transparent text-slate-500 hover:text-slate-900"
  }`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1400px] items-center gap-8 px-6">
          <span className="text-base font-semibold tracking-tight text-slate-900">Owl Harness</span>
          <nav className="flex gap-6">
            <NavLink to="/" end className={navClass}>
              Runs
            </NavLink>
            <NavLink to="/compare" className={navClass}>
              Compare
            </NavLink>
          </nav>
          <Link to="/new" className="ml-auto rounded bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-[#1e40af]">
            New run
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-[1400px] px-6 py-6">
        <Routes>
          <Route path="/" element={<RunsPage />} />
          <Route path="/new" element={<NewRunPage />} />
          <Route path="/jobs/:jobId" element={<JobPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="*" element={<div className="text-sm text-slate-600">Page not found.</div>} />
        </Routes>
      </main>
    </div>
  );
}
