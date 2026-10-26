import { MotionConfig } from "framer-motion";
import { useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import { BottomNavProvider } from "./components/nav";
import { BottomNav, Rail, TopBar } from "./components/Shell";
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

export default function App() {
  const location = useLocation();
  // Runs feed the model colours; refresh them now and then so new models get a colour.
  const [tick, setTick] = useState(0);
  const runs = useAsync(() => api.runs(), [tick, location.pathname]);
  usePolling(true, () => setTick((n) => n + 1), 30000);

  return (
    // reducedMotion="user": framer-motion skips movement when the system asks for reduced motion
    <MotionConfig reducedMotion="user">
      <ModelColorProvider runs={runs.data}>
        <BottomNavProvider>
          <div className="app-backdrop" aria-hidden />
          <Rail />
          <div className="min-h-screen pl-[88px] pr-4 sm:pr-8">
            <div className="mx-auto max-w-[1520px]">
              <TopBar />
              <main className="pb-32 pt-2">
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
          <BottomNav />
        </BottomNavProvider>
      </ModelColorProvider>
    </MotionConfig>
  );
}
