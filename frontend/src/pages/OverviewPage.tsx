import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type RunSummary } from "../api";
import { CiBars } from "../components/charts";
import TradeOffChart from "../components/TradeOffChart";
import JobsBanner from "../components/JobsBanner";
import { Card, Chip, EmptyLine, ErrorNote, Loading, StatusPill } from "../components/ui";
import { dateTime, MODE_LABEL, statText, useAsync } from "../lib";
import { useModelColor } from "../theme";

const finished = (r: RunSummary) => r.status.startsWith("completed") && r.headline.main !== null;

export default function OverviewPage() {
  const [tick, setTick] = useState(0);
  const { data, error, loading } = useAsync(() => api.runs(), [tick]);
  const runs = data ?? [];

  return (
    <div className="space-y-5">
      <JobsBanner onChange={() => setTick((n) => n + 1)} />
      {loading && !data && <Loading />}
      {error && <ErrorNote message={error} />}
      {data && runs.length === 0 && (
        <Card>
          <EmptyLine>No runs yet. Start one from New run.</EmptyLine>
        </Card>
      )}
      {runs.length > 0 && (
        <>
          <Leaderboards runs={runs.filter(finished)} />
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <RecentRuns runs={runs.slice(0, 8)} />
            <TradeOff runs={runs.filter(finished)} />
          </div>
        </>
      )}
    </div>
  );
}

/** One leaderboard per test: each model's best run on its main metric, with the 95% interval. */
function Leaderboards({ runs }: { runs: RunSummary[] }) {
  const colorOf = useModelColor();
  const boards = useMemo(() => {
    const byTask = new Map<string, RunSummary[]>();
    for (const r of runs) byTask.set(r.task, [...(byTask.get(r.task) ?? []), r]);
    return [...byTask.entries()]
      .map(([task, list]) => {
        // Best run per model, but only among that model's largest runs: a 5-sample smoke test
        // scoring 1.000 must not outrank a full run.
        const best = new Map<string, RunSummary>();
        for (const r of list) {
          const cur = best.get(r.model);
          const n = r.metrics.n ?? 0;
          const curN = cur?.metrics.n ?? 0;
          if (!cur || n > curN || (n === curN && r.headline.main!.value > cur.headline.main!.value)) best.set(r.model, r);
        }
        const rows = [...best.values()].sort((a, b) => b.headline.main!.value - a.headline.main!.value);
        const maxN = Math.max(...rows.map((r) => r.metrics.n ?? 0));
        const latest = list.reduce((a, b) => (a.started_at > b.started_at ? a : b)).started_at;
        return { task, rows, maxN, latest, example: list[0].example, mode: list[0].scoring_mode };
      })
      .sort((a, b) => b.latest.localeCompare(a.latest));
  }, [runs]);

  if (boards.length === 0) return null;
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      {boards.map((b) => {
        const main = b.rows[0].headline.main!;
        const scale = main.key === "mean_score";
        return (
          <Card
            key={b.task}
            title={
              <span className="flex flex-wrap items-center gap-2">
                {b.task}
                <Chip>{MODE_LABEL[b.mode]}</Chip>
                {b.example && <Chip tone="accent">Example</Chip>}
              </span>
            }
            right={<span className="text-xs text-faint">best {main.label.toLowerCase()} of each model's largest runs · 95% CI</span>}
          >
            <CiBars
              domain={scale ? [1, 5] : [0, 1]}
              rows={b.rows.map((r) => ({
                label: (r.metrics.n ?? 0) < b.maxN ? `${r.model.length > 24 ? `${r.model.slice(0, 23)}…` : r.model} (n=${r.metrics.n?.toFixed(0)})` : r.model,
                value: r.headline.main!.value,
                lo: r.headline.main!.ci_low,
                hi: r.headline.main!.ci_high,
                color: colorOf(r.model),
                sub: `${r.headline.main!.label}, n = ${r.metrics.n?.toFixed(0) ?? "?"}`,
              }))}
            />
          </Card>
        );
      })}
    </div>
  );
}

function RecentRuns({ runs }: { runs: RunSummary[] }) {
  const colorOf = useModelColor();
  return (
    <Card title="Recent runs" right={<Link to="/runs" className="text-xs font-medium text-accent hover:underline">All runs →</Link>} pad={false}>
      <ul>
        {runs.map((r) => (
          <li key={r.id} className="border-b border-white/[0.06] last:border-0">
            <Link to={`/runs/${r.id}`} className="flex items-center gap-3 px-5 py-3 transition-colors hover:bg-white/[0.04]">
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colorOf(r.model) }} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] font-medium text-ink">{r.model}</div>
                <div className="truncate text-xs text-faint">
                  {r.task} · {dateTime(r.started_at)}
                </div>
              </div>
              <StatusPill status={r.status} />
              <div className="w-24 text-right">
                <div className="num text-[15px] font-semibold text-ink">{statText(r.headline.main)}</div>
                <div className="text-[11px] text-faint">{r.headline.main?.label.toLowerCase() ?? ""}</div>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function TradeOff({ runs }: { runs: RunSummary[] }) {
  return (
    <Card title="Accuracy against median latency" right={<span className="text-xs text-faint">one point per run</span>}>
      <TradeOffChart runs={runs} />
    </Card>
  );
}
