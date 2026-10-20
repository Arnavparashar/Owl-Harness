import { useEffect, useState } from "react";
import { api, type RunDetail, type SampleFilter, type SampleItem } from "../api";
import { ms, useAsync, useDebounced } from "../lib";
import { Card, ErrorNote, th, td } from "./ui";

const FILTERS: { id: SampleFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "correct", label: "Correct" },
  { id: "incorrect", label: "Incorrect" },
  { id: "false_positive", label: "False positives" },
  { id: "false_negative", label: "False negatives" },
  { id: "unparseable", label: "Unparseable" },
];
const PAGE_SIZE = 20;

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-0.5 text-sm text-slate-900">{children}</dd>
    </div>
  );
}

const pre = "max-h-48 overflow-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-slate-50 p-2 font-mono text-xs text-slate-800";

function SamplePanel({ s }: { s: SampleItem }) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-900">Sample {s.sample_id}</h3>
        <span className={`text-sm font-medium ${s.error ? "text-danger" : s.correct ? "text-accent" : "text-danger"}`}>
          {s.error ? "error" : s.correct ? "correct" : "incorrect"}
        </span>
      </div>
      <Field label="Input">
        <pre className={pre}>{s.input}</pre>
      </Field>
      <Field label="Raw output">
        <pre className={pre}>{s.output === null ? "(none)" : JSON.stringify(s.output)}</pre>
      </Field>
      <dl className="grid grid-cols-2 gap-3">
        <Field label="Prediction">{s.pred ?? <span className="text-danger">unparseable</span>}</Field>
        <Field label="Target">{s.target}</Field>
        <Field label="Probability">{s.probability === null ? "–" : s.probability.toFixed(4)}</Field>
        <Field label="Latency">{s.latency_ms === null ? "–" : ms(s.latency_ms)}</Field>
        <Field label="Stop reason">{s.stop_reason ?? "–"}</Field>
      </dl>
      {s.error && (
        <Field label="Error">
          <pre className={`${pre} text-danger`}>{s.error}</pre>
        </Field>
      )}
      <Field label="Metadata">
        <pre className={pre}>{JSON.stringify(s.metadata, null, 2)}</pre>
      </Field>
    </div>
  );
}

export default function SampleExplorer({ run }: { run: RunDetail }) {
  const [filter, setFilter] = useState<SampleFilter>("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string>();
  const q = useDebounced(query);

  useEffect(() => setPage(1), [filter, q]);
  const { data, error } = useAsync(
    () => api.samples(run.id, { filter, q, page, page_size: PAGE_SIZE }),
    [run.id, filter, q, page],
  );

  const items = data?.items ?? [];
  const selected = items.find((i) => i.sample_id === selectedId) ?? items[0];
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <Card title="Samples">
      <div className="-mt-1 mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1" role="tablist">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              role="tab"
              aria-selected={filter === f.id}
              onClick={() => setFilter(f.id)}
              className={`rounded px-2.5 py-1 text-sm ${
                filter === f.id ? "bg-accent font-medium text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search input or output"
          className="w-64 rounded border border-slate-300 px-2.5 py-1 text-sm outline-accent"
        />
      </div>

      {error && <ErrorNote message={error} />}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div>
          <div className="overflow-x-auto rounded border border-slate-200">
            <table className="w-full border-collapse">
              <thead className="border-b border-slate-200 bg-slate-50">
                <tr>
                  <th className={th}>ID</th>
                  <th className={th}>Input</th>
                  <th className={th}>Target</th>
                  <th className={th}>Pred</th>
                  <th className={`${th} text-right`}>Prob.</th>
                  <th className={`${th} text-right`}>Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {items.map((s) => (
                  <tr
                    key={s.sample_id}
                    onClick={() => setSelectedId(s.sample_id)}
                    className={`cursor-pointer ${selected?.sample_id === s.sample_id ? "bg-accent-soft/60" : "hover:bg-slate-50"}`}
                  >
                    <td className={`${td} tabular-nums`}>{s.sample_id}</td>
                    <td className={`${td} max-w-[420px] truncate`} title={s.input}>
                      {s.input}
                    </td>
                    <td className={td}>{s.target}</td>
                    <td className={`${td} font-medium ${s.error || !s.correct ? "text-danger" : ""}`}>
                      {s.error ? "error" : (s.pred ?? "–")}
                    </td>
                    <td className={`${td} text-right tabular-nums`}>{s.probability === null ? "–" : s.probability.toFixed(3)}</td>
                    <td className={`${td} text-right tabular-nums`}>{s.latency_ms === null ? "–" : ms(s.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2 flex items-center justify-between text-sm text-slate-600">
            <span>{data ? `${data.total} sample${data.total === 1 ? "" : "s"}` : ""}</span>
            <div className="flex items-center gap-2">
              <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="rounded border border-slate-300 px-2 py-0.5 disabled:text-slate-300">
                Prev
              </button>
              <span className="tabular-nums">
                {page} / {pages}
              </span>
              <button disabled={page >= pages} onClick={() => setPage(page + 1)} className="rounded border border-slate-300 px-2 py-0.5 disabled:text-slate-300">
                Next
              </button>
            </div>
          </div>
        </div>
        <aside className="rounded border border-slate-200 p-3 lg:max-h-[640px] lg:overflow-y-auto">
          {selected && <SamplePanel s={selected} />}
        </aside>
      </div>
    </Card>
  );
}
