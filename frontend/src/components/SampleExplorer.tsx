import { useEffect, useState, type ReactNode } from "react";
import { api, type RunDetail, type SampleFilter, type SampleItem } from "../api";
import { ms, useAsync, useDebounced } from "../lib";
import { Card, Chip, ErrorNote, Tabs } from "./ui";

const PAGE_SIZE = 25;

function filtersFor(run: RunDetail): { id: SampleFilter; label: string }[] {
  const base: { id: SampleFilter; label: string }[] = [
    { id: "all", label: "All" },
    { id: "correct", label: run.scoring_mode.startsWith("judge") ? "Passed" : "Correct" },
    { id: "incorrect", label: run.scoring_mode.startsWith("judge") ? "Not passed" : "Incorrect" },
  ];
  if (run.scoring_mode === "binary") base.push({ id: "false_positive", label: "False positives" }, { id: "false_negative", label: "False negatives" });
  base.push({ id: "unparseable", label: "Unparseable" });
  return base;
}

/** What the scorer decided, in the words of the scoring mode. */
function verdict(run: RunDetail, s: SampleItem): string {
  if (s.error) return "error";
  if (s.pred === null) return "unparseable";
  if (run.scoring_mode === "judge_passfail") return s.pred === "1" ? "PASS" : "FAIL";
  if (run.scoring_mode === "judge_scale") return `score ${s.pred}`;
  return s.pred;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-1 text-[13px] text-ink">{children}</dd>
    </div>
  );
}

const pre = "max-h-52 overflow-auto whitespace-pre-wrap break-words rounded-[10px] border border-wash/10 bg-white/70 p-2.5 font-mono text-xs leading-relaxed text-sub";

function SamplePanel({ run, s }: { run: RunDetail; s: SampleItem }) {
  const meta = s.metadata as { prompt?: { role: string; content: string }[]; explanation?: string; score?: Record<string, unknown> };
  const judged = run.scoring_mode.startsWith("judge");
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="num text-sm font-semibold text-ink">Sample {s.sample_id}</h3>
        <Chip tone={s.error ? "bad" : s.correct ? "ok" : "bad"}>{s.error ? "error" : s.correct ? (judged ? "passed" : "correct") : judged ? "not passed" : "incorrect"}</Chip>
      </div>
      <Field label="Input">
        <pre className={pre}>{s.input}</pre>
      </Field>
      {meta.prompt && meta.prompt.length > 0 && (
        <Field label="Prompt sent">
          <pre className={pre}>{meta.prompt.map((m) => `[${m.role}]\n${m.content}`).join("\n\n")}</pre>
        </Field>
      )}
      <Field label="Model output">
        <pre className={pre}>{s.output === null ? "(none)" : s.output}</pre>
      </Field>
      <dl className="grid grid-cols-2 gap-3">
        <Field label={judged ? "Verdict" : "Prediction"}>
          <span className={s.pred === null ? "text-bad" : ""}>{verdict(run, s)}</span>
        </Field>
        <Field label={judged ? "Reference" : "Target"}>{s.target || "–"}</Field>
        <Field label="Probability">
          <span className="num">{s.probability === null ? "–" : s.probability.toFixed(4)}</span>
        </Field>
        <Field label="Latency">
          <span className="num">{s.latency_ms === null ? "–" : ms(s.latency_ms)}</span>
        </Field>
        <Field label="Stop reason">{s.stop_reason ?? "–"}</Field>
      </dl>
      {judged && meta.explanation && (
        <Field label="Judge's reply">
          <pre className={pre}>{meta.explanation}</pre>
        </Field>
      )}
      {s.error && (
        <Field label="Error">
          <pre className={`${pre} text-bad`}>{s.error}</pre>
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
  const { data, error } = useAsync(() => api.samples(run.id, { filter, q, page, page_size: PAGE_SIZE }), [run.id, filter, q, page]);

  const items = data?.items ?? [];
  const selected = items.find((i) => i.sample_id === selectedId) ?? items[0];
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <Card
      title="Samples"
      right={
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search input or output" className="input w-64" aria-label="Search samples" />
      }
    >
      <div className="mb-4">
        <Tabs tabs={filtersFor(run)} active={filter} onChange={setFilter} />
      </div>
      {error && <ErrorNote message={error} />}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div>
          <div className="overflow-x-auto">
            <table className="strip-table">
              <thead>
                <tr>
                  <th className="label px-3 py-2.5 text-left">ID</th>
                  <th className="label px-3 py-2.5 text-left">Input</th>
                  <th className="label px-3 py-2.5 text-left">Target</th>
                  <th className="label px-3 py-2.5 text-left">{run.scoring_mode.startsWith("judge") ? "Verdict" : "Pred"}</th>
                  <th className="label px-3 py-2.5 text-right">Prob.</th>
                  <th className="label px-3 py-2.5 text-right">Latency</th>
                </tr>
              </thead>
              <tbody>
                {items.map((s) => (
                  <tr
                    key={s.sample_id}
                    onClick={() => setSelectedId(s.sample_id)}
                    className={`cursor-pointer border-b border-wash/[0.06] ${selected?.sample_id === s.sample_id ? "!bg-emerald/15" : "hover:bg-white"}`}
                  >
                    <td className="num px-3 py-2 text-[13px] text-sub">{s.sample_id}</td>
                    <td className="max-w-[460px] truncate px-3 py-2 text-[13px] text-ink" title={s.input}>
                      {s.input}
                    </td>
                    <td className="max-w-[160px] truncate px-3 py-2 text-[13px] text-sub" title={s.target}>
                      {s.target || "–"}
                    </td>
                    <td className={`max-w-[160px] truncate px-3 py-2 text-[13px] font-medium ${s.error || !s.correct ? "text-bad" : "text-ok"}`}>{verdict(run, s)}</td>
                    <td className="num px-3 py-2 text-right text-[13px] text-sub">{s.probability === null ? "–" : s.probability.toFixed(3)}</td>
                    <td className="num px-3 py-2 text-right text-[13px] text-sub">{s.latency_ms === null ? "–" : ms(s.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data && items.length === 0 && <p className="px-3 py-3 text-sm text-faint">No samples match.</p>}
          </div>
          <div className="mt-3 flex items-center justify-between text-sm text-sub">
            <span className="num">{data ? `${data.total} sample${data.total === 1 ? "" : "s"}` : ""}</span>
            <div className="flex items-center gap-2">
              <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="btn btn-secondary h-8">
                Prev
              </button>
              <span className="num">
                {page} / {pages}
              </span>
              <button disabled={page >= pages} onClick={() => setPage(page + 1)} className="btn btn-secondary h-8">
                Next
              </button>
            </div>
          </div>
        </div>
        <aside className="glass-dense p-4 xl:max-h-[720px] xl:overflow-y-auto">{selected && <SamplePanel run={run} s={selected} />}</aside>
      </div>
    </Card>
  );
}
