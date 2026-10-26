// Wizard step 1: what to test.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type AppConfig, type CustomKind, type CustomTaskSpec, type DatasetInfo, type RunRequest, type TaskInfo } from "../../api";
import { MODE_LABEL, renderTemplate, useAsync } from "../../lib";
import { Chip, ErrorNote, FormField, Tabs } from "../ui";

export interface TestState {
  source: "builtin" | "dataset";
  builtin: string;
  datasetId: string;
  testName: string;
  inputColumn: string;
  targetColumn: string; // "" = no reference (judge only)
  kind: CustomKind;
  positiveLabel: string;
  question: string;
  labels: string[];
  labelsTouched: boolean;
  normalization: { case: boolean; whitespace: boolean; punctuation: boolean };
  rubric: "passfail" | "scale";
  rubricTemplate: string;
  rubricTouched: boolean;
  systemPrompt: string;
  systemTouched: boolean;
  userTemplate: string;
  temperatureHint?: number | null;
}

export const initialTest: TestState = {
  source: "builtin",
  builtin: "",
  datasetId: "",
  testName: "",
  inputColumn: "",
  targetColumn: "",
  kind: "binary",
  positiveLabel: "",
  question: "",
  labels: [],
  labelsTouched: false,
  normalization: { case: true, whitespace: true, punctuation: false },
  rubric: "passfail",
  rubricTemplate: "",
  rubricTouched: false,
  systemPrompt: "",
  systemTouched: false,
  userTemplate: "{input}",
};

const INPUT_GUESS = ["text", "input", "prompt", "question", "sentence", "content", "review", "message", "document"];
const TARGET_GUESS = ["label", "target", "answer", "best_answer", "class", "category", "y", "expected", "output", "sentiment", "topic"];
const POSITIVE_GUESS = ["1", "true", "yes", "positive", "spam", "toxic", "injection", "malicious"];
const pick = (columns: string[], guesses: string[], fallback: string) => columns.find((c) => guesses.includes(c.toLowerCase())) ?? fallback;

/** The system prompt the run will use when the user has not edited it. */
export function defaultSystem(t: TestState, config: AppConfig): string {
  if (t.kind === "binary") return t.question.trim() ? `${t.question.trim()}\n${config.binary_answer_instruction}` : "";
  if (t.kind === "multiclass") return config.multiclass_system_prompt;
  return "";
}
export const effectiveSystem = (t: TestState, config: AppConfig) => (t.systemTouched ? t.systemPrompt : defaultSystem(t, config));
export const effectiveRubric = (t: TestState, config: AppConfig) =>
  t.rubricTouched ? t.rubricTemplate : config.rubrics[t.rubric][t.targetColumn ? "with_reference" : "without_reference"];

export function testProblems(t: TestState, dataset: DatasetInfo | undefined): string[] {
  if (t.source === "builtin") return t.builtin ? [] : ["Choose a test."];
  if (!dataset) return ["Choose a dataset, or upload or import one."];
  const out: string[] = [];
  if (!t.inputColumn) out.push("Choose the input column.");
  if (t.kind !== "judge" && !t.targetColumn) out.push("Choose the target column.");
  if (t.targetColumn && t.targetColumn === t.inputColumn) out.push("The input and target columns must be different.");
  if (!t.testName.trim()) out.push("Give the test a name.");
  if (t.kind === "binary") {
    if (!t.positiveLabel) out.push("Choose which target value means yes.");
    if (!t.question.trim()) out.push("Write the yes/no question.");
  }
  if (t.kind === "multiclass" && t.labelsTouched && t.labels.length < 2) out.push("Keep at least two labels.");
  if (!t.userTemplate.trim()) out.push("The user message template cannot be empty.");
  return out;
}

export function testToRequest(t: TestState, config: AppConfig): Pick<RunRequest, "task" | "custom"> {
  if (t.source === "builtin") return { task: t.builtin };
  const custom: CustomTaskSpec = {
    name: t.testName.trim(),
    dataset_id: t.datasetId,
    input_column: t.inputColumn,
    target_column: t.targetColumn || null,
    kind: t.kind,
    user_template: t.userTemplate,
    system_prompt: t.systemTouched ? t.systemPrompt : t.kind === "binary" || t.kind === "multiclass" ? null : t.systemPrompt || null,
  };
  if (t.kind === "binary") {
    custom.positive_label = t.positiveLabel;
    custom.question = t.question.trim();
  }
  if (t.kind === "multiclass" && t.labelsTouched) custom.labels = t.labels;
  if (t.kind === "exact_match") custom.normalization = t.normalization;
  if (t.kind === "judge") custom.judge = { rubric: t.rubric, template: t.rubricTouched ? effectiveRubric(t, config) : null, provider: null, model: null };
  return { custom };
}

export default function TestStep({
  value: t,
  onChange,
  tasks,
  datasets,
  config,
  reloadDatasets,
}: {
  value: TestState;
  onChange: (patch: Partial<TestState>) => void;
  tasks: TaskInfo[];
  datasets: DatasetInfo[];
  config: AppConfig;
  reloadDatasets: () => void;
}) {
  const task = tasks.find((x) => x.name === t.builtin);
  const dataset = datasets.find((d) => d.id === t.datasetId);
  const detail = useAsync(() => (t.datasetId ? api.dataset(t.datasetId, 5) : Promise.resolve(undefined)), [t.datasetId]);
  const values = useAsync(
    () => (t.datasetId && t.targetColumn && (t.kind === "binary" || t.kind === "multiclass") ? api.datasetValues(t.datasetId, t.targetColumn) : Promise.resolve(undefined)),
    [t.datasetId, t.targetColumn, t.kind],
  );
  const [uploadError, setUploadError] = useState<string>();
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const [previewRow, setPreviewRow] = useState(0);
  const [newLabel, setNewLabel] = useState("");

  const chooseDataset = (d: DatasetInfo | undefined) => {
    if (!d) return onChange({ datasetId: "", inputColumn: "", targetColumn: "", positiveLabel: "", testName: "", labels: [], labelsTouched: false });
    const input = pick(d.columns, INPUT_GUESS, d.columns[0]);
    const target = pick(d.columns.filter((c) => c !== input), TARGET_GUESS, d.columns.filter((c) => c !== input).pop() ?? "");
    onChange({ datasetId: d.id, inputColumn: input, targetColumn: target, positiveLabel: "", testName: d.name, labels: [], labelsTouched: false });
    setPreviewRow(0);
  };

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    setUploadError(undefined);
    try {
      const d = await api.uploadDataset(file);
      reloadDatasets();
      chooseDataset(d);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  // Defaults that follow the data: the likely "yes" value, and the label set for multi-class.
  const valueList = values.data?.values ?? [];
  useEffect(() => {
    if (t.kind === "binary" && values.data && t.positiveLabel === "") {
      const guess = valueList.find((v) => POSITIVE_GUESS.includes(v.value.toLowerCase()))?.value;
      if (guess) onChange({ positiveLabel: guess });
    }
    if (t.kind === "multiclass" && values.data && !t.labelsTouched) {
      onChange({ labels: valueList.map((v) => v.value).filter(Boolean).sort() });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values.data, t.kind]);

  const rows = detail.data?.preview ?? [];
  const row = rows[Math.min(previewRow, rows.length - 1)] ?? {};
  const placeholderValues = useMemo(
    () => ({ ...row, input: row[t.inputColumn] ?? "", labels: t.labels.join(", ") }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(row), t.inputColumn, t.labels.join("|")],
  );
  const system = effectiveSystem(t, config);
  const variables = [...(dataset?.columns ?? []), "input", ...(t.kind === "multiclass" ? ["labels"] : [])];

  const builtins = [...tasks].sort((a, b) => Number(a.example) - Number(b.example) || a.name.localeCompare(b.name));

  return (
    <div className="space-y-5">
      <Tabs
        tabs={[
          { id: "builtin", label: "Built-in and example tests" },
          { id: "dataset", label: `Dataset library (${datasets.length})` },
        ]}
        active={t.source}
        onChange={(id) => onChange({ source: id })}
      />

      {t.source === "builtin" && (
        <div className="grid gap-3 md:grid-cols-2">
          {builtins.map((x) => (
            <button
              key={x.name}
              type="button"
              onClick={() => onChange({ builtin: x.name })}
              className={`glass-dense p-4 text-left transition-colors ${t.builtin === x.name ? "border-emerald bg-emerald/15" : "hover:bg-white"}`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold text-ink">{x.name}</span>
                {x.example && <Chip tone="accent">Example</Chip>}
                <Chip>{MODE_LABEL[x.scoring_mode]}</Chip>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-sub">{x.description}</p>
            </button>
          ))}
          {task?.system_prompt && (
            <div className="md:col-span-2">
              <div className="label mb-1.5">Prompt sent to the model</div>
              <pre className="glass-dense whitespace-pre-wrap p-3 font-mono text-xs leading-relaxed text-sub">{task.system_prompt}</pre>
            </div>
          )}
        </div>
      )}

      {t.source === "dataset" && (
        <div className="space-y-5">
          <div className="flex flex-wrap items-end gap-3">
            <FormField label="Dataset" className="min-w-[260px] flex-1">
              <select className="input" value={t.datasetId} onChange={(e) => chooseDataset(datasets.find((d) => d.id === e.target.value))}>
                <option value="">{datasets.length ? "Choose a dataset…" : "No datasets yet"}</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · {d.rows.toLocaleString()} rows{d.source ? " · from the Hub" : ""}
                  </option>
                ))}
              </select>
            </FormField>
            <input ref={fileInput} type="file" accept=".csv,.tsv,.jsonl,.json,.parquet" className="hidden" aria-label="Upload a dataset file" onChange={(e) => upload(e.target.files?.[0])} />
            <button type="button" className="btn btn-secondary" disabled={uploading} onClick={() => fileInput.current?.click()}>
              {uploading ? "Uploading…" : "Upload a file"}
            </button>
            <Link to="/datasets" className="btn btn-quiet">
              Import from the Hub →
            </Link>
          </div>
          {uploadError && <ErrorNote message={uploadError} />}

          {dataset && (
            <>
              <div className="grid gap-3 md:grid-cols-3">
                <FormField label="Input column" hint="The text the model is asked about: {input} in templates.">
                  <select className="input" value={t.inputColumn} onChange={(e) => onChange({ inputColumn: e.target.value })}>
                    {dataset.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label={t.kind === "judge" ? "Reference answer column" : "Target column"} hint={t.kind === "judge" ? "Optional for judge-graded tests." : "The correct answer."}>
                  <select className="input" value={t.targetColumn} onChange={(e) => onChange({ targetColumn: e.target.value, positiveLabel: "", labelsTouched: false })}>
                    {t.kind === "judge" && <option value="">(no reference answer)</option>}
                    {dataset.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label="Test name" hint="Shown in runs and leaderboards.">
                  <input className="input" value={t.testName} maxLength={60} onChange={(e) => onChange({ testName: e.target.value })} />
                </FormField>
              </div>

              <div>
                <div className="label mb-2">Scoring mode</div>
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                  {(
                    [
                      ["binary", "Yes / no", "Answers 1 or 0. Precision, recall, F1, MCC; ROC and threshold analysis for models with probabilities."],
                      ["multiclass", "Multi-class", "Answers with one label. Macro- and weighted-F1, per-class scores and an N×N confusion matrix."],
                      ["exact_match", "Exact match", "The whole answer must equal the target, after the normalisation you choose."],
                      ["judge", "Judge-graded", "Free text graded by a judge model against a rubric: pass/fail or 1–5."],
                    ] as const
                  ).map(([id, title, text]) => (
                    <label
                      key={id}
                      className={`glass-dense flex cursor-pointer gap-3 p-3.5 transition-colors ${t.kind === id ? "border-emerald bg-emerald/15" : "hover:bg-white"}`}
                    >
                      <input type="radio" name="kind" className="mt-0.5" checked={t.kind === id} onChange={() => onChange({ kind: id, systemTouched: false })} />
                      <span>
                        <span className="block text-[13px] font-semibold text-ink">{title}</span>
                        <span className="mt-1 block text-xs leading-relaxed text-sub">{text}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {t.kind === "binary" && (
                <div className="grid gap-3 md:grid-cols-[280px_1fr]">
                  <FormField label="Target value meaning yes" hint={values.data ? `${values.data.distinct} distinct values; all others count as no.` : undefined}>
                    <select className="input" value={t.positiveLabel} onChange={(e) => onChange({ positiveLabel: e.target.value })}>
                      <option value="">Choose a value…</option>
                      {valueList.map((v) => (
                        <option key={v.value} value={v.value}>
                          {v.value === "" ? "(empty)" : v.value} ({v.count})
                        </option>
                      ))}
                    </select>
                  </FormField>
                  <FormField label="Yes/no question" hint="Unprompted models (classifiers, decision models) receive just this question.">
                    <input className="input" value={t.question} maxLength={4000} placeholder="For example: Is this review positive?" onChange={(e) => onChange({ question: e.target.value })} />
                  </FormField>
                </div>
              )}

              {t.kind === "multiclass" && (
                <div>
                  <div className="label mb-2">Labels</div>
                  <div className="flex flex-wrap items-center gap-2">
                    {t.labels.map((l) => (
                      <span key={l} className="inline-flex h-7 items-center gap-1.5 rounded-full border border-wash/15 bg-wash/[0.06] pl-3 pr-1.5 text-xs text-ink">
                        {l}
                        <button
                          type="button"
                          aria-label={`Remove ${l}`}
                          className="grid h-5 w-5 place-items-center rounded-full text-faint hover:bg-wash/10 hover:text-ink"
                          onClick={() => onChange({ labels: t.labels.filter((x) => x !== l), labelsTouched: true })}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                    <input
                      className="input h-7 w-40 text-xs"
                      placeholder="Add a label, then Enter"
                      value={newLabel}
                      onChange={(e) => setNewLabel(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && newLabel.trim() && !t.labels.some((l) => l.toLowerCase() === newLabel.trim().toLowerCase())) {
                          e.preventDefault();
                          onChange({ labels: [...t.labels, newLabel.trim()], labelsTouched: true });
                          setNewLabel("");
                        }
                      }}
                    />
                  </div>
                  <p className="mt-2 text-xs text-faint">
                    From the distinct values of “{t.targetColumn}”{values.data && values.data.distinct > valueList.length ? ` (first ${valueList.length} of ${values.data.distinct} shown)` : ""}. In templates,{" "}
                    <code className="text-sub">{"{labels}"}</code> lists them. An answer must name exactly one label; otherwise it is unparseable and counts as wrong.
                  </p>
                </div>
              )}

              {t.kind === "exact_match" && (
                <div>
                  <div className="label mb-2">Ignore when comparing</div>
                  <div className="flex flex-wrap gap-5 text-[13px] text-ink">
                    {(
                      [
                        ["case", "letter case"],
                        ["whitespace", "extra spaces and line breaks"],
                        ["punctuation", "punctuation"],
                      ] as const
                    ).map(([k, label]) => (
                      <label key={k} className="flex items-center gap-2">
                        <input type="checkbox" checked={t.normalization[k]} onChange={(e) => onChange({ normalization: { ...t.normalization, [k]: e.target.checked } })} />
                        {label}
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {t.kind === "judge" && (
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-5 text-[13px] text-ink">
                    {(
                      [
                        ["passfail", "Pass / fail"],
                        ["scale", "Score from 1 to 5"],
                      ] as const
                    ).map(([k, label]) => (
                      <label key={k} className="flex items-center gap-2">
                        <input type="radio" name="rubric" checked={t.rubric === k} onChange={() => onChange({ rubric: k, rubricTouched: false })} />
                        {label}
                      </label>
                    ))}
                  </div>
                  <FormField
                    label="Rubric"
                    hint={
                      <>
                        Sent to the judge. Variables: <code className="text-sub">{"{input}"}</code>, <code className="text-sub">{"{target}"}</code> (reference) and{" "}
                        <code className="text-sub">{"{output}"}</code> (the answer). It must end by asking for {t.rubric === "passfail" ? "PASS or FAIL" : "“Score: N”"}. You pick the judge model in the next step.
                      </>
                    }
                  >
                    <textarea className="input min-h-[160px] font-mono text-xs" value={effectiveRubric(t, config)} onChange={(e) => onChange({ rubricTemplate: e.target.value, rubricTouched: true })} />
                  </FormField>
                </div>
              )}

              <div className="grid gap-4 xl:grid-cols-2">
                <div className="space-y-3">
                  <FormField
                    label={
                      <span className="flex items-center justify-between">
                        System prompt
                        {t.systemTouched && (
                          <button type="button" className="text-[11px] font-medium normal-case tracking-normal text-accent hover:underline" onClick={() => onChange({ systemTouched: false })}>
                            Reset to default
                          </button>
                        )}
                      </span>
                    }
                    hint={t.kind === "binary" ? "Built from the question unless you edit it." : t.kind === "multiclass" ? undefined : "Optional."}
                  >
                    <textarea className="input min-h-[96px] font-mono text-xs" value={system} onChange={(e) => onChange({ systemPrompt: e.target.value, systemTouched: true })} />
                  </FormField>
                  <FormField label="User message template">
                    <textarea className="input min-h-[60px] font-mono text-xs" value={t.userTemplate} onChange={(e) => onChange({ userTemplate: e.target.value })} />
                  </FormField>
                  <div className="flex flex-wrap items-center gap-1.5 text-xs text-faint">
                    Variables:
                    {variables.map((v) => (
                      <code key={v} className="rounded bg-wash/[0.07] px-1.5 py-0.5 text-sub">{`{${v}}`}</code>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="mb-1.5 flex items-center justify-between">
                    <span className="label">Live preview</span>
                    {rows.length > 1 && (
                      <span className="flex items-center gap-1 text-xs text-faint">
                        row
                        {rows.map((_, i) => (
                          <button key={i} type="button" onClick={() => setPreviewRow(i)} className={`num h-6 w-6 rounded-md ${i === previewRow ? "bg-wash/15 text-ink" : "hover:bg-wash/10"}`}>
                            {i + 1}
                          </button>
                        ))}
                      </span>
                    )}
                  </div>
                  <div className="glass-dense space-y-3 p-3">
                    {system.trim() && (
                      <div>
                        <div className="text-[11px] font-semibold uppercase tracking-wide text-faint">system</div>
                        <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-sub">{renderTemplate(system, placeholderValues)}</pre>
                      </div>
                    )}
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-faint">user</div>
                      <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ink">{renderTemplate(t.userTemplate, placeholderValues)}</pre>
                    </div>
                    {t.targetColumn && (
                      <div className="text-xs text-faint">
                        expected: <span className="text-sub">{row[t.targetColumn] ?? ""}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
