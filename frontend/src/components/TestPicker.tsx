// Step 1 of a new run: choose what to test. Either a built-in test, or one of your own datasets
// with a column mapping, a scoring mode and a prompt.
import { useEffect, useRef, useState } from "react";
import { api, type AppConfig, type CustomTaskSpec, type DatasetInfo, type RunRequest, type TaskInfo } from "../api";
import { useAsync } from "../lib";
import { btnSecondary, Card, ErrorNote, FormField, inputCls } from "./ui";

export interface TestState {
  source: "builtin" | "dataset";
  builtin: string;
  datasetId: string;
  inputColumn: string;
  targetColumn: string;
  kind: "binary" | "exact_match";
  positiveLabel: string;
  question: string;
  editPrompt: boolean;
  systemPrompt: string;
  caseSensitive: boolean;
  testName: string;
  temperature: string;
  maxTokens: string;
}

export const initialTest: TestState = {
  source: "builtin",
  builtin: "",
  datasetId: "",
  inputColumn: "",
  targetColumn: "",
  kind: "binary",
  positiveLabel: "",
  question: "",
  editPrompt: false,
  systemPrompt: "",
  caseSensitive: false,
  testName: "",
  temperature: "0",
  maxTokens: "",
};

const INPUT_GUESS = ["text", "input", "prompt", "question", "sentence", "content", "review", "message", "document"];
const TARGET_GUESS = ["label", "target", "answer", "class", "category", "y", "expected", "output", "sentiment"];
const POSITIVE_GUESS = ["1", "true", "yes", "positive", "spam", "toxic", "injection", "malicious"];

const pick = (columns: string[], guesses: string[], fallback: string): string =>
  columns.find((c) => guesses.includes(c.toLowerCase())) ?? fallback;

/** The prompt a yes/no test sends: the question plus the answer-format line. */
export const binaryPrompt = (question: string, config: AppConfig): string => `${question.trim()}\n${config.binary_answer_instruction}`;

const optionalNumber = (text: string): number | undefined | "invalid" => {
  if (text.trim() === "") return undefined;
  const n = Number(text);
  return Number.isFinite(n) ? n : "invalid";
};

/** Everything that stops the run from starting, in plain words. */
export function testProblems(t: TestState, dataset: DatasetInfo | undefined): string[] {
  const out: string[] = [];
  if (t.source === "builtin") {
    if (!t.builtin) out.push("Choose a built-in test.");
    return out;
  }
  if (!t.datasetId || !dataset) return ["Upload or choose a dataset."];
  if (!t.inputColumn || !t.targetColumn) out.push("Choose the input and target columns.");
  else if (t.inputColumn === t.targetColumn) out.push("The input and target columns must be different.");
  if (!t.testName.trim()) out.push("Give the test a name.");
  if (t.kind === "binary") {
    if (!t.positiveLabel) out.push("Choose which target value means yes.");
    if (!t.question.trim()) out.push("Write the yes/no question the model should answer.");
  }
  if (optionalNumber(t.temperature) === "invalid") out.push("Temperature must be a number.");
  if (optionalNumber(t.maxTokens) === "invalid") out.push("Max tokens must be a number.");
  return out;
}

export function testToRequest(t: TestState): Pick<RunRequest, "task" | "custom"> {
  if (t.source === "builtin") return { task: t.builtin };
  const temperature = optionalNumber(t.temperature);
  const maxTokens = optionalNumber(t.maxTokens);
  const custom: CustomTaskSpec = {
    name: t.testName.trim(),
    dataset_id: t.datasetId,
    input_column: t.inputColumn,
    target_column: t.targetColumn,
    kind: t.kind,
    temperature: temperature === "invalid" ? 0 : (temperature ?? 0),
    max_tokens: maxTokens === "invalid" ? null : (maxTokens ?? null),
  };
  if (t.kind === "binary") {
    custom.positive_label = t.positiveLabel;
    custom.question = t.question.trim();
    custom.system_prompt = t.editPrompt && t.systemPrompt.trim() ? t.systemPrompt : null;
  } else {
    custom.system_prompt = t.systemPrompt.trim() ? t.systemPrompt : null;
    custom.case_sensitive = t.caseSensitive;
  }
  return { custom };
}

export default function TestPicker({
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
  const selected = datasets.find((d) => d.id === t.datasetId);
  const detail = useAsync(() => (t.datasetId ? api.dataset(t.datasetId) : Promise.resolve(undefined)), [t.datasetId]);
  const values = useAsync(
    () => (t.datasetId && t.targetColumn && t.kind === "binary" ? api.datasetValues(t.datasetId, t.targetColumn) : Promise.resolve(undefined)),
    [t.datasetId, t.targetColumn, t.kind],
  );

  const [uploadError, setUploadError] = useState<string>();
  const [uploading, setUploading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const chooseDataset = (d: DatasetInfo | undefined) => {
    if (!d) {
      onChange({ datasetId: "", inputColumn: "", targetColumn: "", positiveLabel: "", testName: "" });
      return;
    }
    const input = pick(d.columns, INPUT_GUESS, d.columns[0]);
    const target = pick(d.columns, TARGET_GUESS, d.columns.filter((c) => c !== input).pop() ?? d.columns[0]);
    onChange({ datasetId: d.id, inputColumn: input, targetColumn: target, positiveLabel: "", testName: d.name });
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

  const removeDataset = async () => {
    try {
      await api.deleteDataset(t.datasetId);
      setConfirmDelete(false);
      chooseDataset(undefined);
      reloadDatasets();
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e));
    }
  };

  // Once the target column's values are known, preselect the one that most likely means "yes".
  const valueList = values.data?.values ?? [];
  const guessed = valueList.find((v) => POSITIVE_GUESS.includes(v.value.toLowerCase()))?.value;
  useEffect(() => {
    if (t.kind === "binary" && values.data && t.positiveLabel === "" && guessed) onChange({ positiveLabel: guessed });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values.data, t.kind, t.positiveLabel, guessed]);

  const tab = (id: TestState["source"], label: string) => (
    <button
      type="button"
      onClick={() => onChange({ source: id })}
      className={`-mb-px border-b-2 px-0.5 pb-2 text-sm font-medium ${
        t.source === id ? "border-accent text-accent" : "border-transparent text-slate-600 hover:text-slate-900"
      }`}
    >
      {label}
    </button>
  );

  return (
    <Card title="1. What to test">
      <div className="mb-4 flex gap-6 border-b border-slate-200">
        {tab("builtin", "Built-in tests")}
        {tab("dataset", `My datasets (${datasets.length})`)}
      </div>

      {t.source === "builtin" && (
        <div className="space-y-3">
          <FormField label="Test">
            <select className={inputCls} value={t.builtin} onChange={(e) => onChange({ builtin: e.target.value })}>
              <option value="">Choose a test…</option>
              {tasks.map((x) => (
                <option key={x.name} value={x.name}>
                  {x.name}
                </option>
              ))}
            </select>
          </FormField>
          {task && (
            <div className="space-y-2 text-sm text-slate-700">
              <p>{task.description}</p>
              <p className="text-xs text-slate-500">
                {task.binary ? "Yes/no test: reports precision, recall, F1 and ROC." : "Free-text test."}
                {task.expected_size !== null && ` ${task.expected_size} samples.`}
              </p>
              {task.system_prompt && (
                <div>
                  <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Prompt sent to the model</div>
                  <pre className="whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-3 font-mono text-xs leading-5 text-slate-800">
                    {task.system_prompt}
                  </pre>
                </div>
              )}
            </div>
          )}
          <p className="text-xs text-slate-500">
            Want to test your own data or a different question? Use <button type="button" className="text-accent hover:underline" onClick={() => onChange({ source: "dataset" })}>My datasets</button>.
          </p>
        </div>
      )}

      {t.source === "dataset" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-[240px] flex-1">
              <FormField label="Dataset">
                <select
                  className={inputCls}
                  value={t.datasetId}
                  onChange={(e) => chooseDataset(datasets.find((d) => d.id === e.target.value))}
                >
                  <option value="">{datasets.length ? "Choose a dataset…" : "No datasets yet. Upload one."}</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.rows} rows)
                    </option>
                  ))}
                </select>
              </FormField>
            </div>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,.tsv,.jsonl,.json"
              className="hidden"
              aria-label="Upload a dataset file"
              onChange={(e) => upload(e.target.files?.[0])}
            />
            <button type="button" className={btnSecondary} disabled={uploading} onClick={() => fileInput.current?.click()}>
              {uploading ? "Uploading…" : "Upload a file"}
            </button>
            {selected && !confirmDelete && (
              <button type="button" className={btnSecondary} onClick={() => setConfirmDelete(true)}>
                Delete dataset
              </button>
            )}
            {selected && confirmDelete && (
              <span className="flex items-center gap-2 text-sm">
                <span className="text-slate-700">Delete “{selected.name}”? Runs already made are kept.</span>
                <button type="button" className="font-medium text-danger hover:underline" onClick={removeDataset}>
                  Delete
                </button>
                <button type="button" className="text-slate-600 hover:underline" onClick={() => setConfirmDelete(false)}>
                  Keep
                </button>
              </span>
            )}
          </div>
          <p className="-mt-2 text-xs text-slate-500">
            CSV, TSV, JSONL or JSON (a list of objects), UTF-8, up to {Math.round(config.max_upload_bytes / 1024 / 1024)} MB. Files stay on this
            machine.
          </p>
          {uploadError && <ErrorNote message={uploadError} />}

          {selected && (
            <>
              {detail.data?.preview && <Preview rows={detail.data.preview} columns={selected.columns} total={selected.rows} />}

              <div className="grid gap-3 md:grid-cols-3">
                <FormField label="Input column" hint="The text the model sees.">
                  <select className={inputCls} value={t.inputColumn} onChange={(e) => onChange({ inputColumn: e.target.value })}>
                    {selected.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label="Target column" hint="The correct answer.">
                  <select
                    className={inputCls}
                    value={t.targetColumn}
                    onChange={(e) => onChange({ targetColumn: e.target.value, positiveLabel: "" })}
                  >
                    {selected.columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </FormField>
                <FormField label="Test name" hint="Shown in the runs list.">
                  <input className={inputCls} value={t.testName} maxLength={60} onChange={(e) => onChange({ testName: e.target.value })} />
                </FormField>
              </div>

              <fieldset>
                <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">How to score</legend>
                <div className="grid gap-2 md:grid-cols-2">
                  {(
                    [
                      ["binary", "Yes / no question", "The model answers 1 or 0. You get accuracy, precision, recall, F1, confusion matrix and, for models with probabilities, ROC and threshold analysis."],
                      ["exact_match", "Exact match", "The model's whole answer must equal the target text (ignoring case and surrounding spaces by default)."],
                    ] as const
                  ).map(([id, title, text]) => (
                    <label
                      key={id}
                      className={`flex cursor-pointer gap-3 rounded border p-3 ${t.kind === id ? "border-accent bg-accent-soft/40" : "border-slate-200 hover:bg-slate-50"}`}
                    >
                      <input type="radio" name="kind" className="mt-0.5 accent-[#1d4ed8]" checked={t.kind === id} onChange={() => onChange({ kind: id })} />
                      <span>
                        <span className="block text-sm font-medium text-slate-900">{title}</span>
                        <span className="block text-xs text-slate-600">{text}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>

              {t.kind === "binary" && (
                <div className="space-y-3">
                  <FormField
                    label="Which target value means yes?"
                    hint={
                      values.data
                        ? `${values.data.distinct} distinct value${values.data.distinct === 1 ? "" : "s"} in “${t.targetColumn}”. Every other value counts as no.`
                        : values.error
                    }
                  >
                    <select className={inputCls} value={t.positiveLabel} onChange={(e) => onChange({ positiveLabel: e.target.value })}>
                      <option value="">Choose a value…</option>
                      {valueList.map((v) => (
                        <option key={v.value} value={v.value}>
                          {v.value === "" ? "(empty)" : v.value} ({v.count})
                        </option>
                      ))}
                    </select>
                  </FormField>
                  <FormField label="Question" hint="Written so that the answer is yes or no. Yes-answers are scored against the value chosen above.">
                    <textarea
                      className={`${inputCls} min-h-[72px]`}
                      value={t.question}
                      maxLength={4000}
                      placeholder="For example: Is this review positive?"
                      onChange={(e) => onChange({ question: e.target.value })}
                    />
                  </FormField>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Prompt sent to text models</span>
                      <label className="flex items-center gap-1.5 text-xs text-slate-600">
                        <input
                          type="checkbox"
                          className="accent-[#1d4ed8]"
                          checked={t.editPrompt}
                          onChange={(e) => onChange({ editPrompt: e.target.checked, systemPrompt: e.target.checked && !t.systemPrompt ? binaryPrompt(t.question, config) : t.systemPrompt })}
                        />
                        Edit the whole prompt
                      </label>
                    </div>
                    {t.editPrompt ? (
                      <textarea className={`${inputCls} min-h-[96px] font-mono text-xs`} value={t.systemPrompt} maxLength={8000} onChange={(e) => onChange({ systemPrompt: e.target.value })} />
                    ) : (
                      <pre className="whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-3 font-mono text-xs leading-5 text-slate-800">
                        {t.question.trim() ? binaryPrompt(t.question, config) : "(write the question first)"}
                      </pre>
                    )}
                    <p className="mt-1 text-xs text-slate-500">Models like Laya, which are not prompted, receive only the question.</p>
                  </div>
                </div>
              )}

              {t.kind === "exact_match" && (
                <div className="space-y-3">
                  <FormField label="Instructions (optional)" hint="Sent as the system message. For example: Reply with only the capital city, nothing else.">
                    <textarea className={`${inputCls} min-h-[72px]`} value={t.systemPrompt} maxLength={8000} onChange={(e) => onChange({ systemPrompt: e.target.value })} />
                  </FormField>
                  <label className="flex items-center gap-2 text-sm text-slate-700">
                    <input type="checkbox" className="accent-[#1d4ed8]" checked={t.caseSensitive} onChange={(e) => onChange({ caseSensitive: e.target.checked })} />
                    Case sensitive
                  </label>
                </div>
              )}

              <div className="grid gap-3 md:grid-cols-3">
                <FormField label="Temperature" hint="0 gives the most repeatable answers.">
                  <input className={inputCls} inputMode="decimal" value={t.temperature} onChange={(e) => onChange({ temperature: e.target.value })} />
                </FormField>
                <FormField label="Max answer tokens" hint={`Blank uses ${t.kind === "binary" ? config.default_max_tokens.binary : config.default_max_tokens.exact_match}.`}>
                  <input className={inputCls} inputMode="numeric" value={t.maxTokens} placeholder="default" onChange={(e) => onChange({ maxTokens: e.target.value })} />
                </FormField>
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  );
}

function Preview({ rows, columns, total }: { rows: Record<string, string>[]; columns: string[]; total: number }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
        Preview: first {rows.length} of {total} rows
      </div>
      <div className="overflow-x-auto rounded border border-slate-200">
        <table className="w-full border-collapse text-left">
          <thead className="border-b border-slate-200 bg-slate-50">
            <tr>
              {columns.map((c) => (
                <th key={c} className="px-3 py-1.5 text-xs font-semibold text-slate-600">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c} className="max-w-[360px] truncate px-3 py-1 text-xs text-slate-800" title={r[c]}>
                    {r[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
