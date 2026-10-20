// Typed client for the Owl Harness API (see backend/src/owl/api.py).

export type Metrics = Record<string, number>;

export interface RunSummary {
  id: string;
  task: string;
  model: string;
  provider: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  dataset_hash: string;
  owl_version: string;
  positive_class: string | null;
  metrics: Metrics;
  /** 95% Wilson interval for accuracy, or null when the run has no samples. */
  accuracy_ci: [number, number] | null;
}

export interface RunConfig {
  task: string;
  model: string;
  model_args: Record<string, unknown>;
  generate: { temperature: number | null; max_tokens: number | null };
  positive_class: string | null;
  limit: number | null;
  max_concurrency: number;
  timeout_s: number;
  retries: number;
  system_prompt: string | null;
}

export interface HistogramBin {
  lo: number;
  hi: number;
  count: number;
}

export interface RunDetail extends RunSummary {
  config: RunConfig;
  sample_counts: { total: number; errors: number };
  has_probabilities: boolean;
  latency_histogram: HistogramBin[];
}

export interface SampleItem {
  run_id: string;
  sample_id: string;
  input: string;
  target: string;
  output: string | null;
  pred: string | null;
  correct: boolean;
  probability: number | null;
  latency_ms: number | null;
  stop_reason: string | null;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface SamplePage {
  total: number;
  page: number;
  page_size: number;
  items: SampleItem[];
}

export type SampleFilter =
  | "all"
  | "correct"
  | "incorrect"
  | "false_positive"
  | "false_negative"
  | "unparseable";

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ThresholdCurve {
  positive_class: string;
  n: number;
  run_threshold: number;
  points: ThresholdPoint[];
}

export interface Disagreement {
  sample_id: string;
  input: string;
  target: string;
  preds: Record<string, string | null>;
  correct: Record<string, boolean>;
}

export interface RocCurve {
  auc: number;
  points: { fpr: number; tpr: number; threshold: number | null }[];
}

export interface PrCurve {
  average_precision: number;
  baseline: number;
  points: { recall: number; precision: number; threshold: number }[];
}

export interface ScoreBin {
  lo: number;
  hi: number;
  positive: number;
  negative: number;
}

export interface Calibration {
  brier: number;
  ece: number;
  bins: { mean_probability: number; fraction_positive: number; n: number }[];
}

export interface LengthSlice {
  min_chars: number;
  max_chars: number;
  n: number;
  accuracy: number;
  ci_low: number;
  ci_high: number;
}

export interface Analysis {
  roc: RocCurve | null;
  pr: PrCurve | null;
  score_distribution: ScoreBin[] | null;
  calibration: Calibration | null;
  length_slices: LengthSlice[];
  latency_points: { index: number; latency_ms: number; correct: boolean }[];
  latency_by_outcome: {
    correct: { n: number; median_ms: number } | null;
    incorrect: { n: number; median_ms: number } | null;
  };
}

export interface PairwiseStat {
  a: string;
  b: string;
  agreement: number;
  a_only_correct: number;
  b_only_correct: number;
  mcnemar_p: number;
}

export interface Comparison {
  runs: RunSummary[];
  sample_ids: string[];
  /** Per run: was each shared sample (in `sample_ids` order) answered correctly. */
  outcomes: Record<string, boolean[]>;
  coverage: { all_correct: number; all_wrong: number; any_correct: number };
  pairwise: PairwiseStat[];
  roc: Record<string, RocCurve>;
  shared_samples: number;
  total_disagreements: number;
  disagreements: Disagreement[];
}

// --- what can be run ----------------------------------------------------------

export interface ArgSpec {
  name: string;
  type: "string" | "number" | "integer" | "boolean";
  default: string | number | boolean | null;
  description: string;
  affects_models: boolean;
}

export interface ProviderInfo {
  name: string;
  description: string;
  max_concurrency: number;
  binary_only: boolean;
  args: ArgSpec[];
  error?: string;
}

export interface ProviderModels {
  models: string[];
  error: string | null;
}

export interface TaskInfo {
  name: string;
  description: string;
  binary: boolean;
  positive_class: string | null;
  expected_size: number | null;
  system_prompt: string | null;
  temperature: number | null;
  max_tokens: number | null;
}

export interface AppConfig {
  version: string;
  binary_answer_instruction: string;
  max_upload_bytes: number;
  default_max_tokens: { binary: number; exact_match: number };
}

export interface DatasetInfo {
  id: string;
  name: string;
  filename: string;
  format: string;
  rows: number;
  columns: string[];
  size_bytes: number;
  created_at: string;
  preview?: Record<string, string>[];
}

export interface ColumnValues {
  distinct: number;
  values: { value: string; count: number }[];
}

export interface CustomTaskSpec {
  name: string;
  dataset_id: string;
  input_column: string;
  target_column: string;
  kind: "binary" | "exact_match";
  positive_label?: string | null;
  question?: string | null;
  system_prompt?: string | null;
  case_sensitive?: boolean;
  temperature?: number | null;
  max_tokens?: number | null;
}

export interface RunRequest {
  task?: string;
  custom?: CustomTaskSpec;
  models: { provider: string; model: string; args: Record<string, unknown> }[];
  limit?: number | null;
  max_concurrency?: number | null;
  timeout_s?: number;
  retries?: number;
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobItem {
  model: string;
  status: "queued" | "running" | "completed" | "completed_with_errors" | "failed" | "cancelled";
  run_id: string | null;
  done: number;
  total: number;
  correct: number;
  errors: number;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  accuracy: number | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  task: string;
  description: string;
  created_at: string;
  finished_at: string | null;
  cancel_requested: boolean;
  limit: number | null;
  items: JobItem[];
}

export const isActive = (job: Job): boolean => job.status === "queued" || job.status === "running";

// --- transport -----------------------------------------------------------------

async function failure(res: Response): Promise<Error> {
  let detail: unknown = res.statusText;
  try {
    detail = (await res.json()).detail ?? detail;
  } catch {
    /* keep status text */
  }
  // FastAPI validation errors arrive as a list of {loc, msg}
  const text = Array.isArray(detail)
    ? detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`).join("; ")
    : String(detail);
  return new Error(`${res.status}: ${text}`);
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url);
  if (!res.ok) throw await failure(res);
  return res.json() as Promise<T>;
}

async function send<T>(method: "POST" | "DELETE", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res);
  return res.status === 204 ? (undefined as T) : (res.json() as Promise<T>);
}

export const api = {
  runs: () => get<RunSummary[]>("/api/runs"),
  run: (id: string) => get<RunDetail>(`/api/runs/${id}`),
  deleteRun: (id: string) => send<void>("DELETE", `/api/runs/${id}`),
  samples: (id: string, p: { filter: SampleFilter; q?: string; page: number; page_size: number }) =>
    get<SamplePage>(`/api/runs/${id}/samples`, p),
  thresholdCurve: (id: string) => get<ThresholdCurve>(`/api/runs/${id}/threshold-curve`),
  analysis: (id: string) => get<Analysis>(`/api/runs/${id}/analysis`),
  exportUrl: (id: string) => `/api/runs/${id}/export.csv`,
  compare: (ids: string[]) => get<Comparison>("/api/compare", { run_ids: ids.join(",") }),

  config: () => get<AppConfig>("/api/config"),
  providers: () => get<ProviderInfo[]>("/api/providers"),
  providerModels: (name: string, args: Record<string, string>) => get<ProviderModels>(`/api/providers/${name}/models`, args),
  tasks: () => get<TaskInfo[]>("/api/tasks"),

  datasets: () => get<DatasetInfo[]>("/api/datasets"),
  dataset: (id: string) => get<DatasetInfo>(`/api/datasets/${id}`),
  datasetValues: (id: string, column: string) => get<ColumnValues>(`/api/datasets/${id}/values`, { column }),
  deleteDataset: (id: string) => send<void>("DELETE", `/api/datasets/${id}`),
  /** Raw body upload: no multipart, and a non-simple content type so other sites cannot post here. */
  async uploadDataset(file: File): Promise<DatasetInfo> {
    const url = new URL("/api/datasets", window.location.origin);
    url.searchParams.set("filename", file.name);
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
    if (!res.ok) throw await failure(res);
    return res.json() as Promise<DatasetInfo>;
  },

  jobs: () => get<Job[]>("/api/jobs"),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),
  startJob: (req: RunRequest) => send<Job>("POST", "/api/jobs", req),
  cancelJob: (id: string) => send<Job>("POST", `/api/jobs/${id}/cancel`),
};
