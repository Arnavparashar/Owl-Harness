import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import ModelRows, { modelProblems, newRow, rowsToRequest, type ModelRow } from "../components/ModelRows";
import TestPicker, { initialTest, testProblems, testToRequest, type TestState } from "../components/TestPicker";
import { btnPrimary, Card, ErrorNote, FormField, inputCls, Loading } from "../components/ui";
import { useAsync } from "../lib";

const integerIn = (text: string, min: number, max: number): boolean => /^\d+$/.test(text) && Number(text) >= min && Number(text) <= max;

export default function NewRunPage() {
  const navigate = useNavigate();
  const [datasetsTick, setDatasetsTick] = useState(0);
  const config = useAsync(() => api.config(), []);
  const providers = useAsync(() => api.providers(), []);
  const tasks = useAsync(() => api.tasks(), []);
  const datasets = useAsync(() => api.datasets(), [datasetsTick]);

  const [test, setTest] = useState<TestState>(initialTest);
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [limit, setLimit] = useState("");
  const [concurrency, setConcurrency] = useState("");
  const [timeoutS, setTimeoutS] = useState("120");
  const [retries, setRetries] = useState("2");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();

  // Sensible starting point: the first built-in test and one Ollama (or first available) model.
  useEffect(() => {
    if (tasks.data && tasks.data.length > 0 && test.builtin === "") setTest((t) => ({ ...t, builtin: tasks.data![0].name }));
  }, [tasks.data, test.builtin]);
  useEffect(() => {
    if (providers.data && rows.length === 0) {
      const first = providers.data.find((p) => p.name === "ollama" && !p.error) ?? providers.data.find((p) => !p.error);
      if (first) setRows([newRow(first.name)]);
    }
  }, [providers.data, rows.length]);

  const loading = !config.data || !providers.data || !tasks.data || !datasets.data;
  const loadError = config.error ?? providers.error ?? tasks.error ?? datasets.error;

  if (loadError) return <ErrorNote message={loadError} />;
  if (loading) return <Loading />;

  const dataset = datasets.data!.find((d) => d.id === test.datasetId);
  const builtinTask = tasks.data!.find((t) => t.name === test.builtin);
  const isBinary = test.source === "builtin" ? (builtinTask?.binary ?? true) : test.kind === "binary";
  const sampleCount = test.source === "builtin" ? builtinTask?.expected_size : dataset?.rows;

  const problems = [...testProblems(test, dataset), ...modelProblems(rows, providers.data!)];
  rows.forEach((r, i) => {
    if (providers.data!.find((p) => p.name === r.provider)?.binary_only && !isBinary) {
      problems.push(`Model ${i + 1}: ${r.provider} only answers yes/no questions and cannot run this test.`);
    }
  });
  if (limit !== "" && !integerIn(limit, 1, 10_000_000)) problems.push("Samples to run must be a whole number of at least 1, or blank for all.");
  if (concurrency !== "" && !integerIn(concurrency, 1, 64)) problems.push("Parallel requests must be a whole number from 1 to 64, or blank for the default.");
  if (!(Number(timeoutS) > 0 && Number(timeoutS) <= 3600)) problems.push("Timeout must be between 1 and 3600 seconds.");
  if (!integerIn(retries, 0, 10)) problems.push("Retries must be a whole number from 0 to 10.");

  const start = async () => {
    setSubmitting(true);
    setError(undefined);
    try {
      const job = await api.startJob({
        ...testToRequest(test),
        models: rowsToRequest(rows, providers.data!),
        limit: limit === "" ? null : Number(limit),
        max_concurrency: concurrency === "" ? null : Number(concurrency),
        timeout_s: Number(timeoutS),
        retries: Number(retries),
      });
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  const modelCount = rows.length;
  const effective = limit !== "" && sampleCount ? Math.min(Number(limit), sampleCount) : sampleCount;

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">New run</h1>
        <p className="mt-1 text-sm text-slate-600">
          Pick what to test, pick the models, and start. Results appear in <Link to="/" className="text-accent hover:underline">Runs</Link> as soon as the first
          sample finishes.
        </p>
      </div>

      <TestPicker
        value={test}
        onChange={(patch) => setTest((t) => ({ ...t, ...patch }))}
        tasks={tasks.data!}
        datasets={datasets.data!}
        config={config.data!}
        reloadDatasets={() => setDatasetsTick((n) => n + 1)}
      />

      <ModelRows rows={rows} onChange={setRows} providers={providers.data!} binaryOnlyAllowed={isBinary} />

      <Card title="3. Run settings">
        <div className="grid gap-3 md:grid-cols-4">
          <FormField label="Samples to run" hint={sampleCount ? `Blank runs all ${sampleCount}.` : "Blank runs all."}>
            <input className={inputCls} inputMode="numeric" value={limit} placeholder="all" onChange={(e) => setLimit(e.target.value)} />
          </FormField>
          <FormField label="Parallel requests" hint="Blank uses each provider's default (1 for local models).">
            <input className={inputCls} inputMode="numeric" value={concurrency} placeholder="default" onChange={(e) => setConcurrency(e.target.value)} />
          </FormField>
          <FormField label="Timeout (seconds)" hint="Per sample.">
            <input className={inputCls} inputMode="decimal" value={timeoutS} onChange={(e) => setTimeoutS(e.target.value)} />
          </FormField>
          <FormField label="Retries" hint="After a failed sample.">
            <input className={inputCls} inputMode="numeric" value={retries} onChange={(e) => setRetries(e.target.value)} />
          </FormField>
        </div>
      </Card>

      <div className="rounded border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-700">
            {problems.length === 0 ? (
              <>
                Ready: <span className="font-medium">{test.source === "builtin" ? test.builtin : test.testName}</span> on {modelCount} model
                {modelCount === 1 ? "" : "s"}
                {effective ? `, ${effective} samples each` : ""}.
              </>
            ) : (
              "Before you can start:"
            )}
          </p>
          <button type="button" className={btnPrimary} disabled={problems.length > 0 || submitting} onClick={start}>
            {submitting ? "Starting…" : "Start run"}
          </button>
        </div>
        {problems.length > 0 && (
          <ul className="mt-2 list-inside list-disc space-y-0.5 text-sm text-slate-600">
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}
        {error && (
          <div className="mt-3">
            <ErrorNote message={error} />
          </div>
        )}
      </div>
    </div>
  );
}
