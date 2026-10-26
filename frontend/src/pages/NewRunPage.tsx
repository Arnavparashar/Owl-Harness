import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, type RunRequest } from "../api";
import { draftProblems, draftSettings } from "../components/ModelEditor";
import { Card, Chip, ErrorNote, FormField, Loading, PageHeader } from "../components/ui";
import ModelsStep, { candidateNames, initialModels, judgeName, type ModelsState } from "../components/wizard/ModelsStep";
import TestStep, { effectiveRubric, effectiveSystem, initialTest, testProblems, testToRequest, type TestState } from "../components/wizard/TestStep";
import { MODE_LABEL, useAsync } from "../lib";

interface SettingsState {
  temperature: string;
  maxTokens: string;
  seed: string;
  limit: string;
  concurrency: string;
  timeout: string;
  retries: string;
}

const STEPS = ["Test", "Models", "Settings", "Review"] as const;
const isInt = (s: string, min: number, max: number) => /^\d+$/.test(s) && Number(s) >= min && Number(s) <= max;

export default function NewRunPage() {
  const navigate = useNavigate();
  const [datasetsTick, setDatasetsTick] = useState(0);
  const config = useAsync(() => api.config(), []);
  const providers = useAsync(() => api.providers(), []);
  const tasks = useAsync(() => api.tasks(), []);
  const datasets = useAsync(() => api.datasets(), [datasetsTick]);
  const saved = useAsync(() => api.models(), []);

  const [step, setStep] = useState(0);
  const [reached, setReached] = useState(0);
  const [test, setTest] = useState<TestState>(initialTest);
  const [models, setModels] = useState<ModelsState>(initialModels("ollama"));
  const [settings, setSettings] = useState<SettingsState>({ temperature: "", maxTokens: "", seed: "", limit: "", concurrency: "", timeout: "120", retries: "2" });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (tasks.data?.length && !test.builtin) setTest((t) => ({ ...t, builtin: tasks.data!.find((x) => !x.example)?.name ?? tasks.data![0].name }));
  }, [tasks.data, test.builtin]);

  const loadError = config.error ?? providers.error ?? tasks.error ?? datasets.error ?? saved.error;
  if (loadError) return <ErrorNote message={loadError} />;
  if (!config.data || !providers.data || !tasks.data || !datasets.data || !saved.data) return <Loading />;

  const cfg = config.data;
  const dataset = datasets.data.find((d) => d.id === test.datasetId);
  const task = tasks.data.find((t) => t.name === test.builtin);
  const kind = test.source === "builtin" ? (task?.scoring_mode ?? "binary") : test.kind === "judge" ? (test.rubric === "passfail" ? "judge_passfail" : "judge_scale") : test.kind;
  const binary = kind === "binary";
  const judging = test.source === "dataset" && test.kind === "judge";
  const providerOf = (name: string) => providers.data!.find((p) => p.name === name);
  const samples = test.source === "builtin" ? task?.expected_size : dataset?.rows;

  const modelsProblems: string[] = [];
  const count = models.savedIds.length + models.adhoc.length;
  if (count === 0) modelsProblems.push("Choose at least one model.");
  models.adhoc.forEach((d, i) => {
    modelsProblems.push(...draftProblems(d, providers.data!, `Model ${models.savedIds.length + i + 1}`));
    if (!binary && providerOf(d.provider)?.binary_only) modelsProblems.push(`Model ${models.savedIds.length + i + 1}: ${d.provider} only answers yes/no questions.`);
  });
  if (judging) {
    if (models.judgeSource === "saved" && !models.judgeSavedId) modelsProblems.push("Choose the judge.");
    if (models.judgeSource === "adhoc") modelsProblems.push(...draftProblems(models.judgeDraft, providers.data!, "Judge"));
  }

  const settingsProblems: string[] = [];
  if (settings.temperature !== "" && !(Number(settings.temperature) >= 0 && Number(settings.temperature) <= 2)) settingsProblems.push("Temperature must be between 0 and 2.");
  if (settings.maxTokens !== "" && !isInt(settings.maxTokens, 1, 8192)) settingsProblems.push("Max tokens must be a whole number from 1 to 8192.");
  if (settings.seed !== "" && !isInt(settings.seed, 0, 2 ** 31 - 1)) settingsProblems.push("Seed must be a whole number of at least 0.");
  if (settings.limit !== "" && !isInt(settings.limit, 1, 10_000_000)) settingsProblems.push("Samples to run must be a whole number of at least 1.");
  if (settings.concurrency !== "" && !isInt(settings.concurrency, 1, 64)) settingsProblems.push("Parallel requests must be from 1 to 64.");
  if (!(Number(settings.timeout) > 0 && Number(settings.timeout) <= 3600)) settingsProblems.push("Timeout must be between 1 and 3600 seconds.");
  if (!isInt(settings.retries, 0, 10)) settingsProblems.push("Retries must be from 0 to 10.");

  const problems = [testProblems(test, dataset), modelsProblems, settingsProblems, []];
  const blockers = problems[step];

  const buildRequest = (): RunRequest => {
    const req: RunRequest = {
      ...testToRequest(test, cfg),
      models: [
        ...models.savedIds.map((id) => ({ model_id: id })),
        ...models.adhoc.map((d) => ({ provider: d.provider, model: d.model.trim(), args: draftSettings(d, providers.data!) })),
      ],
      limit: settings.limit === "" ? null : Number(settings.limit),
      max_concurrency: settings.concurrency === "" ? null : Number(settings.concurrency),
      timeout_s: Number(settings.timeout),
      retries: Number(settings.retries),
      temperature: settings.temperature === "" ? null : Number(settings.temperature),
      max_tokens: settings.maxTokens === "" ? null : Number(settings.maxTokens),
      seed: settings.seed === "" ? null : Number(settings.seed),
    };
    if (req.custom?.judge) {
      req.custom.judge =
        models.judgeSource === "saved"
          ? { ...req.custom.judge, model_id: models.judgeSavedId }
          : { ...req.custom.judge, provider: models.judgeDraft.provider, model: models.judgeDraft.model.trim(), args: draftSettings(models.judgeDraft, providers.data!) };
    }
    return req;
  };

  const start = async () => {
    setSubmitting(true);
    setError(undefined);
    try {
      const job = await api.startJob(buildRequest());
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  const go = (to: number) => {
    setStep(to);
    setReached((r) => Math.max(r, to));
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <PageHeader title="New run" subtitle="Pick a test, the models and the settings." />
      {/* stepper */}
      <ol className="glass rise grid grid-cols-4 gap-1 !rounded-full p-1.5">
        {STEPS.map((s, i) => {
          const can = i <= reached && problems.slice(0, i).every((p) => p.length === 0);
          return (
            <li key={s}>
              <button
                type="button"
                disabled={!can}
                onClick={() => go(i)}
                className={`relative flex h-11 w-full items-center justify-center gap-2.5 rounded-full text-[13px] font-medium transition-colors ${
                  step === i ? "active-pill" : can ? "text-sub hover:bg-wash/[0.07] hover:text-ink" : "text-faint opacity-60"
                }`}
              >
                <span className={`num grid h-6 w-6 place-items-center rounded-full text-xs font-bold ${step === i ? "bg-white text-ink" : i < step ? "bg-emerald text-white" : "bg-wash/10"}`}>
                  {i < step ? "✓" : i + 1}
                </span>
                {s}
              </button>
            </li>
          );
        })}
      </ol>

      <Card
        title={["1. What to test", "2. Which models", "3. Settings", "4. Review and start"][step]}
        right={
          step === 0 && samples ? (
            <span className="num text-xs text-faint">{samples.toLocaleString()} samples</span>
          ) : step === 1 ? (
            <span className="num text-xs text-faint">
              {count} model{count === 1 ? "" : "s"}
            </span>
          ) : undefined
        }
      >
        {step === 0 && (
          <TestStep value={test} onChange={(p) => setTest((t) => ({ ...t, ...p }))} tasks={tasks.data} datasets={datasets.data} config={cfg} reloadDatasets={() => setDatasetsTick((n) => n + 1)} />
        )}
        {step === 1 && <ModelsStep value={models} onChange={(p) => setModels((m) => ({ ...m, ...p }))} providers={providers.data} saved={saved.data} binaryAllowed={binary} judging={judging} />}
        {step === 2 && <SettingsStep value={settings} onChange={(p) => setSettings((s) => ({ ...s, ...p }))} samples={samples} defaultSeed={cfg.default_seed} />}
        {step === 3 && (
          <Review
            rows={[
              ["Test", test.source === "builtin" ? test.builtin : test.testName],
              ["Scoring", MODE_LABEL[kind]],
              ...(test.source === "dataset"
                ? ([
                    ["Dataset", `${dataset?.name} (${dataset?.rows.toLocaleString()} rows)`],
                    ["Columns", `input: ${test.inputColumn}${test.targetColumn ? ` · target: ${test.targetColumn}` : " · no reference"}`],
                  ] as [string, ReactNode][])
                : []),
              ...(test.source === "dataset" && test.kind === "binary" ? ([["Yes means", `${test.targetColumn} = “${test.positiveLabel}”`]] as [string, ReactNode][]) : []),
              ...(test.source === "dataset" && test.kind === "multiclass" ? ([["Labels", test.labels.join(", ")]] as [string, ReactNode][]) : []),
              ...(test.source === "dataset" && test.kind === "exact_match"
                ? ([["Ignoring", Object.entries(test.normalization).filter(([, v]) => v).map(([k]) => k).join(", ") || "nothing"]] as [string, ReactNode][])
                : []),
              [
                "Models",
                <span className="flex flex-wrap gap-1.5">
                  {candidateNames(models, saved.data).map((n, i) => (
                    <Chip key={i}>{n}</Chip>
                  ))}
                </span>,
              ],
              ...(judging ? ([["Judge", `${judgeName(models, saved.data)} · ${test.rubric === "passfail" ? "pass/fail" : "1–5 scale"}`]] as [string, ReactNode][]) : []),
              ["Samples", settings.limit ? `first ${settings.limit} of ${samples?.toLocaleString() ?? "?"}` : `all ${samples?.toLocaleString() ?? ""}`],
              ["Generation", `temperature ${settings.temperature || "0 (default)"} · seed ${settings.seed || `${cfg.default_seed} (default)`} · max tokens ${settings.maxTokens || "test default"}`],
              ["Limits", `timeout ${settings.timeout} s · ${settings.retries} retries · ${settings.concurrency || "provider default"} parallel`],
            ]}
            prompt={test.source === "dataset" ? effectiveSystem(test, cfg) : task?.system_prompt ?? ""}
            user={test.source === "dataset" ? test.userTemplate : "{input}"}
            rubric={judging ? effectiveRubric(test, cfg) : undefined}
          />
        )}

        {blockers.length > 0 && (
          <ul className="mt-5 space-y-1 border-t border-wash/10 pt-4 text-[13px] text-warn">
            {blockers.map((p) => (
              <li key={p}>· {p}</li>
            ))}
          </ul>
        )}
        {error && (
          <div className="mt-4">
            <ErrorNote message={error} />
          </div>
        )}
        <div className="mt-5 flex items-center justify-between border-t border-wash/10 pt-4">
          <button type="button" className="btn btn-secondary" disabled={step === 0} onClick={() => go(step - 1)}>
            Back
          </button>
          {step < 3 ? (
            <button type="button" className="btn btn-primary" disabled={blockers.length > 0} onClick={() => go(step + 1)}>
              Next: {STEPS[step + 1]}
            </button>
          ) : (
            <button type="button" className="btn btn-primary" disabled={submitting || problems.some((p) => p.length > 0)} onClick={start}>
              {submitting ? "Starting…" : "Start run"}
            </button>
          )}
        </div>
      </Card>
    </div>
  );
}

function SettingsStep({ value: s, onChange, samples, defaultSeed }: { value: SettingsState; onChange: (p: Partial<SettingsState>) => void; samples?: number | null; defaultSeed: number }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <FormField label="Temperature" hint="Blank keeps the test's setting: 0, the most repeatable.">
          <input className="input" inputMode="decimal" placeholder="0" value={s.temperature} onChange={(e) => onChange({ temperature: e.target.value })} />
        </FormField>
        <FormField label="Max answer tokens" hint="Blank keeps the test's default for its scoring mode.">
          <input className="input" inputMode="numeric" placeholder="default" value={s.maxTokens} onChange={(e) => onChange({ maxTokens: e.target.value })} />
        </FormField>
        <FormField label="Seed" hint="Used by providers that support seeded sampling.">
          <input className="input" inputMode="numeric" placeholder={String(defaultSeed)} value={s.seed} onChange={(e) => onChange({ seed: e.target.value })} />
        </FormField>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <FormField label="Samples to run" hint={samples ? `Blank runs all ${samples.toLocaleString()}. A limit takes the first rows.` : "Blank runs all."}>
          <input className="input" inputMode="numeric" placeholder="all" value={s.limit} onChange={(e) => onChange({ limit: e.target.value })} />
        </FormField>
        <FormField label="Timeout (seconds)" hint="Per sample.">
          <input className="input" inputMode="decimal" value={s.timeout} onChange={(e) => onChange({ timeout: e.target.value })} />
        </FormField>
        <FormField label="Retries" hint="After a failed sample.">
          <input className="input" inputMode="numeric" value={s.retries} onChange={(e) => onChange({ retries: e.target.value })} />
        </FormField>
        <FormField label="Parallel requests" hint="Within one model. Blank: the provider default (1 for local models).">
          <input className="input" inputMode="numeric" placeholder="default" value={s.concurrency} onChange={(e) => onChange({ concurrency: e.target.value })} />
        </FormField>
      </div>
    </div>
  );
}

function Review({ rows, prompt, user, rubric }: { rows: [string, ReactNode][]; prompt: string; user: string; rubric?: string }) {
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <dl className="glass-dense grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2.5 p-4 text-[13px]">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-faint">{k}</dt>
            <dd className="min-w-0 break-words text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="space-y-3">
        <div>
          <div className="label mb-1.5">System prompt</div>
          <pre className="glass-dense max-h-48 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs leading-relaxed text-sub">{prompt || "(none)"}</pre>
        </div>
        <div>
          <div className="label mb-1.5">User message</div>
          <pre className="glass-dense whitespace-pre-wrap break-words p-3 font-mono text-xs text-sub">{user}</pre>
        </div>
        {rubric && (
          <div>
            <div className="label mb-1.5">Judge rubric</div>
            <pre className="glass-dense max-h-48 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs leading-relaxed text-sub">{rubric}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
