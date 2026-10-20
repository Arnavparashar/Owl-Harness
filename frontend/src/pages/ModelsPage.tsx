import { useState } from "react";
import { api, type ProbeResult, type ProviderInfo, type SavedModel } from "../api";
import ModelEditor, { draftProblems, draftSettings, emptyDraft, settingsToDraftArgs, type ModelDraft } from "../components/ModelEditor";
import { Card, Chip, ConfirmAction, EmptyLine, ErrorNote, FormField, Loading } from "../components/ui";
import { ms, useAsync } from "../lib";
import { useModelColor } from "../theme";

export default function ModelsPage() {
  const [tick, setTick] = useState(0);
  const models = useAsync(() => api.models(), [tick]);
  const providers = useAsync(() => api.providers(), []);
  const suggestions = useAsync(() => api.modelSuggestions(), [tick]);
  const [editing, setEditing] = useState<SavedModel | "new" | null>(null);
  const [error, setError] = useState<string>();
  const reload = () => setTick((n) => n + 1);

  if (models.error || providers.error) return <ErrorNote message={(models.error ?? providers.error)!} />;
  if (!models.data || !providers.data) return <Loading />;

  const quickAdd = async (s: { name: string; provider: string; model: string; settings: Record<string, unknown> }) => {
    setError(undefined);
    try {
      await api.createModel(s);
      reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-5">
      <div className="glass rise flex flex-wrap items-center justify-between gap-3 px-5 py-4">
        <p className="max-w-3xl text-[13px] leading-relaxed text-sub">
          Saved models can be picked in New run. Any provider plugin installed in this environment appears in the provider list. API keys are never typed here:
          settings name an environment variable and the server reads it.
        </p>
        <button className="btn btn-primary" onClick={() => setEditing("new")}>
          Add model
        </button>
      </div>
      {error && <ErrorNote message={error} />}

      {suggestions.data && suggestions.data.items.length > 0 && (
        <Card title="Installed in Ollama" right={<span className="text-xs text-faint">one click to save</span>}>
          <div className="flex flex-wrap gap-2">
            {suggestions.data.items.map((s) => (
              <button key={s.model} className="btn btn-secondary h-8" onClick={() => quickAdd(s)}>
                + {s.model}
              </button>
            ))}
          </div>
        </Card>
      )}
      {suggestions.data?.error && <p className="text-xs text-faint">Ollama is not reachable, so installed models cannot be listed ({suggestions.data.error}).</p>}

      {editing && (
        <Editor
          key={editing === "new" ? "new" : editing.id}
          initial={editing === "new" ? undefined : editing}
          providers={providers.data}
          onDone={() => {
            setEditing(null);
            reload();
          }}
          onCancel={() => setEditing(null)}
        />
      )}

      {models.data.length === 0 ? (
        <Card>
          <EmptyLine>No saved models yet.</EmptyLine>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {models.data.map((m) => (
            <ModelCard key={m.id} model={m} providers={providers.data!} onEdit={() => setEditing(m)} onDeleted={reload} />
          ))}
        </div>
      )}
    </div>
  );
}

function ModelCard({ model, providers, onEdit, onDeleted }: { model: SavedModel; providers: ProviderInfo[]; onEdit: () => void; onDeleted: () => void }) {
  const colorOf = useModelColor();
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<ProbeResult>();
  const provider = providers.find((p) => p.name === model.provider);
  const settings = Object.entries(model.settings);

  const test = async () => {
    setTesting(true);
    setResult(undefined);
    try {
      setResult(await api.testModel(model.id));
    } catch (e) {
      setResult({ ok: false, error: e instanceof Error ? e.message : String(e), latency_ms: null, setup_ms: null, output: null, probability: null, model_info: {} });
    } finally {
      setTesting(false);
    }
  };

  return (
    <section className="glass rise flex flex-col p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: colorOf(`${model.provider}/${model.model}`) }} />
            <h3 className="truncate text-[15px] font-semibold text-ink">{model.name}</h3>
          </div>
          <p className="mt-1 truncate font-mono text-xs text-sub" title={`${model.provider}/${model.model}`}>
            {model.provider}/{model.model}
          </p>
        </div>
        <Chip>{provider?.display_name ?? `${model.provider} (not installed)`}</Chip>
      </div>

      <dl className="mt-4 space-y-1 text-xs">
        {settings.length === 0 ? (
          <div className="text-faint">provider defaults</div>
        ) : (
          settings.slice(0, 4).map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <dt className="shrink-0 text-faint">{k}</dt>
              <dd className="truncate font-mono text-sub" title={JSON.stringify(v)}>
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </dd>
            </div>
          ))
        )}
        {settings.length > 4 && <div className="text-faint">+{settings.length - 4} more</div>}
      </dl>

      {(testing || result) && (
        <div className={`mt-4 rounded-[10px] border px-3 py-2.5 text-xs ${testing ? "border-white/15 bg-white/[0.05] text-sub" : result?.ok ? "border-ok/40 bg-ok/10" : "border-bad/40 bg-bad/10 text-bad"}`}>
          {testing && "Loading the model and sending one short prompt…"}
          {result?.ok && (
            <div className="space-y-1">
              <div className="num font-medium text-ok">
                Connected · answered in {ms(result.latency_ms)}
                {result.setup_ms !== null && <span className="font-normal text-sub"> (load and warm-up {ms(result.setup_ms)})</span>}
              </div>
              <div className="break-words font-mono text-sub">
                output: {JSON.stringify(result.output)}
                {result.probability !== null && ` · P = ${result.probability.toFixed(3)}`}
              </div>
            </div>
          )}
          {result && !result.ok && <div className="break-words">{result.error}</div>}
        </div>
      )}

      <div className="mt-auto flex flex-wrap items-center gap-2 pt-4">
        <button className="btn btn-secondary h-8" onClick={test} disabled={testing}>
          {testing ? "Testing…" : "Test connection"}
        </button>
        <button className="btn btn-quiet h-8" onClick={onEdit}>
          Edit
        </button>
        <span className="ml-auto">
          <ConfirmAction
            label="Delete"
            question="Delete this saved model? Runs made with it are kept."
            onConfirm={async () => {
              await api.deleteModel(model.id);
              onDeleted();
            }}
          />
        </span>
      </div>
    </section>
  );
}

function Editor({ initial, providers, onDone, onCancel }: { initial?: SavedModel; providers: ProviderInfo[]; onDone: () => void; onCancel: () => void }) {
  const defaultProvider = providers.find((p) => p.name === "ollama" && !p.error)?.name ?? providers.find((p) => !p.error)?.name ?? "mock";
  const [name, setName] = useState(initial?.name ?? "");
  const [draft, setDraft] = useState<ModelDraft>(initial ? { provider: initial.provider, model: initial.model, args: settingsToDraftArgs(initial.settings) } : emptyDraft(defaultProvider));
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<ProbeResult>();
  const [busy, setBusy] = useState<"save" | "test">();
  const problems = draftProblems(draft, providers);

  const body = () => ({ name: name.trim() || draft.model.trim(), provider: draft.provider, model: draft.model.trim(), settings: draftSettings(draft, providers) });

  const save = async () => {
    setBusy("save");
    setError(undefined);
    try {
      if (initial) await api.updateModel(initial.id, body());
      else await api.createModel(body());
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(undefined);
    }
  };
  const test = async () => {
    setBusy("test");
    setResult(undefined);
    try {
      setResult(await api.testUnsaved(body()));
    } finally {
      setBusy(undefined);
    }
  };

  return (
    <Card title={initial ? `Edit ${initial.name}` : "Add a model"}>
      <div className="space-y-4">
        <FormField label="Display name" hint="Blank uses the model name.">
          <input className="input max-w-md" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder={draft.model || "My model"} />
        </FormField>
        <ModelEditor value={draft} onChange={setDraft} providers={providers} />
        {problems.length > 0 && <p className="text-[13px] text-warn">{problems.join(" ")}</p>}
        {error && <ErrorNote message={error} />}
        {result && (
          <div className={`rounded-[10px] border px-3 py-2.5 text-xs ${result.ok ? "border-ok/40 bg-ok/10 text-ok" : "border-bad/40 bg-bad/10 text-bad"}`}>
            {result.ok ? `Connected · answered ${JSON.stringify(result.output)} in ${ms(result.latency_ms)}` : result.error}
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-primary" disabled={problems.length > 0 || busy !== undefined} onClick={save}>
            {busy === "save" ? "Saving…" : initial ? "Save changes" : "Save model"}
          </button>
          <button className="btn btn-secondary" disabled={problems.length > 0 || busy !== undefined} onClick={test}>
            {busy === "test" ? "Testing…" : "Test before saving"}
          </button>
          <button className="btn btn-quiet" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </Card>
  );
}
