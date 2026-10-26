// Wizard step 2: which models, and the judge for judge-graded tests.
import { Link } from "react-router-dom";
import type { ProviderInfo, SavedModel } from "../../api";
import { useModelColor } from "../../theme";
import ModelEditor, { emptyDraft, type ModelDraft } from "../ModelEditor";
import { Chip, EmptyLine } from "../ui";

export interface ModelsState {
  savedIds: string[];
  adhoc: ModelDraft[];
  judgeSource: "saved" | "adhoc";
  judgeSavedId: string;
  judgeDraft: ModelDraft;
}

export const initialModels = (defaultProvider: string): ModelsState => ({
  savedIds: [],
  adhoc: [],
  judgeSource: "saved",
  judgeSavedId: "",
  judgeDraft: emptyDraft(defaultProvider),
});

/** "provider/model" of every model that will be evaluated. */
export function candidateNames(m: ModelsState, saved: SavedModel[]): string[] {
  return [
    ...m.savedIds.map((id) => saved.find((s) => s.id === id)).filter((s): s is SavedModel => !!s).map((s) => `${s.provider}/${s.model}`),
    ...m.adhoc.map((d) => `${d.provider}/${d.model.trim()}`),
  ];
}

export function judgeName(m: ModelsState, saved: SavedModel[]): string | undefined {
  if (m.judgeSource === "saved") {
    const s = saved.find((x) => x.id === m.judgeSavedId);
    return s ? `${s.provider}/${s.model}` : undefined;
  }
  return m.judgeDraft.model.trim() ? `${m.judgeDraft.provider}/${m.judgeDraft.model.trim()}` : undefined;
}

export default function ModelsStep({
  value: m,
  onChange,
  providers,
  saved,
  binaryAllowed,
  judging,
}: {
  value: ModelsState;
  onChange: (patch: Partial<ModelsState>) => void;
  providers: ProviderInfo[];
  saved: SavedModel[];
  /** false when the test is not yes/no: providers that only answer yes/no are unavailable */
  binaryAllowed: boolean;
  judging: boolean;
}) {
  const colorOf = useModelColor();
  const providerOf = (name: string) => providers.find((p) => p.name === name);
  const defaultProvider = providers.find((p) => p.name === "ollama" && !p.error)?.name ?? providers.find((p) => !p.error)?.name ?? "mock";
  const toggle = (id: string) => onChange({ savedIds: m.savedIds.includes(id) ? m.savedIds.filter((x) => x !== id) : [...m.savedIds, id] });
  const judge = judgeName(m, saved);
  const selfJudged = judge !== undefined && candidateNames(m, saved).includes(judge);

  return (
    <div className="space-y-6">
      <section>
        <div className="mb-2 flex items-center justify-between">
          <span className="label">Saved models</span>
          <Link to="/models" className="text-xs font-medium text-accent hover:underline">
            Manage models →
          </Link>
        </div>
        {saved.length === 0 ? (
          <EmptyLine>No saved models yet. Add one below, or save models on the Models page.</EmptyLine>
        ) : (
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {saved.map((s) => {
              const blocked = !binaryAllowed && providerOf(s.provider)?.binary_only;
              const on = m.savedIds.includes(s.id);
              return (
                <label
                  key={s.id}
                  className={`glass-dense flex cursor-pointer items-start gap-3 p-3.5 transition-colors ${blocked ? "cursor-not-allowed opacity-50" : on ? "border-emerald bg-emerald/15" : "hover:bg-white"}`}
                >
                  <input type="checkbox" className="mt-1" checked={on} disabled={!!blocked} onChange={() => toggle(s.id)} />
                  <span className="min-w-0">
                    <span className="flex items-center gap-2 text-[13px] font-semibold text-ink">
                      <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colorOf(`${s.provider}/${s.model}`) }} />
                      <span className="truncate">{s.name}</span>
                    </span>
                    <span className="mt-0.5 block truncate text-xs text-faint">
                      {providerOf(s.provider)?.display_name ?? s.provider} · {s.model}
                    </span>
                    {blocked && <span className="mt-1 block text-xs text-warn">yes/no tests only</span>}
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="label">Other models for this run</span>
          <button type="button" className="btn btn-secondary h-8" disabled={m.adhoc.length + m.savedIds.length >= 10} onClick={() => onChange({ adhoc: [...m.adhoc, emptyDraft(defaultProvider)] })}>
            Add a model
          </button>
        </div>
        {m.adhoc.map((d, i) => (
          <div key={i} className="glass-dense p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[13px] font-semibold text-ink">Model {m.savedIds.length + i + 1}</span>
              <button type="button" className="btn btn-quiet h-7" onClick={() => onChange({ adhoc: m.adhoc.filter((_, j) => j !== i) })}>
                Remove
              </button>
            </div>
            <ModelEditor value={d} providers={providers} binaryOnlyAllowed={binaryAllowed} onChange={(next) => onChange({ adhoc: m.adhoc.map((x, j) => (j === i ? next : x)) })} />
          </div>
        ))}
        <p className="text-xs text-faint">Models run one after another, never at the same time, so their latencies are comparable.</p>
      </section>

      {judging && (
        <section className="glass-dense space-y-3 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-[13px] font-semibold text-ink">Judge</span>
            <div className="flex gap-4 text-[13px] text-ink">
              <label className="flex items-center gap-2">
                <input type="radio" name="judgeSource" checked={m.judgeSource === "saved"} onChange={() => onChange({ judgeSource: "saved" })} /> a saved model
              </label>
              <label className="flex items-center gap-2">
                <input type="radio" name="judgeSource" checked={m.judgeSource === "adhoc"} onChange={() => onChange({ judgeSource: "adhoc" })} /> another model
              </label>
            </div>
          </div>
          {m.judgeSource === "saved" ? (
            <select className="input" value={m.judgeSavedId} onChange={(e) => onChange({ judgeSavedId: e.target.value })}>
              <option value="">Choose the judge…</option>
              {saved
                .filter((s) => !providerOf(s.provider)?.binary_only)
                .map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} ({s.provider}/{s.model})
                  </option>
                ))}
            </select>
          ) : (
            <ModelEditor value={m.judgeDraft} providers={providers} binaryOnlyAllowed={false} onChange={(d) => onChange({ judgeDraft: d })} />
          )}
          {selfJudged && (
            <Chip tone="warn">The judge is also being evaluated. A model grading its own answers tends to be lenient; you can still go ahead.</Chip>
          )}
          <p className="text-xs text-faint">The judge runs at temperature 0. Its full reply is stored with every sample, and its time is not counted in the evaluated model's latency.</p>
        </section>
      )}
    </div>
  );
}
