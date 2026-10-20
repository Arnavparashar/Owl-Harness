// Edits one model: provider, model name (suggested from what the provider reports as
// available) and the provider's own settings. The form is built from the provider's ArgSpecs.
import { useId } from "react";
import { api, type ArgSpec, type ProviderInfo } from "../api";
import { useAsync, useDebounced } from "../lib";
import { FormField } from "./ui";

export interface ModelDraft {
  provider: string;
  model: string;
  /** only settings the user has touched; everything else uses the provider default */
  args: Record<string, string | boolean>;
}

export const emptyDraft = (provider: string): ModelDraft => ({ provider, model: "", args: {} });

/** Typed settings for the API; blank fields are left out so the provider default applies. */
export function draftSettings(d: ModelDraft, providers: ProviderInfo[]): Record<string, unknown> {
  const specs = providers.find((p) => p.name === d.provider)?.args ?? [];
  const out: Record<string, unknown> = {};
  for (const spec of specs) {
    const v = d.args[spec.name];
    if (v === undefined || v === "") continue;
    if (spec.type === "number" || spec.type === "integer") out[spec.name] = Number(v);
    else if (spec.type === "json" && typeof v === "string") out[spec.name] = JSON.parse(v);
    else out[spec.name] = v;
  }
  return out;
}

/** Turn saved settings back into form values. */
export function settingsToDraftArgs(settings: Record<string, unknown>): Record<string, string | boolean> {
  return Object.fromEntries(
    Object.entries(settings).map(([k, v]) => [k, typeof v === "boolean" ? v : typeof v === "object" && v !== null ? JSON.stringify(v, null, 2) : String(v)]),
  );
}

export function draftProblems(d: ModelDraft, providers: ProviderInfo[], label = "Model"): string[] {
  const out: string[] = [];
  if (!d.model.trim()) out.push(`${label}: choose or type a model name.`);
  for (const spec of providers.find((p) => p.name === d.provider)?.args ?? []) {
    const v = d.args[spec.name];
    if (typeof v !== "string" || v === "") continue;
    if ((spec.type === "number" || spec.type === "integer") && !Number.isFinite(Number(v))) out.push(`${label}: ${spec.label || spec.name} must be a number.`);
    if (spec.type === "json") {
      try {
        const parsed = JSON.parse(v);
        if (typeof parsed !== "object" || parsed === null) throw new Error();
      } catch {
        out.push(`${label}: ${spec.label || spec.name} must be a JSON object or list.`);
      }
    }
  }
  return out;
}

export default function ModelEditor({
  value,
  onChange,
  providers,
  binaryOnlyAllowed = true,
}: {
  value: ModelDraft;
  onChange: (d: ModelDraft) => void;
  providers: ProviderInfo[];
  binaryOnlyAllowed?: boolean;
}) {
  const listId = useId();
  const provider = providers.find((p) => p.name === value.provider);
  // Settings that change which models exist (such as a server address) re-run discovery.
  const discovery: Record<string, string> = {};
  for (const spec of provider?.args ?? []) {
    const v = value.args[spec.name];
    if (spec.affects_models && typeof v === "string" && v !== "") discovery[spec.name] = v;
  }
  const query = useDebounced(JSON.stringify(discovery), 400);
  const found = useAsync(() => api.providerModels(value.provider, JSON.parse(query)), [value.provider, query]);
  const setArg = (name: string, v: string | boolean) => onChange({ ...value, args: { ...value.args, [name]: v } });

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-[230px_1fr]">
        <FormField label="Provider">
          <select className="input" value={value.provider} onChange={(e) => onChange({ provider: e.target.value, model: "", args: {} })}>
            {providers.map((p) => (
              <option key={p.name} value={p.name} disabled={!!p.error || (p.binary_only && !binaryOnlyAllowed)}>
                {p.display_name}
                {p.binary_only && !binaryOnlyAllowed ? " (yes/no tests only)" : ""}
                {p.error ? " (failed to load)" : ""}
              </option>
            ))}
          </select>
        </FormField>
        <FormField
          label="Model"
          hint={
            found.loading && !found.data
              ? "Looking for available models…"
              : found.data?.error
                ? `Could not list models (${found.data.error}). Type the model name.`
                : found.data && found.data.models.length > 0
                  ? `${found.data.models.length} available. Pick one or type a name.`
                  : value.provider === "hf"
                    ? "A Hugging Face Hub id, such as deepset/deberta-v3-base-injection, or a local folder."
                    : "Type the model name."
          }
        >
          <input
            className="input"
            list={listId}
            value={value.model}
            placeholder={found.data?.models[0] ?? "model name"}
            onChange={(e) => onChange({ ...value, model: e.target.value })}
            autoComplete="off"
            spellCheck={false}
          />
          <datalist id={listId}>
            {(found.data?.models ?? []).map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </FormField>
      </div>
      {provider?.description && <p className="text-xs leading-relaxed text-faint">{provider.description}</p>}
      {provider && provider.args.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2">
          {provider.args.map((spec) => (
            <ArgField key={spec.name} spec={spec} value={value.args[spec.name]} onChange={(v) => setArg(spec.name, v)} />
          ))}
        </div>
      )}
    </div>
  );
}

function ArgField({ spec, value, onChange }: { spec: ArgSpec; value: string | boolean | undefined; onChange: (v: string | boolean) => void }) {
  const label = spec.label || spec.name;
  if (spec.type === "boolean") {
    return (
      <label className="flex items-start gap-2.5 text-[13px] text-ink md:col-span-2">
        <input type="checkbox" className="mt-0.5" checked={typeof value === "boolean" ? value : spec.default === true} onChange={(e) => onChange(e.target.checked)} />
        <span>
          <span className="font-medium">{label}</span>
          <span className="block text-xs text-faint">{spec.description}</span>
        </span>
      </label>
    );
  }
  if (spec.choices.length > 0) {
    return (
      <FormField label={label} hint={spec.description}>
        <select className="input" value={typeof value === "string" && value !== "" ? value : String(spec.default ?? "")} onChange={(e) => onChange(e.target.value)}>
          {spec.choices.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </FormField>
    );
  }
  if (spec.type === "json") {
    return (
      <FormField label={label} hint={spec.description} className="md:col-span-2">
        <textarea
          className="input min-h-[84px] font-mono text-xs"
          value={typeof value === "string" ? value : ""}
          placeholder={spec.default === null || spec.default === undefined ? "{ }" : JSON.stringify(spec.default)}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
      </FormField>
    );
  }
  return (
    <FormField label={label} hint={spec.description}>
      <input
        className="input"
        value={typeof value === "string" ? value : ""}
        placeholder={spec.default === null || spec.default === undefined ? "" : String(spec.default)}
        inputMode={spec.type === "string" ? undefined : "decimal"}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        spellCheck={false}
      />
    </FormField>
  );
}
