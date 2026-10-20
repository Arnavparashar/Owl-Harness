// Step 2 of a new run: choose the models. Each row is a provider plugin, a model name (suggested
// from what the provider reports as available) and the provider's own settings.
import { useId } from "react";
import { api, type ArgSpec, type ProviderInfo, type RunRequest } from "../api";
import { useAsync, useDebounced } from "../lib";
import { btnSecondary, Card, FormField, inputCls } from "./ui";

export interface ModelRow {
  key: string;
  provider: string;
  model: string;
  /** only settings the user has touched; everything else uses the provider default */
  args: Record<string, string | boolean>;
}

let counter = 0;
export const newRow = (provider: string): ModelRow => ({ key: `row-${++counter}`, provider, model: "", args: {} });

export function rowsToRequest(rows: ModelRow[], providers: ProviderInfo[]): RunRequest["models"] {
  return rows.map((r) => {
    const specs = providers.find((p) => p.name === r.provider)?.args ?? [];
    const args: Record<string, unknown> = {};
    for (const spec of specs) {
      const v = r.args[spec.name];
      if (v === undefined || v === "") continue;
      args[spec.name] = spec.type === "number" || spec.type === "integer" ? Number(v) : v;
    }
    return { provider: r.provider, model: r.model.trim(), args };
  });
}

export function modelProblems(rows: ModelRow[], providers: ProviderInfo[]): string[] {
  const out: string[] = [];
  rows.forEach((r, i) => {
    const label = `Model ${i + 1}`;
    if (!r.model.trim()) out.push(`${label}: choose or type a model name.`);
    const specs = providers.find((p) => p.name === r.provider)?.args ?? [];
    for (const spec of specs) {
      const v = r.args[spec.name];
      if ((spec.type === "number" || spec.type === "integer") && typeof v === "string" && v !== "" && !Number.isFinite(Number(v))) {
        out.push(`${label}: ${spec.name} must be a number.`);
      }
    }
  });
  return out;
}

export default function ModelRows({
  rows,
  onChange,
  providers,
  binaryOnlyAllowed,
}: {
  rows: ModelRow[];
  onChange: (rows: ModelRow[]) => void;
  providers: ProviderInfo[];
  /** false when the chosen test is not a yes/no test: providers that only answer yes/no are disabled */
  binaryOnlyAllowed: boolean;
}) {
  const update = (key: string, patch: Partial<ModelRow>) => onChange(rows.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  const usable = providers.filter((p) => !p.error && (binaryOnlyAllowed || !p.binary_only));

  return (
    <Card
      title="2. Which models"
      right={
        <button type="button" className={btnSecondary} disabled={rows.length >= 10 || usable.length === 0} onClick={() => onChange([...rows, newRow(usable[0].name)])}>
          Add another model
        </button>
      }
    >
      <p className="mb-4 text-sm text-slate-600">
        Models run one after another so their timings are comparable. Providers are plugins: any installed package that registers an{" "}
        <code className="rounded bg-slate-100 px-1 text-xs">owl.providers</code> entry point shows up here.
      </p>
      <div className="space-y-4">
        {rows.map((row, i) => (
          <Row
            key={row.key}
            index={i}
            row={row}
            providers={providers}
            binaryOnlyAllowed={binaryOnlyAllowed}
            onChange={(patch) => update(row.key, patch)}
            onRemove={rows.length > 1 ? () => onChange(rows.filter((r) => r.key !== row.key)) : undefined}
          />
        ))}
      </div>
    </Card>
  );
}

function Row({
  index,
  row,
  providers,
  binaryOnlyAllowed,
  onChange,
  onRemove,
}: {
  index: number;
  row: ModelRow;
  providers: ProviderInfo[];
  binaryOnlyAllowed: boolean;
  onChange: (patch: Partial<ModelRow>) => void;
  onRemove?: () => void;
}) {
  const listId = useId();
  const provider = providers.find((p) => p.name === row.provider);
  // Settings that change which models exist (the server address, for example) re-run discovery.
  const discoveryArgs: Record<string, string> = {};
  for (const spec of provider?.args ?? []) {
    const v = row.args[spec.name];
    if (spec.affects_models && typeof v === "string" && v !== "") discoveryArgs[spec.name] = v;
  }
  const query = useDebounced(JSON.stringify(discoveryArgs), 400);
  const found = useAsync(() => api.providerModels(row.provider, JSON.parse(query)), [row.provider, query]);

  const setArg = (name: string, value: string | boolean) => onChange({ args: { ...row.args, [name]: value } });
  const blocked = provider?.binary_only && !binaryOnlyAllowed;

  return (
    <div className="rounded border border-slate-200 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-900">Model {index + 1}</span>
        {onRemove && (
          <button type="button" className="text-sm text-slate-600 hover:text-danger hover:underline" onClick={onRemove}>
            Remove
          </button>
        )}
      </div>
      <div className="grid gap-3 md:grid-cols-[200px_1fr]">
        <FormField label="Provider">
          <select
            className={inputCls}
            value={row.provider}
            onChange={(e) => onChange({ provider: e.target.value, model: "", args: {} })}
          >
            {providers.map((p) => (
              <option key={p.name} value={p.name} disabled={!!p.error || (p.binary_only && !binaryOnlyAllowed)}>
                {p.name}
                {p.binary_only && !binaryOnlyAllowed ? " (yes/no tests only)" : ""}
              </option>
            ))}
          </select>
        </FormField>
        <FormField
          label="Model"
          hint={
            found.loading && !found.data ? (
              "Looking for available models…"
            ) : found.data?.error ? (
              <span>Could not list models ({found.data.error}). Type the model name instead.</span>
            ) : found.data && found.data.models.length > 0 ? (
              `${found.data.models.length} available. Pick one or type a name.`
            ) : (
              "Type the model name."
            )
          }
        >
          <input
            className={inputCls}
            list={listId}
            value={row.model}
            placeholder={found.data?.models[0] ?? "model name"}
            onChange={(e) => onChange({ model: e.target.value })}
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
      {provider?.description && <p className="mt-2 text-xs text-slate-500">{provider.description}</p>}
      {blocked && <p className="mt-2 text-xs text-danger">This provider only answers yes/no questions and cannot run the chosen test.</p>}

      {provider && provider.args.length > 0 && (
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {provider.args.map((spec) => (
            <ArgField key={spec.name} spec={spec} value={row.args[spec.name]} onChange={(v) => setArg(spec.name, v)} />
          ))}
        </div>
      )}
    </div>
  );
}

function ArgField({ spec, value, onChange }: { spec: ArgSpec; value: string | boolean | undefined; onChange: (v: string | boolean) => void }) {
  if (spec.type === "boolean") {
    return (
      <label className="flex items-start gap-2 text-sm text-slate-800 md:col-span-2">
        <input
          type="checkbox"
          className="mt-0.5 accent-[#1d4ed8]"
          checked={typeof value === "boolean" ? value : spec.default === true}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>
          <span className="font-medium">{spec.name}</span>
          <span className="block text-xs text-slate-500">{spec.description}</span>
        </span>
      </label>
    );
  }
  return (
    <FormField label={spec.name} hint={spec.description}>
      <input
        className={inputCls}
        value={typeof value === "string" ? value : ""}
        placeholder={spec.default === null ? "" : String(spec.default)}
        inputMode={spec.type === "string" ? undefined : "decimal"}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        spellCheck={false}
      />
    </FormField>
  );
}
