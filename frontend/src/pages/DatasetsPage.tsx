import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type DatasetInfo } from "../api";
import { useChartMotion } from "../components/charts";
import { Card, Chip, ConfirmAction, EmptyLine, ErrorNote, FormField, Loading, PageHeader } from "../components/ui";
import { VirtualTable } from "../components/VirtualTable";
import { dateTime, useAsync } from "../lib";
import { AccentDefs, AXIS, GRID, TOOLTIP } from "../theme";

const size = (n: number) => (n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);

export default function DatasetsPage() {
  const [tick, setTick] = useState(0);
  const list = useAsync(() => api.datasets(), [tick]);
  const config = useAsync(() => api.config(), []);
  const [selected, setSelected] = useState<string>();
  const reload = () => setTick((n) => n + 1);

  // A dataset just added is selected once the reloaded list contains it; until then keep waiting
  // rather than falling back to the first row.
  const [params] = useSearchParams();
  const [wanted, setWanted] = useState<string | undefined>(params.get("id") ?? undefined); // from the search pill
  useEffect(() => {
    if (!list.data) return;
    if (wanted) {
      if (list.data.some((d) => d.id === wanted)) {
        setSelected(wanted);
        setWanted(undefined);
      }
      return;
    }
    if (list.data.length > 0 && (!selected || !list.data.some((d) => d.id === selected))) setSelected(list.data[0].id);
  }, [list.data, selected, wanted]);

  const current = list.data?.find((d) => d.id === selected);

  return (
    <div className="space-y-6">
      <PageHeader title="Datasets" subtitle="Your evaluation data, stored on this machine." />
      <div className="grid gap-5 xl:grid-cols-2">
        <Upload
          formats={config.data?.formats ?? ".csv, .tsv, .jsonl, .json or .parquet"}
          maxBytes={config.data?.max_upload_bytes}
          onDone={(d) => {
            setWanted(d.id);
            reload();
          }}
        />
        <HubImport
          onDone={(d) => {
            setWanted(d.id);
            reload();
          }}
        />
      </div>

      <Card title="Library" right={<span className="num text-xs text-faint">{list.data?.length ?? 0} datasets</span>}>
        {list.error && <ErrorNote message={list.error} />}
        {!list.data && <Loading />}
        {list.data && (
          <VirtualTable
            rows={list.data}
            rowKey={(d) => d.id}
            template="minmax(180px,1.4fr) 110px 90px minmax(220px,1.6fr) 90px 170px"
            maxHeight={380}
            onRowClick={(d) => setSelected(d.id)}
            rowClassName={(d) => (d.id === selected ? "!bg-emerald/15" : "hover:bg-white")}
            empty="No datasets yet. Upload a file or import one from the Hugging Face Hub."
            columns={[
              { key: "name", header: "Name", render: (d) => <span className="truncate font-medium">{d.name}</span> },
              { key: "rows", header: "Rows", align: "right", render: (d) => <span className="num text-sub">{d.rows.toLocaleString()}</span> },
              { key: "fmt", header: "Format", render: (d) => <span className="text-sub">{d.format}</span> },
              {
                key: "src",
                header: "Source",
                render: (d) =>
                  d.source ? (
                    <span className="truncate text-sub" title={`revision ${d.source.revision ?? "unknown"}`}>
                      Hub: {d.source.hub} · {d.source.config ? `${d.source.config} · ` : ""}
                      {d.source.split}
                    </span>
                  ) : (
                    <span className="truncate text-sub">upload: {d.filename}</span>
                  ),
              },
              { key: "size", header: "Size", align: "right", render: (d) => <span className="num text-sub">{size(d.size_bytes)}</span> },
              { key: "created", header: "Added", render: (d) => <span className="num text-sub">{dateTime(d.created_at)}</span> },
            ]}
          />
        )}
      </Card>

      {current && <DatasetDetail key={current.id} dataset={current} onDeleted={reload} />}
    </div>
  );
}

function Upload({ formats, maxBytes, onDone }: { formats: string; maxBytes?: number; onDone: (d: DatasetInfo) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const send = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(undefined);
    try {
      onDone(await api.uploadDataset(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  return (
    <Card title="Upload a file">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          send(e.dataTransfer.files?.[0]);
        }}
        className={`flex min-h-36 flex-col items-center justify-center rounded-[14px] border-2 border-dashed px-6 py-6 text-center transition-colors ${
          over ? "border-emerald bg-emerald/10" : "border-wash/15"
        }`}
      >
        <p className="text-sm text-ink">{busy ? "Uploading…" : "Drop a dataset file here"}</p>
        <p className="mt-1 text-xs text-faint">
          {formats}, UTF-8{maxBytes ? `, up to ${Math.round(maxBytes / 1024 / 1024)} MB` : ""}. Text containing commas must be quoted in CSV.
        </p>
        <input ref={input} type="file" accept=".csv,.tsv,.jsonl,.json,.parquet" className="hidden" aria-label="Choose a dataset file" onChange={(e) => send(e.target.files?.[0])} />
        <button className="btn btn-secondary mt-4 h-8" disabled={busy} onClick={() => input.current?.click()}>
          Choose a file
        </button>
      </div>
      {error && (
        <div className="mt-3">
          <ErrorNote message={error} />
        </div>
      )}
    </Card>
  );
}

function HubImport({ onDone }: { onDone: (d: DatasetInfo) => void }) {
  const [dataset, setDataset] = useState("");
  const [config, setConfig] = useState("");
  const [split, setSplit] = useState("test");
  const [maxRows, setMaxRows] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const valid = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/.test(dataset.trim()) && split.trim() !== "" && (maxRows === "" || /^\d+$/.test(maxRows));

  const run = async () => {
    setBusy(true);
    setError(undefined);
    try {
      onDone(await api.importHf({ dataset: dataset.trim(), split: split.trim(), config: config.trim() || null, max_rows: maxRows ? Number(maxRows) : null }));
      setDataset("");
      setConfig("");
      setMaxRows("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="Import from the Hugging Face Hub">
      <div className="grid gap-3 md:grid-cols-2">
        <FormField label="Dataset id" hint="For example fancyzhx/ag_news">
          <input className="input" value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="owner/name" spellCheck={false} />
        </FormField>
        <FormField label="Config" hint="Blank for the default.">
          <input className="input" value={config} onChange={(e) => setConfig(e.target.value)} placeholder="default" spellCheck={false} />
        </FormField>
        <FormField label="Split">
          <input className="input" value={split} onChange={(e) => setSplit(e.target.value)} spellCheck={false} />
        </FormField>
        <FormField label="Row limit" hint="Optional: import only the first rows.">
          <input className="input" inputMode="numeric" value={maxRows} onChange={(e) => setMaxRows(e.target.value)} placeholder="all" />
        </FormField>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button className="btn btn-primary" disabled={!valid || busy} onClick={run}>
          {busy ? "Downloading…" : "Import"}
        </button>
        <span className="text-xs text-faint">Saved here as a fixed snapshot with the Hub revision, so runs on it are reproducible. Class labels are stored by name.</span>
      </div>
      {error && (
        <div className="mt-3">
          <ErrorNote message={error} />
        </div>
      )}
    </Card>
  );
}

function DatasetDetail({ dataset, onDeleted }: { dataset: DatasetInfo; onDeleted: () => void }) {
  const motion = useChartMotion();
  const detail = useAsync(() => api.dataset(dataset.id, 20), [dataset.id]);
  const guess = dataset.columns.find((c) => ["label", "target", "answer", "class", "topic", "category"].includes(c.toLowerCase())) ?? dataset.columns[dataset.columns.length - 1];
  const [column, setColumn] = useState(guess);
  const values = useAsync(() => api.datasetValues(dataset.id, column), [dataset.id, column]);
  const [error, setError] = useState<string>();
  const rows = detail.data?.preview ?? [];

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <Card
        title={
          <span className="flex flex-wrap items-center gap-2">
            {dataset.name}
            <Chip>{dataset.rows.toLocaleString()} rows</Chip>
            <Chip>{dataset.columns.length} columns</Chip>
          </span>
        }
        right={
          <ConfirmAction
            label="Delete dataset"
            question="Delete it? Runs already made on it are kept."
            onConfirm={async () => {
              try {
                await api.deleteDataset(dataset.id);
                onDeleted();
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
          />
        }
      >
        {error && <ErrorNote message={error} />}
        <div className="label mb-2">First {rows.length} rows</div>
        <div className="max-h-[480px] overflow-auto">
          <table className="strip-table sticky-head text-left">
            <thead>
              <tr>
                {dataset.columns.map((c) => (
                  <th key={c} className="label whitespace-nowrap px-3 py-2.5">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t border-wash/[0.06]">
                  {dataset.columns.map((c) => (
                    <td key={c} className="max-w-[340px] truncate px-3 py-1.5 text-xs text-sub" title={r[c]}>
                      {r[c]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {dataset.source && (
          <p className="num mt-3 text-xs text-faint">
            Imported from {dataset.source.hub} ({dataset.source.config ?? "default"}, {dataset.source.split}) at revision {dataset.source.revision?.slice(0, 12) ?? "unknown"}.
          </p>
        )}
      </Card>

      <Card title="Value distribution" right={<select className="input h-8 w-44 text-xs" value={column} onChange={(e) => setColumn(e.target.value)} aria-label="Column">{dataset.columns.map((c) => <option key={c}>{c}</option>)}</select>}>
        {values.error && <ErrorNote message={values.error} />}
        {values.data && (
          <>
            <p className="num mb-3 text-xs text-faint">
              {values.data.distinct.toLocaleString()} distinct value{values.data.distinct === 1 ? "" : "s"}
              {values.data.distinct > values.data.values.length ? `; the ${values.data.values.length} most common shown` : ""}.
              {values.data.distinct > 50 && " Many distinct values: probably free text, better suited to exact match or a judge than to classification."}
            </p>
            {values.data.values.length === 0 ? (
              <EmptyLine>No values.</EmptyLine>
            ) : (
              <ResponsiveContainer width="100%" height={Math.min(560, Math.max(160, values.data.values.slice(0, 15).length * 30 + 30))}>
                <BarChart data={values.data.values.slice(0, 15).map((v) => ({ ...v, value: v.value || "(empty)" }))} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                  <AccentDefs id="dist" />
                  <CartesianGrid stroke={GRID} horizontal={false} />
                  <XAxis type="number" allowDecimals={false} {...AXIS} />
                  <YAxis type="category" dataKey="value" {...AXIS} width={140} tick={{ fill: "#3b3630", fontSize: 12 }} />
                  <Tooltip {...TOOLTIP} formatter={(v) => [String(v), "rows"]} />
                  <Bar dataKey="count" fill="url(#dist)" barSize={16} radius={[0, 5, 5, 0]} {...motion} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
