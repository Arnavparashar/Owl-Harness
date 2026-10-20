import { useEffect, useState } from "react";
import type { RunSummary } from "./api";

// --- formatting -------------------------------------------------------------

export const num = (v: number | undefined, digits = 3): string => (v === undefined ? "–" : v.toFixed(digits));

export const ms = (v: number | undefined): string => {
  if (v === undefined) return "–";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`;
};

export const seconds = (v: number | undefined): string => (v === undefined ? "–" : `${v.toFixed(1)} s`);

export const dateTime = (iso: string | null): string => {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
};

export const pct = (v: number | undefined, digits = 1): string => (v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`);

/** Metrics that follow from a confusion matrix (all 0 when their denominator is 0). */
export function derivedMetrics(tp: number, fp: number, fn: number, tn: number) {
  const div = (a: number, b: number) => (b === 0 ? 0 : a / b);
  const recall = div(tp, tp + fn);
  const specificity = div(tn, tn + fp);
  const mccDenominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return {
    specificity,
    fpr: div(fp, fp + tn),
    npv: div(tn, tn + fn),
    balancedAccuracy: (recall + specificity) / 2,
    mcc: div(tp * tn - fp * fn, mccDenominator),
  };
}

/** Format a p-value; very small ones are shown as a bound rather than a false-precision number. */
export const pValue = (p: number): string => (p < 0.001 ? "< 0.001" : p.toFixed(3));

/** Series colours: the accent, then neutrals. Used for runs in comparison charts. */
export const SERIES_COLORS = ["#1d4ed8", "#334155", "#94a3b8", "#7aa2f0", "#64748b", "#cbd5e1"];

/** Distinct label per run: the model, plus a short run id when the same model appears twice. */
export function runLabels(runs: RunSummary[]): Record<string, string> {
  const counts: Record<string, number> = {};
  for (const r of runs) counts[r.model] = (counts[r.model] ?? 0) + 1;
  return Object.fromEntries(runs.map((r) => [r.id, counts[r.model] > 1 ? `${r.model} (${r.id.slice(-4)})` : r.model]));
}

// --- data loading -----------------------------------------------------------

export interface Async<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
}

/** Run an async loader whenever `deps` change; ignores stale responses. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: undefined, error: undefined, loading: true });
  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    load().then(
      (data) => !cancelled && setState({ data, error: undefined, loading: false }),
      (e: unknown) => !cancelled && setState({ data: undefined, error: e instanceof Error ? e.message : String(e), loading: false }),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** Calls `tick` every `ms` while `active`; used to refresh runs that are still in progress. */
export function usePolling(active: boolean, tick: () => void, ms = 3000): void {
  useEffect(() => {
    if (!active) return;
    const t = setInterval(tick, ms);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ms]);
}

export function useDebounced<T>(value: T, delayMs = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return v;
}
