import { useEffect, useRef, useState } from "react";
import type { ScoringMode, Stat } from "./api";

// --- formatting -------------------------------------------------------------

export const num = (v: number | undefined | null, digits = 3): string => (v === undefined || v === null || Number.isNaN(v) ? "–" : v.toFixed(digits));

export const ms = (v: number | undefined | null): string => {
  if (v === undefined || v === null) return "–";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`;
};

export const seconds = (v: number | undefined | null): string => (v === undefined || v === null ? "–" : `${v.toFixed(1)} s`);

export const pct = (v: number | undefined | null, digits = 1): string => (v === undefined || v === null ? "–" : `${(v * 100).toFixed(digits)}%`);

export const dateTime = (iso: string | null): string => {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
};

export const duration = (secondsLeft: number): string => {
  if (!Number.isFinite(secondsLeft) || secondsLeft < 0) return "–";
  if (secondsLeft < 60) return `${Math.max(1, Math.round(secondsLeft))} s`;
  const m = Math.floor(secondsLeft / 60);
  return m < 60 ? `${m} min ${Math.round(secondsLeft % 60)} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
};

/** Format a p-value; very small ones are shown as a bound rather than a false-precision number. */
export const pValue = (p: number): string => (p < 0.001 ? "< 0.001" : p.toFixed(3));

/** Digits for a metric: a 1-5 mean score reads better with two decimals. */
export const statText = (s: Pick<Stat, "key" | "value"> | null | undefined): string =>
  s ? (s.key === "mean_score" ? s.value.toFixed(2) : s.value.toFixed(3)) : "–";

export const MODE_LABEL: Record<ScoringMode, string> = {
  binary: "Yes / no",
  multiclass: "Multi-class",
  exact_match: "Exact match",
  judge_passfail: "Judge: pass/fail",
  judge_scale: "Judge: 1–5",
};

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

/** Wilson score interval, matching the backend, for numbers recomputed in the browser. */
export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 0];
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/**
 * Fill `{name}` placeholders the same way the backend does (owl/templates.py): only known
 * names are replaced, `{{` and `}}` are literal braces, everything else is left as written.
 */
export function renderTemplate(template: string, values: Record<string, string>): string {
  return template.replace(/\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_ .-]*)\}/g, (token, name: string | undefined) => {
    if (token === "{{") return "{";
    if (token === "}}") return "}";
    return name !== undefined && name in values ? values[name] : token;
  });
}

export const shortHash = (h: string, n = 10): string => h.slice(0, n);

// --- data loading -----------------------------------------------------------

export interface Async<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
}

/** Run an async loader whenever `deps` change; ignores stale responses and keeps the last data while reloading. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: undefined, error: undefined, loading: true });
  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    load().then(
      (data) => !cancelled && setState({ data, error: undefined, loading: false }),
      (e: unknown) => !cancelled && setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e), loading: false })),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** Calls `tick` every `ms` while `active`; used to refresh work that is still in progress. */
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

// --- motion -------------------------------------------------------------------

const reducedQuery = "(prefers-reduced-motion: reduce)";

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => typeof window !== "undefined" && window.matchMedia(reducedQuery).matches);
  useEffect(() => {
    const mq = window.matchMedia(reducedQuery);
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

/** Counts up to `target` once, on first render. Later changes (a moved threshold) apply at once. */
export function useCountUp(target: number, durationMs = 650): number {
  const reduced = useReducedMotion();
  const [value, setValue] = useState(reduced ? target : 0);
  const done = useRef(reduced);
  useEffect(() => {
    if (done.current) {
      setValue(target);
      return;
    }
    let frame = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) frame = requestAnimationFrame(step);
      else done.current = true;
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, durationMs]);
  return value;
}
