import type { ReactNode } from "react";

export function Card({ title, children, right }: { title?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="rounded border border-slate-200 bg-white">
      {title && (
        <header className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded border border-slate-200 bg-white px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">{value}</div>
      <div className="mt-0.5 min-h-4 text-xs tabular-nums text-slate-500">{sub}</div>
    </div>
  );
}

const STATUS_STYLE: Record<string, string> = {
  completed: "text-slate-700",
  running: "text-accent",
  completed_with_errors: "text-danger",
  failed: "text-danger",
  interrupted: "text-slate-500",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm ${STATUS_STYLE[status] ?? "text-slate-700"}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="rounded border border-danger/30 bg-white px-4 py-3 text-sm text-danger">{message}</div>;
}

export function Loading() {
  return <div className="py-8 text-sm text-slate-500">Loading…</div>;
}

export const th = "px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500";
export const td = "px-3 py-1.5 text-sm text-slate-800";

// --- form controls -------------------------------------------------------------

export const inputCls =
  "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent disabled:bg-slate-100 disabled:text-slate-500";
export const btnPrimary =
  "rounded bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-[#1e40af] disabled:cursor-not-allowed disabled:bg-slate-300";
export const btnSecondary =
  "rounded border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400";

export function FormField({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

/** A progress bar: flat, no animation. */
export function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="h-2 w-full overflow-hidden rounded-sm bg-slate-200" role="progressbar" aria-valuenow={value} aria-valuemax={max}>
      <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
    </div>
  );
}
