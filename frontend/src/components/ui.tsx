// Building blocks of the design system. Pages compose these; they never style glass directly.
import { useState, type ReactNode } from "react";
import type { Stat } from "../api";
import { num, statText, useCountUp } from "../lib";

export function Card({
  title,
  right,
  children,
  className = "",
  pad = true,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`glass rise ${className}`}>
      {(title || right) && (
        <header className="flex min-h-12 flex-wrap items-center justify-between gap-2 border-b border-white/10 px-5 py-3">
          {title && <h2 className="text-[13px] font-semibold text-ink">{title}</h2>}
          {right}
        </header>
      )}
      <div className={pad ? "p-5" : ""}>{children}</div>
    </section>
  );
}

/** A denser, more opaque surface for tables, numbers and code. */
export function Dense({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`glass-dense ${className}`}>{children}</div>;
}

export function Chip({
  children,
  onClick,
  title,
  tone = "default",
}: {
  children: ReactNode;
  onClick?: () => void;
  title?: string;
  tone?: "default" | "accent" | "ok" | "warn" | "bad";
}) {
  const tones = {
    default: "border-white/15 bg-white/[0.06] text-sub",
    accent: "border-[#6366f1]/50 bg-[#6366f1]/15 text-[#c7ceff]",
    ok: "border-ok/40 bg-ok/10 text-ok",
    warn: "border-warn/40 bg-warn/10 text-warn",
    bad: "border-bad/40 bg-bad/10 text-bad",
  } as const;
  const cls = `inline-flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium ${tones[tone]}`;
  return onClick ? (
    <button type="button" title={title} onClick={onClick} className={`${cls} hover:border-white/30`}>
      {children}
    </button>
  ) : (
    <span title={title} className={cls}>
      {children}
    </span>
  );
}

const STATUS: Record<string, { tone: "default" | "accent" | "ok" | "warn" | "bad"; label: string }> = {
  completed: { tone: "ok", label: "completed" },
  completed_with_errors: { tone: "warn", label: "completed, some errors" },
  running: { tone: "accent", label: "running" },
  queued: { tone: "default", label: "waiting" },
  failed: { tone: "bad", label: "failed" },
  interrupted: { tone: "warn", label: "interrupted" },
  cancelled: { tone: "default", label: "cancelled" },
};

export function StatusPill({ status }: { status: string }) {
  const s = STATUS[status] ?? { tone: "default" as const, label: status.replace(/_/g, " ") };
  return (
    <Chip tone={s.tone}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {s.label}
    </Chip>
  );
}

/** A number that counts up on first render. */
export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const shown = useCountUp(value);
  return <>{format(shown)}</>;
}

export function MetricTile({ label, value, sub, digits = 3, text }: { label: string; value?: number | null; sub?: ReactNode; digits?: number; text?: string }) {
  return (
    <div className="glass-dense px-4 py-3">
      <div className="label">{label}</div>
      <div className="num mt-1 text-xl font-semibold text-ink">
        {text ?? (value === undefined || value === null ? "–" : <CountUp value={value} format={(v) => v.toFixed(digits)} />)}
      </div>
      {sub && <div className="num mt-0.5 text-xs text-faint">{sub}</div>}
    </div>
  );
}

export function Tip({ tip, children }: { tip: ReactNode; children: ReactNode }) {
  return (
    <span className="tip inline-flex" tabIndex={0}>
      {children}
      <span role="tooltip" className="tip-body">
        {tip}
      </span>
    </span>
  );
}

/**
 * The headline number of a run: large, with its standard error and 95% interval, and a
 * "small sample" badge when the interval is 0.10 wide or more (for 0-1 metrics; 0.5 for a 1-5 mean).
 */
export function HeroMetric({ stat, size = "lg", note }: { stat: Stat | null; size?: "lg" | "md"; note?: ReactNode }) {
  if (!stat) return null;
  const width = stat.ci_low !== null && stat.ci_high !== null ? stat.ci_high - stat.ci_low : null;
  const wide = width !== null && width >= (stat.key === "mean_score" ? 0.5 : 0.1);
  const digits = stat.key === "mean_score" ? 2 : 3;
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="label">{stat.label}</span>
        {wide && (
          <Tip
            tip={
              <>
                The 95% interval is {num(width, digits)} wide: with this many samples the true value could plausibly be anywhere from{" "}
                {num(stat.ci_low, digits)} to {num(stat.ci_high, digits)}. Run more samples to narrow it before drawing conclusions.
              </>
            }
          >
            <Chip tone="warn">small sample</Chip>
          </Tip>
        )}
      </div>
      <div className={`num mt-1 font-semibold tracking-tight ${size === "lg" ? "gradient-text text-6xl" : "text-4xl text-ink"}`}>
        <CountUp value={stat.value} format={(v) => statText({ key: stat.key, value: v })} />
      </div>
      <div className="num mt-2 text-sm text-sub">
        {stat.stderr !== null && <>± {num(stat.stderr, digits)} stderr</>}
        {stat.ci_low !== null && stat.ci_high !== null && (
          <>
            {stat.stderr !== null && <span className="mx-2 text-faint">·</span>}
            95% CI {num(stat.ci_low, digits)}–{num(stat.ci_high, digits)}
          </>
        )}
      </div>
      {note && <div className="mt-1 text-xs text-faint">{note}</div>}
    </div>
  );
}

export function Tabs<T extends string>({ tabs, active, onChange }: { tabs: { id: T; label: string }[]; active: T; onChange: (id: T) => void }) {
  return (
    <nav className="flex flex-wrap gap-1 rounded-[12px] border border-white/10 bg-white/[0.04] p-1" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          className={`relative h-8 rounded-[9px] px-3.5 text-[13px] font-medium transition-colors ${
            active === t.id ? "bg-white/[0.1] text-ink" : "text-sub hover:text-ink"
          }`}
        >
          {t.label}
          {active === t.id && <span className="accent-bar absolute inset-x-3 -bottom-[3px] h-[2px] rounded-full" />}
        </button>
      ))}
    </nav>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="rounded-[12px] border border-bad/40 bg-bad/10 px-4 py-3 text-sm text-bad">{message}</div>;
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="py-8 text-sm text-faint">{what}…</div>;
}

/** A plain line of text for an empty list; no illustrations. */
export function EmptyLine({ children }: { children: ReactNode }) {
  return <p className="py-3 text-sm text-faint">{children}</p>;
}

export function FormField({ label, hint, children, className = "" }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={`block ${className}`}>
      <span className="label mb-1.5 block">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-xs leading-relaxed text-faint">{hint}</span>}
    </label>
  );
}

/** A progress bar with a moving sheen while `running`. */
export function ProgressBar({ value, max, running = false }: { value: number; max: number; running?: boolean }) {
  const p = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/10" role="progressbar" aria-valuenow={value} aria-valuemax={max}>
      <div className={`progress-fill h-full rounded-full ${running ? "running" : ""}`} style={{ width: `${p}%` }} />
    </div>
  );
}

/** A destructive action that asks once, inline, before acting (never a browser dialog). */
export function ConfirmAction({ label, question, onConfirm, disabled }: { label: string; question: string; onConfirm: () => void | Promise<void>; disabled?: boolean }) {
  const [asking, setAsking] = useState(false);
  if (!asking) {
    return (
      <button type="button" className="btn btn-danger" disabled={disabled} onClick={() => setAsking(true)}>
        {label}
      </button>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2 text-sm">
      <span className="text-sub">{question}</span>
      <button
        type="button"
        className="btn btn-danger border border-bad/40"
        onClick={async () => {
          await onConfirm();
          setAsking(false);
        }}
      >
        {label}
      </button>
      <button type="button" className="btn btn-quiet" onClick={() => setAsking(false)}>
        Keep
      </button>
    </span>
  );
}

/** Shows a short form of `text`; clicking copies the whole of it. */
export function CopyChip({ text, display, label }: { text: string; display: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Chip
      title={`${label}: ${text}. Click to copy.`}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        } catch {
          /* clipboard unavailable: the full value is in the title */
        }
      }}
    >
      <span className="text-faint">{label}</span>
      <span className="num font-mono">{copied ? "copied" : display}</span>
    </Chip>
  );
}

export const th = "label px-3 py-2.5 text-left";
export const td = "px-3 py-2 text-[13px] text-ink";
