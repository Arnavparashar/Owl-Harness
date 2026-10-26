// Building blocks of the design system. Pages compose these; they never style glass directly.
import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Check, ListFilter, Search } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Stat } from "../api";
import { num, statText, useCountUp } from "../lib";

const lift = { y: -4 };
const spring = { type: "spring", stiffness: 380, damping: 30 } as const;

/** The round button in a card's top-right corner that opens the card's detail page. */
export function CornerLink({ to, label, white = false }: { to: string; label: string; white?: boolean }) {
  return (
    <Link to={to} aria-label={label} title={label} className={`icon-btn shrink-0 ${white ? "icon-btn-hero" : ""}`}>
      <ArrowUpRight size={16} strokeWidth={2.2} />
    </Link>
  );
}

export function Card({
  title,
  right,
  children,
  className = "",
  pad = true,
  to,
  toLabel,
  hero = false,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
  /** detail page opened by the corner button */
  to?: string;
  toLabel?: string;
  hero?: boolean;
}) {
  return (
    <motion.section
      className={`${hero ? "glass-hero" : "glass hoverable"} rise ${className}`}
      whileHover={lift}
      transition={spring}
    >
      {(title || right || to) && (
        <header className={`flex min-h-14 flex-wrap items-center justify-between gap-2 px-7 pt-6 ${pad ? "" : "pb-4"}`}>
          {title && <h2 className={`text-[15px] font-semibold ${hero ? "text-white" : "text-ink"}`}>{title}</h2>}
          <div className="ml-auto flex items-center gap-2">
            {right}
            {to && <CornerLink to={to} label={toLabel ?? "Open"} white={hero} />}
          </div>
        </header>
      )}
      <div className={pad ? "px-7 pb-7 pt-4" : ""}>{children}</div>
    </motion.section>
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
    default: "border-wash/15 bg-white/70 text-sub",
    accent: "border-emerald/40 bg-[#d5eee3] text-accent",
    ok: "border-ok/35 bg-ok/12 text-ok",
    warn: "border-warn/35 bg-[#fbe8c8] text-warn",
    bad: "border-bad/35 bg-[#fbdcdc] text-bad",
  } as const;
  const cls = `inline-flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium ${tones[tone]}`;
  return onClick ? (
    <button type="button" title={title} onClick={onClick} className={`${cls} hover:border-wash/35`}>
      {children}
    </button>
  ) : (
    <span title={title} className={cls}>
      {children}
    </span>
  );
}

const STATUS: Record<string, { dot: string; label: string }> = {
  completed: { dot: "#059669", label: "completed" },
  completed_with_errors: { dot: "#d97706", label: "completed, some errors" },
  running: { dot: "#d97706", label: "running" },
  queued: { dot: "#d97706", label: "waiting" },
  failed: { dot: "#dc2626", label: "failed" },
  interrupted: { dot: "#dc2626", label: "interrupted" },
  cancelled: { dot: "#8a7f70", label: "cancelled" },
};

/** Status as a small coloured dot with a text label: green done, amber pending, red failed. */
export function StatusPill({ status }: { status: string }) {
  const s = STATUS[status] ?? { dot: "#8a7f70", label: status.replace(/_/g, " ") };
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-[13px] text-sub">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: s.dot }} />
      {s.label}
    </span>
  );
}

/** A number that counts up on first render. */
export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const shown = useCountUp(value);
  return <>{format(shown)}</>;
}

/** Green with an up arrow for an improvement, red with a down arrow for a decline. */
export function DeltaPill({ change, higherIsBetter = true, onHero = false }: { change: number; higherIsBetter?: boolean; onHero?: boolean }) {
  const up = change >= 0;
  const good = up === higherIsBetter;
  const Icon = up ? ArrowUpRight : ArrowDownRight;
  return (
    <span
      className={`num inline-flex h-6 items-center gap-1 rounded-full px-2 text-xs font-semibold ${onHero ? (good ? "bg-white text-[#065f46]" : "bg-white text-[#991b1b]") : good ? "bg-[#d5eee3] text-ok" : "bg-[#fbdcdc] text-bad"}`}
      title="compared with the previous period of the same length"
    >
      <Icon size={13} strokeWidth={2.4} />
      {Math.abs(change * 100).toFixed(change !== 0 && Math.abs(change) < 0.1 ? 1 : 0)}%
    </span>
  );
}

/**
 * A KPI card: small label at the top left, a large bold value at the bottom left with a
 * caption under it, and an optional delta pill at the bottom right.
 */
export function KpiCard({
  label,
  value,
  format,
  caption,
  change,
  hero = false,
  to,
  toLabel,
}: {
  label: string;
  value: number | null;
  format: (v: number) => string;
  caption?: ReactNode;
  /** relative change against the previous period; omitted hides the pill */
  change?: number | null;
  hero?: boolean;
  to?: string;
  toLabel?: string;
}) {
  return (
    <motion.section className={`${hero ? "glass-hero" : "glass hoverable"} rise flex min-h-[168px] flex-col justify-between p-6`} whileHover={lift} transition={spring}>
      <div className="flex items-start justify-between gap-2">
        <span className={`text-[13px] ${hero ? "text-white" : "text-faint"}`}>{label}</span>
        {to && <CornerLink to={to} label={toLabel ?? label} white={hero} />}
      </div>
      <div className="flex items-end justify-between gap-3">
        <div className="min-w-0">
          <div className={`num text-[34px] font-bold leading-none tracking-tight ${hero ? "text-white" : "text-ink"}`}>
            {value === null ? "–" : <CountUp value={value} format={format} />}
          </div>
          {caption && <div className={`mt-2 truncate text-xs ${hero ? "text-white" : "text-faint"}`}>{caption}</div>}
        </div>
        {change !== undefined && change !== null && Number.isFinite(change) && (
          <DeltaPill change={change} onHero={hero} />
        )}
      </div>
    </motion.section>
  );
}

export function MetricTile({ label, value, sub, digits = 3, text }: { label: string; value?: number | null; sub?: ReactNode; digits?: number; text?: string }) {
  return (
    <div className="glass-dense px-5 py-4">
      <div className="label">{label}</div>
      <div className="num mt-1.5 text-xl font-bold text-ink">
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
 * The headline number of a run, with its standard error and 95% interval, and a
 * "small sample" badge when the interval is 0.10 wide or more (0.5 for a 1-5 mean).
 * `onHero` renders it for the filled hero card.
 */
export function HeroMetric({ stat, size = "lg", note, onHero = false }: { stat: Stat | null; size?: "lg" | "md"; note?: ReactNode; onHero?: boolean }) {
  if (!stat) return null;
  const width = stat.ci_low !== null && stat.ci_high !== null ? stat.ci_high - stat.ci_low : null;
  const wide = width !== null && width >= (stat.key === "mean_score" ? 0.5 : 0.1);
  const digits = stat.key === "mean_score" ? 2 : 3;
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className={`text-[13px] ${onHero ? "text-white" : "text-faint"}`}>{stat.label}</span>
        {wide && (
          <Tip
            tip={
              <>
                The 95% interval is {num(width, digits)} wide: with this many samples the true value could plausibly be anywhere from {num(stat.ci_low, digits)} to{" "}
                {num(stat.ci_high, digits)}. Run more samples to narrow it before drawing conclusions.
              </>
            }
          >
            <span className={`inline-flex h-6 items-center rounded-full px-2.5 text-xs font-semibold ${onHero ? "bg-white text-[#7c3a0c]" : "bg-[#fbe8c8] text-warn"}`}>small sample</span>
          </Tip>
        )}
      </div>
      <div className={`num mt-2 font-bold leading-none tracking-tight ${size === "lg" ? "text-6xl" : "text-[40px]"} ${onHero ? "text-white" : "text-ink"}`}>
        <CountUp value={stat.value} format={(v) => statText({ key: stat.key, value: v })} />
      </div>
      <div className={`num mt-3 text-sm ${onHero ? "text-white" : "text-sub"}`}>
        {stat.stderr !== null && <>± {num(stat.stderr, digits)} stderr</>}
        {stat.ci_low !== null && stat.ci_high !== null && (
          <>
            {stat.stderr !== null && <span className="mx-2 opacity-70">·</span>}
            95% CI {num(stat.ci_low, digits)}–{num(stat.ci_high, digits)}
          </>
        )}
      </div>
      {note && <div className={`mt-1.5 text-xs ${onHero ? "text-white" : "text-faint"}`}>{note}</div>}
    </div>
  );
}

export function Tabs<T extends string>({ tabs, active, onChange }: { tabs: { id: T; label: string }[]; active: T; onChange: (id: T) => void }) {
  return (
    <nav className="pill inline-flex h-11 gap-1 p-1" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          className={`h-9 rounded-full px-4 text-[13px] font-medium transition-colors ${active === t.id ? "active-pill" : "text-sub hover:text-ink"}`}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}

export type Period = "7d" | "30d" | "all";
export const PERIODS: { id: Period; label: string; days: number | null }[] = [
  { id: "7d", label: "Last 7 days", days: 7 },
  { id: "30d", label: "Last 30 days", days: 30 },
  { id: "all", label: "All time", days: null },
];

/** The glass period-selector pill in a page header. */
export function PeriodSelector({ value, onChange }: { value: Period; onChange: (p: Period) => void }) {
  return (
    <div className="pill h-10 gap-0.5 p-1" role="radiogroup" aria-label="Period">
      {PERIODS.map((p) => (
        <button
          key={p.id}
          role="radio"
          aria-checked={value === p.id}
          onClick={() => onChange(p.id)}
          className={`h-8 rounded-full px-3.5 text-xs font-medium transition-colors ${value === p.id ? "active-pill" : "text-faint hover:text-ink"}`}
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}

/** A large bold title with a small muted subtitle underneath, and controls on the right. */
export function PageHeader({ title, subtitle, right }: { title: ReactNode; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-[32px] font-bold leading-tight tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1 text-sm font-light text-faint">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="rounded-[16px] border border-bad/35 bg-white/70 px-4 py-3 text-sm text-bad">{message}</div>;
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
    <div className="h-2.5 w-full overflow-hidden rounded-full bg-wash/15" role="progressbar" aria-valuenow={value} aria-valuemax={max}>
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
        className="btn btn-danger border border-bad/35"
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

// --- table controls -------------------------------------------------------------------

/** A glass pill with a small search icon. */
export function SearchField({ value, onChange, placeholder = "Search", label }: { value: string; onChange: (v: string) => void; placeholder?: string; label?: string }) {
  return (
    <label className="pill h-10 min-w-0 flex-1 sm:max-w-[260px]">
      <Search size={15} className="shrink-0 text-faint" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={label ?? placeholder}
        className="w-full min-w-0 bg-transparent text-[13px] text-ink outline-none placeholder:text-faint"
      />
    </label>
  );
}

/** A glass "Filter" pill that opens a short list of choices. The active choice is marked in emerald. */
export function FilterPill({
  value,
  options,
  onChange,
  label = "Filter",
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const current = options.find((o) => o.value === value);
  const filtered = value !== "" && current;
  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`pill h-10 text-[13px] font-medium ${filtered ? "text-accent" : "text-ink"}`}
      >
        <ListFilter size={15} className="shrink-0" />
        {filtered ? current.label : label}
      </button>
      {open && (
        <div role="listbox" className="popover absolute right-0 top-12 z-40 min-w-[200px] p-1.5">
          {options.map((o) => (
            <button
              key={o.value}
              role="option"
              aria-selected={o.value === value}
              type="button"
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
              className="flex w-full items-center justify-between gap-3 rounded-[12px] px-3 py-2 text-left text-[13px] text-ink hover:bg-wash/[0.07]"
            >
              {o.label}
              {o.value === value && <Check size={14} className="text-accent" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
