// Per-sample outcome grid for a comparison, drawn on a canvas: one row per run, one column per
// sample. A canvas keeps this fast for thousands of samples, where one element per cell would not be.
import { useEffect, useRef, useState } from "react";
import { BAD } from "../theme";

const ROW = 22;
const GAP = 6;
const CORRECT = "rgba(5,150,105,0.55)";

export function OutcomeGrid({
  runs,
  sampleIds,
  outcomes,
}: {
  runs: { id: string; label: string; color: string }[];
  sampleIds: string[];
  outcomes: Record<string, boolean[]>;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<{ run: number; sample: number } | null>(null);
  const n = sampleIds.length;
  const height = runs.length * (ROW + GAP) - GAP;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || width === 0 || n === 0) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(width * dpr);
    c.height = Math.floor(height * dpr);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const cell = width / n;
    const gap = cell > 4 ? 1 : 0;
    runs.forEach((r, ri) => {
      const y = ri * (ROW + GAP);
      const row = outcomes[r.id] ?? [];
      for (let k = 0; k < n; k++) {
        ctx.fillStyle = row[k] ? CORRECT : BAD;
        ctx.fillRect(k * cell, y, Math.max(cell - gap, 0.6), ROW);
      }
    });
  }, [width, height, n, runs, outcomes]);

  const locate = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const run = Math.floor(y / (ROW + GAP));
    const sample = Math.floor((x / rect.width) * n);
    if (run >= 0 && run < runs.length && sample >= 0 && sample < n && y - run * (ROW + GAP) <= ROW) setHover({ run, sample });
    else setHover(null);
  };

  return (
    <div>
      <div className="flex gap-3">
        <div className="w-44 shrink-0">
          {runs.map((r) => (
            <div key={r.id} className="flex items-center gap-2 truncate text-[13px] text-sub" style={{ height: ROW, marginBottom: GAP }} title={r.label}>
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: r.color }} />
              <span className="truncate">{r.label}</span>
            </div>
          ))}
        </div>
        <div ref={wrap} className="min-w-0 flex-1">
          <canvas
            ref={canvas}
            style={{ width: "100%", height }}
            onMouseMove={locate}
            onMouseLeave={() => setHover(null)}
            aria-label={`Outcome of ${n} samples for ${runs.length} runs`}
            role="img"
          />
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-faint">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: CORRECT }} /> correct
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: BAD }} /> incorrect
        </span>
        <span className="num text-sub">
          {hover
            ? `${runs[hover.run].label}, sample ${sampleIds[hover.sample]}: ${outcomes[runs[hover.run].id]?.[hover.sample] ? "correct" : "incorrect"}`
            : `${n} samples in dataset order. A column red in every row is a sample no model got right.`}
        </span>
      </div>
    </div>
  );
}
