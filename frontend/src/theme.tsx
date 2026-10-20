// Chart theme and model colours, shared by every page.
import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { RunSummary } from "./api";

/**
 * Colour-blind-safe palette for comparing models (Okabe and Ito), ordered so the first
 * colours read best on the dark background. A model keeps its colour on every page.
 */
export const MODEL_PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#D55E00", "#5B8DEF", "#B8C2D2"];

export const ACCENT_FROM = "#818cf8";
export const ACCENT_TO = "#22d3ee";
export const OK = "#4ade80";
export const WARN = "#fbbf24";
export const BAD = "#fb7185";
export const INK = "#eef2f8";
export const SUB = "#cdd5e1";
export const FAINT = "#bcc4d2";

export const AXIS = { stroke: "#7c8aa3", fontSize: 12, tickLine: false, tick: { fill: FAINT } } as const;
export const GRID = "rgba(255,255,255,0.08)";
export const TOOLTIP = {
  contentStyle: {
    background: "rgba(17,23,40,0.97)",
    border: "1px solid rgba(255,255,255,0.16)",
    borderRadius: 10,
    color: INK,
    fontSize: 12,
    boxShadow: "0 12px 32px rgba(0,0,0,0.35)",
  },
  labelStyle: { color: SUB, marginBottom: 4 },
  itemStyle: { color: INK },
  cursor: { fill: "rgba(255,255,255,0.05)" },
} as const;
export const LEGEND_STYLE = { color: SUB, fontSize: 12 };

/** SVG gradient used by the main chart series. Put it inside a chart and fill with url(#accent). */
export function AccentDefs({ id = "accent", vertical = false }: { id?: string; vertical?: boolean }) {
  return (
    <defs>
      <linearGradient id={id} x1="0" y1={vertical ? "1" : "0"} x2={vertical ? "0" : "1"} y2="0">
        <stop offset="0%" stopColor={ACCENT_FROM} />
        <stop offset="100%" stopColor={ACCENT_TO} />
      </linearGradient>
    </defs>
  );
}

// --- model colours ------------------------------------------------------------------

const ColorContext = createContext<(model: string) => string>(() => MODEL_PALETTE[0]);

/**
 * Assigns colours by the order models first appeared (oldest run first), so a model's colour
 * stays the same across pages and does not shift when newer models are added.
 */
export function ModelColorProvider({ runs, children }: { runs: RunSummary[] | undefined; children: ReactNode }) {
  const colorOf = useMemo(() => {
    const order: string[] = [];
    for (const r of [...(runs ?? [])].sort((a, b) => a.started_at.localeCompare(b.started_at))) {
      if (!order.includes(r.model)) order.push(r.model);
    }
    return (model: string) => {
      const i = order.indexOf(model);
      if (i >= 0) return MODEL_PALETTE[i % MODEL_PALETTE.length];
      // a model without runs yet: a stable pick from its name
      let h = 0;
      for (const c of model) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      return MODEL_PALETTE[h % MODEL_PALETTE.length];
    };
  }, [runs]);
  return <ColorContext.Provider value={colorOf}>{children}</ColorContext.Provider>;
}

export const useModelColor = () => useContext(ColorContext);
