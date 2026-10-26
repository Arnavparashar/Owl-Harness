// Chart theme and model colours, shared by every page.
import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { RunSummary } from "./api";

/**
 * Chart palette for models: emerald and black first, then colour-blind-safe hues (after Okabe-Ito,
 * darkened to read on light glass). Colours are assigned by the order models first appeared, so a
 * model keeps its colour on every page.
 */
export const MODEL_PALETTE = ["#059669", "#1c1a17", "#d97706", "#0072b2", "#cc79a7", "#d55e00", "#56b4e9", "#8a7f70"];

export const ACCENT_FROM = "#059669";
export const ACCENT_TO = "#0d9488";
export const EMERALD = "#059669";
export const BLACK = "#1c1a17";
export const OK = "#059669";
export const WARN = "#d97706";
export const BAD = "#dc2626";
export const INK = "#1f1c18";
export const SUB = "#38332d";
export const FAINT = "#403b34";

export const AXIS = { stroke: "rgba(58,50,38,0.14)", fontSize: 11, tickLine: false, axisLine: false, tick: { fill: FAINT, fontSize: 11 } } as const;
export const GRID = "rgba(58,50,38,0.1)";
/** The floating white glass pill tooltip, with dark bold text. */
export const TOOLTIP = {
  contentStyle: {
    background: "rgba(255,255,255,0.94)",
    border: "1px solid rgba(255,255,255,0.9)",
    borderRadius: 14,
    color: INK,
    fontSize: 12,
    fontWeight: 700,
    padding: "6px 12px",
    boxShadow: "0 12px 28px rgba(96,74,44,0.25)",
  },
  labelStyle: { color: SUB, fontWeight: 600, marginBottom: 2 },
  itemStyle: { color: INK, fontWeight: 700, padding: 0 },
  cursor: { fill: "rgba(58,50,38,0.06)" },
} as const;
export const LEGEND_STYLE = { color: SUB, fontSize: 12 };
/** Legends use small coloured dots beside each label. */
/** Legend text is set in dark ink: recharts would otherwise paint it in the series colour (pink on glass fails AA). */
export const LEGEND = {
  iconType: "circle",
  iconSize: 8,
  wrapperStyle: LEGEND_STYLE,
  formatter: (value: string) => <span style={{ color: SUB }}>{value}</span>,
} as const;

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
