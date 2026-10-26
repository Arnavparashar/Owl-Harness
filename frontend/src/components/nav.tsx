// In-page tabs live in the floating bottom pill, not in the page. A page registers its tabs
// with useBottomTabs; the shell draws them. Pages with no tabs register nothing.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

export interface BottomTabs {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}

interface NavContext {
  current: BottomTabs | null;
  set: (value: BottomTabs | null) => void;
}

const Ctx = createContext<NavContext>({ current: null, set: () => undefined });

export function BottomNavProvider({ children }: { children: ReactNode }) {
  const [current, setCurrent] = useState<BottomTabs | null>(null);
  const set = useCallback((value: BottomTabs | null) => setCurrent(value), []);
  const value = useMemo(() => ({ current, set }), [current, set]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useBottomNav = () => useContext(Ctx).current;

/** Show `tabs` in the bottom pill while the calling page is mounted. */
export function useBottomTabs<T extends string>(tabs: { id: T; label: string }[], active: T, onChange: (id: T) => void) {
  const { set } = useContext(Ctx);
  const callback = useRef(onChange);
  callback.current = onChange;
  // re-register only when what is shown changes, not on every render
  const key = `${tabs.map((t) => `${t.id}:${t.label}`).join("|")}@${active}`;
  useEffect(() => {
    set({ tabs, active, onChange: (id) => callback.current(id as T) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, set]);
  useEffect(() => () => set(null), [set]);
}
