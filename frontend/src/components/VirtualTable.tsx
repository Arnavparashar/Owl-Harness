// A table that only renders the rows in view, so thousands of rows scroll smoothly.
// Fixed row height; columns are laid out with a CSS grid template.
import { useState, type ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  align?: "left" | "right" | "center";
  onSort?: () => void;
  sorted?: "asc" | "desc" | null;
}

export function VirtualTable<T>({
  rows,
  columns,
  template,
  rowKey,
  rowHeight = 44,
  maxHeight = 560,
  rowClassName,
  onRowClick,
  empty,
}: {
  rows: T[];
  columns: Column<T>[];
  /** grid-template-columns for header and rows, e.g. "40px 1fr 120px" */
  template: string;
  rowKey: (row: T) => string;
  rowHeight?: number;
  maxHeight?: number;
  rowClassName?: (row: T) => string;
  onRowClick?: (row: T) => void;
  empty?: ReactNode;
}) {
  const [scrollTop, setScrollTop] = useState(0);
  const header = 38;
  const viewport = Math.min(maxHeight, header + Math.max(1, rows.length) * rowHeight);
  const overscan = 6;
  const first = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const last = Math.min(rows.length, Math.ceil((scrollTop + viewport) / rowHeight) + overscan);
  const align = (a?: string) => (a === "right" ? "justify-end text-right" : a === "center" ? "justify-center text-center" : "justify-start text-left");

  return (
    <div className="glass-dense overflow-hidden">
      <div className="overflow-auto" style={{ height: viewport }} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)} role="table">
        <div
          className="sticky top-0 z-10 grid items-center border-b border-white/10 bg-[#1a2033]/95 px-2"
          style={{ gridTemplateColumns: template, height: header, minWidth: "fit-content" }}
          role="row"
        >
          {columns.map((c) => (
            <div key={c.key} role="columnheader" className={`label flex items-center px-2 ${align(c.align)}`}>
              {c.onSort ? (
                <button type="button" onClick={c.onSort} className="label inline-flex items-center gap-1 hover:text-ink">
                  {c.header}
                  <span className="text-accent">{c.sorted === "asc" ? "↑" : c.sorted === "desc" ? "↓" : ""}</span>
                </button>
              ) : (
                c.header
              )}
            </div>
          ))}
        </div>
        {rows.length === 0 ? (
          <div className="px-4 py-3 text-sm text-faint">{empty ?? "Nothing to show."}</div>
        ) : (
          <div style={{ height: rows.length * rowHeight, position: "relative", minWidth: "fit-content" }}>
            {rows.slice(first, last).map((row, i) => (
              <div
                key={rowKey(row)}
                role="row"
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={`absolute inset-x-0 grid items-center border-b border-white/[0.06] px-2 ${onRowClick ? "cursor-pointer" : ""} ${
                  rowClassName?.(row) ?? "hover:bg-white/[0.04]"
                }`}
                style={{ top: (first + i) * rowHeight, height: rowHeight, gridTemplateColumns: template }}
              >
                {columns.map((c) => (
                  <div key={c.key} role="cell" className={`flex min-w-0 items-center px-2 text-[13px] text-ink ${align(c.align)}`}>
                    {c.render(row)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
