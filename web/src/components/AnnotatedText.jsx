import React, { useRef, useEffect, useCallback } from "react";
import styles from "./CaseStudy.module.css";

/* ─── Label → background color mapping ─── */
const LABEL_COLORS = {
  core:                "rgba(76,175,80,0.25)",
  closure:             "rgba(76,175,80,0.20)",
  return_site:         "rgba(255,152,0,0.18)",
  productive_explore:  "rgba(158,158,158,0.12)",
  productive_exploit:  "rgba(158,158,158,0.12)",
  drift:               "rgba(244,67,54,0.25)",
  explore:             "rgba(158,158,158,0.08)",
  exploit:             "rgba(158,158,158,0.08)",
};

function getLabelColor(label) {
  if (!label) return "transparent";
  return LABEL_COLORS[label] ?? "rgba(158,158,158,0.06)";
}

/* ─── Legend items ─── */
const LEGEND = [
  { label: "Core",        color: "rgba(76,175,80,0.45)" },
  { label: "Closure",     color: "rgba(76,175,80,0.35)" },
  { label: "Return Site", color: "rgba(255,152,0,0.35)" },
  { label: "Drift",       color: "rgba(244,67,54,0.45)" },
  { label: "Productive",  color: "rgba(158,158,158,0.25)" },
];

/**
 * AnnotatedText — renders CoT full text with per-slice color highlighting,
 * a minimap bar, and return-edge pulse animation.
 *
 * Props:
 *   fullText      — the raw CoT text string
 *   items         — [{slice_idx, char_start, char_end}, ...]
 *   labels        — per-slice label array from effectiveness.labels
 *   activeEdge    — {i, j} or null — two slice indices to pulse-highlight
 *   onSliceClick  — (sliceIdx) => void
 */
export default function AnnotatedText({
  fullText, items, labels,
  activeEdge, onSliceClick,
  showLegend = true,
}) {
  const textRef = useRef(null);
  const sliceRefs = useRef({});

  // Auto-scroll to active edge endpoints
  useEffect(() => {
    if (!activeEdge) return;
    const el = sliceRefs.current[activeEdge.i];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [activeEdge]);

  const setSliceRef = useCallback((idx, el) => {
    if (el) sliceRefs.current[idx] = el;
  }, []);

  if (!fullText || !items || items.length === 0) {
    return <div className={styles.textArea} style={{ color: "var(--color-text-muted)" }}>No text data available</div>;
  }

  // Build slices — each maps to a text range
  const slices = items.map((item) => {
    const idx = item.slice_idx;
    const text = fullText.slice(item.char_start, item.char_end);
    const label = labels?.[idx] ?? null;
    return { idx, text, label, color: getLabelColor(label) };
  });

  const isActive = (idx) =>
    activeEdge && (idx === activeEdge.i || idx === activeEdge.j);

  return (
    <>
      {/* Legend */}
      {showLegend && (
        <div className={styles.legend}>
          {LEGEND.map((l) => (
            <div key={l.label} className={styles.legendItem}>
              <div className={styles.legendSwatch} style={{ background: l.color }} />
              {l.label}
            </div>
          ))}
        </div>
      )}

      {/* Minimap bar */}
      <div className={styles.minimap} title="Sequence minimap — click to scroll">
        {slices.map((s) => (
          <div
            key={s.idx}
            className={styles.minimapSlice}
            style={{ background: s.color === "transparent" ? "var(--color-border-light)" : s.color }}
            onClick={() => {
              const el = sliceRefs.current[s.idx];
              if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
          />
        ))}
      </div>

      {/* Full text with colored spans */}
      <div className={styles.textArea} ref={textRef}>
        {slices.map((s) => (
          <span
            key={s.idx}
            ref={(el) => setSliceRef(s.idx, el)}
            className={`${styles.sliceSpan}${isActive(s.idx) ? ` ${styles.sliceActive}` : ""}`}
            style={{ backgroundColor: s.color }}
            title={`Slice ${s.idx}${s.label ? ` (${s.label})` : ""}`}
            onClick={() => onSliceClick?.(s.idx)}
          >
            {s.text}
          </span>
        ))}
      </div>
    </>
  );
}
