import React, { useMemo, useEffect, useRef } from "react";

const EXPLORE_BG        = "rgba(91,141,239,0.08)";
const EXPLOIT_BG        = "rgba(224,90,71,0.08)";
const EXPLORE_BG_HOVER  = "rgba(91,141,239,0.22)";
const EXPLOIT_BG_HOVER  = "rgba(224,90,71,0.22)";
const EXPLORE_BG_SEL    = "rgba(91,141,239,0.35)";
const EXPLOIT_BG_SEL    = "rgba(224,90,71,0.35)";
const EXPLORE_BORDER    = "rgba(91,141,239,0.6)";
const EXPLOIT_BORDER    = "rgba(224,90,71,0.6)";

// Answer-tail focus styles
const TAIL_BG           = "rgba(255,152,0,0.18)";
const TAIL_BG_HOVER     = "rgba(255,152,0,0.35)";
const TAIL_BG_SEL       = "rgba(255,152,0,0.50)";
const TAIL_BORDER       = "rgba(255,152,0,0.8)";
const NONTAIL_OPACITY   = 0.25;

export default function SegmentedText({
  textBundle, hmmStates,
  hoveredSlice, selectedSlice,
  onHoverSlice, onClickSlice,
  focusMode = "global", answerIsland = null,
}) {
  const selectedRef = useRef(null);
  const isTailFocus = focusMode === "answer-tail" && answerIsland;

  // Auto-scroll to selected segment
  useEffect(() => {
    if (selectedSlice != null && selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedSlice]);

  const segments = useMemo(() => {
    if (!textBundle?.items?.length || !textBundle.full_text) return null;
    const items = textBundle.items.slice().sort((a, b) => a.char_start - b.char_start);
    const text = textBundle.full_text;
    const result = [];
    let cursor = 0;

    for (const item of items) {
      if (item.char_start > cursor) {
        result.push({ type: "gap", text: text.slice(cursor, item.char_start) });
      }
      result.push({
        type: "slice",
        sliceIdx: item.slice_idx,
        text: text.slice(item.char_start, item.char_end),
      });
      cursor = item.char_end;
    }
    if (cursor < text.length) {
      result.push({ type: "gap", text: text.slice(cursor) });
    }
    return result;
  }, [textBundle]);

  if (!segments) return null;

  return (
    <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
      {segments.map((seg, i) => {
        if (seg.type === "gap") {
          return (
            <span key={i} style={{
              color: "var(--color-text-muted)",
              opacity: isTailFocus ? NONTAIL_OPACITY : 1,
            }}>{seg.text}</span>
          );
        }

        const isExplore  = hmmStates?.[seg.sliceIdx] === 0;
        const isHovered  = hoveredSlice === seg.sliceIdx;
        const isSelected = selectedSlice === seg.sliceIdx;
        const inTail     = isTailFocus &&
          seg.sliceIdx >= answerIsland.tailStart &&
          seg.sliceIdx <= answerIsland.tailEnd;
        const outOfTail  = isTailFocus && !inTail;

        let bg, border, opacity = 1, fontWeight = "normal";

        if (isTailFocus) {
          if (outOfTail) {
            // Dim everything outside the tail
            bg = "transparent";
            border = "2px solid transparent";
            opacity = NONTAIL_OPACITY;
          } else if (isSelected) {
            bg = TAIL_BG_SEL; border = `2px solid ${TAIL_BORDER}`; fontWeight = "600";
          } else if (isHovered) {
            bg = TAIL_BG_HOVER; border = `2px solid ${TAIL_BORDER}`;
          } else {
            bg = TAIL_BG; border = `2px solid ${TAIL_BORDER}`;
          }
        } else {
          if (isSelected) {
            bg = isExplore ? EXPLORE_BG_SEL : EXPLOIT_BG_SEL;
            border = `2px solid ${isExplore ? EXPLORE_BORDER : EXPLOIT_BORDER}`;
            fontWeight = "600";
          } else if (isHovered) {
            bg = isExplore ? EXPLORE_BG_HOVER : EXPLOIT_BG_HOVER;
            border = `2px solid ${isExplore ? EXPLORE_BORDER : EXPLOIT_BORDER}`;
          } else {
            bg = isExplore ? EXPLORE_BG : EXPLOIT_BG;
            border = "2px solid transparent";
          }
        }

        return (
          <span
            key={i}
            ref={isSelected ? selectedRef : undefined}
            data-slice={seg.sliceIdx}
            onMouseEnter={() => onHoverSlice(seg.sliceIdx)}
            onMouseLeave={() => onHoverSlice(null)}
            onClick={() => onClickSlice(seg.sliceIdx)}
            style={{
              cursor: "pointer",
              background: bg,
              borderBottom: border,
              borderRadius: 2,
              opacity,
              fontWeight,
              transition: "background 0.1s, border-color 0.1s, opacity 0.15s",
            }}
          >{seg.text}</span>
        );
      })}
    </div>
  );
}
