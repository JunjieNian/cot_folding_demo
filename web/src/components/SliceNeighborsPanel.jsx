import React, { useState, useEffect, useRef } from "react";
import { getSliceNeighbors } from "../api";

export default function SliceNeighborsPanel({ problemId, sampleId, selectedSlice, onSliceClick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const cacheRef = useRef({ key: null, data: null });

  // Fetch neighbors once per sample (lazy on first expand)
  useEffect(() => {
    if (problemId == null || sampleId == null) return;
    const key = `${problemId}-${sampleId}`;
    if (cacheRef.current.key === key) {
      setData(cacheRef.current.data);
      return;
    }
    setData(null);
  }, [problemId, sampleId]);

  useEffect(() => {
    if (!open || data || loading) return;
    if (problemId == null || sampleId == null) return;
    const key = `${problemId}-${sampleId}`;
    if (cacheRef.current.key === key) {
      setData(cacheRef.current.data);
      return;
    }
    setLoading(true);
    const controller = new AbortController();
    getSliceNeighbors(problemId, sampleId, controller.signal)
      .then((d) => {
        cacheRef.current = { key, data: d };
        setData(d);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [open, data, loading, problemId, sampleId]);

  if (typeof selectedSlice !== "number") return null;

  const slice = data?.slices?.[selectedSlice];

  const pctDiff = slice
    ? Math.round(((slice.structural_mean_text_sim - slice.sequential_mean_text_sim) / slice.sequential_mean_text_sim) * 100)
    : null;

  return (
    <div style={{
      borderBottom: "1px solid var(--color-border-light)",
      fontSize: 11,
      color: "var(--color-text-faint)",
    }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%", background: "none", border: "none", cursor: "pointer",
          padding: "6px 12px", textAlign: "left",
          fontSize: 11, color: "var(--color-text)", fontWeight: 500,
          display: "flex", alignItems: "center", gap: 4,
        }}
      >
        <span style={{ fontSize: 9 }}>{open ? "\u25BC" : "\u25B6"}</span>
        Structural Neighbors
        {slice && pctDiff != null && (
          <span style={{
            marginLeft: "auto", fontSize: 10, fontFamily: "monospace",
            color: pctDiff > 0 ? "#4CAF50" : "var(--color-text-faint)",
          }}>
            {pctDiff > 0 ? "+" : ""}{pctDiff}%
          </span>
        )}
      </button>

      {open && (
        <div style={{ padding: "0 12px 8px" }}>
          {loading && <div style={{ fontSize: 10, color: "var(--color-text-faint)" }}>Loading neighbors...</div>}
          {!loading && !slice && data && (
            <div style={{ fontSize: 10, color: "var(--color-text-faint)" }}>No neighbor data for this slice</div>
          )}
          {!loading && !data && !loading && (
            <div style={{ fontSize: 10, color: "var(--color-text-faint)" }}>Neighbor data unavailable</div>
          )}
          {slice && (
            <>
              <NeighborRow
                label="Nearest by structure (top-5)"
                indices={slice.structural_top5}
                sims={slice.structural_sims}
                meanSim={slice.structural_mean_text_sim}
                onSliceClick={onSliceClick}
              />
              <NeighborRow
                label="Nearest by position"
                indices={slice.sequential_top5}
                sims={slice.sequential_sims}
                meanSim={slice.sequential_mean_text_sim}
                onSliceClick={onSliceClick}
              />
              {pctDiff != null && (
                <div style={{
                  marginTop: 4, fontSize: 10, fontFamily: "monospace",
                  color: pctDiff > 0 ? "#4CAF50" : "var(--color-text-faint)",
                }}>
                  structural {pctDiff > 0 ? ">" : pctDiff === 0 ? "=" : "<"} sequential ({pctDiff > 0 ? "+" : ""}{pctDiff}%)
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function NeighborRow({ label, indices, sims, meanSim, onSliceClick }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: "var(--color-text-faint)", marginBottom: 2 }}>{label}:</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 6px", fontFamily: "monospace", fontSize: 10 }}>
        {indices.map((idx, i) => (
          <span
            key={i}
            onClick={() => onSliceClick?.(idx)}
            style={{
              cursor: "pointer", color: "var(--color-primary)",
              textDecoration: "underline", textDecorationStyle: "dotted",
            }}
            title={`Jump to slice ${idx}`}
          >
            S{idx} <span style={{ color: "var(--color-text-faint)" }}>{sims[i].toFixed(2)}</span>
          </span>
        ))}
      </div>
      <div style={{ fontSize: 10, color: "var(--color-text-faint)", marginTop: 1 }}>
        avg text sim: <span style={{ fontWeight: 600, fontFamily: "monospace" }}>{meanSim.toFixed(3)}</span>
      </div>
    </div>
  );
}
