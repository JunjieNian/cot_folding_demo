import React, { useMemo } from "react";
import Plot from "./Plot";

function ContactMap({ data, decodedSimilarity = null, similarityLoading = false, onSliceClick }) {
  const { similarity, similarity_shape, hmm_states } = data;
  const n = similarity_shape[0];

  // Use pre-decoded similarity from parent, or raw array from data
  const flat = decodedSimilarity || similarity || null;

  // Reconstruct 2D matrix
  const z = useMemo(() => {
    if (!flat) return null;
    const matrix = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) row.push(flat[i * n + j]);
      matrix.push(row);
    }
    return matrix;
  }, [flat, n]);

  // HMM state color band as shapes
  const shapes = useMemo(() => {
    const s = [];
    const bw = Math.max(2, n / 40);
    for (let i = 0; i < n; i++) {
      const color = hmm_states[i] === 0 ? "#4285F4" : "#EA4335";
      // Top band
      s.push({
        type: "rect", x0: i - 0.5, x1: i + 0.5, y0: -bw - 0.5, y1: -0.5,
        fillcolor: color, line: { width: 0 }, layer: "above",
      });
      // Left band
      s.push({
        type: "rect", x0: -bw - 0.5, x1: -0.5, y0: i - 0.5, y1: i + 0.5,
        fillcolor: color, line: { width: 0 }, layer: "above",
      });
    }
    return s;
  }, [hmm_states, n]);

  // Loading state: similarity not yet available
  if (!z) {
    return (
      <div style={{
        width: "100%", height: "100%",
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--color-text-secondary, #888)",
        fontSize: 14,
      }}>
        {similarityLoading ? "Loading contact map..." : "Contact map loads in Detail view"}
      </div>
    );
  }

  return (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <div style={{ aspectRatio: "1 / 1", maxWidth: "100%", maxHeight: "100%", width: "100%" }}>
        <Plot
          data={[{
            z,
            type: "heatmap",
            colorscale: "RdYlBu",
            reversescale: true,
            zmin: 0, zmax: 1,
            hovertemplate: "Slice %{x} vs %{y}<br>Similarity: %{z:.4f}<extra></extra>",
            colorbar: { title: "Sim", len: 0.9 },
          }]}
          layout={{
            title: { text: "Contact Map (Similarity)", font: { size: 14 } },
            xaxis: { title: "Slice Index", range: [-Math.max(2, n / 40) - 1, n - 0.5] },
            yaxis: { title: "Slice Index", range: [n - 0.5, -Math.max(2, n / 40) - 1], scaleanchor: "x" },
            shapes,
            margin: { l: 60, r: 20, t: 40, b: 50 },
            paper_bgcolor: "transparent",
          }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ responsive: true }}
          onClick={(event) => {
            if (onSliceClick && event.points && event.points.length > 0) {
              const pt = event.points[0];
              const sliceIdx = Math.round(pt.x);
              if (sliceIdx >= 0 && sliceIdx < n) onSliceClick(sliceIdx);
            }
          }}
        />
      </div>
    </div>
  );
}

export default React.memo(ContactMap);
