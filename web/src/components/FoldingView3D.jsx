import React, { useMemo } from "react";
import Plot from "./Plot";

const EXPLORE = "#5B8DEF";
const EXPLOIT = "#E05A47";
const BG = "transparent";

function FoldingView3D({ data, colorMode = "entropy", onSliceClick, compact = false, focusMode = "global", answerIsland = null, hoveredSlice = null }) {
  const { mds_coords_3d, similarity, similarity_shape, hmm_states, entropy, confidence, mds_stress_3d, effectiveness } = data;
  if (!mds_coords_3d) return <div style={{ padding: 40, textAlign: "center", color: "#999" }}>No 3D data</div>;

  const n = similarity_shape[0];
  const unitLabel = data.unit_label || "slice";
  const unitTitle = unitLabel.charAt(0).toUpperCase() + unitLabel.slice(1);
  const hasEffectiveness = effectiveness && effectiveness.scores;

  // Compute tail range directly from hmm_states — last contiguous exploit run
  const tailRange = useMemo(() => {
    if (focusMode !== "answer-tail") return null;
    if (!hmm_states || hmm_states.length < 2) return null;
    const len = hmm_states.length;
    if (hmm_states[len - 1] !== 1) return null;
    let start = len - 1;
    while (start > 0 && hmm_states[start - 1] === 1) start--;
    if (len - 1 - start < 1) return null;
    return { tailStart: start, tailEnd: len - 1 };
  }, [hmm_states, focusMode]);

  const isTailFocus = tailRange !== null;
  const isExFocus = false;

  const { traces, cameraOverride } = useMemo(() => {
    const xs = mds_coords_3d.map((p) => p[0]);
    const ys = mds_coords_3d.map((p) => p[1]);
    const zs = mds_coords_3d.map((p) => p[2]);
    const result = [];
    let camOverride = null;

    // Backbone segments by HMM state
    let start = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || hmm_states[i] !== hmm_states[start]) {
        const end = Math.min(i - 1, n - 1);
        const sx = [], sy = [], sz = [];
        for (let j = start; j <= end; j++) { sx.push(xs[j]); sy.push(ys[j]); sz.push(zs[j]); }
        const isE = hmm_states[start] === 0;

        let lineWidth = 4;
        let lineColor = isE ? EXPLORE : EXPLOIT;
        let lineOpacity = 1;

        if (isExFocus) {
          lineWidth = 6;
        }

        if (isTailFocus) {
          // Check if this segment overlaps with tail
          const segOverlapsTail = end >= tailRange.tailStart && start <= tailRange.tailEnd;
          lineOpacity = segOverlapsTail ? 1 : 0.1;
          lineColor = segOverlapsTail
            ? (isE ? EXPLORE : EXPLOIT)
            : `rgba(128,128,128,0.1)`;
        }

        result.push({
          x: sx, y: sy, z: sz, mode: "lines", type: "scatter3d",
          line: { color: lineColor, width: lineWidth },
          opacity: lineOpacity,
          hoverinfo: "skip", showlegend: false,
        });
        start = i;
      }
    }

    // Nodes
    const entMin = Math.min(...entropy), entMax = Math.max(...entropy);
    const entRange = entMax - entMin || 1e-8;
    const entNorm = entropy.map((e) => (e - entMin) / entRange);

    let nodeColor, nodeColorscale, nodeColorbar, nodeSize;
    const sz = compact ? 0.45 : 1;  // compact mode: shrink nodes so edges stay visible

    if (colorMode === "effectiveness" && hasEffectiveness) {
      nodeColor = effectiveness.scores;
      nodeColorscale = [[0, "#D32F2F"], [0.25, "#FF5722"], [0.5, "#FF9800"], [0.75, "#8BC34A"], [1, "#2E7D32"]];
      nodeColorbar = { title: { text: "Effective", font: { size: 10 } }, thickness: 10, len: 0.5 };
      nodeSize = effectiveness.scores.map((s) => (4 + 14 * s) * sz);
    } else if (colorMode === "entropy") {
      nodeColor = entropy;
      nodeColorscale = [[0, "#FFF7EC"], [0.3, "#FDD49E"], [0.6, "#F16913"], [1, "#8C2D04"]];
      nodeColorbar = { title: { text: "Entropy", font: { size: 10 } }, thickness: 10, len: 0.5 };
      nodeSize = entNorm.map((e) => (4 + 12 * e) * sz);
    } else if (colorMode === "confidence") {
      nodeColor = confidence;
      nodeColorscale = [[0, "#EFF3FF"], [0.3, "#BDD7E7"], [0.6, "#3182BD"], [1, "#08519C"]];
      nodeColorbar = { title: { text: "Conf.", font: { size: 10 } }, thickness: 10, len: 0.5 };
      nodeSize = 7 * sz;
    } else {
      nodeColor = hmm_states.map((s) => s === 0 ? EXPLORE : EXPLOIT);
      nodeColorscale = undefined;
      nodeColorbar = undefined;
      nodeSize = 6 * sz;
    }

    // Answer-tail focus: adjust node size and opacity
    let nodeOpacity = 0.9;
    if (isTailFocus) {
      nodeOpacity = Array.from({ length: n }, (_, i) =>
        i >= tailRange.tailStart && i <= tailRange.tailEnd ? 0.95 : 0.12
      );
      // Enlarge tail nodes, shrink others
      const baseSize = Array.isArray(nodeSize) ? nodeSize : Array(n).fill(nodeSize);
      nodeSize = baseSize.map((s, i) =>
        i >= tailRange.tailStart && i <= tailRange.tailEnd ? s * 1.5 : s * 0.6
      );

      // Camera: aim at tail centroid
      const tailCoords = mds_coords_3d.slice(tailRange.tailStart, tailRange.tailEnd + 1);
      const tc = tailCoords.reduce((acc, c) => [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]], [0, 0, 0])
        .map((v) => v / tailCoords.length);
      camOverride = {
        eye: { x: tc[0] + 0.8, y: tc[1] + 0.8, z: tc[2] + 0.5 },
        center: { x: tc[0], y: tc[1], z: tc[2] },
      };
    }

    result.push({
      x: xs, y: ys, z: zs, mode: "markers", type: "scatter3d",
      marker: {
        size: nodeSize,
        color: nodeColor,
        colorscale: nodeColorscale,
        colorbar: nodeColorbar,
        opacity: nodeOpacity,
        line: { color: "rgba(128,128,128,0.3)", width: 0.5 },
      },
      customdata: Array.from({ length: n }, (_, i) => i),
      text: Array.from({ length: n }, (_, i) => {
        let txt = `<b>${unitTitle} ${i}</b><br>${hmm_states[i] === 0 ? "Explore" : "Exploit"}<br>Entropy: ${entropy[i].toFixed(3)}<br>Conf: ${confidence[i].toFixed(2)}`;
        if (hasEffectiveness) txt += `<br>Eff: ${(effectiveness.scores[i] * 100).toFixed(0)}%<br>Label: ${effectiveness.labels[i]}`;
        return txt;
      }),
      hoverinfo: "text", showlegend: false,
    });

    // Start/End markers
    result.push({
      x: [xs[0]], y: [ys[0]], z: [zs[0]], mode: "markers+text", type: "scatter3d",
      marker: { size: 10, color: "#2E7D32", symbol: "diamond" },
      text: ["N"], textposition: "top center", textfont: { size: 10, color: "#2E7D32" },
      hoverinfo: "skip", showlegend: false,
    });
    result.push({
      x: [xs[n - 1]], y: [ys[n - 1]], z: [zs[n - 1]], mode: "markers+text", type: "scatter3d",
      marker: { size: 10, color: "#E65100", symbol: "square" },
      text: ["C"], textposition: "top center", textfont: { size: 10, color: "#E65100" },
      hoverinfo: "skip", showlegend: false,
    });

    // Hovered node highlight (from text panel hover) — bright glow
    if (hoveredSlice != null && hoveredSlice >= 0 && hoveredSlice < n) {
      // Large glow sphere
      result.push({
        x: [xs[hoveredSlice]], y: [ys[hoveredSlice]], z: [zs[hoveredSlice]],
        mode: "markers", type: "scatter3d",
        marker: { size: 22, color: "rgba(255,214,0,0.35)", line: { color: "#FFD600", width: 2.5 }, symbol: "circle" },
        hoverinfo: "skip", showlegend: false,
      });
      // Inner bright dot
      result.push({
        x: [xs[hoveredSlice]], y: [ys[hoveredSlice]], z: [zs[hoveredSlice]],
        mode: "markers", type: "scatter3d",
        marker: { size: 10, color: "#FFD600", line: { color: "#FF6F00", width: 2 }, symbol: "circle" },
        hoverinfo: "skip", showlegend: false,
      });
    }

    return { traces: result, cameraOverride: camOverride };
  }, [mds_coords_3d, hmm_states, entropy, confidence, colorMode, effectiveness, hasEffectiveness, n, tailRange, isTailFocus, hoveredSlice]);

  return (
    <div style={{ width: "100%", height: "100%", background: BG, borderRadius: 6, overflow: "hidden" }}>
      <Plot
        data={traces}
        layout={{
          scene: {
            xaxis: { visible: false },
            yaxis: { visible: false },
            zaxis: { visible: false },
            bgcolor: BG,
            camera: cameraOverride || { eye: { x: 1.5, y: 1.5, z: 1.0 } },
          },
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: BG,
          annotations: [
            {
              x: 1, y: 1, xref: "paper", yref: "paper", xanchor: "right", yanchor: "top",
              text: `<b>3D MDS</b> stress: ${(mds_stress_3d ?? 0).toFixed(3)}` +
                (data.is_correct != null
                  ? `<br><b style="color:${data.is_correct ? "#4CAF50" : "#F44336"}">${data.is_correct ? "\u2713 Correct" : "\u2717 Incorrect"}</b>`
                  : ""),
              showarrow: false, font: { size: 11 },
              bgcolor: "rgba(128,128,128,0.15)", borderpad: 4,
            },
          ],
        }}
        useResizeHandler style={{ width: "100%", height: "100%" }}
        config={{ responsive: true }}
        onClick={(event) => {
          if (onSliceClick && event.points?.[0]?.customdata != null) {
            onSliceClick(event.points[0].customdata);
          }
        }}
      />
    </div>
  );
}

export default React.memo(FoldingView3D);
