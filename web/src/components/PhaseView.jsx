import React, { useMemo } from "react";
import Plot from "./Plot";

const EXPLORE = "#5B8DEF";
const EXPLOIT = "#E05A47";
const BOND_COLOR = "rgba(220,180,60,0.25)";
const BG = "transparent";

function PhaseView({ data, onSliceClick, compact = false }) {
  const phases = data?.phases;
  if (!phases || !phases.items || phases.count === 0) {
    return <div style={{ padding: 40, textAlign: "center", color: "#999" }}>No phase data</div>;
  }

  const items = phases.items;
  const n = phases.count;
  const unitLabel = data.unit_label || "slice";

  // ═══════════════════════════════════════════════
  //  TOP PLOT: Phase-level folded structure
  // ═══════════════════════════════════════════════

  const { topTraces, topAnnotations } = useMemo(() => {
    // Normalize MDS coords to [-1, 1]
    const rawX = items.map((p) => p.mds_x);
    const rawY = items.map((p) => p.mds_y);
    const cx = (Math.min(...rawX) + Math.max(...rawX)) / 2;
    const cy = (Math.min(...rawY) + Math.max(...rawY)) / 2;
    const span = Math.max(Math.max(...rawX) - Math.min(...rawX), Math.max(...rawY) - Math.min(...rawY)) || 1;
    const s = 1.0 / span;
    const mx = rawX.map((x) => (x - cx) * s);
    const my = rawY.map((y) => (y - cy) * s);

    const traces = [];
    const annotations = [];

    // ── Backbone: blue (explore) / red (exploit) ──
    for (let i = 0; i < n - 1; i++) {
      const color = items[i].state === 0 ? EXPLORE : EXPLOIT;

      traces.push({
        x: [mx[i], mx[i + 1]], y: [my[i], my[i + 1]], mode: "lines",
        line: { color, width: 3, shape: "spline" },
        hoverinfo: "skip", showlegend: false,
      });
    }

    // ── Phase nodes ──
    const sizes = items.map((p) => {
      const base = 10 + Math.sqrt(p.length) * 4;
      return Math.min(55, base);
    });

    traces.push({
      x: mx, y: my, mode: "markers+text",
      marker: {
        size: sizes,
        color: items.map((p) => p.state === 0 ? EXPLORE : EXPLOIT),
        opacity: 0.9,
        symbol: "circle",
        line: {
          color: "rgba(200,200,200,0.8)",
          width: 1.5,
        },
      },
      text: items.map(() => ""),
      textfont: { size: 9, color: "rgba(200,200,200,0.9)", family: "Arial Black" },
      customdata: items.map((p, i) => ({ start: p.start, end: p.end, phaseIdx: i, stateName: p.state_name })),
      hovertext: items.map((p, i) =>
        `<b>Phase ${i + 1}: ${p.state_name.toUpperCase()}</b><br>` +
        `${unitLabel}s ${p.slice_range} (${p.length} ${unitLabel}s)<br>` +
        `<br>Entropy: ${p.mean_entropy.toFixed(3)}` +
        `<br>Confidence: ${p.mean_confidence.toFixed(3)}`
      ),
      hoverinfo: "text",
      showlegend: false,
    });

    // ── Start / End markers ──
    traces.push({
      x: [mx[0]], y: [my[0]], mode: "markers+text",
      marker: { size: 13, color: "#2E7D32", symbol: "diamond", line: { color: "rgba(128,128,128,0.3)", width: 2 } },
      text: ["N"], textposition: "top center",
      textfont: { size: 10, color: "#2E7D32", family: "Arial Black" },
      hovertext: [`START (Phase 1, ${items[0].state_name})`],
      hoverinfo: "text", showlegend: false,
    });
    traces.push({
      x: [mx[n - 1]], y: [my[n - 1]], mode: "markers+text",
      marker: { size: 13, color: "#E65100", symbol: "square", line: { color: "rgba(128,128,128,0.3)", width: 2 } },
      text: ["C"], textposition: "top center",
      textfont: { size: 10, color: "#E65100", family: "Arial Black" },
      hovertext: [`END (Phase ${n}, ${items[n - 1].state_name})`],
      hoverinfo: "text", showlegend: false,
    });

    // ── Direction arrows (every 3 phases) ──
    const step = Math.max(1, Math.floor(n / 10));
    for (let i = 0; i < n - 1; i += step) {
      annotations.push({
        x: mx[i + 1], y: my[i + 1],
        ax: mx[i], ay: my[i],
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true,
        arrowhead: 3, arrowsize: 1.2, arrowwidth: 1.5,
        arrowcolor: items[i].state === 0 ? "rgba(91,141,239,0.5)" : "rgba(224,90,71,0.5)",
      });
    }

    return { topTraces: traces, topAnnotations: annotations };
  }, [items, n, phases.similarity]);

  // ═══════════════════════════════════════════════
  //  BOTTOM PLOT: Annotation tracks (like FoldingArcDiagram)
  // ═══════════════════════════════════════════════

  const bottomTraces = useMemo(() => {
    const traces = [];
    const idx = Array.from({ length: n }, (_, i) => i);

    // Track 1: State bar (explore/exploit) at y=[2, 2.5]
    for (let i = 0; i < n; i++) {
      traces.push({
        x: [i], y: [2.5], type: "bar",
        marker: { color: items[i].state === 0 ? EXPLORE : EXPLOIT, line: { width: 0 } },
        width: 0.95, base: [2],
        hovertemplate: `Phase ${i + 1}: ${items[i].state_name}<extra></extra>`,
        showlegend: false,
      });
    }

    // Track 2: Length bar at y=[0.9, variable], width proportional
    const maxLen = Math.max(...items.map((p) => p.length));
    traces.push({
      x: idx, y: idx.map(() => 0.9), mode: "lines",
      line: { color: "transparent", width: 0 }, hoverinfo: "skip", showlegend: false,
    });
    traces.push({
      x: idx,
      y: items.map((p) => 0.9 + (p.length / maxLen) * 0.85),
      mode: "lines", fill: "tonexty",
      line: { color: "#78909C", width: 1.5 },
      fillcolor: "rgba(120,144,156,0.2)",
      hovertemplate: items.map((p, i) => `Phase ${i + 1}<br>${p.length} ${unitLabel}s<extra></extra>`),
      showlegend: false,
    });

    // Track 3: Entropy area
    const entMax = Math.max(...items.map((p) => p.mean_entropy)) || 1;
    traces.push({
      x: idx, y: idx.map(() => -0.1), mode: "lines",
      line: { color: "transparent", width: 0 }, hoverinfo: "skip", showlegend: false,
    });
    traces.push({
      x: idx,
      y: items.map((p) => -0.1 + (p.mean_entropy / entMax) * 0.85),
      mode: "lines", fill: "tonexty",
      line: { color: "#F16913", width: 1.5 },
      fillcolor: "rgba(241,105,19,0.2)",
      hovertemplate: items.map((p, i) => `Phase ${i + 1}<br>Entropy: ${p.mean_entropy.toFixed(3)}<extra></extra>`),
      showlegend: false,
    });

    // Track 4: Confidence area
    const confMax = Math.max(...items.map((p) => p.mean_confidence)) || 1;
    traces.push({
      x: idx, y: idx.map(() => -1.2), mode: "lines",
      line: { color: "transparent", width: 0 }, hoverinfo: "skip", showlegend: false,
    });
    traces.push({
      x: idx,
      y: items.map((p) => -1.2 + (p.mean_confidence / confMax) * 0.85),
      mode: "lines", fill: "tonexty",
      line: { color: "#3182BD", width: 1.5 },
      fillcolor: "rgba(49,130,189,0.2)",
      hovertemplate: items.map((p, i) => `Phase ${i + 1}<br>Confidence: ${p.mean_confidence.toFixed(3)}<extra></extra>`),
      showlegend: false,
    });

    return traces;
  }, [items, n]);

  const bottomAnnotations = [
    { x: -0.01, y: 2.25, xref: "paper", text: "State", showarrow: false, font: { size: 9, color: "#888" }, xanchor: "right" },
    { x: -0.01, y: 1.3, xref: "paper", text: "Length", showarrow: false, font: { size: 9, color: "#78909C" }, xanchor: "right" },
    { x: -0.01, y: 0.3, xref: "paper", text: "Entropy", showarrow: false, font: { size: 9, color: "#F16913" }, xanchor: "right" },
    { x: -0.01, y: -0.8, xref: "paper", text: "Conf.", showarrow: false, font: { size: 9, color: "#3182BD" }, xanchor: "right" },
  ];

  // Summary
  const explorePhases = items.filter((p) => p.state === 0);
  const exploitPhases = items.filter((p) => p.state === 1);

  const pad = 0.15;
  const allMdsX = items.map((p) => p.mds_x);
  const allMdsY = items.map((p) => p.mds_y);

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: BG, borderRadius: 6, overflow: "hidden" }}>

      {/* ── Folded Phase Structure ── */}
      <div style={{ flex: compact ? 1 : 7, minHeight: 0 }}>
        <Plot
          data={topTraces}
          layout={{
            xaxis: { visible: false },
            yaxis: { visible: false, scaleanchor: "x" },
            annotations: [
              ...topAnnotations,
              {
                x: 1, y: 1, xref: "paper", yref: "paper", xanchor: "right", yanchor: "top",
                text: (data.is_correct != null
                  ? `<b style="color:${data.is_correct ? "#4CAF50" : "#F44336"}">${data.is_correct ? "\u2713 Correct" : "\u2717 Incorrect"}</b>` +
                    (data.nfs != null ? `  NFS:${data.nfs.toFixed(1)}` : "") + `<br>`
                  : "") +
                  `<b>${n} Phases</b>  ` +
                  `<span style="color:${EXPLORE}">${explorePhases.length}E</span> ` +
                  `<span style="color:${EXPLOIT}">${exploitPhases.length}X</span>`,
                showarrow: false,
                font: { size: 11 },
                bgcolor: "rgba(128,128,128,0.15)", borderpad: 5,
              },
            ],
            legend: { x: 1, y: 1, xanchor: "right", yanchor: "top", bgcolor: "rgba(128,128,128,0.15)", font: { size: 10 } },
            margin: { l: 5, r: 5, t: 5, b: 5 },
            paper_bgcolor: BG, plot_bgcolor: BG,
          }}
          useResizeHandler style={{ width: "100%", height: "100%" }}
          config={{ responsive: true, displayModeBar: false }}
          onClick={(event) => {
            if (onSliceClick && event.points?.[0]?.customdata != null) {
              onSliceClick(event.points[0].customdata);
            }
          }}
        />
      </div>

      {!compact && <>
        <div style={{ height: 1, background: "var(--color-border)", margin: "0 16px" }} />
        <div style={{ flex: 3, minHeight: 0 }}>
          <Plot
            data={bottomTraces}
            layout={{
              xaxis: {
                range: [-0.8, n - 0.2],
                tickmode: "array",
                tickvals: items.map((_, i) => i),
                ticktext: items.map((_, i) => `${i + 1}`),
                tickfont: { size: 8, color: "#aaa" },
                zeroline: false, showgrid: false,
              },
              yaxis: { visible: false, range: [-1.5, 3.0] },
              annotations: bottomAnnotations,
              bargap: 0,
              margin: { l: 50, r: 15, t: 5, b: 22 },
              paper_bgcolor: BG, plot_bgcolor: BG,
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </>}
    </div>
  );
}

export default React.memo(PhaseView);
