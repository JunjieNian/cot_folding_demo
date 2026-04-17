import React, { useMemo } from "react";
import Plot from "./Plot";

const CHART_MARGIN = { t: 36, r: 50, b: 50, l: 60 };

function basePlotLayout(title, darkMode) {
  return {
    title: { text: title, font: { size: 13 } },
    margin: CHART_MARGIN,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: darkMode ? "#ccc" : "#333", size: 11 },
    xaxis: {
      title: "RL Step",
      gridcolor: darkMode ? "#333" : "#eee",
      zeroline: false,
    },
    yaxis: {
      gridcolor: darkMode ? "#333" : "#eee",
      zeroline: false,
    },
    legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center", font: { size: 10 } },
    hovermode: "x unified",
  };
}

export default function TrajectoryView({ trajectoryData, problemsMeta, darkMode, onProblemClick }) {
  const data = trajectoryData;

  const charts = useMemo(() => {
    if (!data || !data.dynamics) return null;

    const dyn = data.dynamics;
    const steps = dyn.map((d) => d.rl_step);

    // Chart 1: Accuracy + AUROC vs RL Step
    const chart1Data = [
      {
        x: steps, y: dyn.map((d) => d.accuracy),
        name: "Accuracy (%)", type: "scatter", mode: "lines+markers",
        line: { color: "#1A73E8", width: 2 }, marker: { size: 6 },
      },
      {
        x: steps, y: dyn.map((d) => d.auroc),
        name: "AUROC", type: "scatter", mode: "lines+markers",
        line: { color: "#E05A47", width: 2 }, marker: { size: 6 },
        yaxis: "y2",
      },
    ];
    const chart1Layout = {
      ...basePlotLayout("Accuracy & AUROC vs. RL Step", darkMode),
      yaxis: { ...basePlotLayout("", darkMode).yaxis, title: "Accuracy (%)" },
      yaxis2: { title: "AUROC", overlaying: "y", side: "right", gridcolor: "transparent" },
    };

    // Chart 2: NFS Components (B, H, D*) vs RL Step
    const chart2Data = [
      {
        x: steps, y: dyn.map((d) => d.B_mean),
        name: "B (Bundle)", type: "scatter", mode: "lines+markers",
        line: { color: "#4CAF50", width: 2 }, marker: { size: 5 },
      },
      {
        x: steps, y: dyn.map((d) => d.H_mean),
        name: "H (Header)", type: "scatter", mode: "lines+markers",
        line: { color: "#FF9800", width: 2 }, marker: { size: 5 },
      },
      {
        x: steps, y: dyn.map((d) => d.D_star_mean),
        name: "D* (Drift)", type: "scatter", mode: "lines+markers",
        line: { color: "#F44336", width: 2 }, marker: { size: 5 },
      },
    ];
    const chart2Layout = {
      ...basePlotLayout("NFS Primitives vs. RL Step", darkMode),
      yaxis: { ...basePlotLayout("", darkMode).yaxis, title: "Mean Score" },
    };

    return { chart1Data, chart1Layout, chart2Data, chart2Layout };
  }, [data, darkMode]);

  if (!data || !charts) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-muted)" }}>
        Loading trajectory data...
      </div>
    );
  }

  const categories = problemsMeta?.problems || [];

  return (
    <div style={{ padding: 16, overflow: "auto", height: "100%" }}>
      {/* 1×2 Chart Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
        <div style={{ height: 320, background: "var(--color-surface)", borderRadius: 8, border: "1px solid var(--color-border)" }}>
          <Plot data={charts.chart1Data} layout={charts.chart1Layout} config={{ displayModeBar: false, responsive: true }} />
        </div>
        <div style={{ height: 320, background: "var(--color-surface)", borderRadius: 8, border: "1px solid var(--color-border)" }}>
          <Plot data={charts.chart2Data} layout={charts.chart2Layout} config={{ displayModeBar: false, responsive: true }} />
        </div>
      </div>

      {/* Problems table — clickable rows jump to Compare */}
      {categories.length > 0 && (
        <div style={{ background: "var(--color-surface)", borderRadius: 8, border: "1px solid var(--color-border)", padding: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: "var(--color-text)" }}>
            Selected Problems ({categories.length})
          </div>
          <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--color-border)", color: "var(--color-text-muted)" }}>
                <th style={{ textAlign: "left", padding: "4px 8px" }}>ID</th>
                <th style={{ textAlign: "left", padding: "4px 8px" }}>Category</th>
                <th style={{ textAlign: "right", padding: "4px 8px" }}></th>
              </tr>
            </thead>
            <tbody>
              {categories.map((p) => (
                <tr
                  key={p.id}
                  style={{ borderBottom: "1px solid var(--color-border-light)", cursor: onProblemClick ? "pointer" : "default" }}
                  onClick={() => onProblemClick?.(p.id)}
                >
                  <td style={{ padding: "3px 8px", fontFamily: "monospace", color: "var(--color-text)" }}>
                    {p.short_id}
                  </td>
                  <td style={{ padding: "3px 8px" }}>
                    <span style={{
                      padding: "1px 6px", borderRadius: 8, fontSize: 10, fontWeight: 600,
                      background: p.category === "h_rises_first" ? "#E8F5E9"
                        : p.category === "d_collapses_first" ? "#FBE9E7" : "#F3E5F5",
                      color: p.category === "h_rises_first" ? "#2E7D32"
                        : p.category === "d_collapses_first" ? "#D84315" : "#6A1B9A",
                    }}>
                      {p.category === "h_rises_first" ? "H rises first"
                        : p.category === "d_collapses_first" ? "D collapses first" : "Other"}
                    </span>
                  </td>
                  <td style={{ padding: "3px 8px", textAlign: "right", fontSize: 10, color: "var(--color-primary)" }}>
                    {onProblemClick ? "Compare \u2192" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
