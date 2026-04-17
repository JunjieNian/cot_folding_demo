import React, { useMemo } from "react";
import Plot from "./Plot";

const CHART_MARGIN = { t: 36, r: 50, b: 50, l: 60 };
const PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"];

function basePlotLayout(title, darkMode) {
  return {
    title: { text: title, font: { size: 13 } },
    margin: CHART_MARGIN,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: darkMode ? "#ccc" : "#333", size: 11 },
    xaxis: {
      gridcolor: darkMode ? "#333" : "#eee",
      zeroline: false,
    },
    yaxis: {
      gridcolor: darkMode ? "#333" : "#eee",
      zeroline: false,
    },
    legend: { orientation: "h", y: -0.22, x: 0.5, xanchor: "center", font: { size: 9 } },
    hovermode: "closest",
  };
}

export default function ModelRankingView({ rankingData, darkMode }) {
  const panels = useMemo(() => {
    if (!rankingData || !rankingData.rankings) return null;

    const {
      rankings, checkpoints, rl_steps, accuracy,
      metric_values, rank_matrix, normalized, peak_checkpoint,
    } = rankingData;

    // ── Panel A: Spearman rho horizontal bar chart ──
    const top20 = rankings.slice(0, 20);
    const barData = [{
      type: "bar",
      orientation: "h",
      y: top20.map((r) => r.metric).reverse(),
      x: top20.map((r) => r.spearman_rho).reverse(),
      marker: {
        color: top20.map((r) => r.spearman_rho > 0 ? "#2166ac" : "#b2182b").reverse(),
      },
      text: top20.map((r) =>
        `\u03c1=${r.spearman_rho.toFixed(3)}  best=${r.predicted_best}`
      ).reverse(),
      textposition: "outside",
      textfont: { size: 9 },
      hovertemplate: "%{y}<br>\u03c1=%{x:.3f}<extra></extra>",
    }];
    const barLayout = {
      ...basePlotLayout("A. Spearman \u03c1 with Accuracy", darkMode),
      margin: { ...CHART_MARGIN, l: 160 },
      xaxis: {
        ...basePlotLayout("", darkMode).xaxis,
        title: "Spearman \u03c1",
        range: [-1.05, 1.05],
      },
      yaxis: { ...basePlotLayout("", darkMode).yaxis, automargin: true },
      shapes: [{
        type: "line", x0: 0, x1: 0, y0: -0.5, y1: top20.length - 0.5,
        line: { color: "grey", width: 0.5 },
      }],
    };

    // ── Panel B: Top-4 scatter subplots ──
    const top4 = rankings.slice(0, 4);
    const peakIdx = checkpoints.indexOf(peak_checkpoint);
    const shortNames = checkpoints.map((n) => n.replace("step-", "s"));

    const scatterData = [];
    const scatterLayout = {
      ...basePlotLayout("B. Top-4 Metrics vs Accuracy", darkMode),
      margin: { t: 36, r: 20, b: 50, l: 50 },
      showlegend: false,
      hovermode: "closest",
    };

    // Create 2x2 subplot grid via xaxis/yaxis domain
    const domains = [
      { x: [0, 0.45], y: [0.55, 1] },
      { x: [0.55, 1], y: [0.55, 1] },
      { x: [0, 0.45], y: [0, 0.42] },
      { x: [0.55, 1], y: [0, 0.42] },
    ];

    top4.forEach((r, i) => {
      const mname = r.metric;
      const mvec = metric_values[mname];
      if (!mvec) return;
      const xKey = i === 0 ? "xaxis" : `xaxis${i + 1}`;
      const yKey = i === 0 ? "yaxis" : `yaxis${i + 1}`;
      const xRef = i === 0 ? "x" : `x${i + 1}`;
      const yRef = i === 0 ? "y" : `y${i + 1}`;

      // All points
      scatterData.push({
        type: "scatter", mode: "markers+text",
        x: mvec, y: accuracy,
        text: shortNames,
        textposition: "top right",
        textfont: { size: 7 },
        marker: { color: "#4393c3", size: 7 },
        xaxis: xRef, yaxis: yRef,
        hovertemplate: `${mname}: %{x:.4f}<br>Accuracy: %{y:.2f}%<extra></extra>`,
      });
      // Highlight peak
      if (peakIdx >= 0) {
        scatterData.push({
          type: "scatter", mode: "markers",
          x: [mvec[peakIdx]], y: [accuracy[peakIdx]],
          marker: { color: "#d6604d", size: 12, symbol: "star" },
          xaxis: xRef, yaxis: yRef,
          hoverinfo: "skip", showlegend: false,
        });
      }

      scatterLayout[xKey] = {
        domain: domains[i].x,
        title: { text: mname, font: { size: 9 } },
        gridcolor: darkMode ? "#333" : "#eee",
        zeroline: false,
      };
      scatterLayout[yKey] = {
        domain: domains[i].y,
        title: { text: i % 2 === 0 ? "Accuracy (%)" : "", font: { size: 9 } },
        gridcolor: darkMode ? "#333" : "#eee",
        zeroline: false,
      };
      // Subplot title as annotation
      scatterLayout.annotations = scatterLayout.annotations || [];
      scatterLayout.annotations.push({
        text: `\u03c1=${r.spearman_rho.toFixed(3)}`,
        xref: "paper", yref: "paper",
        x: (domains[i].x[0] + domains[i].x[1]) / 2,
        y: domains[i].y[1] + 0.02,
        showarrow: false,
        font: { size: 9, color: darkMode ? "#aaa" : "#555" },
      });
    });

    // ── Panel C: Ranking heatmap ──
    const rmKeys = Object.keys(rank_matrix);
    const rmValues = rmKeys.map((k) => rank_matrix[k]);
    const nCkpt = checkpoints.length;

    // Build annotations for cell values
    const heatAnnotations = [];
    rmKeys.forEach((_, ri) => {
      rmValues[ri].forEach((val, ci) => {
        heatAnnotations.push({
          x: ci, y: ri,
          text: String(val),
          showarrow: false,
          font: {
            size: 9,
            color: val <= 3 || val >= 9 ? "white" : "black",
          },
        });
      });
    });

    const heatData = [{
      type: "heatmap",
      z: rmValues,
      x: shortNames,
      y: rmKeys.map((k, i) => {
        if (i === 0) return k;
        const r = rankings.find((rk) => rk.metric === k);
        return r ? `${k}  \u03c1=${r.spearman_rho.toFixed(2)}` : k;
      }),
      colorscale: "RdYlGn",
      reversescale: true,
      zmin: 1, zmax: nCkpt,
      colorbar: { title: "Rank", len: 0.8 },
      hovertemplate: "%{y}<br>%{x}: rank %{z}<extra></extra>",
    }];
    const heatLayout = {
      ...basePlotLayout("C. Checkpoint Ranking (1=best)", darkMode),
      margin: { t: 36, r: 80, b: 60, l: 180 },
      annotations: heatAnnotations,
      yaxis: { ...basePlotLayout("", darkMode).yaxis, automargin: true, tickfont: { size: 9 } },
      xaxis: { ...basePlotLayout("", darkMode).xaxis, tickangle: -45 },
    };

    // ── Panel D: Normalized trajectory overlay ──
    const normKeys = Object.keys(normalized);
    const trajData = [];

    // Accuracy line (bold)
    trajData.push({
      type: "scatter", mode: "lines+markers",
      x: rl_steps, y: normalized.accuracy,
      name: "Accuracy",
      line: { color: darkMode ? "#fff" : "#000", width: 2.5 },
      marker: { size: 6 },
    });

    // Top-5 metric lines
    normKeys.filter((k) => k !== "accuracy").forEach((k, i) => {
      const r = rankings.find((rk) => rk.metric === k);
      const rho = r ? r.spearman_rho.toFixed(2) : "";
      trajData.push({
        type: "scatter", mode: "lines+markers",
        x: rl_steps, y: normalized[k],
        name: `${k} (\u03c1=${rho})`,
        line: { color: PALETTE[i % PALETTE.length], width: 1.5 },
        marker: { size: 4 },
        opacity: 0.85,
      });
    });

    const peakStep = peak_checkpoint === "base" ? 0 : parseInt(peak_checkpoint.split("-")[1], 10);
    const trajLayout = {
      ...basePlotLayout("D. Normalised Metric Trajectories", darkMode),
      xaxis: {
        ...basePlotLayout("", darkMode).xaxis,
        title: "RL Step",
        range: [-30, 1030],
      },
      yaxis: {
        ...basePlotLayout("", darkMode).yaxis,
        title: "Min-Max Normalised",
        range: [-0.05, 1.05],
      },
      shapes: [{
        type: "line",
        x0: peakStep, x1: peakStep, y0: -0.05, y1: 1.05,
        line: { dash: "dash", color: "grey", width: 1 },
      }],
      annotations: [{
        x: peakStep + 20, y: 0.02,
        text: `peak (${peak_checkpoint})`,
        showarrow: false,
        font: { size: 9, color: "grey" },
      }],
    };

    return {
      barData, barLayout,
      scatterData, scatterLayout,
      heatData, heatLayout,
      trajData, trajLayout,
    };
  }, [rankingData, darkMode]);

  if (!rankingData || !panels) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-muted)" }}>
        Loading ranking data...
      </div>
    );
  }

  const cellStyle = {
    height: 360,
    background: "var(--color-surface)",
    borderRadius: 8,
    border: "1px solid var(--color-border)",
  };

  return (
    <div style={{ padding: 16, overflow: "auto", height: "100%" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={cellStyle}>
          <Plot data={panels.barData} layout={panels.barLayout} config={{ displayModeBar: false, responsive: true }} />
        </div>
        <div style={cellStyle}>
          <Plot data={panels.scatterData} layout={panels.scatterLayout} config={{ displayModeBar: false, responsive: true }} />
        </div>
        <div style={cellStyle}>
          <Plot data={panels.heatData} layout={panels.heatLayout} config={{ displayModeBar: false, responsive: true }} />
        </div>
        <div style={cellStyle}>
          <Plot data={panels.trajData} layout={panels.trajLayout} config={{ displayModeBar: false, responsive: true }} />
        </div>
      </div>
    </div>
  );
}
