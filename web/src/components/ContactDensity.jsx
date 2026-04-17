import React from "react";
import Plot from "./Plot";

function ContactDensity({ data }) {
  const { contact_density, n_slices, unit_label = "slice" } = data;
  const { gaps, ee, xx, cross } = contact_density;
  const longRangeThreshold = Math.floor(n_slices / 4);
  const unitTitle = unit_label.charAt(0).toUpperCase() + unit_label.slice(1);

  const traces = [
    {
      x: gaps, y: ee, mode: "lines", name: "Explore-Explore",
      line: { color: "#4285F4", width: 2 },
      hovertemplate: "Gap %{x}<br>E-E Sim: %{y:.4f}<extra></extra>",
    },
    {
      x: gaps, y: xx, mode: "lines", name: "Exploit-Exploit",
      line: { color: "#EA4335", width: 2 },
      hovertemplate: "Gap %{x}<br>X-X Sim: %{y:.4f}<extra></extra>",
    },
    {
      x: gaps, y: cross, mode: "lines", name: "Cross",
      line: { color: "#9E9E9E", width: 2 },
      hovertemplate: "Gap %{x}<br>Cross Sim: %{y:.4f}<extra></extra>",
    },
  ];

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: "Contact Density by Sequence Gap", font: { size: 14 } },
        xaxis: { title: `${unitTitle} Gap |i - j|`, range: [1, n_slices - 1] },
        yaxis: { title: "Mean Similarity" },
        shapes: [{
          type: "line", x0: longRangeThreshold, x1: longRangeThreshold,
          y0: 0, y1: 1, yref: "paper",
          line: { color: "black", width: 1, dash: "dash" },
        }],
        annotations: [{
          x: longRangeThreshold + 2, y: 1, yref: "paper",
          text: `N/4=${longRangeThreshold}`, showarrow: false,
          font: { size: 10, color: "#666" },
        }],
        legend: { x: 0.98, y: 0.98, xanchor: "right" },
        margin: { l: 60, r: 20, t: 40, b: 50 },
        paper_bgcolor: "transparent",
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ responsive: true }}
    />
  );
}

export default React.memo(ContactDensity);
