import React, { useMemo } from "react";
import Plot from "./Plot";

function MDSPlot({ data, onSliceClick }) {
  const { mds_coords, mds_coords_shape, hmm_states, entropy, mds_stress } = data;
  const n = mds_coords_shape[0];

  // Reconstruct coordinates (mds_coords is [[x,y], ...] nested array)
  const x = useMemo(() => mds_coords.map((p) => p[0]), [mds_coords]);
  const y = useMemo(() => mds_coords.map((p) => p[1]), [mds_coords]);

  // Normalize entropy for point sizes
  const sizes = useMemo(() => {
    const eMin = Math.min(...entropy);
    const eMax = Math.max(...entropy);
    const range = eMax - eMin || 1e-8;
    return entropy.map((e) => 6 + 16 * ((e - eMin) / range));
  }, [entropy]);

  // Split by state
  const traces = useMemo(() => {
    const explore = { x: [], y: [], size: [], text: [], customdata: [] };
    const exploit = { x: [], y: [], size: [], text: [], customdata: [] };

    for (let i = 0; i < n; i++) {
      const target = hmm_states[i] === 0 ? explore : exploit;
      target.x.push(x[i]);
      target.y.push(y[i]);
      target.size.push(sizes[i]);
      target.text.push(`Slice ${i}<br>Entropy: ${(entropy[i] ?? 0).toFixed(3)}`);
      target.customdata.push(i);
    }

    return [
      // Connection line
      {
        x, y, mode: "lines", line: { color: "#ccc", width: 1 },
        hoverinfo: "skip", showlegend: false,
      },
      // Explore
      {
        x: explore.x, y: explore.y, mode: "markers",
        marker: { color: "#4285F4", size: explore.size, line: { color: "rgba(128,128,128,0.3)", width: 0.5 } },
        text: explore.text, hoverinfo: "text", name: "Exploration",
        customdata: explore.customdata,
      },
      // Exploit
      {
        x: exploit.x, y: exploit.y, mode: "markers",
        marker: { color: "#EA4335", size: exploit.size, line: { color: "rgba(128,128,128,0.3)", width: 0.5 } },
        text: exploit.text, hoverinfo: "text", name: "Exploitation",
        customdata: exploit.customdata,
      },
    ];
  }, [x, y, sizes, hmm_states, entropy, n]);

  // Annotations for START and END
  const annotations = [
    {
      x: x[0], y: y[0], text: "START", showarrow: true,
      arrowhead: 2, ax: 20, ay: -20, font: { size: 11, color: "#333" },
      bgcolor: "#FFFFCC", bordercolor: "#999",
    },
    {
      x: x[n - 1], y: y[n - 1], text: "END", showarrow: true,
      arrowhead: 2, ax: 20, ay: 15, font: { size: 11, color: "#333" },
      bgcolor: "#CCFFCC", bordercolor: "#999",
    },
  ];

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: `MDS 2D Folding (Stress = ${mds_stress.toFixed(4)})`, font: { size: 14 } },
        xaxis: { title: "MDS Dim 1", zeroline: false },
        yaxis: { title: "MDS Dim 2", zeroline: false, scaleanchor: "x" },
        annotations,
        legend: { x: 0.98, y: 0.98, xanchor: "right" },
        margin: { l: 50, r: 20, t: 40, b: 50 },
        paper_bgcolor: "transparent",
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ responsive: true }}
      onClick={(event) => {
        if (onSliceClick && event.points && event.points.length > 0) {
          const pt = event.points[0];
          if (pt.customdata != null) onSliceClick(pt.customdata);
        }
      }}
    />
  );
}

export default React.memo(MDSPlot);
