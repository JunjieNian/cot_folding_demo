import React, { useMemo } from "react";
import Plot from "./Plot";

const EXPLORE = "#5B8DEF";
const EXPLOIT = "#E05A47";
const BG = "transparent";

function PhaseView3D({ data, onSliceClick, compact = false }) {
  const phases = data?.phases;
  if (!phases?.items?.length || !phases.mds_coords_3d) {
    return <div style={{ padding: 40, textAlign: "center", color: "#999" }}>No 3D phase data</div>;
  }

  const items = phases.items;
  const n = phases.count;
  const unitLabel = data.unit_label || "slice";

  const traces = useMemo(() => {
    const xs = items.map((p) => p.mds_3d[0]);
    const ys = items.map((p) => p.mds_3d[1]);
    const zs = items.map((p) => p.mds_3d[2]);
    const result = [];

    // Backbone lines colored by state (explore/exploit)
    for (let i = 0; i < n - 1; i++) {
      const color = items[i].state === 0 ? EXPLORE : EXPLOIT;

      result.push({
        x: [xs[i], xs[i + 1]], y: [ys[i], ys[i + 1]], z: [zs[i], zs[i + 1]],
        mode: "lines", type: "scatter3d",
        line: { color, width: 5 },
        hoverinfo: "skip", showlegend: false,
      });
    }

    // Phase nodes — color = explore/exploit, size = phase length
    const sizes = items.map((p) => Math.min(35, 8 + Math.sqrt(p.length) * 3));
    result.push({
      x: xs, y: ys, z: zs, mode: "markers", type: "scatter3d",
      marker: {
        size: sizes,
        color: items.map((p) => p.state === 0 ? EXPLORE : EXPLOIT),
        opacity: 0.9,
        line: { color: "rgba(128,128,128,0.3)", width: 1 },
      },
      customdata: items.map((p, i) => ({ start: p.start, end: p.end, phaseIdx: i, stateName: p.state_name })),
      text: items.map((p, i) =>
        `<b>Phase ${i + 1}: ${p.state_name}</b><br>` +
        `${p.length} ${unitLabel}s (${p.slice_range})<br>` +
        `Entropy: ${p.mean_entropy.toFixed(3)}<br>` +
        `Conf: ${p.mean_confidence.toFixed(3)}`
      ),
      hoverinfo: "text", showlegend: false,
    });

    // Start/End
    result.push({
      x: [xs[0]], y: [ys[0]], z: [zs[0]], mode: "markers", type: "scatter3d",
      marker: { size: 10, color: "#2E7D32", symbol: "diamond" },
      hovertext: ["START"], hoverinfo: "text", showlegend: false,
    });
    result.push({
      x: [xs[n - 1]], y: [ys[n - 1]], z: [zs[n - 1]], mode: "markers", type: "scatter3d",
      marker: { size: 10, color: "#E65100", symbol: "square" },
      hovertext: ["END"], hoverinfo: "text", showlegend: false,
    });

    return result;
  }, [items, n]);

  const explorePhases = items.filter((p) => p.state === 0);
  const exploitPhases = items.filter((p) => p.state === 1);
  const driftPhases = items.filter((p) => p.dominant_label === "drift");

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
            camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } },
          },
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: BG,
          annotations: [{
            x: 1, y: 1, xref: "paper", yref: "paper", xanchor: "right", yanchor: "top",
            text: (data.is_correct != null
              ? `<b style="color:${data.is_correct ? "#4CAF50" : "#F44336"}">${data.is_correct ? "\u2713 Correct" : "\u2717 Incorrect"}</b>` +
                (data.nfs != null ? ` NFS:${data.nfs.toFixed(1)}` : "") + "<br>"
              : "") +
              `<b>${n} Phases</b> ` +
              `<span style="color:${EXPLORE}">${explorePhases.length}E</span> ` +
              `<span style="color:${EXPLOIT}">${exploitPhases.length}X</span>` +
              (driftPhases.length > 0 ? ` <span style="color:#D32F2F">${driftPhases.length} drift</span>` : "") +
              `<br>3D stress: ${phases.mds_stress_3d ?? "?"}`,
            showarrow: false, font: { size: 11 },
            bgcolor: "rgba(128,128,128,0.15)", borderpad: 5,
          }],
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

export default React.memo(PhaseView3D);
