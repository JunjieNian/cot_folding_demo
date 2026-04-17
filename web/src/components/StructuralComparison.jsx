import React, { useEffect, useState, useMemo, Suspense, lazy } from "react";
import Plot from "./Plot";
import { getStructuralComparison, getSampleBundle } from "../api";

const FoldingArcDiagram = lazy(() => import("./FoldingArcDiagram"));

const GREEN = "#4CAF50";
const RED = "#F44336";
const BLUE = "#1A73E8";

function StructuralComparison({ problemId, onSampleClick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [correctFolding, setCorrectFolding] = useState(null);
  const [incorrectFolding, setIncorrectFolding] = useState(null);
  const [foldingLoading, setFoldingLoading] = useState(false);

  useEffect(() => {
    if (problemId == null) return;
    setLoading(true);
    setError(null);
    setCorrectFolding(null);
    setIncorrectFolding(null);
    getStructuralComparison(problemId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [problemId]);

  // Load representative correct/incorrect folding data once comparison data arrives
  useEffect(() => {
    if (!data) return;
    const c = data.correct;
    const ic = data.incorrect;
    if (!c?.sample_ids?.length && !ic?.sample_ids?.length) return;

    setFoldingLoading(true);
    const promises = [];

    const pickMedian = (group) => {
      if (!group?.sample_ids?.length) return null;
      const sorted = group.sample_ids.map((sid, i) => ({ sid, nfs: group.nfs_values[i] }))
        .sort((a, b) => a.nfs - b.nfs);
      return sorted[Math.floor(sorted.length / 2)].sid;
    };

    const cSid = pickMedian(c);
    const icSid = pickMedian(ic);

    if (cSid != null) promises.push(
      getSampleBundle(problemId, cSid)
        .then((d) => setCorrectFolding({ ...d.folding, _sid: cSid }))
        .catch(() => null)
    );
    if (icSid != null) promises.push(
      getSampleBundle(problemId, icSid)
        .then((d) => setIncorrectFolding({ ...d.folding, _sid: icSid }))
        .catch(() => null)
    );

    Promise.all(promises).finally(() => setFoldingLoading(false));
  }, [data, problemId]);

  if (loading) return <div style={styles.center}>Loading structural comparison...</div>;
  if (error) return <div style={{ ...styles.center, color: RED }}>{error}</div>;
  if (!data) return <div style={styles.center}>Select a problem to compare</div>;

  const { correct: c, incorrect: ic, effects, accuracy, bin_labels } = data;
  const hasC = c && c.count > 0;
  const hasIC = ic && ic.count > 0;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h3 style={{ margin: 0, fontSize: 15 }}>
          P{problemId} Structural Comparison
        </h3>
        <span style={styles.accuracy}>
          Accuracy: <b>{accuracy}%</b>
          {hasC && <span style={{ color: GREEN, marginLeft: 8 }}>{c.count} correct</span>}
          {hasIC && <span style={{ color: RED, marginLeft: 8 }}>{ic.count} incorrect</span>}
        </span>
      </div>

      {/* Side-by-side Folding Diagrams */}
      {(correctFolding || incorrectFolding) && (
        <div style={{ display: "flex", gap: 8, marginBottom: 8, height: 420 }}>
          {correctFolding && (
            <div style={{ flex: 1, ...styles.card, display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "6px 12px", borderBottom: "1px solid var(--color-border-light)", fontSize: 12, fontWeight: 600, color: GREEN, display: "flex", justifyContent: "space-between" }}>
                <span>Correct Sample S{correctFolding._sid} ({correctFolding.n_slices} slices)</span>
                <button onClick={() => onSampleClick?.(correctFolding._sid)} style={{ background: "none", border: "1px solid var(--color-border)", borderRadius: 4, cursor: "pointer", fontSize: 11, padding: "2px 8px" }}>Inspect</button>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <Suspense fallback={<div style={styles.center}>Loading...</div>}>
                  <FoldingArcDiagram data={correctFolding} colorMode="effectiveness" />
                </Suspense>
              </div>
            </div>
          )}
          {incorrectFolding && (
            <div style={{ flex: 1, ...styles.card, display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "6px 12px", borderBottom: "1px solid var(--color-border-light)", fontSize: 12, fontWeight: 600, color: RED, display: "flex", justifyContent: "space-between" }}>
                <span>Incorrect Sample S{incorrectFolding._sid} ({incorrectFolding.n_slices} slices)</span>
                <button onClick={() => onSampleClick?.(incorrectFolding._sid)} style={{ background: "none", border: "1px solid var(--color-border)", borderRadius: 4, cursor: "pointer", fontSize: 11, padding: "2px 8px" }}>Inspect</button>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <Suspense fallback={<div style={styles.center}>Loading...</div>}>
                  <FoldingArcDiagram data={incorrectFolding} colorMode="effectiveness" />
                </Suspense>
              </div>
            </div>
          )}
        </div>
      )}
      {foldingLoading && <div style={{ textAlign: "center", color: "#999", padding: 8, fontSize: 12 }}>Loading folding diagrams...</div>}

      <div style={styles.grid}>
        {/* 1. Exploit Profile Over Time */}
        <div style={styles.card}>
          <Plot
            data={[
              ...(hasC ? [{
                x: bin_labels,
                y: c.mean_profile,
                name: `Correct (n=${c.count})`,
                type: "scatter", mode: "lines+markers",
                line: { color: GREEN, width: 3 },
                marker: { size: 8 },
                error_y: { type: "data", array: c.std_profile, visible: true, color: "rgba(76,175,80,0.3)" },
              }] : []),
              ...(hasIC ? [{
                x: bin_labels,
                y: ic.mean_profile,
                name: `Incorrect (n=${ic.count})`,
                type: "scatter", mode: "lines+markers",
                line: { color: RED, width: 3, dash: "dash" },
                marker: { size: 8, symbol: "diamond" },
                error_y: { type: "data", array: ic.std_profile, visible: true, color: "rgba(244,67,54,0.3)" },
              }] : []),
            ]}
            layout={{
              title: { text: "Exploit Ratio Over Reasoning Progress", font: { size: 14 } },
              xaxis: { title: "Reasoning Progress", tickangle: -30 },
              yaxis: { title: "Exploit Ratio", range: [0, 1] },
              legend: { x: 0.02, y: 0.98 },
              shapes: [{
                type: "line", x0: 0, x1: 1, y0: 0.5, y1: 0.5,
                xref: "paper", line: { dash: "dot", color: "#ccc", width: 1 },
              }],
              annotations: [{
                x: "90-100%", y: hasC ? c.mean_profile[9] : 0,
                text: hasC ? `${(c.mean_profile[9] * 100).toFixed(0)}%` : "",
                showarrow: true, arrowhead: 2, ax: 30, ay: -20,
                font: { color: GREEN, size: 12, family: "Arial Black" },
              }],
              margin: { l: 50, r: 20, t: 40, b: 60 },
              paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* 2. Length Distribution */}
        <div style={styles.card}>
          <Plot
            data={[
              ...(hasC ? [{
                x: c.lengths, type: "histogram", nbinsx: 25, opacity: 0.7,
                marker: { color: GREEN }, name: "Correct",
              }] : []),
              ...(hasIC ? [{
                x: ic.lengths, type: "histogram", nbinsx: 25, opacity: 0.7,
                marker: { color: RED }, name: "Incorrect",
              }] : []),
            ]}
            layout={{
              title: { text: "Reasoning Length Distribution", font: { size: 14 } },
              xaxis: { title: "Length (slices)" }, yaxis: { title: "Count" },
              barmode: "overlay",
              legend: { x: 0.7, y: 0.98 },
              margin: { l: 50, r: 20, t: 40, b: 50 },
              paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* 3. NFS Distribution */}
        <div style={styles.card}>
          <Plot
            data={[
              ...(hasC ? [{
                x: c.nfs_values, type: "histogram", nbinsx: 20, opacity: 0.7,
                marker: { color: GREEN }, name: `Correct (\u03BC=${c.mean_nfs})`,
              }] : []),
              ...(hasIC ? [{
                x: ic.nfs_values, type: "histogram", nbinsx: 20, opacity: 0.7,
                marker: { color: RED }, name: `Incorrect (\u03BC=${ic.mean_nfs})`,
              }] : []),
            ]}
            layout={{
              title: { text: "NFS Distribution", font: { size: 14 } },
              xaxis: { title: "NFS" }, yaxis: { title: "Count" },
              barmode: "overlay",
              legend: { x: 0.02, y: 0.98 },
              margin: { l: 50, r: 20, t: 40, b: 50 },
              paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* 4. Effect Size Summary */}
        <div style={styles.card}>
          <Plot
            data={[{
              y: effects.map((e) => e.metric),
              x: effects.map((e) => e.cohens_d),
              type: "bar",
              orientation: "h",
              marker: {
                color: effects.map((e) => e.cohens_d > 0 ? GREEN : RED),
              },
              text: effects.map((e) => `d=${e.cohens_d}`),
              textposition: "outside",
              hovertemplate: effects.map((e) =>
                `${e.metric}<br>Correct: ${e.correct_mean}<br>Incorrect: ${e.incorrect_mean}<br>Cohen's d: ${e.cohens_d}<extra></extra>`
              ),
            }]}
            layout={{
              title: { text: "Effect Size (Cohen's d)", font: { size: 14 } },
              xaxis: { title: "Cohen's d (positive = correct higher)", zeroline: true,
                       zerolinecolor: "#333", zerolinewidth: 2 },
              yaxis: { automargin: true },
              margin: { l: 120, r: 60, t: 40, b: 50 },
              paper_bgcolor: "transparent", plot_bgcolor: "transparent",
              shapes: [
                { type: "line", x0: -0.8, x1: -0.8, y0: -0.5, y1: effects.length - 0.5,
                  line: { dash: "dot", color: "#aaa" } },
                { type: "line", x0: 0.8, x1: 0.8, y0: -0.5, y1: effects.length - 0.5,
                  line: { dash: "dot", color: "#aaa" } },
              ],
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* 5. Individual Sample Profiles (scatter) */}
        <div style={{ ...styles.card, gridColumn: "1 / -1" }}>
          <Plot
            data={[
              ...(hasC ? [{
                x: c.lengths,
                y: c.profiles.map((p) => p[9]),
                text: c.sample_ids.map((sid, i) => `S${sid}<br>NFS: ${c.nfs_values[i]}<br>Length: ${c.lengths[i]}`),
                customdata: c.sample_ids,
                mode: "markers",
                marker: { color: GREEN, size: 7, opacity: 0.6, line: { width: 0.5, color: "white" } },
                name: "Correct",
                hoverinfo: "text",
              }] : []),
              ...(hasIC ? [{
                x: ic.lengths,
                y: ic.profiles.map((p) => p[9]),
                text: ic.sample_ids.map((sid, i) => `S${sid}<br>NFS: ${ic.nfs_values[i]}<br>Length: ${ic.lengths[i]}`),
                customdata: ic.sample_ids,
                mode: "markers",
                marker: { color: RED, size: 7, opacity: 0.6, symbol: "diamond",
                          line: { width: 0.5, color: "white" } },
                name: "Incorrect",
                hoverinfo: "text",
              }] : []),
            ]}
            layout={{
              title: { text: "Each Sample: Length vs Final Exploit Ratio (click to inspect)", font: { size: 14 } },
              xaxis: { title: "Reasoning Length (slices)", type: "log" },
              yaxis: { title: "Final 10% Exploit Ratio", range: [0, 1.05] },
              legend: { x: 0.02, y: 0.98 },
              shapes: [{
                type: "line", x0: 0, x1: 1, y0: 0.5, y1: 0.5,
                xref: "paper", line: { dash: "dot", color: "#ccc" },
              }],
              margin: { l: 50, r: 20, t: 40, b: 50 },
              paper_bgcolor: "transparent", plot_bgcolor: "transparent",
            }}
            useResizeHandler style={{ width: "100%", height: 320 }}
            config={{ responsive: true, displayModeBar: false }}
            onClick={(event) => {
              if (onSampleClick && event.points?.[0]?.customdata != null) {
                onSampleClick(event.points[0].customdata);
              }
            }}
          />
        </div>

        {/* 6. Summary Stats Table */}
        <div style={{ ...styles.card, gridColumn: "1 / -1" }}>
          <div style={{ padding: 16, overflow: "auto" }}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Metric</th>
                  <th style={{ ...styles.th, color: GREEN }}>Correct</th>
                  <th style={{ ...styles.th, color: RED }}>Incorrect</th>
                  <th style={styles.th}>Difference</th>
                  <th style={styles.th}>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                <SummaryRow label="Avg Length" c={hasC ? `${c.mean_length} \u00B1 ${c.std_length}` : "\u2014"} ic={hasIC ? `${ic.mean_length} \u00B1 ${ic.std_length}` : "\u2014"}
                  diff={hasC && hasIC ? c.mean_length - ic.mean_length : null}
                  interp="Shorter = more focused reasoning" />
                <SummaryRow label="Final Exploit %" c={hasC ? `${(c.mean_profile[9]*100).toFixed(0)}%` : "\u2014"} ic={hasIC ? `${(ic.mean_profile[9]*100).toFixed(0)}%` : "\u2014"}
                  diff={hasC && hasIC ? (c.mean_profile[9] - ic.mean_profile[9]) * 100 : null}
                  interp="Higher = confident conclusion" positive />
                <SummaryRow label="Mean Similarity" c={hasC ? c.mean_sim : "\u2014"} ic={hasIC ? ic.mean_sim : "\u2014"}
                  diff={hasC && hasIC ? c.mean_sim - ic.mean_sim : null}
                  interp="Higher = coherent reasoning" positive />
                <SummaryRow label="Transitions" c={hasC ? c.mean_transitions : "\u2014"} ic={hasIC ? ic.mean_transitions : "\u2014"}
                  diff={hasC && hasIC ? c.mean_transitions - ic.mean_transitions : null}
                  interp="Fewer = less hesitation" />
                <SummaryRow label="Explore Ratio" c={hasC ? `${(c.mean_explore_ratio*100).toFixed(0)}%` : "\u2014"} ic={hasIC ? `${(ic.mean_explore_ratio*100).toFixed(0)}%` : "\u2014"}
                  diff={hasC && hasIC ? (c.mean_explore_ratio - ic.mean_explore_ratio) * 100 : null}
                  interp="Lower = more exploitation" />
                <SummaryRow label="NFS" c={hasC ? c.mean_nfs : "\u2014"} ic={hasIC ? ic.mean_nfs : "\u2014"}
                  diff={hasC && hasIC ? c.mean_nfs - ic.mean_nfs : null}
                  interp="Higher = better structural quality" positive />
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function SummaryRow({ label, c, ic, diff, interp, positive }) {
  let diffColor = "#888";
  let diffStr = "\u2014";
  if (diff != null) {
    const good = positive ? diff > 0 : diff < 0;
    diffColor = good ? GREEN : Math.abs(diff) < 0.01 ? "#888" : RED;
    diffStr = (diff > 0 ? "+" : "") + (typeof diff === "number" ? diff.toFixed(2) : diff);
  }
  return (
    <tr>
      <td style={styles.td}><b>{label}</b></td>
      <td style={styles.td}>{c}</td>
      <td style={styles.td}>{ic}</td>
      <td style={{ ...styles.td, color: diffColor, fontWeight: 600 }}>{diffStr}</td>
      <td style={{ ...styles.td, color: "#888", fontSize: 11 }}>{interp}</td>
    </tr>
  );
}

const styles = {
  container: { height: "100%", display: "flex", flexDirection: "column", overflow: "auto", padding: 8 },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "8px 12px", background: "var(--color-surface)", borderRadius: 6,
    border: "1px solid var(--color-border)", marginBottom: 8,
  },
  accuracy: { fontSize: 13, color: "var(--color-text-secondary)" },
  grid: {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, flex: 1,
  },
  card: {
    background: "var(--color-surface)", borderRadius: 6,
    border: "1px solid var(--color-border)", overflow: "hidden", minHeight: 280,
  },
  center: { padding: 40, textAlign: "center", color: "#999", fontSize: 14 },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: { textAlign: "left", padding: "6px 10px", borderBottom: "2px solid var(--color-border)", fontSize: 12 },
  td: { padding: "5px 10px", borderBottom: "1px solid var(--color-border-light)", fontFamily: "monospace", fontSize: 12 },
};

export default React.memo(StructuralComparison);
