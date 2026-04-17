import React, { useEffect, useState, Suspense, lazy } from "react";
import { getSampleBundle, getStructuralComparison } from "../api";

const FoldingArcDiagram = lazy(() => import("./FoldingArcDiagram"));
const PhaseView = lazy(() => import("./PhaseView"));

const GREEN = "#4CAF50";
const RED = "#F44336";

function ComparisonView({ problemId, samples, colorMode, onSampleClick }) {
  const [compData, setCompData] = useState(null);
  const [correctSid, setCorrectSid] = useState(null);
  const [incorrectSid, setIncorrectSid] = useState(null);
  const [correctFolding, setCorrectFolding] = useState(null);
  const [incorrectFolding, setIncorrectFolding] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load comparison data
  useEffect(() => {
    if (problemId == null) return;
    setLoading(true);
    setCorrectFolding(null);
    setIncorrectFolding(null);
    setCompData(null);
    setCorrectSid(null);
    setIncorrectSid(null);

    getStructuralComparison(problemId)
      .then((data) => {
        setCompData(data);
        const pickMedian = (group) => {
          if (!group?.sample_ids?.length) return null;
          const sorted = group.sample_ids.map((sid, i) => ({ sid, nfs: group.nfs_values[i] }))
            .sort((a, b) => a.nfs - b.nfs);
          return sorted[Math.floor(sorted.length / 2)].sid;
        };
        const cSid = pickMedian(data.correct);
        const icSid = pickMedian(data.incorrect);
        if (cSid != null) setCorrectSid(cSid);
        if (icSid != null) setIncorrectSid(icSid);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [problemId]);

  // Load correct sample folding when selection changes
  useEffect(() => {
    if (problemId == null || correctSid == null) { setCorrectFolding(null); return; }
    getSampleBundle(problemId, correctSid)
      .then((d) => setCorrectFolding(d.folding))
      .catch(() => setCorrectFolding(null));
  }, [problemId, correctSid]);

  // Load incorrect sample folding when selection changes
  useEffect(() => {
    if (problemId == null || incorrectSid == null) { setIncorrectFolding(null); return; }
    getSampleBundle(problemId, incorrectSid)
      .then((d) => setIncorrectFolding(d.folding))
      .catch(() => setIncorrectFolding(null));
  }, [problemId, incorrectSid]);

  if (loading) return <div style={S.center}>Loading comparison for P{problemId}...</div>;

  const hasC = !!correctFolding;
  const hasIC = !!incorrectFolding;
  if (!hasC && !hasIC) return <div style={S.center}>No correct/incorrect samples to compare for P{problemId}</div>;

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <b>P{problemId} Correct vs Incorrect</b>
        {compData && (
          <span style={{ fontSize: 12, color: "#888" }}>
            Accuracy: <b>{compData.accuracy}%</b>
            {compData.correct && <span style={{ color: GREEN, marginLeft: 8 }}>{compData.correct.count}C</span>}
            {compData.incorrect && <span style={{ color: RED, marginLeft: 8 }}>{compData.incorrect.count}I</span>}
          </span>
        )}
      </div>

      {/* 2x2 grid */}
      <div style={S.grid}>
        <Panel
          label="Correct Folding"
          info={hasC ? `${correctFolding.n_slices} slices` : ""}
          color={GREEN}
          sampleIds={compData?.correct?.sample_ids}
          nfsValues={compData?.correct?.nfs_values}
          selectedSid={correctSid}
          onChangeSid={setCorrectSid}
          onInspect={hasC ? () => onSampleClick?.(correctFolding.sample_id) : null}
        >
          {hasC ? (
            <Suspense fallback={<div style={S.center}>Loading...</div>}>
              <FoldingArcDiagram data={correctFolding} colorMode={colorMode} compact />
            </Suspense>
          ) : <div style={S.center}>No correct samples</div>}
        </Panel>

        <Panel
          label="Incorrect Folding"
          info={hasIC ? `${incorrectFolding.n_slices} slices` : ""}
          color={RED}
          sampleIds={compData?.incorrect?.sample_ids}
          nfsValues={compData?.incorrect?.nfs_values}
          selectedSid={incorrectSid}
          onChangeSid={setIncorrectSid}
          onInspect={hasIC ? () => onSampleClick?.(incorrectFolding.sample_id) : null}
        >
          {hasIC ? (
            <Suspense fallback={<div style={S.center}>Loading...</div>}>
              <FoldingArcDiagram data={incorrectFolding} colorMode={colorMode} compact />
            </Suspense>
          ) : <div style={S.center}>No incorrect samples</div>}
        </Panel>

        <Panel label="Correct Phase" info={hasC ? `${correctFolding.phases?.count || "?"} phases` : ""} color={GREEN}>
          {hasC && correctFolding.phases ? (
            <Suspense fallback={<div style={S.center}>Loading...</div>}>
              <PhaseView data={correctFolding} compact />
            </Suspense>
          ) : <div style={S.center}>{"\u2014"}</div>}
        </Panel>

        <Panel label="Incorrect Phase" info={hasIC ? `${incorrectFolding.phases?.count || "?"} phases` : ""} color={RED}>
          {hasIC && incorrectFolding.phases ? (
            <Suspense fallback={<div style={S.center}>Loading...</div>}>
              <PhaseView data={incorrectFolding} compact />
            </Suspense>
          ) : <div style={S.center}>{"\u2014"}</div>}
        </Panel>
      </div>
    </div>
  );
}

function Panel({ label, info, color, sampleIds, nfsValues, selectedSid, onChangeSid, onInspect, children }) {
  return (
    <div style={S.panel}>
      <div style={{ ...S.panelHeader, borderLeftColor: color }}>
        <span style={{ color, display: "flex", alignItems: "center", gap: 6 }}>
          {label}
          {sampleIds && sampleIds.length > 0 && onChangeSid && (
            <select
              value={selectedSid ?? ""}
              onChange={(e) => onChangeSid(Number(e.target.value))}
              style={S.sampleSelect}
            >
              {sampleIds.map((sid, i) => (
                <option key={sid} value={sid}>
                  S{sid}{nfsValues ? ` (NFS ${nfsValues[i]?.toFixed(1)})` : ""}
                </option>
              ))}
            </select>
          )}
          {info && <span style={{ color: "#999", fontWeight: 400 }}>{info}</span>}
        </span>
        {onInspect && (
          <button onClick={onInspect} style={S.inspectBtn}>Inspect</button>
        )}
      </div>
      <div style={S.panelContent}>
        {children || <div style={S.center}>{"\u2014"}</div>}
      </div>
    </div>
  );
}

const S = {
  root: { height: "100%", display: "flex", flexDirection: "column", overflow: "hidden", padding: 6, gap: 6 },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "6px 12px", background: "var(--color-surface)", borderRadius: 6,
    border: "1px solid var(--color-border)", fontSize: 14,
  },
  grid: {
    flex: 1, display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gridTemplateRows: "1fr 1fr",
    gap: 6, minHeight: 0,
  },
  panel: {
    background: "var(--color-surface)", borderRadius: 6,
    border: "1px solid var(--color-border)", overflow: "hidden",
    display: "flex", flexDirection: "column", minHeight: 0,
  },
  panelHeader: {
    padding: "5px 10px", fontSize: 11, fontWeight: 600,
    borderBottom: "1px solid var(--color-border-light)",
    borderLeft: "3px solid", display: "flex",
    justifyContent: "space-between", alignItems: "center",
    flexShrink: 0,
  },
  panelContent: { flex: 1, minHeight: 0, overflow: "hidden" },
  inspectBtn: {
    background: "none", border: "1px solid var(--color-border)", borderRadius: 4,
    cursor: "pointer", fontSize: 10, padding: "2px 8px", color: "var(--color-text-secondary)",
  },
  center: { padding: 24, textAlign: "center", color: "#999", fontSize: 13 },
  sampleSelect: {
    padding: "2px 4px", fontSize: 10, borderRadius: 3,
    border: "1px solid var(--color-border)", background: "var(--color-surface)", color: "var(--color-text)",
    maxWidth: 130, cursor: "pointer",
  },
};

export default React.memo(ComparisonView);
