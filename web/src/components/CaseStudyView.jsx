import React, { useState, useEffect, useCallback } from "react";
import { getProblemCompare, getSampleBundle, getSampleText } from "../api";
import AnnotatedText from "./AnnotatedText";
import styles from "./CaseStudy.module.css";

/* ─── Recommended problems with good contrast ─── */
const RECOMMENDED = [70, 73, 61];

/* ─── NfsScoreCard — compact score + component badges ─── */
function NfsScoreCard({ isCorrect, sampleId, nfs, components }) {
  const B = components?.B ?? 0;
  const H = components?.H ?? 0;
  const D0 = components?.D0 ?? 0;
  const G = components?.G ?? 0;

  return (
    <div className={styles.scoreCard}>
      <div className={styles.scoreHeader}>
        <span className={`${styles.scoreIcon} ${isCorrect ? styles.correct : styles.incorrect}`}>
          {isCorrect ? "\u2713" : "\u2717"}
        </span>
        <span className={styles.scoreSampleId}>
          {isCorrect ? "Correct" : "Incorrect"} S{sampleId}
        </span>
        <span className={styles.scoreNfs}>NFS {nfs?.toFixed(2) ?? "—"}</span>
      </div>
      <div className={styles.componentRow}>
        <span className={styles.componentBadge}>B={B.toFixed(2)}</span>
        <span className={styles.componentBadge}>H={H.toFixed(2)}</span>
        <span className={styles.componentBadge}>D*={D0.toFixed(2)}</span>
        <span className={styles.componentBadge}>G={G.toFixed(2)}</span>
      </div>
    </div>
  );
}

/* ─── ReturnEdgeList — top return edges, clickable ─── */
function ReturnEdgeList({ edges, activeEdge, onEdgeClick }) {
  if (!edges || edges.length === 0) {
    return (
      <div className={styles.edgesSection}>
        <div className={styles.edgesSectionTitle}>Return Edges</div>
        <div className={styles.edgeCard} style={{ color: "var(--color-text-faint)", cursor: "default" }}>
          No return edges detected
        </div>
      </div>
    );
  }

  const top = edges
    .slice()
    .sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0))
    .slice(0, 10);

  return (
    <div className={styles.edgesSection}>
      <div className={styles.edgesSectionTitle}>
        Top Return Edges ({edges.length} total)
      </div>
      {top.map((e, k) => {
        const isActive = activeEdge && activeEdge.i === e.i && activeEdge.j === e.j;
        return (
          <div
            key={k}
            className={`${styles.edgeCard}${isActive ? ` ${styles.active}` : ""}`}
            onClick={() => onEdgeClick(isActive ? null : e)}
          >
            <span className={styles.edgeIndices}>S{e.i} ↔ S{e.j}</span>
            <span className={styles.edgeSim}>sim {(e.similarity ?? 0).toFixed(3)}</span>
            <span className={styles.edgeType}>{e.type ?? `gap ${e.gap}`}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ─── SamplePanel — one side of the comparison ─── */
function SamplePanel({ isCorrect, sampleId, nfs, bundle, textData, onSampleClick }) {
  const [activeEdge, setActiveEdge] = useState(null);

  const labels = bundle?.folding?.effectiveness?.labels ?? [];
  const coreIndices = bundle?.functional?.core?.indices ?? [];
  const returnEdges = bundle?.functional?.return_edges ?? [];
  const driftBranches = bundle?.functional?.drift_branches ?? [];
  const nfsComponents = bundle?.folding?.effectiveness?.nfs_components ?? {};

  const fullText = textData?.full_text ?? "";
  const items = textData?.items ?? [];

  return (
    <div className={`${styles.samplePanel} ${isCorrect ? styles.correct : styles.incorrect}`}>
      <NfsScoreCard
        isCorrect={isCorrect}
        sampleId={sampleId}
        nfs={nfs}
        components={nfsComponents}
      />
      <AnnotatedText
        fullText={fullText}
        items={items}
        labels={labels}
        coreIndices={coreIndices}
        returnEdges={returnEdges}
        driftBranches={driftBranches}
        activeEdge={activeEdge}
        onSliceClick={() => onSampleClick?.(sampleId)}
      />
      <ReturnEdgeList
        edges={returnEdges}
        activeEdge={activeEdge}
        onEdgeClick={setActiveEdge}
      />
    </div>
  );
}

/* ─── Main CaseStudyView ─── */
export default function CaseStudyView({ problemId, onProblemClick, onSampleClick }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Compare data
  const [compareData, setCompareData] = useState(null);
  // Best correct sample
  const [bestId, setBestId] = useState(null);
  const [bestNfs, setBestNfs] = useState(null);
  const [bestBundle, setBestBundle] = useState(null);
  const [bestText, setBestText] = useState(null);
  // Worst incorrect sample
  const [worstId, setWorstId] = useState(null);
  const [worstNfs, setWorstNfs] = useState(null);
  const [worstBundle, setWorstBundle] = useState(null);
  const [worstText, setWorstText] = useState(null);

  const loadData = useCallback(async (pid) => {
    setLoading(true);
    setError(null);
    setBestBundle(null);
    setBestText(null);
    setWorstBundle(null);
    setWorstText(null);

    try {
      const cmp = await getProblemCompare(pid);
      setCompareData(cmp);

      const correct = cmp.correct ?? {};
      const incorrect = cmp.incorrect ?? {};
      const correctIds = correct.sample_ids ?? [];
      const incorrectIds = incorrect.sample_ids ?? [];
      const correctNfs = correct.nfs_values ?? [];
      const incorrectNfs = incorrect.nfs_values ?? [];

      // Pick best correct (highest NFS) and worst incorrect (lowest NFS)
      let bIdx = -1, bNfs = -Infinity;
      correctNfs.forEach((v, i) => { if (v > bNfs) { bNfs = v; bIdx = i; } });

      let wIdx = -1, wNfs = Infinity;
      incorrectNfs.forEach((v, i) => { if (v < wNfs) { wNfs = v; wIdx = i; } });

      const hasBest = bIdx >= 0 && correctIds[bIdx] != null;
      const hasWorst = wIdx >= 0 && incorrectIds[wIdx] != null;

      if (!hasBest && !hasWorst) {
        // Fallback: try to pick from whichever side has data
        if (correctIds.length > 0) {
          bIdx = 0; bNfs = correctNfs[0] ?? 0;
        } else if (incorrectIds.length > 0) {
          wIdx = 0; wNfs = incorrectNfs[0] ?? 0;
        }
      }

      const bSid = hasBest ? correctIds[bIdx] : (bIdx >= 0 ? correctIds[bIdx] : null);
      const wSid = hasWorst ? incorrectIds[wIdx] : (wIdx >= 0 ? incorrectIds[wIdx] : null);

      setBestId(bSid);
      setBestNfs(hasBest ? bNfs : (bIdx >= 0 ? bNfs : null));
      setWorstId(wSid);
      setWorstNfs(hasWorst ? wNfs : (wIdx >= 0 ? wNfs : null));

      // Parallel fetch bundles + text for both samples
      const fetches = [];
      if (bSid != null) {
        fetches.push(
          getSampleBundle(pid, bSid).then(setBestBundle),
          getSampleText(pid, bSid).then(setBestText),
        );
      }
      if (wSid != null) {
        fetches.push(
          getSampleBundle(pid, wSid).then(setWorstBundle),
          getSampleText(pid, wSid).then(setWorstText),
        );
      }
      await Promise.all(fetches);
    } catch (e) {
      if (e.name !== "AbortError") {
        setError(e.message || "Failed to load case study data");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (problemId == null) return;
    loadData(problemId);
  }, [problemId, loadData]);

  // Compute accuracy
  const nCorrect = compareData?.correct?.sample_ids?.length ?? 0;
  const nIncorrect = compareData?.incorrect?.sample_ids?.length ?? 0;
  const total = nCorrect + nIncorrect;
  const accuracy = total > 0 ? ((nCorrect / total) * 100).toFixed(0) : "—";

  // No contrast case: all correct or all incorrect
  const noContrast = !loading && !error && (bestId == null || worstId == null);

  if (loading) {
    return <div className={styles.loading}>Loading Case Study for P{problemId}...</div>;
  }

  if (error) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyTitle}>Failed to load</div>
        <div className={styles.emptyHint}>{error}</div>
      </div>
    );
  }

  if (noContrast) {
    const allCorrect = nCorrect > 0 && nIncorrect === 0;
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyTitle}>
          P{problemId}: {allCorrect ? "All samples correct" : "All samples incorrect"}
        </div>
        <div className={styles.emptyHint}>
          Case Study requires both correct and incorrect samples for comparison.
          <br />Try one of these problems with good contrast:
        </div>
        {RECOMMENDED.map((pid) => (
          <button
            key={pid}
            className={styles.suggestBtn}
            onClick={() => onProblemClick?.(pid)}
            style={{ marginTop: 4 }}
          >
            Problem {pid}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <span className={styles.headerTitle}>Case Study: Problem {problemId}</span>
        <span className={styles.headerSub}>
          NFS primitives (Core / Return Edges / Drift) mapped to original CoT text
        </span>
        <span className={styles.headerAccuracy}>
          Accuracy: {accuracy}% ({nCorrect}/{total})
        </span>
      </div>

      {/* Side-by-side panels */}
      <div className={styles.pairRow}>
        {bestId != null && (
          <SamplePanel
            isCorrect
            sampleId={bestId}
            nfs={bestNfs}
            bundle={bestBundle}
            textData={bestText}
            onSampleClick={(sid) => onSampleClick?.(sid)}
          />
        )}
        {worstId != null && (
          <SamplePanel
            isCorrect={false}
            sampleId={worstId}
            nfs={worstNfs}
            bundle={worstBundle}
            textData={worstText}
            onSampleClick={(sid) => onSampleClick?.(sid)}
          />
        )}
      </div>
    </div>
  );
}
