import React, { useState, useEffect, useCallback, Suspense, lazy } from "react";
import { getProblemCompare, getSampleBundle, getSampleText, getComparePresets } from "../api";
import AnnotatedText from "./AnnotatedText";
import styles from "./CaseStudy.module.css";

const FoldingArcDiagram = lazy(() => import("./FoldingArcDiagram"));
const FoldingView3D = lazy(() => import("./FoldingView3D"));

const RECOMMENDED = [70, 73, 61];

const LEGEND = [
  { label: "Core",        color: "rgba(76,175,80,0.45)" },
  { label: "Closure",     color: "rgba(76,175,80,0.35)" },
  { label: "Return Site", color: "rgba(255,152,0,0.35)" },
  { label: "Drift",       color: "rgba(244,67,54,0.45)" },
  { label: "Productive",  color: "rgba(158,158,158,0.25)" },
];

/* NFS Score Card — compact single row */
function NfsScoreCard({ isCorrect, sampleId, nfs, components }) {
  const B = components?.B ?? 0, H = components?.H ?? 0,
        D0 = components?.D0 ?? 0, G = components?.G ?? 0;
  return (
    <div className={styles.scoreCardCompact}>
      <span className={`${styles.scoreIcon} ${isCorrect ? styles.correct : styles.incorrect}`}>
        {isCorrect ? "\u2713" : "\u2717"}
      </span>
      <span className={styles.scoreSampleId}>
        {isCorrect ? "Correct" : "Incorrect"} S{sampleId}
      </span>
      <span className={styles.scoreNfsCompact}>NFS {nfs?.toFixed(2) ?? "\u2014"}</span>
      {[`B=${B.toFixed(2)}`, `H=${H.toFixed(2)}`, `D*=${D0.toFixed(2)}`, `G=${G.toFixed(2)}`].map(t => (
        <span key={t} className={styles.componentBadge}>{t}</span>
      ))}
    </div>
  );
}

/* Return Edge List — collapsible drawer */
function ReturnEdgeList({ edges, activeEdge, onEdgeClick }) {
  const [open, setOpen] = useState(false);
  const count = edges?.length ?? 0;

  const top = (edges ?? [])
    .slice()
    .sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0))
    .slice(0, 10);

  return (
    <div className={styles.edgesDrawer}>
      <button className={styles.edgesDrawerToggle} onClick={() => setOpen(v => !v)}>
        Return Edges ({count}) {open ? "\u25b2" : "\u25bc"}
      </button>
      {open && (
        <div className={styles.edgesDrawerBody}>
          {count === 0 ? (
            <div className={styles.edgeCard} style={{ color: "var(--color-text-faint)", cursor: "default" }}>
              No return edges detected
            </div>
          ) : (
            top.map((e, k) => {
              const isActive = activeEdge && activeEdge.i === e.i && activeEdge.j === e.j;
              return (
                <div
                  key={k}
                  className={`${styles.edgeCard}${isActive ? ` ${styles.active}` : ""}`}
                  onClick={() => onEdgeClick(isActive ? null : e)}
                >
                  <span className={styles.edgeIndices}>S{e.i} \u2194 S{e.j}</span>
                  <span className={styles.edgeSim}>sim {(e.similarity ?? 0).toFixed(3)}</span>
                  <span className={styles.edgeType}>{e.type ?? `gap ${e.gap}`}</span>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

/* Text Panel — NFS card + annotated text + return edges */
function TextPanel({ isCorrect, sampleId, nfs, bundle, textData, onSampleClick }) {
  const [activeEdge, setActiveEdge] = useState(null);

  const labels = bundle?.folding?.effectiveness?.labels ?? [];
  const returnEdges = bundle?.functional?.return_edges ?? [];
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
        activeEdge={activeEdge}
        onSliceClick={() => onSampleClick?.(sampleId)}
        showLegend={false}
      />
      <ReturnEdgeList
        edges={returnEdges}
        activeEdge={activeEdge}
        onEdgeClick={setActiveEdge}
      />
    </div>
  );
}

function Loader() {
  return <div style={{ padding: 20, textAlign: "center", color: "#999" }}>Loading...</div>;
}

/* Single full-width canvas — no "split" mode in compare */
function MainCanvas({ bundle, lens }) {
  if (!bundle) return null;
  const data = bundle.folding;
  if (lens === "3d") {
    return (
      <Suspense fallback={<Loader />}>
        <FoldingView3D data={data} colorMode="effectiveness" compact />
      </Suspense>
    );
  }
  return (
    <Suspense fallback={<Loader />}>
      <FoldingArcDiagram data={data} colorMode="effectiveness" compact miniature />
    </Suspense>
  );
}

export default function ComparePage({ problemId, onProblemClick, onSampleClick }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lens, setLens] = useState("2d");
  const [activeGraph, setActiveGraph] = useState("correct"); // "correct" | "incorrect"
  const [peeking, setPeeking] = useState(false); // Shift-hold temporary peek
  const [presets, setPresets] = useState([]);

  const [compareData, setCompareData] = useState(null);
  const [bestId, setBestId] = useState(null);
  const [bestNfs, setBestNfs] = useState(null);
  const [bestBundle, setBestBundle] = useState(null);
  const [bestText, setBestText] = useState(null);
  const [worstId, setWorstId] = useState(null);
  const [worstNfs, setWorstNfs] = useState(null);
  const [worstBundle, setWorstBundle] = useState(null);
  const [worstText, setWorstText] = useState(null);

  // Shift-hold peek: temporarily show the other side
  useEffect(() => {
    const down = (e) => { if (e.key === "Shift") setPeeking(true); };
    const up   = (e) => { if (e.key === "Shift") setPeeking(false); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup",   up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup",   up);
    };
  }, []);

  // Load presets once
  useEffect(() => {
    getComparePresets().then((data) => {
      setPresets(data?.presets ?? []);
    }).catch(() => {});
  }, []);

  const loadData = useCallback(async (pid, presetsCache) => {
    setLoading(true);
    setError(null);
    setBestBundle(null);
    setBestText(null);
    setWorstBundle(null);
    setWorstText(null);
    setActiveGraph("correct");

    try {
      const cmp = await getProblemCompare(pid);
      setCompareData(cmp);

      const correct = cmp.correct ?? {};
      const incorrect = cmp.incorrect ?? {};
      const correctIds = correct.sample_ids ?? [];
      const incorrectIds = incorrect.sample_ids ?? [];
      const correctNfs = correct.nfs_values ?? [];
      const incorrectNfs = incorrect.nfs_values ?? [];

      // Check for curated preset
      const preset = (presetsCache ?? []).find((p) => p.problem_id === pid);

      let bSid = null, bNfs = null, wSid = null, wNfs = null;

      if (preset) {
        // Use curated pair
        const cIdx = correctIds.indexOf(preset.correct_sample_id);
        const iIdx = incorrectIds.indexOf(preset.incorrect_sample_id);
        if (cIdx >= 0) { bSid = preset.correct_sample_id; bNfs = correctNfs[cIdx] ?? 0; }
        if (iIdx >= 0) { wSid = preset.incorrect_sample_id; wNfs = incorrectNfs[iIdx] ?? 0; }
      }

      // Fallback: NFS-based selection
      if (bSid == null) {
        let bIdx = -1, bestNfsVal = -Infinity;
        correctNfs.forEach((v, i) => { if (v > bestNfsVal) { bestNfsVal = v; bIdx = i; } });
        if (bIdx >= 0 && correctIds[bIdx] != null) {
          bSid = correctIds[bIdx]; bNfs = bestNfsVal;
        }
      }
      if (wSid == null) {
        let wIdx = -1, worstNfsVal = Infinity;
        incorrectNfs.forEach((v, i) => { if (v < worstNfsVal) { worstNfsVal = v; wIdx = i; } });
        if (wIdx >= 0 && incorrectIds[wIdx] != null) {
          wSid = incorrectIds[wIdx]; wNfs = worstNfsVal;
        }
      }

      // Last resort: pick first available
      if (bSid == null && wSid == null) {
        if (correctIds.length > 0) { bSid = correctIds[0]; bNfs = correctNfs[0] ?? 0; }
        else if (incorrectIds.length > 0) { wSid = incorrectIds[0]; wNfs = incorrectNfs[0] ?? 0; }
      }

      setBestId(bSid);
      setBestNfs(bNfs);
      setWorstId(wSid);
      setWorstNfs(wNfs);

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
        setError(e.message || "Failed to load comparison data");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (problemId == null) return;
    loadData(problemId, presets);
  }, [problemId, loadData, presets]);

  const nCorrect   = compareData?.correct?.sample_ids?.length   ?? 0;
  const nIncorrect = compareData?.incorrect?.sample_ids?.length ?? 0;
  const total      = nCorrect + nIncorrect;
  const accuracy   = total > 0 ? ((nCorrect / total) * 100).toFixed(0) : "\u2014";

  const noContrast = !loading && !error && (bestId == null || worstId == null);

  // Which side the main graph currently displays (accounting for peek)
  const shownSide     = peeking ? (activeGraph === "correct" ? "incorrect" : "correct") : activeGraph;
  const shownBundle   = shownSide === "correct" ? bestBundle  : worstBundle;
  const shownId       = shownSide === "correct" ? bestId      : worstId;
  const shownIsCorrect = shownSide === "correct";

  if (loading) {
    return <div className={styles.loading}>Loading comparison for P{problemId}...</div>;
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
          Comparison requires both correct and incorrect samples.
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
      {/* ── Header ── */}
      <div className={styles.header}>
        <span className={styles.headerTitle}>P{problemId}</span>
        <span className={styles.headerSub}>
          Accuracy {accuracy}%&nbsp;&nbsp;&middot;&nbsp;&nbsp;{nCorrect}{"✓"}&nbsp;/&nbsp;{nIncorrect}{"✗"}
        </span>
        {/* 2D / 3D lens toggle (no split in compare mode) */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 2 }}>
          {["2d", "3d"].map(k => (
            <button
              key={k}
              onClick={() => setLens(k)}
              style={{
                background: lens === k ? "var(--color-primary)" : "none",
                color: lens === k ? "#fff" : "var(--color-text-muted)",
                border: "1px solid var(--color-border)",
                borderRadius: 4,
                padding: "2px 9px",
                fontSize: 10,
                fontWeight: 600,
                cursor: "pointer",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
                transition: "background 0.12s, color 0.12s",
              }}
            >
              {k}
            </button>
          ))}
        </div>
      </div>

      {/* ── Preset navigation bar ── */}
      {presets.length > 0 && (
        <div className={styles.presetBar}>
          <span className={styles.presetLabel}>Recommended:</span>
          {presets.map((p) => (
            <button
              key={p.problem_id}
              className={`${styles.presetBtn}${p.problem_id === problemId ? ` ${styles.presetBtnActive}` : ""}`}
              onClick={() => onProblemClick?.(p.problem_id)}
            >
              P{p.problem_id}
              <span className={styles.presetBtnSub}>{p.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── Shared main graph stage ── */}
      {(bestBundle || worstBundle) && (
        <div className={styles.graphStage}>
          {/* Tab bar: Correct | Incorrect | spacer | hint | Inspect */}
          <div className={styles.graphStageTabs}>
            {bestId != null && (
              <button
                className={[
                  styles.graphStageTab,
                  styles.graphStageTabCorrect,
                  activeGraph === "correct" && !peeking ? styles.graphStageTabActive : "",
                ].join(" ")}
                onClick={() => setActiveGraph("correct")}
              >
                {"✓"} Correct S{bestId}
                <span className={styles.graphStageTabNfs}>NFS {bestNfs?.toFixed(2)}</span>
                {bestBundle && (
                  <span className={styles.graphStageTabSlices}>
                    {bestBundle.folding.n_slices} slices
                  </span>
                )}
              </button>
            )}
            {worstId != null && (
              <button
                className={[
                  styles.graphStageTab,
                  styles.graphStageTabIncorrect,
                  activeGraph === "incorrect" && !peeking ? styles.graphStageTabActive : "",
                ].join(" ")}
                onClick={() => setActiveGraph("incorrect")}
              >
                {"✗"} Incorrect S{worstId}
                <span className={styles.graphStageTabNfs}>NFS {worstNfs?.toFixed(2)}</span>
                {worstBundle && (
                  <span className={styles.graphStageTabSlices}>
                    {worstBundle.folding.n_slices} slices
                  </span>
                )}
              </button>
            )}
            <div style={{ flex: 1 }} />
            <span className={styles.graphStagePeekHint}>Hold Shift to peek</span>
            {shownId != null && (
              <button
                className={styles.graphStageInspect}
                onClick={() => onSampleClick?.(shownId)}
              >
                Inspect {"→"}
              </button>
            )}
          </div>

          {/* Canvas */}
          <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
            {peeking && (
              <div className={styles.peekBadge}>
                Peeking {shownIsCorrect ? "Correct" : "Incorrect"}
              </div>
            )}
            <MainCanvas bundle={shownBundle} lens={lens} />
          </div>
        </div>
      )}

      {/* ── Shared legend ── */}
      <div className={styles.sharedLegend}>
        {LEGEND.map((l) => (
          <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: l.color, flexShrink: 0 }} />
            {l.label}
          </div>
        ))}
      </div>

      {/* ── Side-by-side text panels ── */}
      <div className={styles.pairRow}>
        {bestId != null && (
          <TextPanel
            isCorrect
            sampleId={bestId}
            nfs={bestNfs}
            bundle={bestBundle}
            textData={bestText}
            onSampleClick={(sid) => onSampleClick?.(sid)}
          />
        )}
        {worstId != null && (
          <TextPanel
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
