import React, { Suspense, lazy, useEffect, useState, useRef } from "react";
import useFoldingState from "./hooks/useFoldingState";
import useURLState from "./hooks/useURLState";
import useKeyboardShortcuts from "./hooks/useKeyboardShortcuts";
import useAnswerIslandDetection from "./hooks/useAnswerIslandDetection";
import Sidebar from "./components/Sidebar";
import LensToggle from "./components/LensToggle";
import FocusToggle from "./components/FocusToggle";
import SegmentedText from "./components/SegmentedText";
import SliceNeighborsPanel from "./components/SliceNeighborsPanel";
import { ChartSkeleton } from "./components/Skeleton";
import styles from "./components/App.module.css";

const BatchOverview = lazy(() => import("./components/BatchOverview"));
const FoldingArcDiagram = lazy(() => import("./components/FoldingArcDiagram"));
const ComparePage = lazy(() => import("./components/ComparePage"));
const FoldingView3D = lazy(() => import("./components/FoldingView3D"));
const TrajectoryView = lazy(() => import("./components/TrajectoryView"));

function truncateId(id) {
  const s = String(id);
  return s.length > 12 ? s.slice(0, 8) + "..." : s;
}

export default function App() {
  const [overviewMounted, setOverviewMounted] = useState(false);
  const [toastMsg, setToastMsg] = useState(null);
  const [hoveredSlice, setHoveredSlice] = useState(null);
  const toastTimer = useRef(null);
  const state = useFoldingState();
  const {
    datasets, activeDataset, semanticValidation,
    problems, selectedProblem, samples, selectedSample,
    foldingData, loading, error,
    decodedSimilarity,
    flowData, functionalData,
    textBundle,
    selectedSlice, sliceTextData,
    showOverview, view, colorMode, sidebarCollapsed, darkMode,
    lensMode, focusMode,
    // RL-specific
    checkpoints, selectedCheckpoint, trajectoryData, problemsMeta, checkpointSampleCorrectness,
    setSelectedProblem, setSelectedSample,
    setShowOverview, setView, setColorMode,
    setSelectedSlice, setSliceTextData,
    setLensMode, setFocusMode,
    handleSliceClick, handleDatasetSwitch,
    handleCheckpointChange,
    toggleDarkMode, toggleSidebar,
  } = state;

  // URL state persistence
  useURLState({
    activeDataset, selectedProblem, selectedSample, view, colorMode,
    handleDatasetSwitch, setSelectedProblem, setSelectedSample, setView, setColorMode,
    datasets,
    selectedCheckpoint, handleCheckpointChange,
  });

  // Keyboard shortcuts
  const { showHelp, setShowHelp } = useKeyboardShortcuts({
    problems, selectedProblem, setSelectedProblem,
    samples, selectedSample, setSelectedSample,
    view, setView, setShowOverview,
    lensMode, setLensMode, focusMode, setFocusMode,
  });

  // Answer island detection
  const currentSample = samples?.find((s) => s.sample_id === selectedSample);
  const answerIsland = useAnswerIslandDetection(foldingData, currentSample);

  // Auto-switch to split when answer island detected
  const prevIsland = useRef(null);
  useEffect(() => {
    if (answerIsland && !prevIsland.current) {
      if (lensMode === "2d") setLensMode("split");
      setToastMsg("Terminal branch candidate detected");
      if (toastTimer.current) clearTimeout(toastTimer.current);
      toastTimer.current = setTimeout(() => setToastMsg(null), 5000);
    }
    prevIsland.current = answerIsland;
  }, [answerIsland, lensMode, setLensMode]);

  useEffect(() => {
    if (showOverview) {
      setOverviewMounted(true);
    }
  }, [showOverview]);

  const unitLabel = foldingData?.unit_label ?? "slice";
  const unitTitle = unitLabel.charAt(0).toUpperCase() + unitLabel.slice(1);

  // When hovering text in default view, clear on sample change
  useEffect(() => setHoveredSlice(null), [selectedSample]);

  return (
    <div className={styles.root}>
      {/* Sidebar */}
      <Sidebar
        datasets={datasets} activeDataset={activeDataset} handleDatasetSwitch={handleDatasetSwitch}
        view={view} setView={setView}
        problems={problems} selectedProblem={selectedProblem} setSelectedProblem={setSelectedProblem}
        samples={samples} selectedSample={selectedSample} setSelectedSample={setSelectedSample}
        colorMode={colorMode} setColorMode={setColorMode}
        functionalData={functionalData} foldingData={foldingData}
        semanticValidation={semanticValidation}
        currentSample={currentSample}
        showOverview={showOverview} setShowOverview={setShowOverview}
        collapsed={sidebarCollapsed} onToggleCollapse={toggleSidebar}
        darkMode={darkMode} onToggleDarkMode={toggleDarkMode}
        checkpoints={checkpoints} selectedCheckpoint={selectedCheckpoint}
        onCheckpointChange={handleCheckpointChange} trajectoryData={trajectoryData}
        checkpointSampleCorrectness={checkpointSampleCorrectness}
      />

      {/* Main content */}
      <div className={styles.main}>
        {loading && (
          <div className={styles.loading} role="status" aria-live="polite">
            Loading {truncateId(selectedProblem)} S{selectedSample}...
          </div>
        )}
        {error && (
          <div className={styles.error} role="alert">
            {error}
          </div>
        )}

        {view === "trajectory" && activeDataset === "rl" ? (
          <Suspense fallback={<ChartSkeleton />}>
            <TrajectoryView
              trajectoryData={trajectoryData}
              problemsMeta={problemsMeta}
              darkMode={darkMode}
              onProblemClick={(pid) => { setSelectedProblem(pid); setView("structure"); }}
            />
          </Suspense>
        ) : view === "structure" && selectedProblem != null ? (
          <Suspense fallback={<ChartSkeleton />}>
            <ComparePage
              problemId={selectedProblem}
              onProblemClick={(pid) => { setSelectedProblem(pid); }}
              onSampleClick={(sid) => { setSelectedSample(sid); setView("arc"); }}
            />
          </Suspense>
        ) : foldingData ? (
          <div className={styles.contentArea}>
            {/* Inspect center: toolbar + canvas */}
            <div className={styles.inspectCenter}>
              {/* Toolbar: lens toggle + focus toggle */}
              <div className={styles.inspectToolbar}>
                <LensToggle lensMode={lensMode} setLensMode={setLensMode} />
                <FocusToggle focusMode={focusMode} setFocusMode={setFocusMode} answerIsland={answerIsland} />
              </div>

              {/* Main canvas — switches by lensMode */}
              {lensMode === "2d" ? (
                <div className={styles.inspectArc}>
                  <Suspense fallback={<ChartSkeleton />}>
                    <FoldingArcDiagram
                      data={foldingData} colorMode={colorMode}
                      decodedSimilarity={decodedSimilarity}
                      onSliceClick={handleSliceClick}
                      focusMode={focusMode} answerIsland={answerIsland}
                      hoveredSlice={hoveredSlice}
                      compact
                    />
                  </Suspense>
                </div>
              ) : lensMode === "3d" ? (
                <div className={styles.inspectArc}>
                  <Suspense fallback={<ChartSkeleton />}>
                    <FoldingView3D
                      data={foldingData} colorMode={colorMode}
                      onSliceClick={handleSliceClick}
                      focusMode={focusMode} answerIsland={answerIsland}
                      hoveredSlice={hoveredSlice}
                    />
                  </Suspense>
                </div>
              ) : (
                /* split */
                <div className={styles.inspectSplit}>
                  <div className={styles.inspectSplitMain}>
                    <Suspense fallback={<ChartSkeleton />}>
                      <FoldingArcDiagram
                        data={foldingData} colorMode={colorMode}
                        decodedSimilarity={decodedSimilarity}
                        onSliceClick={handleSliceClick}
                        focusMode={focusMode} answerIsland={answerIsland}
                        hoveredSlice={hoveredSlice}
                        compact
                      />
                    </Suspense>
                  </div>
                  <div className={styles.inspectSplitInset}>
                    <Suspense fallback={<ChartSkeleton />}>
                      <FoldingView3D
                        data={foldingData} colorMode={colorMode}
                        onSliceClick={handleSliceClick}
                        focusMode={focusMode} answerIsland={answerIsland}
                        hoveredSlice={hoveredSlice}
                        compact
                      />
                    </Suspense>
                  </div>
                </div>
              )}
            </div>

            {/* Text panel — always visible, always interactive */}
            <div className={styles.textPanel}>
              {textBundle?.full_text ? (
                <>
                  <div className={styles.textPanelHeader}>
                    <span className={styles.textPanelTitle}>
                      CoT Text
                      <span className={styles.textPanelMeta}>
                        {" "}{textBundle.items?.length ?? 0} {unitLabel}s
                        {selectedSlice != null
                          ? ` \u2014 ${unitTitle} ${selectedSlice}`
                          : " \u2014 hover to locate on map"}
                      </span>
                    </span>
                    {selectedSlice != null && (
                      <button
                        className={styles.textPanelClose}
                        onClick={() => { setSelectedSlice(null); setSliceTextData(null); setHoveredSlice(null); }}
                        aria-label="Deselect slice"
                      >
                        {"\u00D7"}
                      </button>
                    )}
                  </div>
                  {selectedSlice != null && (
                    <SliceNeighborsPanel
                      problemId={selectedProblem}
                      sampleId={selectedSample}
                      selectedSlice={typeof selectedSlice === "number" ? selectedSlice : null}
                      onSliceClick={handleSliceClick}
                    />
                  )}
                  <div className={styles.textPanelBody}>
                    <SegmentedText
                      textBundle={textBundle}
                      hmmStates={foldingData.hmm_states}
                      hoveredSlice={hoveredSlice}
                      selectedSlice={typeof selectedSlice === "number" ? selectedSlice : null}
                      onHoverSlice={setHoveredSlice}
                      onClickSlice={handleSliceClick}
                      focusMode={focusMode}
                      answerIsland={answerIsland}
                    />
                  </div>
                </>
              ) : (
                <div className={styles.textPanelBody} style={{ display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-faint)", textAlign: "center", padding: 24 }}>
                  <div style={{ fontSize: 13, lineHeight: 1.6 }}>Loading text...</div>
                </div>
              )}
            </div>
          </div>
        ) : (
          !loading && <div className={styles.placeholder}>Select a problem and sample</div>
        )}
      </div>

      {overviewMounted && (
        <Suspense fallback={null}>
          <BatchOverview
            open={showOverview}
            onClose={() => setShowOverview(false)}
            onProblemClick={(pid) => { setSelectedProblem(pid); setShowOverview(false); }}
          />
        </Suspense>
      )}

      {/* Toast notification */}
      {toastMsg && <div className={styles.toast}>{toastMsg}</div>}

      {/* Keyboard shortcuts help modal */}
      {showHelp && (
        <div className={styles.textPanel} style={{
          position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
          width: 340, height: "auto", zIndex: 1001, boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
          borderRadius: 8, border: "1px solid var(--color-border)",
        }}>
          <div className={styles.textPanelHeader}>
            <span className={styles.textPanelTitle}>Keyboard Shortcuts</span>
            <button className={styles.textPanelClose} onClick={() => setShowHelp(false)} aria-label="Close help">{"\u00D7"}</button>
          </div>
          <div style={{ padding: 12, fontSize: 12, lineHeight: 2 }}>
            <div><kbd>{"\u2190"}/{"\u2192"}</kbd> \u2014 Switch sample</div>
            <div><kbd>{"\u2191"}/{"\u2193"}</kbd> \u2014 Switch problem</div>
            <div style={{ marginTop: 4, fontWeight: 600, color: "var(--color-text-faint)", fontSize: 10, textTransform: "uppercase" }}>Pages</div>
            <div><kbd>1</kbd> \u2014 Inspect</div>
            <div><kbd>2</kbd> \u2014 Compare</div>
            <div><kbd>3</kbd> \u2014 Training (RL)</div>
            <div style={{ marginTop: 4, fontWeight: 600, color: "var(--color-text-faint)", fontSize: 10, textTransform: "uppercase" }}>Lens (Inspect)</div>
            <div><kbd>D</kbd> \u2014 2D</div>
            <div><kbd>T</kbd> \u2014 3D</div>
            <div><kbd>S</kbd> \u2014 Split</div>
            <div style={{ marginTop: 4, fontWeight: 600, color: "var(--color-text-faint)", fontSize: 10, textTransform: "uppercase" }}>Focus (Inspect)</div>
            <div><kbd>G</kbd> \u2014 Global</div>
            <div><kbd>A</kbd> \u2014 Answer tail</div>
            <div style={{ marginTop: 4 }}><kbd>B</kbd> \u2014 Batch overview</div>
            <div><kbd>?</kbd> \u2014 This help</div>
            <div><kbd>Esc</kbd> \u2014 Close</div>
          </div>
        </div>
      )}
    </div>
  );
}
