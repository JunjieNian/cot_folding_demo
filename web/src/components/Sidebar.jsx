import React, { useState } from "react";
import ProblemSelector from "./ProblemSelector";
import SampleSelector from "./SampleSelector";
import CheckpointSlider from "./CheckpointSlider";
import styles from "./Sidebar.module.css";

function StatRow({ label, value, highlight, color }) {
  return (
    <div className={styles.statRow}>
      <span className={styles.statLabel} style={color ? { color } : undefined}>{label}</span>
      <span className={`${styles.statValue}${highlight ? ` ${styles.highlight}` : ""}`}>{value}</span>
    </div>
  );
}

export default function Sidebar({
  datasets, activeDataset, handleDatasetSwitch,
  view, setView,
  problems, selectedProblem, setSelectedProblem,
  samples, selectedSample, setSelectedSample,
  colorMode, setColorMode,
  functionalData, foldingData,
  semanticValidation, currentSample,
  showOverview, setShowOverview,
  collapsed, onToggleCollapse,
  darkMode, onToggleDarkMode,
  // RL-specific props
  checkpoints, selectedCheckpoint, onCheckpointChange, trajectoryData, checkpointSampleCorrectness,
}) {
  const [moreMetrics, setMoreMetrics] = useState(false);
  const [showValidationDetail, setShowValidationDetail] = useState(false);
  const m = foldingData?.metrics;
  const hasFunc = foldingData && foldingData.entropy.some((e) => e !== 0);
  const unitLabel = foldingData?.unit_label ?? "slice";
  const unitTitle = unitLabel.charAt(0).toUpperCase() + unitLabel.slice(1);
  const isRL = activeDataset === "rl";

  return (
    <div className={`${styles.sidebar}${collapsed ? ` ${styles.collapsed}` : ""}`}>
      <button
        className={styles.collapseBtn}
        onClick={onToggleCollapse}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? "\u25B6" : "\u25C0"}
      </button>

      {!collapsed && (
        <>
          {/* Title */}
          <div className={styles.title}>COT Folding Map</div>

          {/* Dataset switcher */}
          <div style={{ marginBottom: 8 }}>
            <div className={styles.pillGroup}>
              {datasets.map((ds) => (
                <button
                  key={ds}
                  className={`${styles.pill}${ds === activeDataset ? ` ${styles.active}` : ""}`}
                  onClick={() => handleDatasetSwitch(ds)}
                  aria-label={`Switch to ${ds} dataset`}
                >
                  {ds === "aime24" ? "AIME24" : "RL"}
                </button>
              ))}
            </div>
          </div>

          {/* Checkpoint slider (RL only) */}
          {isRL && checkpoints && checkpoints.length > 0 && (
            <CheckpointSlider
              checkpoints={checkpoints}
              selected={selectedCheckpoint}
              onChange={onCheckpointChange}
              sampleCorrectness={checkpointSampleCorrectness}
            />
          )}

          {/* Theme toggle */}
          <button className={styles.themeToggle} onClick={onToggleDarkMode} aria-label="Toggle dark mode">
            {darkMode ? "\u2600 Light" : "\u263E Dark"}
          </button>

          {/* View toggle — single row: Inspect | Compare | Training(RL) */}
          <div className={styles.pillGroup} style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <button className={`${styles.pill}${view === "arc" ? ` ${styles.active}` : ""}`} onClick={() => setView("arc")}>
              Inspect
            </button>
            <button className={`${styles.pill}${view === "structure" ? ` ${styles.active}` : ""}`} onClick={() => setView("structure")}>
              Compare
            </button>
            {isRL && (
              <button className={`${styles.pill}${view === "trajectory" ? ` ${styles.active}` : ""}`} onClick={() => setView("trajectory")}>
                Training
              </button>
            )}
          </div>

          <ProblemSelector problems={problems} selected={selectedProblem} onChange={setSelectedProblem} />
          <SampleSelector samples={samples} selected={selectedSample} onChange={setSelectedSample} />

          {/* Color mode */}
          {view === "arc" && hasFunc && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>Node Color</div>
              <div className={styles.colorPillGroup}>
                {[
                  { mode: "entropy", label: "Entropy", color: "#E05A47" },
                  { mode: "confidence", label: "Confidence", color: "#3182BD" },
                  { mode: "state", label: "State", color: "#666" },
                ].map(({ mode, label, color }) => (
                  <button
                    key={mode}
                    className={styles.colorPill}
                    style={colorMode === mode
                      ? { border: `2px solid ${color}`, background: color, color: "#fff" }
                      : undefined
                    }
                    onClick={() => setColorMode(mode)}
                    aria-label={`Color by ${label}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Correctness */}
          {foldingData && foldingData.is_correct != null && (
            <div className={styles.divider} style={{
              padding: "6px 0",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{
                fontSize: 13, fontWeight: 700,
                color: foldingData.is_correct ? "#4CAF50" : "#F44336",
              }}>
                {foldingData.is_correct ? "\u2713 Correct" : "\u2717 Incorrect"}
              </span>
              {foldingData.nfs != null && (
                <span style={{ fontSize: 11, fontWeight: 600, color: "#1A73E8", fontFamily: "monospace" }}>
                  NFS {foldingData.nfs.toFixed(2)}
                </span>
              )}
            </div>
          )}
          {/* Layer 2: per-sample alignment badge */}
          {currentSample && currentSample.alignment_rho != null && (
            <div style={{
              fontSize: 10, color: "var(--color-text-faint)", fontFamily: "monospace",
              padding: "2px 0 4px",
            }}>
              Align ρ {currentSample.alignment_rho.toFixed(3)}
              {currentSample.partial_rho != null && (
                <>  Partial {currentSample.partial_rho.toFixed(3)}</>
              )}
            </div>
          )}
          {foldingData && foldingData.answer != null && (
            <div style={{ fontSize: 10, color: "#888", marginBottom: 4, wordBreak: "break-all" }}>
              Answer: <b style={{ color: "var(--color-text)" }}>{foldingData.answer}</b>
            </div>
          )}
          {/* Problem prompt preview */}
          {(() => {
            const prob = problems.find((p) => p.problem_id === selectedProblem);
            if (!prob || !prob.short_prompt) return null;
            return (
              <div style={{ fontSize: 10, color: "#888", marginBottom: 4, lineHeight: 1.4 }}>
                <span style={{ color: "var(--color-text-faint)", fontWeight: 600 }}>Q: </span>
                {prob.short_prompt}
                {prob.ground_truth && (
                  <div style={{ marginTop: 2 }}>
                    <span style={{ color: "var(--color-text-faint)", fontWeight: 600 }}>GT: </span>
                    <b style={{ color: "var(--color-text)", fontFamily: "monospace" }}>{prob.ground_truth}</b>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Compressed Metrics — key stats always visible, rest in "More" */}
          {m && (
            <div className={styles.statsSection}>
              <div className={styles.sectionLabel}>Metrics</div>
              {foldingData.nfs != null && (
                <StatRow label="NFS" value={foldingData.nfs.toFixed(3)} highlight />
              )}
              <StatRow label="Folding" value={m.folding_degree.toFixed(3)} highlight />
              <StatRow label="Explore/Exploit" value={`${(m.n_explore / m.n_slices * 100).toFixed(0)}% / ${(m.n_exploit / m.n_slices * 100).toFixed(0)}%`} />
              {moreMetrics && (<>
                <StatRow label={unitTitle} value={m.n_slices} />
                <StatRow label="Contacts" value={`${m.long_range_contacts}/${m.total_contacts}`} />
                <StatRow label="Stress" value={foldingData.mds_stress.toFixed(3)} />
                <StatRow label="Explore" value={`${m.n_explore} (${(m.n_explore / m.n_slices * 100).toFixed(0)}%)`} color="#5B8DEF" />
                <StatRow label="Exploit" value={`${m.n_exploit} (${(m.n_exploit / m.n_slices * 100).toFixed(0)}%)`} color="#E05A47" />
                <StatRow label="Transitions" value={m.n_transitions} />
                {foldingData.original_n_slices != null && unitLabel !== "slice" && (
                  <StatRow label="Orig. Slices" value={foldingData.original_n_slices} />
                )}
              </>)}
              <button
                onClick={() => setMoreMetrics((v) => !v)}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  fontSize: 10, color: "var(--color-primary)", padding: "2px 0", marginTop: 2,
                }}
              >
                {moreMetrics ? "Less metrics" : "More metrics..."}
              </button>
            </div>
          )}

          {/* Legend */}
          {view === "arc" && foldingData && (
            <div className={styles.legend}>
              <div className={styles.sectionLabel}>Legend</div>
              <div><span style={{ color: "#5B8DEF" }}>{"\u2501\u2501"}</span> Explore backbone</div>
              <div><span style={{ color: "#E05A47" }}>{"\u2501\u2501"}</span> Exploit backbone</div>
              <div><span style={{ color: "#DcB43c" }}>----</span> Folding bonds</div>
              <div>Node size = entropy</div>
              {colorMode === "entropy" && hasFunc && <div>Node opacity = confidence</div>}
              <div><span style={{ color: "#2E7D32" }}>{"\u25C6"} N</span> start  <span style={{ color: "#E65100" }}>{"\u25A0"} C</span> end</div>
            </div>
          )}

          {/* Layer 1: global validation entry */}
          {semanticValidation && (
            <div style={{
              padding: "8px 0 4px",
              borderTop: "1px solid var(--color-border-light)",
              marginTop: 4,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text)", marginBottom: 4 }}>
                Validated Locality
              </div>
              <div style={{ fontSize: 10, color: "var(--color-text-faint)", fontFamily: "monospace", lineHeight: 1.8 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Structure–text ρ</span>
                  <span style={{ fontWeight: 600 }}>{semanticValidation.embedding_rho.mean}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Position-controlled</span>
                  <span style={{ fontWeight: 600 }}>{semanticValidation.partial_rho.mean}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Top-5 struct vs adj</span>
                  <span style={{ fontWeight: 600 }}>
                    {semanticValidation.topk.structural}/{semanticValidation.topk.adjacent}
                  </span>
                </div>
                {semanticValidation.source_model_rho && (
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Source-model ρ</span>
                    <span style={{ fontWeight: 600 }}>{semanticValidation.source_model_rho.mean}</span>
                  </div>
                )}
              </div>
              <button
                onClick={() => setShowValidationDetail(true)}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  fontSize: 10, color: "var(--color-primary)", padding: "2px 0", marginTop: 2,
                }}
              >
                More details...
              </button>
            </div>
          )}

          {/* Validation detail modal */}
          {showValidationDetail && (
            <div
              onClick={() => setShowValidationDetail(false)}
              style={{
                position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
                background: "rgba(0,0,0,0.5)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 1000,
              }}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                style={{
                  background: "var(--color-surface)",
                  borderRadius: "var(--radius-lg)",
                  padding: 24,
                  maxWidth: 700, width: "90%",
                  maxHeight: "85vh", overflow: "auto",
                  boxShadow: "0 4px 24px rgba(0,0,0,0.2)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                  <h3 style={{ margin: 0, fontSize: 16, color: "var(--color-text)" }}>
                    Structure–Text Alignment
                  </h3>
                  <button
                    onClick={() => setShowValidationDetail(false)}
                    style={{
                      background: "none", border: "none", cursor: "pointer",
                      fontSize: 20, color: "var(--color-text-faint)", lineHeight: 1,
                    }}
                  >{"\u00D7"}</button>
                </div>
                <p style={{ fontSize: 12, color: "var(--color-text-faint)", lineHeight: 1.6, margin: "0 0 16px" }}>
                  Structural similarity (from hidden-layer activation cosine distance) correlates strongly with
                  text semantic similarity across {semanticValidation.n_samples} samples,
                  validated with {semanticValidation.method}.
                </p>
                <img
                  src="./data/aime24/semantic_validation_binned.png"
                  alt="Structural vs text similarity binned plot"
                  style={{ width: "100%", borderRadius: 6, border: "1px solid var(--color-border-light)" }}
                />
                {semanticValidation.shuffle_controls && (
                  <div style={{ marginTop: 16 }}>
                    <h4 style={{ margin: "0 0 8px", fontSize: 13, color: "var(--color-text)" }}>
                      Shuffle Controls (Negative Control)
                    </h4>
                    <p style={{ fontSize: 11, color: "var(--color-text-faint)", lineHeight: 1.5, margin: "0 0 8px" }}>
                      Shuffling text or structure breaks the correlation, confirming it is not a statistical artifact.
                    </p>
                    <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 1.8, color: "var(--color-text-faint)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Observed ρ</span>
                        <span style={{ fontWeight: 600 }}>{semanticValidation.shuffle_controls.text.observed_mean}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Shuffled text ρ</span>
                        <span style={{ fontWeight: 600 }}>{semanticValidation.shuffle_controls.text.shuffled_mean}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Shuffled struct ρ</span>
                        <span style={{ fontWeight: 600 }}>{semanticValidation.shuffle_controls.structure.shuffled_mean}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Mean z-score (text)</span>
                        <span style={{ fontWeight: 600 }}>{semanticValidation.shuffle_controls.text.mean_z}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Mean z-score (struct)</span>
                        <span style={{ fontWeight: 600 }}>{semanticValidation.shuffle_controls.structure.mean_z}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className={styles.spacer} />

          {/* Batch Overview — downgraded to small text link */}
          <div style={{ textAlign: "center", padding: "4px 0" }}>
            <button
              onClick={() => setShowOverview(true)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                fontSize: 11, color: "var(--color-text-faint)", textDecoration: "underline",
              }}
              aria-label="Open batch overview"
            >
              Advanced statistics...
            </button>
          </div>
        </>
      )}
    </div>
  );
}
