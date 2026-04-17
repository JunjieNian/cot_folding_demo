import React, { useCallback } from "react";
import styles from "./CheckpointSlider.module.css";

const LABEL_STEPS = new Set([0, 500, 1000]);

export default function CheckpointSlider({ checkpoints, selected, onChange, sampleCorrectness }) {
  if (!checkpoints || checkpoints.length === 0) return null;

  const selectedIdx = checkpoints.findIndex((c) => c.name === selected);
  const current = checkpoints[selectedIdx] || checkpoints[0];

  const handleKeyDown = useCallback((e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = Math.min(selectedIdx + 1, checkpoints.length - 1);
      if (next !== selectedIdx) onChange(checkpoints[next].name);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      const prev = Math.max(selectedIdx - 1, 0);
      if (prev !== selectedIdx) onChange(checkpoints[prev].name);
    }
  }, [selectedIdx, checkpoints, onChange]);

  // Compute dot color based on per-sample correctness at each checkpoint
  function getDotColor(ckptName) {
    if (!sampleCorrectness || sampleCorrectness[ckptName] == null) return undefined;
    return sampleCorrectness[ckptName] ? "#2ecc40" : "#ff4136";
  }

  return (
    <div
      className={styles.wrapper}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      role="slider"
      aria-label="RL Checkpoint"
      aria-valuemin={0}
      aria-valuemax={checkpoints.length - 1}
      aria-valuenow={selectedIdx}
      aria-valuetext={current.name}
    >
      <div className={styles.label}>RL Checkpoint</div>

      {/* Timeline track */}
      <div className={styles.track}>
        {/* Progress fill */}
        {checkpoints.length > 1 && (
          <div
            className={styles.fill}
            style={{ width: `${(selectedIdx / (checkpoints.length - 1)) * 100}%` }}
          />
        )}

        {/* Dots */}
        {checkpoints.map((ckpt, i) => {
          const isSelected = ckpt.name === selected;
          const dotColor = getDotColor(ckpt.name);
          const corrVal = sampleCorrectness?.[ckpt.name];
          const corrLabel = corrVal != null ? ` (sample: ${corrVal ? "\u2713" : "\u2717"})` : "";
          const dotStyle = { left: `${(i / (checkpoints.length - 1)) * 100}%` };

          // Apply color to dot: override border and background
          if (dotColor) {
            dotStyle.borderColor = dotColor;
            if (isSelected) {
              dotStyle.background = dotColor;
              dotStyle.boxShadow = `0 0 0 3px ${dotColor}33`;
            } else {
              dotStyle.background = dotColor;
            }
          }

          return (
            <button
              key={ckpt.name}
              className={`${styles.dot}${isSelected ? ` ${styles.active}` : ""}`}
              style={dotStyle}
              onClick={() => onChange(ckpt.name)}
              title={`${ckpt.name} — acc ${ckpt.accuracy?.toFixed(1)}%${corrLabel}`}
              aria-label={`Switch to checkpoint ${ckpt.name}`}
              tabIndex={-1}
            />
          );
        })}

        {/* Step labels */}
        {checkpoints.map((ckpt, i) => {
          if (!LABEL_STEPS.has(ckpt.rl_step)) return null;
          return (
            <span
              key={`label-${ckpt.name}`}
              className={styles.stepLabel}
              style={{ left: `${(i / (checkpoints.length - 1)) * 100}%` }}
            >
              {ckpt.name === "base" ? "base" : ckpt.rl_step}
            </span>
          );
        })}
      </div>

      {/* Current checkpoint info */}
      <div className={styles.info}>
        <span className={styles.infoName}>{current.name}</span>
        <span className={styles.infoMetric}>
          Acc {current.accuracy?.toFixed(1)}%
        </span>
        {current.auroc != null && (
          <span className={styles.infoMetric}>
            AUROC {current.auroc?.toFixed(3)}
          </span>
        )}
      </div>
    </div>
  );
}
