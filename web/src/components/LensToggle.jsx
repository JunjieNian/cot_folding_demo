import React from "react";
import styles from "./Sidebar.module.css";

const MODES = [
  { key: "2d", label: "2D" },
  { key: "3d", label: "3D" },
  { key: "split", label: "Split" },
];

export default function LensToggle({ lensMode, setLensMode, compact = false }) {
  return (
    <div className={styles.pillGroup} style={compact ? { fontSize: 10 } : undefined}>
      {MODES.map(({ key, label }) => (
        <button
          key={key}
          className={`${styles.pill}${lensMode === key ? ` ${styles.active}` : ""}`}
          onClick={() => setLensMode(key)}
          style={compact ? { padding: "2px 8px", fontSize: 10 } : undefined}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
