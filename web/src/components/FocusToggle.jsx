import React from "react";
import styles from "./Sidebar.module.css";

const MODES = [
  { key: "global", label: "Global" },
  { key: "answer-tail", label: "Answer tail" },
];

export default function FocusToggle({ focusMode, setFocusMode, answerIsland }) {
  return (
    <div className={styles.pillGroup}>
      {MODES.map(({ key, label }) => (
        <button
          key={key}
          className={`${styles.pill}${focusMode === key ? ` ${styles.active}` : ""}`}
          onClick={() => setFocusMode(key)}
          style={{ position: "relative" }}
        >
          {label}
          {key === "answer-tail" && answerIsland && (
            <span style={{
              position: "absolute", top: 1, right: 1,
              width: 6, height: 6, borderRadius: "50%",
              background: "#FF9800",
            }} />
          )}
        </button>
      ))}
    </div>
  );
}
