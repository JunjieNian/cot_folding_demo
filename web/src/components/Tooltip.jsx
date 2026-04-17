import React, { useState } from "react";
import styles from "./Tooltip.module.css";

export default function Tooltip({ text, children }) {
  const [visible, setVisible] = useState(false);

  return (
    <span
      className={styles.wrapper}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className={styles.trigger}>{children}</span>
      {visible && <span className={styles.tooltip} role="tooltip">{text}</span>}
    </span>
  );
}
