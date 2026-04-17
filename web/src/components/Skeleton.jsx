import React from "react";
import styles from "./Skeleton.module.css";

export function ChartSkeleton() {
  return <div className={`${styles.skeleton} ${styles.chart}`} />;
}

export function TextSkeleton({ lines = 3 }) {
  return (
    <div>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className={`${styles.skeleton} ${i === lines - 1 ? styles.textShort : styles.text}`} />
      ))}
    </div>
  );
}

export function SidebarSkeleton() {
  return (
    <div className={styles.sidebar}>
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className={`${styles.skeleton} ${styles.sidebarItem}`} />
      ))}
    </div>
  );
}

export default { ChartSkeleton, TextSkeleton, SidebarSkeleton };
