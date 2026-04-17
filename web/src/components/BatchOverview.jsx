import React, { useEffect, useState, useMemo, useCallback } from "react";
import Plot from "./Plot";
import { getBatchOverview } from "../api";
import styles from "./BatchOverview.module.css";

const NFS_VALIDATION = {
  formula: "NFS* = 100 \u00D7 (B \u00D7 H \u00D7 (1\u2212D*))^{1/3}",
  discrimination: { auroc: 0.7646, auprc: 0.8784, baseline_auprc: 0.7536, cohens_d: 0.937 },
  selective_accuracy: [
    { label: "Top 10%", value: 0.9219 },
    { label: "Top 20%", value: 0.9297 },
    { label: "Top 30%", value: 0.9271 },
    { label: "Top 50%", value: 0.8885 },
    { label: "Overall",  value: 0.7536 },
  ],
  ranking: [
    { label: "Hit@1", value: 0.6667 },
    { label: "Hit@3", value: 0.8333 },
    { label: "Hit@5", value: 0.8667 },
    { label: "Pairwise", value: 0.5901 },
  ],
  voting: { majority: 0.80, weighted: 0.80, top1: 0.6667 },
};

function BatchOverview({ open, onClose, onProblemClick }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState("problem_id");
  const [sortAsc, setSortAsc] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (open && !data) {
      getBatchOverview().then(setData).catch((e) => setError(e.message));
    }
  }, [open, data]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const handleSort = useCallback((key) => {
    setSortKey((prev) => {
      if (prev === key) { setSortAsc((a) => !a); return key; }
      setSortAsc(true);
      return key;
    });
  }, []);

  const sortedStats = useMemo(() => {
    if (!data?.problem_stats) return [];
    let rows = [...data.problem_stats];
    if (filter) {
      rows = rows.filter((p) => String(p.problem_id).includes(filter));
    }
    rows.sort((a, b) => {
      const va = a[sortKey], vb = b[sortKey];
      const cmp = typeof va === "number" ? va - vb : String(va).localeCompare(String(vb));
      return sortAsc ? cmp : -cmp;
    });
    return rows;
  }, [data, sortKey, sortAsc, filter]);

  if (!open) return null;

  const columns = [
    { key: "problem_id", label: "Problem" },
    { key: "n_samples", label: "Samples" },
    { key: "avg_slices", label: "Avg Slices" },
    { key: "min_slices", label: "Min" },
    { key: "max_slices", label: "Max" },
  ];

  return (
    <div className={styles.overlay} onClick={onClose} role="dialog" aria-modal="true" aria-label="Batch Overview">
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>Batch Overview</h2>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close">{"\u2715"}</button>
        </div>

        {error && <div className={styles.error}>{error}</div>}
        {!data && !error && <div>Loading...</div>}

        {data && (
          <>
            <div className={styles.statsRow}>
              <Stat label="Problems" value={data.n_problems} />
              <Stat label="Total Samples" value={data.n_samples} />
              <Stat label="Total Time" value={`${data.total_time_s}s`} />
              {data.clustering_summary && (
                <>
                  <Stat label="Mean Separation" value={data.clustering_summary.mean_separation?.toFixed(4)} />
                  <Stat label="Clustering Cohen's d (E/X sep.)" value={data.clustering_summary.mean_cohens_d?.toFixed(4)} />
                  <Stat label="Conclusion" value={data.clustering_summary.conclusion} />
                </>
              )}
            </div>

            {data.distributions && (
              <div className={styles.chartRow}>
                <div className={styles.chartCol}>
                  <Plot
                    data={[{
                      x: data.distributions.separation.values,
                      type: "histogram", nbinsx: 40,
                      marker: { color: "#4285F4" },
                    }]}
                    layout={{
                      title: { text: `Separation (mean=${data.distributions.separation.mean})`, font: { size: 13 } },
                      xaxis: { title: "Separation" }, yaxis: { title: "Count" },
                      margin: { l: 50, r: 20, t: 40, b: 40 }, height: 280,
                      paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                    }}
                    useResizeHandler style={{ width: "100%" }} config={{ responsive: true }}
                  />
                </div>
                <div className={styles.chartCol}>
                  <Plot
                    data={[{
                      x: data.distributions.cohens_d.values,
                      type: "histogram", nbinsx: 40,
                      marker: { color: "#EA4335" },
                    }]}
                    layout={{
                      title: { text: `Clustering Cohen's d (E/X sep.) (mean=${data.distributions.cohens_d.mean})`, font: { size: 13 } },
                      xaxis: { title: "Clustering Cohen's d (E/X sep.)" }, yaxis: { title: "Count" },
                      margin: { l: 50, r: 20, t: 40, b: 40 }, height: 280,
                      paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                    }}
                    useResizeHandler style={{ width: "100%" }} config={{ responsive: true }}
                  />
                </div>
              </div>
            )}

            {/* NFS Distribution */}
            {data.nfs_distribution && (
              <div className={styles.chartRow} style={{ marginTop: 16 }}>
                <div className={styles.chartCol}>
                  <Plot
                    data={[
                      {
                        x: data.nfs_distribution.correct || [],
                        type: "histogram", nbinsx: 30, opacity: 0.7,
                        marker: { color: "#4CAF50" }, name: "Correct",
                      },
                      {
                        x: data.nfs_distribution.incorrect || [],
                        type: "histogram", nbinsx: 30, opacity: 0.7,
                        marker: { color: "#F44336" }, name: "Incorrect",
                      },
                    ]}
                    layout={{
                      title: { text: "NFS Distribution (Correct vs Incorrect)", font: { size: 13 } },
                      xaxis: { title: "NFS" }, yaxis: { title: "Count" },
                      barmode: "overlay",
                      margin: { l: 50, r: 20, t: 40, b: 40 }, height: 280,
                      paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                    }}
                    useResizeHandler style={{ width: "100%" }} config={{ responsive: true }}
                  />
                </div>
              </div>
            )}

            <div className={styles.nfsValidation}>
              <h3>NFS Validation Metrics</h3>
              <div className={styles.formula}>{NFS_VALIDATION.formula}</div>

              <div className={styles.statsRow}>
                <Stat label="AUROC" value={NFS_VALIDATION.discrimination.auroc.toFixed(4)} />
                <Stat label="AUPRC" value={`${NFS_VALIDATION.discrimination.auprc.toFixed(4)} (baseline: ${NFS_VALIDATION.discrimination.baseline_auprc})`} />
                <Stat label="NFS Cohen's d (correct vs incorrect)" value={NFS_VALIDATION.discrimination.cohens_d.toFixed(3)} />
              </div>

              <div className={styles.metricsGrid}>
                <div>
                  <h4>Selective Accuracy</h4>
                  <table className={styles.compactTable}>
                    <tbody>
                      {NFS_VALIDATION.selective_accuracy.map((r) => (
                        <tr key={r.label}><td>{r.label}</td><td>{r.value.toFixed(4)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h4>Ranking (Hit@k)</h4>
                  <table className={styles.compactTable}>
                    <tbody>
                      {NFS_VALIDATION.ranking.map((r) => (
                        <tr key={r.label}><td>{r.label}</td><td>{r.value.toFixed(4)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h4>Voting</h4>
                  <table className={styles.compactTable}>
                    <tbody>
                      <tr><td>Majority</td><td>{NFS_VALIDATION.voting.majority.toFixed(2)}</td></tr>
                      <tr><td>Weighted</td><td>{NFS_VALIDATION.voting.weighted.toFixed(2)}</td></tr>
                      <tr><td>Top-1</td><td>{NFS_VALIDATION.voting.top1.toFixed(4)}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {data.problem_stats && (
              <div className={styles.tableSection}>
                <h3>Per-Problem Statistics</h3>
                <input
                  className={styles.filterInput}
                  placeholder="Filter by problem ID..."
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  aria-label="Filter problems"
                />
                <table className={styles.table}>
                  <thead>
                    <tr>
                      {columns.map((col) => (
                        <th key={col.key} onClick={() => handleSort(col.key)} aria-sort={sortKey === col.key ? (sortAsc ? "ascending" : "descending") : "none"}>
                          {col.label}
                          {sortKey === col.key && <span className={styles.sortIcon}>{sortAsc ? "\u25B2" : "\u25BC"}</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedStats.map((p) => (
                      <tr
                        key={p.problem_id}
                        onClick={() => onProblemClick?.(p.problem_id)}
                        tabIndex={0}
                        onKeyDown={(e) => { if (e.key === "Enter") onProblemClick?.(p.problem_id); }}
                        aria-label={`Problem ${p.problem_id}`}
                      >
                        <td>P{p.problem_id}</td>
                        <td>{p.n_samples}</td>
                        <td>{p.avg_slices}</td>
                        <td>{p.min_slices}</td>
                        <td>{p.max_slices}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className={styles.stat}>
      <div className={styles.statLabel}>{label}</div>
      <div className={styles.statValue}>{value}</div>
    </div>
  );
}

export default React.memo(BatchOverview);
