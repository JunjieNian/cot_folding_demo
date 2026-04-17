import React from "react";
import Tooltip from "./Tooltip";
import styles from "./MetricsPanel.module.css";

function MetricsPanel({ data, clustering, functionalData }) {
  const m = data.metrics;
  const n = m.n_slices;
  const unitLabelPlural = m.unit_label_plural || "slices";

  return (
    <div className={styles.panel}>
      <div className={styles.header}>Folding Metrics</div>
      <Row label="Folding Degree" value={m.folding_degree.toFixed(4)} tooltip="Long-range contacts / total contacts" />
      <div className={styles.sub}>long-range contacts / total contacts</div>
      <Row label="Contact Order" value={m.contact_order.toFixed(4)} tooltip="Weighted avg sequence distance / N" />
      <div className={styles.sub}>weighted avg seq distance / N</div>
      <Row label="Radius of Gyration" value={m.radius_of_gyration.toFixed(4)} tooltip="Spread of coordinates in MDS space" />
      <div className={styles.sub}>spread in MDS space</div>
      <Row label="MDS Stress" value={m.mds_stress.toFixed(4)} tooltip="Kruskal stress-1: embedding distortion" />
      <div className={styles.sub}>embedding distortion</div>
      <Row label="Contact Threshold" value={m.contact_threshold.toFixed(4)} tooltip="mean + 1 std of upper triangle similarity" />
      <div className={styles.sub}>mean + 1 std of upper-tri</div>

      <div className={styles.header}>Sequence Stats</div>
      <Row label={`Total ${unitLabelPlural}`} value={n} />
      {data.original_n_slices != null && unitLabelPlural !== "slices" && (
        <Row label="Original slices" value={data.original_n_slices} />
      )}
      <Row label="Exploration" value={`${m.n_explore} (${(m.n_explore / n * 100).toFixed(1)}%)`} />
      <Row label="Exploitation" value={`${m.n_exploit} (${(m.n_exploit / n * 100).toFixed(1)}%)`} />
      <Row label="Transitions" value={m.n_transitions} tooltip="Number of explore/exploit state changes" />
      <Row label="Long-range contacts" value={`${m.long_range_contacts} / ${m.total_contacts}`} />

      {clustering && (
        <>
          <div className={styles.header}>Clustering Stats</div>
          <Row label="Separation" value={(clustering.separation ?? 0).toFixed(4)} tooltip="Difference between within-state and cross-state similarity" />
          <Row label="Cohen's d (E/X sep.)" value={(clustering.cohens_d ?? 0).toFixed(4)} tooltip="Explore vs Exploit state separation effect size in similarity space" />
          <Row label="p-value" value={clustering.p_value != null ? clustering.p_value.toExponential(2) : "N/A"} />
          <Row label="Within mean" value={(clustering.within_mean ?? 0).toFixed(4)} />
          <Row label="Cross mean" value={(clustering.cross_mean ?? 0).toFixed(4)} />
        </>
      )}

      {functionalData && (
        <>
          <div className={styles.header}>NFS Analysis</div>
          <Row label="NFS Score" value={functionalData.nfs?.toFixed(2) ?? "N/A"} tooltip="Normalized Folding Score" />
          <Row label="Correct" value={functionalData.is_correct ? "\u2713 Yes" : "\u2717 No"} />
          <Row label="Answer" value={functionalData.answer ?? "N/A"} />
          <div className={styles.sub} style={{ fontFamily: "monospace" }}>
            NFS* = 100 {"\u00D7"} (B {"\u00D7"} H {"\u00D7"} (1{"\u2212"}D*))^(1/3)
          </div>

          <div className={styles.header} style={{ fontSize: "var(--font-xl)" }}>Components</div>
          <Row label="B (backbone)" value={functionalData.nfs_components?.B?.toFixed(4) ?? "N/A"} tooltip="s_core * f_core: core strength times core fraction" />
          <Row label="H (hydrogen)" value={functionalData.nfs_components?.H?.toFixed(4) ?? "N/A"} tooltip="Mean return edge similarity" />
          <Row label="D\u2080 (drift)" value={functionalData.nfs_components?.D0?.toFixed(4) ?? "N/A"} tooltip="Weighted unresolved drift" />
          <Row label="G (gate)" value={functionalData.nfs_components?.G?.toFixed(4) ?? "N/A"} tooltip="(1+C)/2 convergence gate" />
          <Row label="D* (drift+closure)" value={functionalData.nfs_components?.D_star?.toFixed(4) ?? "N/A"} tooltip="1 - G*(1-D0): combined drift penalty" />

          <div className={styles.header} style={{ fontSize: "var(--font-xl)" }}>Primitives</div>
          <Row label="Core size" value={functionalData.core?.indices?.length ?? "N/A"} tooltip="Number of slices in core" />
          <Row label="Core internal sim." value={functionalData.core?.internal_similarity?.toFixed(4) ?? "N/A"} tooltip="Average pairwise similarity within core" />
          <Row label="Core exploit frac." value={functionalData.core?.fraction_of_exploit != null ? `${(functionalData.core.fraction_of_exploit * 100).toFixed(1)}%` : "N/A"} tooltip="Fraction of core slices in exploit state" />
          <Row label="Return edges" value={functionalData.return_edges?.length ?? "N/A"} tooltip="Number of long-range return connections" />
          <Row label="Catalytic frac." value={functionalData.contact_summary?.catalytic_fraction != null ? `${(functionalData.contact_summary.catalytic_fraction * 100).toFixed(1)}%` : "N/A"} tooltip="Fraction of contacts that are catalytic" />
          <Row label="Drift branches" value={functionalData.drift_branches?.length ?? "N/A"} tooltip="Total non-core segments" />
          <Row label="True drifts" value={functionalData.drift_branches?.filter(b => b.is_drift).length ?? "N/A"} tooltip="Drift branches with is_drift=true" />
          <Row label="Closure coeff." value={functionalData.final_closure?.closure_coefficient?.toFixed(4) ?? "N/A"} tooltip="Final convergence coefficient" />
          <Row label="s_close" value={functionalData.final_closure?.s_close?.toFixed(4) ?? "N/A"} tooltip="Closure similarity score" />
        </>
      )}
    </div>
  );
}

function Row({ label, value, tooltip }) {
  return (
    <div className={styles.row}>
      <span className={styles.label}>
        {tooltip ? <Tooltip text={tooltip}>{label}</Tooltip> : label}
      </span>
      <span className={styles.value}>{value}</span>
    </div>
  );
}

export default React.memo(MetricsPanel);
