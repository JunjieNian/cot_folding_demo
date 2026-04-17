#!/usr/bin/env python3
"""
Effectiveness & Vascular Flow 判别力验证 + 相关性分析

对所有 AIME24 样本系统评估 effectiveness 和 flow 指标：
  - 判别力：AUROC / AUPRC / Cohen's d / Mann-Whitney U (correct vs incorrect)
  - 相关性：与 NFS 的 Pearson / Spearman 相关 + 热力图

数据来源：primitives_analysis.json + nfs_analysis.json + npy 文件。
复用 FoldingEngine._compute_effectiveness 和 get_flow_data 的计算逻辑。

用法:
  python -m analysis.effectiveness_flow_eval
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from project_paths import (
    REPO_ROOT,
    default_batch_dir_for_benchmark,
    default_results_dir,
)
from nfs_pipeline.fold_score import compute_auroc_auprc

BATCH_DIR = default_batch_dir_for_benchmark("aime24")
RESULTS_DIR = default_results_dir()
OUT_JSON = BATCH_DIR / "effectiveness_flow_eval.json"
OUT_PNG = RESULTS_DIR / "effectiveness_flow_eval.png"


# ═══════════════════════════════════════════════════════════════
#  1. Data collection — compute metrics from pre-existing JSON + npy
# ═══════════════════════════════════════════════════════════════

def _load_json(name: str) -> dict | None:
    p = BATCH_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _compute_effectiveness_from_primitives(prim: dict, n: int, hmm_states: np.ndarray,
                                           similarity: np.ndarray) -> dict:
    """Replicates FoldingEngine._compute_effectiveness logic from primitives."""
    core_indices = set(prim["core"]["indices"])
    return_edges = prim.get("return_edges", [])
    return_endpoints = set()
    for e in return_edges:
        if e["i"] < n:
            return_endpoints.add(e["i"])
        if e["j"] < n:
            return_endpoints.add(e["j"])

    drift_set = set()
    for br in prim.get("drift_branches", []):
        if br.get("is_drift", False):
            start = br.get("start", br.get("seg_id", 0))
            length = br.get("length", 1)
            for idx in range(start, min(start + length, n)):
                drift_set.add(idx)

    fc = prim.get("final_closure", {})
    closure_start = fc.get("last_exploit_start", n)
    closure_end = fc.get("last_exploit_end", n)
    closure_set = set(range(closure_start, min(closure_end, n)))
    closure_coeff = fc.get("closure_coefficient", 0.0)

    # Core similarity
    core_list = list(core_indices)
    if core_list:
        core_sim = np.mean(similarity[:, core_list], axis=1)
    else:
        core_sim = np.zeros(n)
    cs_min, cs_max = core_sim.min(), core_sim.max()
    cs_range = cs_max - cs_min if cs_max > cs_min else 1e-8
    core_sim_norm = (core_sim - cs_min) / cs_range

    # Return edge contribution
    return_counts = np.zeros(n)
    for e in return_edges:
        if e["i"] < n:
            return_counts[e["i"]] += e.get("similarity", 0)
        if e["j"] < n:
            return_counts[e["j"]] += e.get("similarity", 0)
    rc_max = return_counts.max() if return_counts.max() > 0 else 1e-8
    return_norm = return_counts / rc_max

    # Combined score + labels
    scores = np.zeros(n)
    labels = []
    for i in range(n):
        if i in core_indices:
            scores[i] = 0.7 + 0.3 * core_sim_norm[i]
        elif i in closure_set:
            scores[i] = 0.6 + 0.4 * closure_coeff
        elif i in drift_set:
            scores[i] = 0.1 + 0.15 * core_sim_norm[i]
        elif i in return_endpoints:
            scores[i] = 0.5 + 0.3 * return_norm[i] + 0.2 * core_sim_norm[i]
        else:
            scores[i] = 0.2 + 0.5 * core_sim_norm[i] + 0.3 * return_norm[i]

        if i in drift_set:
            labels.append("drift")
        elif i in core_indices:
            labels.append("core")
        elif i in closure_set:
            labels.append("closure")
        elif i in return_endpoints:
            labels.append("return_site")
        elif hmm_states[i] == 0 and scores[i] > 0.4:
            labels.append("productive_explore")
        elif hmm_states[i] == 1 and scores[i] > 0.5:
            labels.append("productive_exploit")
        elif hmm_states[i] == 0:
            labels.append("explore")
        else:
            labels.append("exploit")

    productive_count = sum(1 for l in labels if l in
                          ("core", "closure", "productive_explore", "productive_exploit", "return_site"))

    return {
        "productive_fraction": round(productive_count / n, 4) if n > 0 else 0,
        "n_drift_slices": len(drift_set),
        "closure_coefficient": round(closure_coeff, 4),
        "n_return_edges": len(return_edges),
    }


def _compute_flow_from_npy(dist_matrix: np.ndarray, hmm_states: np.ndarray) -> dict:
    """Replicates FoldingEngine.get_flow_data logic using similarity-derived gradients.

    Without NAD cache, we derive entropy/confidence proxies from the distance matrix:
      entropy_proxy[i] = mean distance to neighbours (local disorder)
      confidence_proxy[i] = 1 - entropy_proxy[i]
    """
    n = len(hmm_states)
    similarity = 1.0 - dist_matrix

    # Entropy proxy: mean distance to k nearest neighbours
    k = min(5, n - 1)
    sorted_dist = np.sort(dist_matrix, axis=1)[:, 1:k + 1]  # exclude self
    entropy_proxy = sorted_dist.mean(axis=1)
    confidence_proxy = 1.0 - entropy_proxy

    # Sequence gradients
    d_entropy = np.diff(entropy_proxy)
    d_confidence = np.diff(confidence_proxy)

    ent_std = d_entropy.std() if d_entropy.std() > 1e-12 else 1.0
    conf_std = d_confidence.std() if d_confidence.std() > 1e-12 else 1.0
    d_ent_norm = d_entropy / ent_std
    d_conf_norm = d_confidence / conf_std

    # Flow classification
    flow_type = []
    for i in range(n - 1):
        de = d_ent_norm[i]
        dc = d_conf_norm[i]
        if abs(de) < 0.3 and abs(dc) < 0.3:
            flow_type.append("capillary")
        elif abs(de) < 0.3 and dc > 0.3:
            flow_type.append("shunt")
        elif de > 0 or dc < -0.3:
            flow_type.append("arterial")
        elif de < 0 or dc > 0.3:
            flow_type.append("venous")
        else:
            if abs(de) >= abs(dc):
                flow_type.append("arterial" if de > 0 else "venous")
            else:
                flow_type.append("venous" if dc > 0 else "arterial")

    n_edges = n - 1
    art_count = flow_type.count("arterial")
    ven_count = flow_type.count("venous")
    cap_count = flow_type.count("capillary")

    # Congestion: 3+ consecutive high-entropy explore with no state change
    congestion_count = 0
    median_ent = np.median(entropy_proxy)
    run = 0
    for i in range(n):
        if hmm_states[i] == 0 and entropy_proxy[i] > median_ent:
            run += 1
        else:
            if run >= 3:
                congestion_count += 1
            run = 0
    if run >= 3:
        congestion_count += 1

    return {
        "arterial_fraction": round(art_count / n_edges, 4) if n_edges > 0 else 0,
        "venous_fraction": round(ven_count / n_edges, 4) if n_edges > 0 else 0,
        "capillary_fraction": round(cap_count / n_edges, 4) if n_edges > 0 else 0,
        "congestion_count": congestion_count,
    }


def collect_metrics() -> list[dict]:
    """Load all data and compute per-sample metrics."""
    nfs_data = _load_json("nfs_analysis.json")
    primitives_data = _load_json("primitives_analysis.json")
    if not nfs_data or not primitives_data:
        print("  ERROR: Required JSON files not found")
        return []

    nfs_samples = nfs_data.get("samples", [])

    # Build primitives lookup
    prim_lookup: dict[tuple, dict] = {}
    for rec in primitives_data.get("samples", []):
        pid = int(rec["problem_id"]) if isinstance(rec["problem_id"], str) else rec["problem_id"]
        prim_lookup[(pid, rec["sample_id"])] = rec

    records: list[dict] = []
    n_ok = n_fail_eff = n_fail_flow = 0

    for s in nfs_samples:
        pid = s["problem_id"]
        sid = s["sample_id"]

        rec = {
            "problem_id": pid,
            "sample_id": sid,
            "is_correct": bool(s.get("is_correct", False)),
            "NFS": float(s.get("NFS", 0)),
        }

        # Load npy files
        dist_file = BATCH_DIR / f"dist_p{pid}_s{sid}.npy"
        hmm_file = BATCH_DIR / f"hmm_p{pid}_s{sid}.npy"
        if not dist_file.exists() or not hmm_file.exists():
            n_fail_eff += 1
            continue

        dist_matrix = np.load(dist_file)
        hmm_states = np.load(hmm_file)
        n = len(hmm_states)
        similarity = 1.0 - dist_matrix

        # Effectiveness
        prim = prim_lookup.get((pid, sid))
        if prim:
            eff = _compute_effectiveness_from_primitives(prim, n, hmm_states, similarity)
            rec["eff_productive_fraction"] = eff["productive_fraction"]
            rec["eff_n_drift_slices"] = eff["n_drift_slices"]
            rec["eff_closure_coeff"] = eff["closure_coefficient"]
            rec["eff_n_return_edges"] = eff["n_return_edges"]
        else:
            n_fail_eff += 1
            rec["eff_productive_fraction"] = 0
            rec["eff_n_drift_slices"] = 0
            rec["eff_closure_coeff"] = 0
            rec["eff_n_return_edges"] = 0

        # Flow
        try:
            flow = _compute_flow_from_npy(dist_matrix, hmm_states)
            rec["flow_arterial_fraction"] = flow["arterial_fraction"]
            rec["flow_venous_fraction"] = flow["venous_fraction"]
            rec["flow_capillary_fraction"] = flow["capillary_fraction"]
            rec["flow_congestion_count"] = flow["congestion_count"]
        except Exception:
            n_fail_flow += 1
            rec["flow_arterial_fraction"] = 0
            rec["flow_venous_fraction"] = 0
            rec["flow_capillary_fraction"] = 0
            rec["flow_congestion_count"] = 0

        records.append(rec)
        n_ok += 1

        if n_ok % 200 == 0:
            print(f"    ... {n_ok}/{len(nfs_samples)}")

    print(f"  Collected {n_ok} samples "
          f"(eff_fail={n_fail_eff}, flow_fail={n_fail_flow})")
    return records


# ═══════════════════════════════════════════════════════════════
#  2. Discrimination analysis
# ═══════════════════════════════════════════════════════════════

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def discrimination_analysis(records: list[dict], metric_keys: list[str]) -> dict:
    """AUROC, AUPRC, Cohen's d, Mann-Whitney U for each metric."""
    from scipy.stats import mannwhitneyu

    labels = np.array([int(r["is_correct"]) for r in records])
    results = {}

    for key in metric_keys:
        vals = np.array([r.get(key, 0) for r in records], dtype=np.float64)
        correct_vals = vals[labels == 1]
        incorrect_vals = vals[labels == 0]

        if len(correct_vals) < 2 or len(incorrect_vals) < 2:
            results[key] = {"auroc": None, "note": "insufficient samples"}
            continue

        # AUROC / AUPRC
        try:
            auroc, auprc = compute_auroc_auprc(vals, labels)
        except Exception:
            auroc, auprc = np.nan, np.nan

        # Cohen's d
        d = _cohens_d(correct_vals, incorrect_vals)

        # Mann-Whitney U
        try:
            stat, p_value = mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
        except Exception:
            stat, p_value = np.nan, np.nan

        results[key] = {
            "auroc": round(float(auroc), 4) if np.isfinite(auroc) else None,
            "auprc": round(float(auprc), 4) if np.isfinite(auprc) else None,
            "cohens_d": round(d, 4),
            "p_value": round(float(p_value), 6) if np.isfinite(p_value) else None,
            "correct_mean": round(float(correct_vals.mean()), 4),
            "correct_std": round(float(correct_vals.std()), 4),
            "incorrect_mean": round(float(incorrect_vals.mean()), 4),
            "incorrect_std": round(float(incorrect_vals.std()), 4),
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  3. Correlation analysis
# ═══════════════════════════════════════════════════════════════

def correlation_analysis(records: list[dict], metric_keys: list[str]) -> dict:
    """Pearson & Spearman correlation of each metric vs NFS."""
    from scipy.stats import pearsonr, spearmanr

    nfs = np.array([r["NFS"] for r in records], dtype=np.float64)
    results = {}

    for key in metric_keys:
        vals = np.array([r.get(key, 0) for r in records], dtype=np.float64)
        if vals.std() < 1e-12 or nfs.std() < 1e-12:
            results[key] = {"pearson_r": None, "spearman_rho": None, "note": "zero variance"}
            continue
        try:
            pr, pp = pearsonr(vals, nfs)
            sr, sp = spearmanr(vals, nfs)
        except Exception:
            pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan

        results[key] = {
            "pearson_r": round(float(pr), 4) if np.isfinite(pr) else None,
            "pearson_p": round(float(pp), 6) if np.isfinite(pp) else None,
            "spearman_rho": round(float(sr), 4) if np.isfinite(sr) else None,
            "spearman_p": round(float(sp), 6) if np.isfinite(sp) else None,
        }

    return results


def flow_type_chi2(records: list[dict]) -> dict | None:
    """Chi-squared test: flow type distribution vs correctness."""
    from scipy.stats import chi2_contingency

    flow_keys = ["flow_arterial_fraction", "flow_venous_fraction",
                 "flow_capillary_fraction"]
    if not all(k in records[0] for k in flow_keys):
        return None

    labels = np.array([int(r["is_correct"]) for r in records])
    result = {}
    for fk in flow_keys:
        vals = np.array([r[fk] for r in records])
        t1, t2 = np.percentile(vals, [33.3, 66.7])
        bins = np.digitize(vals, [t1, t2])
        try:
            contingency = np.zeros((2, 3), dtype=int)
            for label, b in zip(labels, bins):
                contingency[label, b] += 1
            chi2, p, dof, expected = chi2_contingency(contingency)
            result[fk] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p), 6),
                "dof": int(dof),
            }
        except Exception:
            result[fk] = {"chi2": None, "note": "computation failed"}

    return result


# ═══════════════════════════════════════════════════════════════
#  4. Visualization
# ═══════════════════════════════════════════════════════════════

def make_plots(records: list[dict], all_keys: list[str],
               disc_results: dict, corr_results: dict):
    """Generate correlation heatmap + distribution boxplots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    active_keys = [k for k in all_keys if k in records[0]]
    if len(active_keys) < 2:
        print("  Too few metrics for heatmap — skipping plots")
        return

    n_metrics = len(active_keys)
    corr_matrix = np.zeros((n_metrics, n_metrics))
    data_arrays = {}
    for k in active_keys:
        data_arrays[k] = np.array([r.get(k, 0) for r in records], dtype=np.float64)

    from scipy.stats import spearmanr
    for i, ki in enumerate(active_keys):
        for j, kj in enumerate(active_keys):
            if data_arrays[ki].std() < 1e-12 or data_arrays[kj].std() < 1e-12:
                corr_matrix[i, j] = 0
            else:
                rho, _ = spearmanr(data_arrays[ki], data_arrays[kj])
                corr_matrix[i, j] = rho if np.isfinite(rho) else 0

    short_names = [k.replace("eff_", "E:").replace("flow_", "F:") for k in active_keys]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.2]})

    # Panel 1: Heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(n_metrics):
        for j in range(n_metrics):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman \u03c1")
    ax.set_title("Metric Correlation Matrix", fontsize=11)

    # Panel 2: Boxplots (correct vs incorrect)
    ax2 = axes[1]
    labels_arr = np.array([int(r["is_correct"]) for r in records])
    box_data_correct = []
    box_data_incorrect = []
    for k in active_keys:
        arr = data_arrays[k]
        box_data_correct.append(arr[labels_arr == 1])
        box_data_incorrect.append(arr[labels_arr == 0])

    positions_c = np.arange(n_metrics) * 3
    positions_i = positions_c + 1

    bp_c = ax2.boxplot(box_data_correct, positions=positions_c, widths=0.8,
                       patch_artist=True, showfliers=False)
    bp_i = ax2.boxplot(box_data_incorrect, positions=positions_i, widths=0.8,
                       patch_artist=True, showfliers=False)
    for patch in bp_c["boxes"]:
        patch.set_facecolor("#4CAF50")
        patch.set_alpha(0.6)
    for patch in bp_i["boxes"]:
        patch.set_facecolor("#F44336")
        patch.set_alpha(0.6)

    ax2.set_xticks(positions_c + 0.5)
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax2.legend([bp_c["boxes"][0], bp_i["boxes"][0]], ["Correct", "Incorrect"], fontsize=9)
    ax2.set_title("Correct vs Incorrect Distribution", fontsize=11)

    for idx, k in enumerate(active_keys):
        dr = disc_results.get(k, {})
        auroc = dr.get("auroc")
        if auroc is not None:
            ax2.text(positions_c[idx] + 0.5, ax2.get_ylim()[1] * 0.95,
                     f"AUC={auroc:.2f}", ha="center", fontsize=7, color="#333")

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {OUT_PNG}")


# ═══════════════════════════════════════════════════════════════
#  5. Formatted terminal output
# ═══════════════════════════════════════════════════════════════

def print_table(disc_results: dict, corr_results: dict, chi2_results: dict | None):
    header = f"{'Metric':<32} {'AUROC':>7} {'AUPRC':>7} {'Cohen d':>8} {'p-value':>10} {'r(NFS)':>7} {'\u03c1(NFS)':>7}"
    print("\n" + "=" * len(header))
    print("  Effectiveness & Flow Evaluation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key in disc_results:
        d = disc_results[key]
        c = corr_results.get(key, {})
        auroc = f"{d['auroc']:.4f}" if d.get("auroc") is not None else "  N/A"
        auprc = f"{d['auprc']:.4f}" if d.get("auprc") is not None else "  N/A"
        cd = f"{d['cohens_d']:+.4f}" if d.get("cohens_d") is not None else "  N/A"
        pv = f"{d['p_value']:.6f}" if d.get("p_value") is not None else "    N/A"
        pr = f"{c['pearson_r']:+.4f}" if c.get("pearson_r") is not None else "  N/A"
        sr = f"{c['spearman_rho']:+.4f}" if c.get("spearman_rho") is not None else "  N/A"
        print(f"{key:<32} {auroc:>7} {auprc:>7} {cd:>8} {pv:>10} {pr:>7} {sr:>7}")

    print("-" * len(header))

    if chi2_results:
        print("\n  Flow Type Chi-Squared Tests (vs Correctness):")
        for fk, res in chi2_results.items():
            if res.get("chi2") is not None:
                print(f"    {fk:<32}  \u03c7\u00b2={res['chi2']:.2f}  p={res['p_value']:.4f}  dof={res['dof']}")
            else:
                print(f"    {fk:<32}  {res.get('note', 'N/A')}")

    print()


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 60)
    print("  Effectiveness & Flow Metric Evaluation")
    print("=" * 60)

    # 1. Collect per-sample metrics
    print("\n[1/4] Computing per-sample metrics from JSON + npy...")
    records = collect_metrics()

    if not records:
        print("  ERROR: No records collected — aborting")
        return

    # 2. Define metric groups
    eff_keys = [
        "eff_productive_fraction",
        "eff_n_drift_slices",
        "eff_closure_coeff",
        "eff_n_return_edges",
    ]
    flow_keys = [
        "flow_arterial_fraction",
        "flow_venous_fraction",
        "flow_capillary_fraction",
        "flow_congestion_count",
    ]
    all_keys = eff_keys + flow_keys

    # 3. Discrimination + correlation
    print("\n[2/4] Running discrimination & correlation analysis...")
    disc = discrimination_analysis(records, all_keys)
    corr = correlation_analysis(records, all_keys)
    chi2 = flow_type_chi2(records)

    # 4. Print results
    print_table(disc, corr, chi2)

    # 5. Save JSON
    output = {
        "n_samples": len(records),
        "discrimination": disc,
        "correlation_with_nfs": corr,
        "chi2_flow_vs_correctness": chi2,
        "metric_keys": {"effectiveness": eff_keys, "flow": flow_keys},
    }
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[3/4] Results saved to {OUT_JSON}")

    # 6. Plots
    print("\n[4/4] Generating plots...")
    make_plots(records, all_keys, disc, corr)

    print("\nDone.")


if __name__ == "__main__":
    main()
