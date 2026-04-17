#!/usr/bin/env python3
"""Model Ranking Visualization: Label-Free Metric vs Accuracy Correlation.

Reads rl_dynamics.json and nfs_trajectory.json, computes Spearman/Pearson/Kendall
correlations between label-free checkpoint metrics and accuracy, and produces a
4-panel figure summarising the ranking quality.

Usage:
    python -m analysis.model_ranking_viz
    python -m analysis.model_ranking_viz --top 15
    python -m analysis.model_ranking_viz --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats

# ── path setup ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import REPO_ROOT  # noqa: E402

CROSS_DIR = REPO_ROOT / "batch_results_rl" / "cross_checkpoint"
DYNAMICS_JSON = CROSS_DIR / "rl_dynamics.json"
NFS_JSON = CROSS_DIR / "nfs_trajectory.json"

PEAK_CHECKPOINT = "step-600"


# ── 1. Data loading ────────────────────────────────────────────────────
def load_ranking_data() -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Return (checkpoint_names, accuracy_vec, {metric_name: value_vec}).

    All vectors are aligned by checkpoint order from the JSON files.
    """
    with open(DYNAMICS_JSON) as f:
        dyn = json.load(f)
    with open(NFS_JSON) as f:
        nfs = json.load(f)

    traj = dyn["trajectory"]
    nfs_ckpts = nfs["checkpoints"]

    # Build lookup by name for nfs data
    nfs_by_name: dict[str, dict] = {c["name"]: c for c in nfs_ckpts}

    names = [t["name"] for t in traj]
    accuracy = np.array([t["accuracy"] for t in traj])

    metrics: dict[str, np.ndarray] = {}

    # --- from dynamics (trajectory) ---
    metrics["-B_mean (slice)"] = np.array([-t["B_mean"] for t in traj])
    metrics["-H_mean (slice)"] = np.array([-t["H_mean"] for t in traj])
    metrics["-D* (slice)"] = np.array([-t["D_star_mean"] for t in traj])
    metrics["-NFS_mean (slice)"] = np.array([-t["NFS_mean"] for t in traj])
    metrics["-beta_nfs"] = np.array([-t["beta_nfs"] for t in traj])
    metrics["L"] = np.array([t["L"] for t in traj])
    metrics["R"] = np.array([t["R"] for t in traj])
    metrics["H/B (slice)"] = np.array(
        [t["H_mean"] / t["B_mean"] if t["B_mean"] != 0 else 0.0 for t in traj]
    )

    # --- from nfs_trajectory (three granularities) ---
    for granularity in ("segment_union", "segment_avg"):
        tag = "seg_union" if granularity == "segment_union" else "seg_avg"
        nfs_vals = [nfs_by_name[n][granularity] for n in names]

        metrics[f"nfs_mean ({tag})"] = np.array([v["nfs_mean"] for v in nfs_vals])
        metrics[f"-B_mean ({tag})"] = np.array([-v["B_mean"] for v in nfs_vals])
        metrics[f"H_mean ({tag})"] = np.array([v["H_mean"] for v in nfs_vals])
        metrics[f"D_star_mean ({tag})"] = np.array([v["D_star_mean"] for v in nfs_vals])
        b_arr = np.array([v["B_mean"] for v in nfs_vals])
        h_arr = np.array([v["H_mean"] for v in nfs_vals])
        metrics[f"H/B ({tag})"] = np.where(b_arr != 0, h_arr / b_arr, 0.0)

    # slice-level nfs_std and CV from nfs_trajectory
    slice_vals = [nfs_by_name[n]["slice"] for n in names]
    metrics["nfs_std (slice)"] = np.array([v["nfs_std"] for v in slice_vals])
    nfs_mean_slice = np.array([abs(v["nfs_mean"]) for v in slice_vals])
    nfs_std_slice = np.array([v["nfs_std"] for v in slice_vals])
    metrics["NFS_cv (slice)"] = np.where(
        nfs_mean_slice != 0, nfs_std_slice / nfs_mean_slice, 0.0
    )

    return names, accuracy, metrics


# ── 2. Correlation computation ──────────────────────────────────────────
def compute_correlations(
    names: list[str],
    accuracy: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict]:
    """Compute Spearman/Pearson/Kendall for each metric vs accuracy.

    Returns list of dicts sorted by |Spearman ρ| descending.
    """
    # Ground-truth best checkpoint
    best_idx = int(np.argmax(accuracy))
    gt_best = names[best_idx]

    results = []
    for mname, mvec in metrics.items():
        sp_rho, sp_p = stats.spearmanr(mvec, accuracy)
        pe_r, pe_p = stats.pearsonr(mvec, accuracy)
        ke_tau, ke_p = stats.kendalltau(mvec, accuracy)

        # Predicted best: checkpoint with highest metric value
        pred_idx = int(np.argmax(mvec))
        pred_best = names[pred_idx]

        results.append({
            "metric": mname,
            "spearman_rho": float(sp_rho),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
            "kendall_tau": float(ke_tau),
            "kendall_p": float(ke_p),
            "predicted_best": pred_best,
            "top1_correct": pred_best == gt_best,
            "gt_best": gt_best,
        })

    results.sort(key=lambda r: abs(r["spearman_rho"]), reverse=True)
    return results


# ── 3. Visualization ───────────────────────────────────────────────────
def make_ranking_figure(
    names: list[str],
    accuracy: np.ndarray,
    metrics: dict[str, np.ndarray],
    corr_results: list[dict],
    top_n: int = 20,
) -> plt.Figure:
    """Build 2×2 figure with bar chart, scatter, heatmap, trajectory."""

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.22)

    _panel_a(fig, gs[0, 0], corr_results, top_n)
    _panel_b(fig, gs[0, 1], names, accuracy, metrics, corr_results)
    _panel_c(fig, gs[1, 0], names, accuracy, metrics, corr_results)
    _panel_d(fig, gs[1, 1], names, accuracy, metrics, corr_results)

    fig.suptitle(
        "Model Ranking: Label-Free Metrics vs Accuracy",
        fontsize=16, fontweight="bold", y=1.01,
    )
    return fig


def _panel_a(fig, gs_slot, corr_results: list[dict], top_n: int):
    """Panel A — Spearman ρ bar chart."""
    ax = fig.add_subplot(gs_slot)
    top = corr_results[:top_n][::-1]  # reverse for bottom-up plotting

    y_labels = [r["metric"] for r in top]
    rhos = [r["spearman_rho"] for r in top]
    colours = ["#2166ac" if v > 0 else "#b2182b" for v in rhos]

    bars = ax.barh(range(len(top)), rhos, color=colours, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(0, color="grey", linewidth=0.5)

    for i, r in enumerate(top):
        offset = 0.02 if r["spearman_rho"] >= 0 else -0.02
        ha = "left" if r["spearman_rho"] >= 0 else "right"
        ax.text(
            r["spearman_rho"] + offset, i,
            f"ρ={r['spearman_rho']:.3f}  best={r['predicted_best']}",
            va="center", ha=ha, fontsize=6.5,
        )

    ax.set_title("A. Label-Free Checkpoint Ranking:\nSpearman ρ with Accuracy", fontsize=10)


def _panel_b(fig, gs_slot, names, accuracy, metrics, corr_results):
    """Panel B — Top-4 scatter plots with regression lines."""
    inner = gs_slot.subgridspec(2, 2, hspace=0.35, wspace=0.30)
    top4 = corr_results[:4]

    peak_idx = names.index(PEAK_CHECKPOINT)

    for idx, r in enumerate(top4):
        ax = fig.add_subplot(inner[idx // 2, idx % 2])
        mname = r["metric"]
        mvec = metrics[mname]

        # Scatter
        ax.scatter(
            mvec, accuracy, s=28, c="#4393c3", zorder=3, edgecolors="white", linewidths=0.5
        )
        # Highlight peak
        ax.scatter(
            [mvec[peak_idx]], [accuracy[peak_idx]],
            s=80, c="#d6604d", marker="*", zorder=4, label=PEAK_CHECKPOINT,
        )
        # Label each point
        for i, n in enumerate(names):
            label = n.replace("step-", "s")
            ax.annotate(
                label, (mvec[i], accuracy[i]),
                fontsize=5.5, textcoords="offset points", xytext=(3, 3),
            )

        # Regression line
        slope, intercept, r_val, _, _ = stats.linregress(mvec, accuracy)
        x_fit = np.linspace(mvec.min(), mvec.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="#999999", linewidth=1)
        ax.text(
            0.05, 0.92, f"r²={r_val**2:.3f}",
            transform=ax.transAxes, fontsize=7, color="#555555",
        )

        ax.set_xlabel(mname, fontsize=7)
        ax.set_ylabel("Accuracy (%)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"ρ={r['spearman_rho']:.3f}", fontsize=8)

    # Add panel title via first axes
    fig.axes[-4].set_title(
        f"B. Top-4 Metrics vs Accuracy\nρ={top4[0]['spearman_rho']:.3f}", fontsize=8
    )


def _panel_c(fig, gs_slot, names, accuracy, metrics, corr_results):
    """Panel C — Ranking comparison heatmap."""
    ax = fig.add_subplot(gs_slot)
    top10 = corr_results[:10]
    n_ckpt = len(names)

    # Ground-truth accuracy ranking (1 = best)
    acc_order = np.argsort(-accuracy)
    gt_rank = np.empty(n_ckpt, dtype=int)
    for rank, idx in enumerate(acc_order):
        gt_rank[idx] = rank + 1

    # Build rank matrix: rows = [GT, method1, method2, ...], cols = checkpoints
    row_labels = ["Accuracy (GT)"] + [r["metric"] for r in top10]
    rank_matrix = np.zeros((len(row_labels), n_ckpt), dtype=int)
    rank_matrix[0] = gt_rank

    for ri, r in enumerate(top10):
        mvec = metrics[r["metric"]]
        order = np.argsort(-mvec)
        ranks = np.empty(n_ckpt, dtype=int)
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        rank_matrix[ri + 1] = ranks

    # Color map: 1 (green) → 11 (red)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=1, vmax=n_ckpt)

    im = ax.imshow(rank_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    short_names = [n.replace("step-", "s") for n in names]
    ax.set_xticks(range(n_ckpt))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))

    # Annotate y-tick labels with rho values
    ylabels = ["Accuracy (GT)"]
    for r in top10:
        ylabels.append(f"{r['metric']}  ρ={r['spearman_rho']:.2f}")
    ax.set_yticklabels(ylabels, fontsize=7)

    # Cell text
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            val = rank_matrix[i, j]
            color = "white" if val <= 3 or val >= 9 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    ax.set_title("C. Checkpoint Ranking Comparison (1=best)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Rank")


def _panel_d(fig, gs_slot, names, accuracy, metrics, corr_results):
    """Panel D — Normalised trajectory overlay."""
    ax = fig.add_subplot(gs_slot)
    top5 = corr_results[:5]

    rl_steps = []
    for n in names:
        if n == "base":
            rl_steps.append(0)
        else:
            rl_steps.append(int(n.split("-")[1]))
    rl_steps = np.array(rl_steps)

    def _minmax(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)

    # Accuracy curve (bold black)
    ax.plot(
        rl_steps, _minmax(accuracy), "k-o", linewidth=2.2, markersize=5,
        label="Accuracy", zorder=5,
    )

    # Top-5 metric curves
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, r in enumerate(top5):
        mvec = metrics[r["metric"]]
        ax.plot(
            rl_steps, _minmax(mvec), "-s", color=palette[i], linewidth=1.2,
            markersize=3.5, alpha=0.85,
            label=f"{r['metric']} (ρ={r['spearman_rho']:.2f})",
        )

    # Peak checkpoint vertical line
    peak_step = 600
    ax.axvline(peak_step, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        peak_step + 15, 0.02, f"peak ({PEAK_CHECKPOINT})",
        fontsize=7, color="grey",
    )

    ax.set_xlabel("RL Step")
    ax.set_ylabel("Min-Max Normalised Value")
    ax.set_title("D. Normalised Metric Trajectories", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.set_xlim(-30, 1030)
    ax.set_ylim(-0.05, 1.05)


# ── 4. Output helpers ──────────────────────────────────────────────────
def print_ranking_table(corr_results: list[dict]):
    """Pretty-print the ranking table to stdout."""
    header = (
        f"{'Rank':>4}  {'Metric':<26} {'Spearman ρ':>10} {'Pearson r':>10} "
        f"{'Kendall τ':>10} {'Pred Best':<12} {'Top1?':<5}"
    )
    print("\n" + "=" * len(header))
    print("  Label-Free Checkpoint Ranking (sorted by |Spearman ρ|)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(corr_results, 1):
        check = "✓" if r["top1_correct"] else ""
        print(
            f"{i:>4}  {r['metric']:<26} {r['spearman_rho']:>+10.4f} "
            f"{r['pearson_r']:>+10.4f} {r['kendall_tau']:>+10.4f} "
            f"{r['predicted_best']:<12} {check:<5}"
        )
    print("-" * len(header))
    print(f"  Ground-truth best: {corr_results[0]['gt_best']}")
    print()


def save_json(
    corr_results: list[dict],
    out_dir: Path,
    names: list[str],
    accuracy: np.ndarray,
    metrics: dict[str, np.ndarray],
):
    """Write model_ranking.json to the cross-checkpoint directory.

    Includes metric_values, rank_matrix, and normalized fields so the
    frontend can render all panels with zero computation.
    """
    out_path = CROSS_DIR / "model_ranking.json"

    # rl_steps: numeric x-axis values
    rl_steps = []
    for n in names:
        rl_steps.append(0 if n == "base" else int(n.split("-")[1]))

    # metric_values: raw values per metric (Panel B scatter)
    metric_values = {k: v.tolist() for k, v in metrics.items()}

    # rank_matrix: rank per checkpoint for GT + top-10 (Panel C heatmap)
    n_ckpt = len(names)
    top10 = corr_results[:10]

    acc_order = np.argsort(-accuracy)
    gt_rank = np.empty(n_ckpt, dtype=int)
    for rank, idx in enumerate(acc_order):
        gt_rank[idx] = rank + 1

    rank_matrix = {"Accuracy (GT)": gt_rank.tolist()}
    for r in top10:
        mvec = metrics[r["metric"]]
        order = np.argsort(-mvec)
        ranks = np.empty(n_ckpt, dtype=int)
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1
        rank_matrix[r["metric"]] = ranks.tolist()

    # normalized: min-max to [0,1] for accuracy + top-5 metrics (Panel D trajectory)
    def _minmax(v: np.ndarray) -> list[float]:
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            return [round(float((x - lo) / (hi - lo)), 4) for x in v]
        return [0.5] * len(v)

    normalized = {"accuracy": _minmax(accuracy)}
    for r in corr_results[:5]:
        normalized[r["metric"]] = _minmax(metrics[r["metric"]])

    payload = {
        "description": "Label-free checkpoint ranking correlations with accuracy",
        "peak_checkpoint": PEAK_CHECKPOINT,
        "checkpoints": names,
        "rl_steps": rl_steps,
        "accuracy": accuracy.tolist(),
        "n_metrics": len(corr_results),
        "rankings": corr_results,
        "metric_values": metric_values,
        "rank_matrix": rank_matrix,
        "normalized": normalized,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")


# ── 5. CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualise label-free checkpoint ranking correlations."
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top methods to display in bar chart (default: 20)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for PNG (default: results/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("[1/4] Loading ranking data …")
    names, accuracy, metrics = load_ranking_data()
    print(f"      {len(names)} checkpoints, {len(metrics)} metrics")

    # Correlations
    print("[2/4] Computing correlations …")
    corr_results = compute_correlations(names, accuracy, metrics)

    # Terminal table
    print_ranking_table(corr_results)

    # Figure
    print("[3/4] Generating figure …")
    fig = make_ranking_figure(names, accuracy, metrics, corr_results, top_n=args.top)
    png_path = out_dir / "model_ranking.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")

    # JSON
    print("[4/4] Saving JSON …")
    save_json(corr_results, out_dir, names, accuracy, metrics)

    print("Done.")


if __name__ == "__main__":
    main()
