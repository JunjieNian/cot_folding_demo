#!/usr/bin/env python3
"""
Shuffle Controls: Negative control experiments proving structure-text correlation
is not a statistical artifact.

Two controls:
  1a. Shuffle Text: preserve structural similarity, randomize text embedding order
  1b. Shuffle Structure: preserve text similarity, randomize structural similarity order

For each sample, we repeat 10 shuffles and compute:
  - observed rho (from Tier 2)
  - shuffle mean/std rho
  - z-score = (observed - shuffle_mean) / shuffle_std
  - empirical p-value

Output:
  results/controls/shuffle_text.csv
  results/controls/shuffle_structure.csv
  results/controls/shuffle_summary.json
  results/summary/shuffle_controls_plot.png
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──
PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Set CUDA_VISIBLE_DEVICES before importing torch if you need a specific GPU
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared functions from run_experiment.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import load_all_samples, encode_all_slices


N_SHUFFLES = 10
RNG_SEED = 42


def run_shuffle_text(samples, embeddings, sample_offsets):
    """
    Shuffle Text control: preserve struct_sim, randomize text embedding order.

    For each sample, permute the embedding indices N_SHUFFLES times,
    recompute text_sim from permuted embeddings, and correlate with struct_sim.
    """
    print("\n── Shuffle Text Control ──")
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED)

    results = []
    for idx, s in enumerate(samples):
        n = s["n_slices"]
        if n < 5:
            continue

        start, count = sample_offsets[idx]
        emb = embeddings[start:start + count]

        # Observed correlation (from original embeddings)
        text_sim = emb @ emb.T
        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        text_vals = text_sim[iu]
        observed_rho, _ = stats.spearmanr(struct_vals, text_vals)

        # Shuffle N times
        shuffled_rhos = []
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(n)
            emb_shuffled = emb[perm]
            text_sim_shuffled = emb_shuffled @ emb_shuffled.T
            text_vals_shuffled = text_sim_shuffled[iu]
            rho_s, _ = stats.spearmanr(struct_vals, text_vals_shuffled)
            if not np.isnan(rho_s):
                shuffled_rhos.append(rho_s)

        if len(shuffled_rhos) < 2:
            continue

        shuffle_mean = np.mean(shuffled_rhos)
        shuffle_std = np.std(shuffled_rhos)
        z_score = (observed_rho - shuffle_mean) / max(shuffle_std, 1e-10)

        # Empirical p: fraction of shuffled rhos >= observed
        p_empirical = np.mean([r >= observed_rho for r in shuffled_rhos])

        results.append({
            "problem_id": s["problem_id"],
            "sample_id": s["sample_id"],
            "n_slices": n,
            "is_correct": s["is_correct"],
            "observed_rho": observed_rho,
            "shuffle_mean_rho": shuffle_mean,
            "shuffle_std_rho": shuffle_std,
            "z_score": z_score,
            "p_empirical": p_empirical,
        })

    elapsed = time.time() - t0
    rhos = [r["observed_rho"] for r in results]
    shuffled_means = [r["shuffle_mean_rho"] for r in results]
    z_scores = [r["z_score"] for r in results]
    print(f"  {len(results)} samples in {elapsed:.1f}s")
    print(f"  Observed ρ mean: {np.mean(rhos):.4f}")
    print(f"  Shuffled ρ mean: {np.mean(shuffled_means):.4f}")
    print(f"  Z-score mean: {np.mean(z_scores):.2f}")
    print(f"  Frac p < 0.05: {np.mean([r['p_empirical'] < 0.05 for r in results]):.2%}")

    out_dir = RESULTS_DIR / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "shuffle_text.csv", index=False)
    print(f"  Saved shuffle_text.csv")
    return results


def run_shuffle_structure(samples, embeddings, sample_offsets):
    """
    Shuffle Structure control: preserve text_sim, randomize struct_sim order.

    For each sample, permute the struct_sim indices (row+col) N_SHUFFLES times,
    and correlate with original text_sim.
    """
    print("\n── Shuffle Structure Control ──")
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED + 1)

    results = []
    for idx, s in enumerate(samples):
        n = s["n_slices"]
        if n < 5:
            continue

        start, count = sample_offsets[idx]
        emb = embeddings[start:start + count]
        text_sim = emb @ emb.T
        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        text_vals = text_sim[iu]
        observed_rho, _ = stats.spearmanr(struct_vals, text_vals)

        # Shuffle struct N times
        shuffled_rhos = []
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(n)
            struct_sim_shuffled = s["struct_sim"][np.ix_(perm, perm)]
            struct_vals_shuffled = struct_sim_shuffled[iu]
            rho_s, _ = stats.spearmanr(struct_vals_shuffled, text_vals)
            if not np.isnan(rho_s):
                shuffled_rhos.append(rho_s)

        if len(shuffled_rhos) < 2:
            continue

        shuffle_mean = np.mean(shuffled_rhos)
        shuffle_std = np.std(shuffled_rhos)
        z_score = (observed_rho - shuffle_mean) / max(shuffle_std, 1e-10)
        p_empirical = np.mean([r >= observed_rho for r in shuffled_rhos])

        results.append({
            "problem_id": s["problem_id"],
            "sample_id": s["sample_id"],
            "n_slices": n,
            "is_correct": s["is_correct"],
            "observed_rho": observed_rho,
            "shuffle_mean_rho": shuffle_mean,
            "shuffle_std_rho": shuffle_std,
            "z_score": z_score,
            "p_empirical": p_empirical,
        })

    elapsed = time.time() - t0
    rhos = [r["observed_rho"] for r in results]
    shuffled_means = [r["shuffle_mean_rho"] for r in results]
    z_scores = [r["z_score"] for r in results]
    print(f"  {len(results)} samples in {elapsed:.1f}s")
    print(f"  Observed ρ mean: {np.mean(rhos):.4f}")
    print(f"  Shuffled ρ mean: {np.mean(shuffled_means):.4f}")
    print(f"  Z-score mean: {np.mean(z_scores):.2f}")
    print(f"  Frac p < 0.05: {np.mean([r['p_empirical'] < 0.05 for r in results]):.2%}")

    out_dir = RESULTS_DIR / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "shuffle_structure.csv", index=False)
    print(f"  Saved shuffle_structure.csv")
    return results


def generate_shuffle_summary(text_results, struct_results):
    """Generate summary JSON for both shuffle controls."""
    def summarize(results, label):
        observed = [r["observed_rho"] for r in results]
        shuffled = [r["shuffle_mean_rho"] for r in results]
        z_scores = [r["z_score"] for r in results]
        return {
            "observed_mean": round(float(np.mean(observed)), 4),
            "observed_std": round(float(np.std(observed)), 4),
            "shuffled_mean": round(float(np.mean(shuffled)), 4),
            "shuffled_std": round(float(np.std(shuffled)), 4),
            "mean_z": round(float(np.mean(z_scores)), 2),
            "median_z": round(float(np.median(z_scores)), 2),
            "frac_p05": round(float(np.mean([r["p_empirical"] < 0.05 for r in results])), 4),
            "n_samples": len(results),
        }

    summary = {
        "text": summarize(text_results, "text"),
        "structure": summarize(struct_results, "structure"),
    }

    out_path = RESULTS_DIR / "controls" / "shuffle_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved shuffle_summary.json")
    print(f"  Text shuffle: observed={summary['text']['observed_mean']}, "
          f"shuffled={summary['text']['shuffled_mean']}, z={summary['text']['mean_z']}")
    print(f"  Structure shuffle: observed={summary['structure']['observed_mean']}, "
          f"shuffled={summary['structure']['shuffled_mean']}, z={summary['structure']['mean_z']}")
    return summary


def generate_shuffle_plot(text_results, struct_results):
    """
    Two-panel plot:
      Left:  observed vs shuffled ρ distributions (overlaid histograms)
      Right: z-score distributions for both controls
    """
    print("\n── Generating shuffle controls plot ──")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: observed vs shuffled distributions ──
    observed_text = [r["observed_rho"] for r in text_results]
    shuffled_text = [r["shuffle_mean_rho"] for r in text_results]
    observed_struct = [r["observed_rho"] for r in struct_results]
    shuffled_struct = [r["shuffle_mean_rho"] for r in struct_results]

    bins = np.linspace(-0.3, 0.9, 60)
    ax1.hist(observed_text, bins=bins, alpha=0.6, color="#3498db", label="Observed ρ", density=True)
    ax1.hist(shuffled_text, bins=bins, alpha=0.6, color="#e74c3c", label="Shuffled text ρ (mean)", density=True)
    ax1.hist(shuffled_struct, bins=bins, alpha=0.4, color="#e67e22",
             label="Shuffled struct ρ (mean)", density=True, histtype="step", linewidth=2)
    ax1.axvline(np.mean(observed_text), color="#2c3e50", linestyle="--", linewidth=1.5,
                label=f"Observed mean = {np.mean(observed_text):.3f}")
    ax1.axvline(np.mean(shuffled_text), color="#c0392b", linestyle=":", linewidth=1.5,
                label=f"Shuffled text mean = {np.mean(shuffled_text):.3f}")
    ax1.set_xlabel("Spearman ρ", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Observed vs Shuffled Correlations", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Right panel: z-score distributions ──
    z_text = [r["z_score"] for r in text_results]
    z_struct = [r["z_score"] for r in struct_results]

    z_bins = np.linspace(0, max(max(z_text), max(z_struct)) * 1.1, 50)
    ax2.hist(z_text, bins=z_bins, alpha=0.6, color="#3498db", label=f"Shuffle text (mean z={np.mean(z_text):.1f})")
    ax2.hist(z_struct, bins=z_bins, alpha=0.6, color="#e67e22", label=f"Shuffle struct (mean z={np.mean(z_struct):.1f})")
    ax2.axvline(5, color="#e74c3c", linestyle="--", linewidth=1.5, label="z = 5 threshold")
    ax2.set_xlabel("z-score", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("z-score Distributions (Shuffle Controls)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = RESULTS_DIR / "summary" / "shuffle_controls_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print("=" * 70)
    print("  Shuffle Controls: Negative Control Experiments")
    print("=" * 70)
    t_start = time.time()

    # 1. Load data
    samples, total_slices = load_all_samples()

    # 2. Encode all slices (one-time, reused for both controls)
    embeddings, sample_offsets = encode_all_slices(samples, total_slices)

    # 3. Run shuffle text
    text_results = run_shuffle_text(samples, embeddings, sample_offsets)

    # 4. Run shuffle structure
    struct_results = run_shuffle_structure(samples, embeddings, sample_offsets)

    # 5. Summary
    summary = generate_shuffle_summary(text_results, struct_results)

    # 6. Plot
    generate_shuffle_plot(text_results, struct_results)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Shuffle controls complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
