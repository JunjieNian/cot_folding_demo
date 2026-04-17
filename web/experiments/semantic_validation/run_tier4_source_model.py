#!/usr/bin/env python3
"""
Tier 4: Source-Model Embedding Similarity (Supplementary)

Uses the DeepSeek-R1 neuron activation w_sum vectors as sparse embeddings
and computes cosine similarity between slices. This is a same-model check:
structural similarity (Jaccard on neuron keys) vs embedding similarity
(cosine on weighted neuron activations).

Note: This is NOT independent validation (same model produces both signals).
It supplements the independent Tier 1-3 results.

Usage:
    python run_tier4_source_model.py
"""

import os
import sys
import json
import time
import base64
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, sparse
from sklearn.preprocessing import normalize

# ── Paths ──
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "public" / "data" / "aime24"
INDEX_FILE = DATA_DIR / "problems.index.json"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Set via environment variable, e.g.:
#   export INTRA_COT_CACHE_BASE=/path/to/cache/DeepSeek-R1-0528-Qwen3-8B
# The full path should point to the neuron output cache directory containing a
# "rows/" subdirectory.
CACHE_DIR = Path(
    os.environ.get(
        "TIER4_CACHE_DIR",
        os.environ.get("INTRA_COT_CACHE_BASE", ""),
    )
)
if str(CACHE_DIR) == "":
    CACHE_DIR = PROJ_ROOT.parent.parent / "public" / "data" / "cache"
    print(
        "WARNING: TIER4_CACHE_DIR / INTRA_COT_CACHE_BASE not set. "
        f"Falling back to {CACHE_DIR}"
    )
ROWS_DIR = CACHE_DIR / "rows"


def decode_sim_b64(b64_str: str, n: int) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    mat = np.zeros((n, n), dtype=np.float32)
    k = 0
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            v = raw[k] / 255.0
            mat[i, j] = v
            mat[j, i] = v
            k += 1
    return mat


def load_cache_arrays():
    """Load the global cache arrays (mmap for efficiency)."""
    print("Loading cache arrays (mmap)...")
    t0 = time.time()
    arrays = {
        "sample_row_ptr": np.fromfile(str(ROWS_DIR / "sample_row_ptr.int64"), dtype="<i8"),
        "row_ptr": np.fromfile(str(ROWS_DIR / "row_ptr.int64"), dtype="<i8"),
        "slice_ids": np.fromfile(str(ROWS_DIR / "slice_ids.int32"), dtype="<i4"),
        "keys": np.fromfile(str(ROWS_DIR / "keys.uint32"), dtype="<u4"),
        "w_sum": np.fromfile(str(ROWS_DIR / "w_sum.float16"), dtype="<f2"),
    }
    elapsed = time.time() - t0
    n_samples = len(arrays["sample_row_ptr"]) - 1
    n_slices = len(arrays["slice_ids"])
    print(f"  Loaded: {n_samples} samples, {n_slices} slices in {elapsed:.1f}s")
    return arrays


def build_sparse_cosine_sim(arrays, sample_idx, n_slices):
    """
    Build sparse (n_slices, vocab) matrix from cache, then compute cosine similarity.
    Returns dense n_slices × n_slices cosine similarity matrix.
    """
    sample_row_ptr = arrays["sample_row_ptr"]
    row_ptr = arrays["row_ptr"]
    keys = arrays["keys"]
    w_sum = arrays["w_sum"]

    row_start = int(sample_row_ptr[sample_idx])
    row_end = int(sample_row_ptr[sample_idx + 1])
    actual_n = row_end - row_start
    assert actual_n == n_slices, f"Expected {n_slices} slices, got {actual_n}"

    # Build COO sparse matrix
    row_indices = []
    col_indices = []
    values = []

    for local_i in range(n_slices):
        global_row = row_start + local_i
        k_start = int(row_ptr[global_row])
        k_end = int(row_ptr[global_row + 1])

        K = k_end - k_start
        row_indices.extend([local_i] * K)
        col_indices.extend(keys[k_start:k_end].tolist())
        values.extend(w_sum[k_start:k_end].astype(np.float32).tolist())

    # Vocab size: use max key + 1 (local to this sample for efficiency)
    if len(col_indices) == 0:
        return np.eye(n_slices, dtype=np.float32)

    col_arr = np.array(col_indices, dtype=np.int64)
    # Remap columns to contiguous indices to save memory
    unique_cols, inverse = np.unique(col_arr, return_inverse=True)
    n_features = len(unique_cols)

    mat = sparse.coo_matrix(
        (np.array(values, dtype=np.float32),
         (np.array(row_indices, dtype=np.int64), inverse)),
        shape=(n_slices, n_features),
    ).tocsr()

    # L2-normalize rows → cosine sim = dot product
    mat_normed = normalize(mat, norm="l2", axis=1)
    sim = (mat_normed @ mat_normed.T).toarray()

    return sim


def run_tier4(arrays):
    """Compute source-model embedding cosine similarity and correlate with structural sim."""
    print("\n── Tier 4: Source-Model Sparse Cosine ──")
    t0 = time.time()

    with open(INDEX_FILE) as f:
        index = json.load(f)

    results = []
    sample_idx = 0
    total = sum(len(p["samples"]) for p in index["problems"])

    for prob in index["problems"]:
        pid = prob["problem_id"]
        prob_dir = SAMPLES_DIR / f"p{pid}"

        for sinfo in prob["samples"]:
            sid = sinfo["sample_id"]
            n = sinfo["n_slices"]

            if n < 5:
                sample_idx += 1
                continue

            # Load structural similarity
            sim_path = prob_dir / f"s{sid}.sim.b64"
            with open(sim_path) as f:
                b64_str = f.read().strip()
            struct_sim = decode_sim_b64(b64_str, n)

            # Build source-model cosine similarity
            source_sim = build_sparse_cosine_sim(arrays, sample_idx, n)

            # Correlate upper triangles
            iu = np.triu_indices(n, k=1)
            struct_vals = struct_sim[iu]
            source_vals = source_sim[iu]

            rho, pval = stats.spearmanr(struct_vals, source_vals)

            results.append({
                "problem_id": pid,
                "sample_id": sid,
                "n_slices": n,
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "is_correct": sinfo.get("is_correct", False),
            })

            sample_idx += 1
            if sample_idx % 200 == 0:
                elapsed = time.time() - t0
                rate = sample_idx / elapsed
                eta = (total - sample_idx) / rate if rate > 0 else 0
                print(f"  [{sample_idx}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    rhos = [r["spearman_rho"] for r in results if not np.isnan(r["spearman_rho"])]
    print(f"  {len(results)} samples processed in {elapsed:.1f}s")
    print(f"  Spearman ρ: mean={np.mean(rhos):.4f}, median={np.median(rhos):.4f}, "
          f"std={np.std(rhos):.4f}")
    print(f"  Fraction p<0.05: {np.mean([r['p_value'] < 0.05 for r in results]):.2%}")

    # Save results
    out_dir = RESULTS_DIR / "tier4_source_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "per_sample_correlations.csv", index=False)
    print(f"  Saved to {out_dir / 'per_sample_correlations.csv'}")

    return results


def run_topk_source(arrays, k=5):
    """
    Top-k retrieval using source-model cosine similarity as the text-level ground truth
    (instead of MiniLM embeddings).
    """
    print(f"\n── Tier 4 Control: Top-k Retrieval with Source Embeddings (k={k}) ──")
    t0 = time.time()

    with open(INDEX_FILE) as f:
        index = json.load(f)

    structural_scores = []
    adjacent_scores = []
    random_scores = []
    rng = np.random.RandomState(42)

    sample_idx = 0
    total = sum(len(p["samples"]) for p in index["problems"])

    for prob in index["problems"]:
        pid = prob["problem_id"]
        prob_dir = SAMPLES_DIR / f"p{pid}"

        for sinfo in prob["samples"]:
            sid = sinfo["sample_id"]
            n = sinfo["n_slices"]

            if n < k + 2:
                sample_idx += 1
                continue

            # Load structural similarity
            sim_path = prob_dir / f"s{sid}.sim.b64"
            with open(sim_path) as f:
                b64_str = f.read().strip()
            struct_sim = decode_sim_b64(b64_str, n)

            # Build source-model cosine similarity
            source_sim = build_sparse_cosine_sim(arrays, sample_idx, n)

            for i in range(n):
                ss = struct_sim[i].copy()
                ss[i] = -1
                topk_struct = np.argsort(ss)[-k:]
                structural_scores.append(np.mean(source_sim[i, topk_struct]))

                dists = np.abs(np.arange(n) - i).astype(float)
                dists[i] = n + 1
                topk_adj = np.argsort(dists)[:k]
                adjacent_scores.append(np.mean(source_sim[i, topk_adj]))

                candidates = np.delete(np.arange(n), i)
                topk_rand = rng.choice(candidates, size=k, replace=False)
                random_scores.append(np.mean(source_sim[i, topk_rand]))

            sample_idx += 1
            if sample_idx % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{sample_idx}/{total}] {elapsed:.0f}s")

    structural_scores = np.array(structural_scores)
    adjacent_scores = np.array(adjacent_scores)
    random_scores = np.array(random_scores)

    elapsed = time.time() - t0
    print(f"  {len(structural_scores)} query slices in {elapsed:.1f}s")
    print(f"  Structural-nearest: mean={structural_scores.mean():.4f}")
    print(f"  Adjacent:           mean={adjacent_scores.mean():.4f}")
    print(f"  Random:             mean={random_scores.mean():.4f}")

    stat_vs_adj, p_vs_adj = stats.wilcoxon(structural_scores, adjacent_scores, alternative="greater")
    stat_vs_rand, p_vs_rand = stats.wilcoxon(structural_scores, random_scores, alternative="greater")

    print(f"  Wilcoxon structural > adjacent: p={p_vs_adj:.2e}")
    print(f"  Wilcoxon structural > random:   p={p_vs_rand:.2e}")

    topk_results = {
        "structural_mean": float(structural_scores.mean()),
        "adjacent_mean": float(adjacent_scores.mean()),
        "random_mean": float(random_scores.mean()),
        "structural_std": float(structural_scores.std()),
        "adjacent_std": float(adjacent_scores.std()),
        "random_std": float(random_scores.std()),
        "wilcoxon_struct_vs_adj_p": float(p_vs_adj),
        "wilcoxon_struct_vs_rand_p": float(p_vs_rand),
        "n_queries": len(structural_scores),
    }

    out_dir = RESULTS_DIR / "tier4_source_model"
    with open(out_dir / "topk_retrieval.json", "w") as f:
        json.dump(topk_results, f, indent=2)

    return topk_results


def update_summary(tier4_results, topk_results):
    """Update the summary_statistics.json with Tier 4 results."""
    summary_path = RESULTS_DIR / "summary" / "summary_statistics.json"
    with open(summary_path) as f:
        summary = json.load(f)

    rhos = [r["spearman_rho"] for r in tier4_results if not np.isnan(r["spearman_rho"])]
    summary["tier4_source_model"] = {
        "mean_rho": float(np.mean(rhos)),
        "median_rho": float(np.median(rhos)),
        "std_rho": float(np.std(rhos)),
        "frac_p05": float(np.mean([r["p_value"] < 0.05 for r in tier4_results])),
        "method": "sparse cosine on DeepSeek-R1 neuron w_sum",
        "note": "same-model supplementary (not independent validation)",
    }
    summary["tier4_topk_retrieval"] = topk_results

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Updated {summary_path}")


def main():
    print("=" * 60)
    print("  Tier 4: Source-Model Sparse Embedding Similarity")
    print("  (Same-model supplementary validation)")
    print("=" * 60)

    arrays = load_cache_arrays()
    tier4_results = run_tier4(arrays)
    topk_results = run_topk_source(arrays)
    update_summary(tier4_results, topk_results)

    print(f"\n{'=' * 60}")
    rhos = [r["spearman_rho"] for r in tier4_results if not np.isnan(r["spearman_rho"])]
    print(f"  Tier 4 (Source-model cosine) mean ρ: {np.mean(rhos):.4f}")
    print(f"  Top-5 structural nearest sim:        {topk_results['structural_mean']:.4f}")
    print(f"  Top-5 sequential adjacent sim:       {topk_results['adjacent_mean']:.4f}")
    print(f"  Top-5 random baseline sim:           {topk_results['random_mean']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
