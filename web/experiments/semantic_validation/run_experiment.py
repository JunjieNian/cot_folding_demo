#!/usr/bin/env python3
"""
Semantic Validation Experiment: Folding Structure Similarity vs Text Semantic Similarity

Validates that structural similarity (from hidden-layer activation cosine distance)
captures genuine semantic relationships by comparing against text-level similarity.

Dataset: deepseek / aime24 (30 problems × 64 samples = 1920 samples, ~880k slices)
"""

import os
import sys
import json
import time
import base64
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "public" / "data" / "aime24"
INDEX_FILE = DATA_DIR / "problems.index.json"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Set CUDA_VISIBLE_DEVICES before importing torch if you need a specific GPU
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def decode_sim_b64(b64_str: str, n: int) -> np.ndarray:
    """Decode base64 upper-triangle packed uint8 → full n×n float symmetric matrix."""
    raw = base64.b64decode(b64_str)
    expected = n * (n - 1) // 2
    assert len(raw) == expected, f"Expected {expected} bytes, got {len(raw)}"
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


def load_all_samples():
    """Load all samples from the aime24 dataset."""
    print("Loading all samples...")
    t0 = time.time()

    with open(INDEX_FILE) as f:
        index = json.load(f)

    samples = []
    total_slices = 0

    for prob in index["problems"]:
        pid = prob["problem_id"]
        prob_dir = SAMPLES_DIR / f"p{pid}"

        for sinfo in prob["samples"]:
            sid = sinfo["sample_id"]
            n_slices = sinfo["n_slices"]

            # Load text
            text_path = prob_dir / f"s{sid}.text.json"
            with open(text_path) as f:
                text_data = json.load(f)

            # Extract per-slice text
            full_text = text_data["full_text"]
            slice_texts = []
            for item in text_data["items"]:
                txt = full_text[item["char_start"]:item["char_end"]]
                slice_texts.append(txt)

            assert len(slice_texts) == n_slices, \
                f"p{pid}/s{sid}: expected {n_slices} slices, got {len(slice_texts)} items"

            # Load structural similarity
            sim_path = prob_dir / f"s{sid}.sim.b64"
            with open(sim_path) as f:
                b64_str = f.read().strip()
            struct_sim = decode_sim_b64(b64_str, n_slices)

            # Load bundle for HMM states and is_correct
            bundle_path = prob_dir / f"s{sid}.bundle.json"
            with open(bundle_path) as f:
                bundle = json.load(f)
            folding = bundle["folding"]
            hmm_states = np.array(folding["hmm_states"])
            is_correct = folding.get("is_correct", False)

            samples.append({
                "problem_id": pid,
                "sample_id": sid,
                "n_slices": n_slices,
                "slice_texts": slice_texts,
                "struct_sim": struct_sim,
                "hmm_states": hmm_states,
                "is_correct": is_correct,
            })
            total_slices += n_slices

    elapsed = time.time() - t0
    print(f"  Loaded {len(samples)} samples, {total_slices} total slices in {elapsed:.1f}s")
    return samples, total_slices


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 1: TF-IDF Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier1_tfidf(samples):
    """Compute TF-IDF cosine similarity per sample and correlate with structural similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    print("\n── Tier 1: TF-IDF Baseline ──")
    t0 = time.time()

    results = []
    for s in samples:
        n = s["n_slices"]
        if n < 5:
            continue

        vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        try:
            tfidf_mat = vec.fit_transform(s["slice_texts"])
        except ValueError:
            # All empty documents
            continue
        text_sim = sk_cosine(tfidf_mat).astype(np.float32)

        # Extract upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        text_vals = text_sim[iu]

        rho, pval = stats.spearmanr(struct_vals, text_vals)
        results.append({
            "problem_id": s["problem_id"],
            "sample_id": s["sample_id"],
            "n_slices": n,
            "spearman_rho": rho,
            "p_value": pval,
            "is_correct": s["is_correct"],
        })

    # Store text_sim back in samples for later use
    # (Re-run is cheap so we just store the results)
    elapsed = time.time() - t0
    rhos = [r["spearman_rho"] for r in results if not np.isnan(r["spearman_rho"])]
    print(f"  {len(results)} samples processed in {elapsed:.1f}s")
    print(f"  Spearman ρ: mean={np.mean(rhos):.4f}, median={np.median(rhos):.4f}, "
          f"std={np.std(rhos):.4f}")
    print(f"  Fraction p<0.05: {np.mean([r['p_value'] < 0.05 for r in results]):.2%}")

    # Save per-sample results
    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "tier1_tfidf" / "per_sample_correlations.csv", index=False
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 2: Sentence Embedding (all-MiniLM-L6-v2)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_all_slices(samples, total_slices):
    """Encode all slices using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print("\n── Encoding all slices with all-MiniLM-L6-v2 ──")
    t0 = time.time()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Gather all texts with their sample/slice indices
    all_texts = []
    sample_offsets = []  # (start_idx, n_slices) per sample
    offset = 0
    for s in samples:
        sample_offsets.append((offset, s["n_slices"]))
        all_texts.extend(s["slice_texts"])
        offset += s["n_slices"]

    assert len(all_texts) == total_slices

    # Batch encode on GPU
    embeddings = model.encode(
        all_texts,
        batch_size=512,
        show_progress_bar=True,
        device="cuda",
        normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    elapsed = time.time() - t0
    print(f"  Encoded {len(all_texts)} slices in {elapsed:.1f}s, shape={embeddings.shape}")
    return embeddings, sample_offsets


def run_tier2_embedding(samples, embeddings, sample_offsets):
    """Compute embedding cosine similarity per sample and correlate."""
    print("\n── Tier 2: Embedding Cosine ──")
    t0 = time.time()

    results = []
    for idx, s in enumerate(samples):
        n = s["n_slices"]
        if n < 5:
            continue

        start, count = sample_offsets[idx]
        emb = embeddings[start:start + count]
        # Since embeddings are L2-normalized, cosine sim = dot product
        text_sim = emb @ emb.T

        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        text_vals = text_sim[iu]

        rho, pval = stats.spearmanr(struct_vals, text_vals)
        results.append({
            "problem_id": s["problem_id"],
            "sample_id": s["sample_id"],
            "n_slices": n,
            "spearman_rho": rho,
            "p_value": pval,
            "is_correct": s["is_correct"],
        })

        # Store text_sim for controls
        s["emb_text_sim"] = text_sim

    elapsed = time.time() - t0
    rhos = [r["spearman_rho"] for r in results if not np.isnan(r["spearman_rho"])]
    print(f"  {len(results)} samples processed in {elapsed:.1f}s")
    print(f"  Spearman ρ: mean={np.mean(rhos):.4f}, median={np.median(rhos):.4f}, "
          f"std={np.std(rhos):.4f}")
    print(f"  Fraction p<0.05: {np.mean([r['p_value'] < 0.05 for r in results]):.2%}")

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "tier2_embedding" / "per_sample_correlations.csv", index=False
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 3: Cross-Encoder Reranking (sampled pairs)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_pairs_for_tier3(samples, n_bins=10, pairs_per_bin=500):
    """Sample pairs stratified by structural similarity bins, with position-matching."""
    print("\n── Sampling pairs for Tier 3 ──")
    rng = np.random.RandomState(42)

    # Collect all pairs with their metadata
    all_struct = []
    all_gaps = []
    all_texts_i = []
    all_texts_j = []
    all_meta = []  # (sample_idx, i, j)

    for s_idx, s in enumerate(samples):
        n = s["n_slices"]
        if n < 5:
            continue
        iu_i, iu_j = np.triu_indices(n, k=1)
        for idx in range(len(iu_i)):
            i, j = int(iu_i[idx]), int(iu_j[idx])
            all_struct.append(s["struct_sim"][i, j])
            all_gaps.append(abs(i - j) / n)
            all_texts_i.append(s["slice_texts"][i])
            all_texts_j.append(s["slice_texts"][j])
            all_meta.append((s_idx, i, j))

    all_struct = np.array(all_struct)
    all_gaps = np.array(all_gaps)

    # Bin by structural similarity
    bin_edges = np.linspace(0, 0.7, n_bins + 1)
    bin_indices = np.digitize(all_struct, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    sampled_indices = []
    for b in range(n_bins):
        mask = bin_indices == b
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue

        # Within each bin, sample weighted by gap distribution to avoid position bias
        # Use inverse frequency weighting: less common gaps get higher weight
        gaps_in_bin = all_gaps[candidates]
        gap_bins = np.digitize(gaps_in_bin, np.linspace(0, 1, 11)) - 1
        gap_counts = np.bincount(gap_bins, minlength=10).astype(float)
        gap_counts[gap_counts == 0] = 1
        weights = 1.0 / gap_counts[gap_bins]
        weights /= weights.sum()

        n_sample = min(pairs_per_bin, len(candidates))
        chosen = rng.choice(len(candidates), size=n_sample, replace=False, p=weights)
        sampled_indices.extend(candidates[chosen])

    sampled_indices = np.array(sampled_indices)
    print(f"  Sampled {len(sampled_indices)} pairs across {n_bins} bins")

    pairs_data = {
        "texts_i": [all_texts_i[i] for i in sampled_indices],
        "texts_j": [all_texts_j[i] for i in sampled_indices],
        "struct_sim": all_struct[sampled_indices],
        "gaps": all_gaps[sampled_indices],
        "meta": [all_meta[i] for i in sampled_indices],
    }
    return pairs_data


def run_tier3_crossencoder(pairs_data):
    """Score sampled pairs with a cross-encoder and correlate with structural similarity."""
    from sentence_transformers import CrossEncoder

    print("\n── Tier 3: Cross-Encoder ──")
    t0 = time.time()

    model = CrossEncoder("cross-encoder/stsb-distilroberta-base", device="cuda")

    # Prepare sentence pairs
    sentence_pairs = list(zip(pairs_data["texts_i"], pairs_data["texts_j"]))

    # Score in batches
    scores = model.predict(sentence_pairs, batch_size=256, show_progress_bar=True)
    scores = np.array(scores, dtype=np.float32)

    # Normalize scores to [0, 1] range (cross-encoder outputs can vary)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    struct_vals = pairs_data["struct_sim"]
    rho, pval = stats.spearmanr(struct_vals, scores_norm)

    elapsed = time.time() - t0
    print(f"  Scored {len(sentence_pairs)} pairs in {elapsed:.1f}s")
    print(f"  Spearman ρ = {rho:.4f}, p = {pval:.2e}")

    # Save results
    df = pd.DataFrame({
        "struct_sim": struct_vals,
        "cross_encoder_score": scores,
        "cross_encoder_score_norm": scores_norm,
        "gap": pairs_data["gaps"],
    })
    df.to_csv(RESULTS_DIR / "tier3_crossencoder" / "pair_scores.csv", index=False)

    return {
        "spearman_rho": rho,
        "p_value": pval,
        "n_pairs": len(sentence_pairs),
        "scores": scores_norm,
        "struct_vals": struct_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Controls
# ═══════════════════════════════════════════════════════════════════════════════

def run_partial_spearman(samples):
    """
    Partial Spearman: regress out position proximity from both structural and text similarity,
    then correlate the residuals.
    """
    print("\n── Control: Partial Spearman (position debiased) ──")
    t0 = time.time()

    results = []
    for s in samples:
        n = s["n_slices"]
        if n < 5 or "emb_text_sim" not in s:
            continue

        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        text_vals = s["emb_text_sim"][iu]

        # Position proximity: 1 - |i-j|/n
        ii, jj = iu
        pos_prox = 1.0 - np.abs(ii - jj).astype(np.float32) / n

        # Rank all three
        r_struct = stats.rankdata(struct_vals)
        r_text = stats.rankdata(text_vals)
        r_pos = stats.rankdata(pos_prox)

        # Residualize: regress out position from each
        # Using simple OLS: residual = y - slope * x - intercept
        def residualize(y, x):
            slope, intercept, _, _, _ = stats.linregress(x, y)
            return y - slope * x - intercept

        resid_struct = residualize(r_struct, r_pos)
        resid_text = residualize(r_text, r_pos)

        # Pearson on residuals = partial Spearman
        rho, pval = stats.pearsonr(resid_struct, resid_text)

        results.append({
            "problem_id": s["problem_id"],
            "sample_id": s["sample_id"],
            "n_slices": n,
            "partial_spearman_rho": rho,
            "p_value": pval,
            "is_correct": s["is_correct"],
        })

    elapsed = time.time() - t0
    rhos = [r["partial_spearman_rho"] for r in results if not np.isnan(r["partial_spearman_rho"])]
    print(f"  {len(results)} samples in {elapsed:.1f}s")
    print(f"  Partial Spearman ρ: mean={np.mean(rhos):.4f}, median={np.median(rhos):.4f}, "
          f"std={np.std(rhos):.4f}")
    print(f"  Fraction p<0.05: {np.mean([r['p_value'] < 0.05 for r in results]):.2%}")

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "controls" / "partial_spearman.csv", index=False
    )
    return results


def run_topk_retrieval(samples, k=5):
    """
    Top-k retrieval comparison: for each slice, compare text similarity of
    structural-nearest vs sequential-adjacent vs random neighbors.
    """
    print(f"\n── Control: Top-k Retrieval (k={k}) ──")
    t0 = time.time()
    rng = np.random.RandomState(42)

    structural_scores = []
    adjacent_scores = []
    random_scores = []

    for s in samples:
        n = s["n_slices"]
        if n < k + 2 or "emb_text_sim" not in s:
            continue

        struct_sim = s["struct_sim"].copy()
        text_sim = s["emb_text_sim"]

        for i in range(n):
            # Zero out self
            ss = struct_sim[i].copy()
            ss[i] = -1

            # Structural top-k
            topk_struct = np.argsort(ss)[-k:]
            structural_scores.append(np.mean(text_sim[i, topk_struct]))

            # Adjacent top-k (nearest by position)
            dists = np.abs(np.arange(n) - i).astype(float)
            dists[i] = n + 1  # exclude self
            topk_adj = np.argsort(dists)[:k]
            adjacent_scores.append(np.mean(text_sim[i, topk_adj]))

            # Random top-k
            candidates = np.delete(np.arange(n), i)
            topk_rand = rng.choice(candidates, size=k, replace=False)
            random_scores.append(np.mean(text_sim[i, topk_rand]))

    structural_scores = np.array(structural_scores)
    adjacent_scores = np.array(adjacent_scores)
    random_scores = np.array(random_scores)

    elapsed = time.time() - t0
    print(f"  {len(structural_scores)} query slices in {elapsed:.1f}s")
    print(f"  Structural-nearest: mean={structural_scores.mean():.4f}")
    print(f"  Adjacent:           mean={adjacent_scores.mean():.4f}")
    print(f"  Random:             mean={random_scores.mean():.4f}")

    # Wilcoxon signed-rank tests
    stat_vs_adj, p_vs_adj = stats.wilcoxon(structural_scores, adjacent_scores, alternative="greater")
    stat_vs_rand, p_vs_rand = stats.wilcoxon(structural_scores, random_scores, alternative="greater")
    stat_adj_vs_rand, p_adj_vs_rand = stats.wilcoxon(adjacent_scores, random_scores, alternative="greater")

    print(f"  Wilcoxon structural > adjacent: p={p_vs_adj:.2e}")
    print(f"  Wilcoxon structural > random:   p={p_vs_rand:.2e}")
    print(f"  Wilcoxon adjacent > random:     p={p_adj_vs_rand:.2e}")

    results = {
        "structural_mean": float(structural_scores.mean()),
        "adjacent_mean": float(adjacent_scores.mean()),
        "random_mean": float(random_scores.mean()),
        "structural_std": float(structural_scores.std()),
        "adjacent_std": float(adjacent_scores.std()),
        "random_std": float(random_scores.std()),
        "wilcoxon_struct_vs_adj_p": float(p_vs_adj),
        "wilcoxon_struct_vs_rand_p": float(p_vs_rand),
        "wilcoxon_adj_vs_rand_p": float(p_adj_vs_rand),
        "n_queries": len(structural_scores),
    }

    with open(RESULTS_DIR / "controls" / "topk_retrieval.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, structural_scores, adjacent_scores, random_scores


def run_hmm_state_analysis(samples):
    """
    Analyze Spearman correlations stratified by HMM state pairs:
    explore-explore, exploit-exploit, cross-state.
    """
    print("\n── Control: HMM State Stratification ──")
    t0 = time.time()

    # hmm_states: 0=explore, 1=exploit
    strata = {"explore-explore": [], "exploit-exploit": [], "cross-state": []}

    for s in samples:
        n = s["n_slices"]
        if n < 5 or "emb_text_sim" not in s:
            continue

        hmm = s["hmm_states"]
        iu_i, iu_j = np.triu_indices(n, k=1)

        # Classify pairs
        for stratum_name, mask_fn in [
            ("explore-explore", lambda i, j: hmm[i] == 0 and hmm[j] == 0),
            ("exploit-exploit", lambda i, j: hmm[i] == 1 and hmm[j] == 1),
            ("cross-state",    lambda i, j: hmm[i] != hmm[j]),
        ]:
            mask = np.array([mask_fn(int(iu_i[k]), int(iu_j[k])) for k in range(len(iu_i))])
            if mask.sum() < 10:
                continue

            struct_vals = s["struct_sim"][iu_i[mask], iu_j[mask]]
            text_vals = s["emb_text_sim"][iu_i[mask], iu_j[mask]]
            rho, pval = stats.spearmanr(struct_vals, text_vals)

            if not np.isnan(rho):
                strata[stratum_name].append({
                    "problem_id": s["problem_id"],
                    "sample_id": s["sample_id"],
                    "spearman_rho": rho,
                    "p_value": pval,
                    "n_pairs": int(mask.sum()),
                })

    elapsed = time.time() - t0
    results = {}
    for name, vals in strata.items():
        rhos = [v["spearman_rho"] for v in vals]
        if rhos:
            results[name] = {
                "mean_rho": float(np.mean(rhos)),
                "median_rho": float(np.median(rhos)),
                "std_rho": float(np.std(rhos)),
                "n_samples": len(rhos),
                "frac_p05": float(np.mean([v["p_value"] < 0.05 for v in vals])),
            }
            print(f"  {name}: mean ρ={results[name]['mean_rho']:.4f}, "
                  f"n={results[name]['n_samples']}, frac p<0.05={results[name]['frac_p05']:.2%}")

    with open(RESULTS_DIR / "controls" / "hmm_state_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Done in {elapsed:.1f}s")
    return results


def run_correctness_comparison(tier2_results):
    """Compare Spearman ρ distributions between correct and incorrect samples."""
    print("\n── Control: Correct vs Incorrect Samples ──")

    correct = [r["spearman_rho"] for r in tier2_results if r["is_correct"] and not np.isnan(r["spearman_rho"])]
    incorrect = [r["spearman_rho"] for r in tier2_results if not r["is_correct"] and not np.isnan(r["spearman_rho"])]

    print(f"  Correct samples ({len(correct)}): mean ρ={np.mean(correct):.4f}, std={np.std(correct):.4f}")
    print(f"  Incorrect samples ({len(incorrect)}): mean ρ={np.mean(incorrect):.4f}, std={np.std(incorrect):.4f}")

    stat, pval = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
    print(f"  Mann-Whitney U test: p={pval:.4e}")

    results = {
        "correct_mean": float(np.mean(correct)),
        "correct_std": float(np.std(correct)),
        "correct_n": len(correct),
        "incorrect_mean": float(np.mean(incorrect)),
        "incorrect_std": float(np.std(incorrect)),
        "incorrect_n": len(incorrect),
        "mannwhitney_p": float(pval),
    }

    with open(RESULTS_DIR / "controls" / "correctness_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_combined_binned_plot(samples, tier3_results):
    """
    Main figure: X = structural similarity bin, Y = mean text similarity.
    Three lines: TF-IDF, embedding, cross-encoder.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    print("\n── Generating combined binned plot ──")

    n_bins = 20
    bin_edges = np.linspace(0, 0.7, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Accumulate per-bin text similarities for TF-IDF and embedding
    tfidf_sums = np.zeros(n_bins)
    tfidf_counts = np.zeros(n_bins)
    emb_sums = np.zeros(n_bins)
    emb_counts = np.zeros(n_bins)

    for s in samples:
        n = s["n_slices"]
        if n < 5 or "emb_text_sim" not in s:
            continue

        iu = np.triu_indices(n, k=1)
        struct_vals = s["struct_sim"][iu]
        emb_vals = s["emb_text_sim"][iu]

        # TF-IDF
        vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        try:
            tfidf_mat = vec.fit_transform(s["slice_texts"])
            tfidf_sim = sk_cosine(tfidf_mat).astype(np.float32)
            tfidf_vals = tfidf_sim[iu]
        except ValueError:
            tfidf_vals = None

        bins = np.digitize(struct_vals, bin_edges) - 1
        bins = np.clip(bins, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bins == b
            if mask.sum() == 0:
                continue
            emb_sums[b] += emb_vals[mask].sum()
            emb_counts[b] += mask.sum()
            if tfidf_vals is not None:
                tfidf_sums[b] += tfidf_vals[mask].sum()
                tfidf_counts[b] += mask.sum()

    tfidf_means = np.divide(tfidf_sums, tfidf_counts, where=tfidf_counts > 0)
    emb_means = np.divide(emb_sums, emb_counts, where=emb_counts > 0)

    # Cross-encoder: bin the sampled pairs
    ce_struct = tier3_results["struct_vals"]
    ce_scores = tier3_results["scores"]
    ce_bins = np.digitize(ce_struct, bin_edges) - 1
    ce_bins = np.clip(ce_bins, 0, n_bins - 1)
    ce_means = np.zeros(n_bins)
    ce_counts_arr = np.zeros(n_bins)
    for b in range(n_bins):
        mask = ce_bins == b
        if mask.sum() > 0:
            ce_means[b] = ce_scores[mask].mean()
            ce_counts_arr[b] = mask.sum()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    valid_tfidf = tfidf_counts > 0
    valid_emb = emb_counts > 0
    valid_ce = ce_counts_arr > 0

    ax.plot(bin_centers[valid_tfidf], tfidf_means[valid_tfidf], "o-",
            color="#e74c3c", label="TF-IDF cosine", linewidth=2, markersize=5)
    ax.plot(bin_centers[valid_emb], emb_means[valid_emb], "s-",
            color="#3498db", label="Embedding cosine (MiniLM)", linewidth=2, markersize=5)
    ax.plot(bin_centers[valid_ce], ce_means[valid_ce], "^-",
            color="#2ecc71", label="Cross-encoder (DistilRoBERTa)", linewidth=2, markersize=5)

    ax.set_xlabel("Structural Similarity (binned)", fontsize=13)
    ax.set_ylabel("Mean Text Similarity", fontsize=13)
    ax.set_title("Structural Similarity vs Text Semantic Similarity", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary" / "combined_binned_plot.png", dpi=150)
    plt.close(fig)
    print("  Saved combined_binned_plot.png")


def generate_correlation_histogram(tier2_results, partial_results):
    """Histogram of Spearman ρ distributions: raw and partial."""
    print("── Generating correlation histogram ──")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw Spearman
    rhos_raw = [r["spearman_rho"] for r in tier2_results if not np.isnan(r["spearman_rho"])]
    axes[0].hist(rhos_raw, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].axvline(np.mean(rhos_raw), color="red", linestyle="--", linewidth=2,
                     label=f"Mean = {np.mean(rhos_raw):.3f}")
    axes[0].set_xlabel("Spearman ρ", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Raw Spearman ρ (Embedding)", fontsize=13)
    axes[0].legend(fontsize=11)

    # Partial Spearman
    rhos_partial = [r["partial_spearman_rho"] for r in partial_results
                    if not np.isnan(r["partial_spearman_rho"])]
    axes[1].hist(rhos_partial, bins=50, color="#e67e22", alpha=0.7, edgecolor="white")
    axes[1].axvline(np.mean(rhos_partial), color="red", linestyle="--", linewidth=2,
                     label=f"Mean = {np.mean(rhos_partial):.3f}")
    axes[1].set_xlabel("Partial Spearman ρ", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Partial Spearman ρ (Position Debiased)", fontsize=13)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary" / "correlation_histogram.png", dpi=150)
    plt.close(fig)
    print("  Saved correlation_histogram.png")


def generate_position_control_plot(topk_results):
    """Bar chart: structural-nearest vs adjacent vs random text similarity."""
    print("── Generating position control plot ──")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels = ["Structural\nNearest", "Sequential\nAdjacent", "Random"]
    means = [topk_results["structural_mean"], topk_results["adjacent_mean"],
             topk_results["random_mean"]]
    stds = [topk_results["structural_std"], topk_results["adjacent_std"],
            topk_results["random_std"]]
    # Use SEM for error bars
    n = topk_results["n_queries"]
    sems = [s / np.sqrt(n) for s in stds]

    colors = ["#3498db", "#e67e22", "#95a5a6"]
    bars = ax.bar(labels, means, yerr=sems, color=colors, capsize=5, edgecolor="white",
                  linewidth=1.5)

    # Annotate significance
    y_max = max(means) + max(sems) * 2
    p_struct_adj = topk_results["wilcoxon_struct_vs_adj_p"]
    sig_str = "***" if p_struct_adj < 0.001 else "**" if p_struct_adj < 0.01 else "*" if p_struct_adj < 0.05 else "ns"
    ax.annotate(sig_str, xy=(0.5, y_max), fontsize=14, ha="center", fontweight="bold")

    ax.set_ylabel("Mean Text Similarity (Embedding)", fontsize=12)
    ax.set_title("Top-5 Retrieval: Text Similarity by Neighbor Type", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary" / "position_control_plot.png", dpi=150)
    plt.close(fig)
    print("  Saved position_control_plot.png")


def generate_topk_plot(samples, k_values=None):
    """Plot retrieval overlap between structural and sequential nearest as a function of k."""
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10, 15, 20]

    print("── Generating top-k retrieval overlap plot ──")

    overlap_struct_adj = []  # Mean overlap fraction per k
    text_sim_struct = []     # Mean text sim of structural-nearest per k
    text_sim_adj = []        # Mean text sim of adjacent per k

    for k in k_values:
        overlaps_k = []
        ts_struct_k = []
        ts_adj_k = []

        for s in samples:
            n = s["n_slices"]
            if n < k + 2 or "emb_text_sim" not in s:
                continue

            struct_sim = s["struct_sim"]
            text_sim = s["emb_text_sim"]

            for i in range(n):
                ss = struct_sim[i].copy()
                ss[i] = -1
                topk_struct = set(np.argsort(ss)[-k:])

                dists = np.abs(np.arange(n) - i).astype(float)
                dists[i] = n + 1
                topk_adj = set(np.argsort(dists)[:k])

                overlaps_k.append(len(topk_struct & topk_adj) / k)

                topk_struct_arr = np.array(list(topk_struct))
                topk_adj_arr = np.array(list(topk_adj))
                ts_struct_k.append(np.mean(text_sim[i, topk_struct_arr]))
                ts_adj_k.append(np.mean(text_sim[i, topk_adj_arr]))

        overlap_struct_adj.append(np.mean(overlaps_k))
        text_sim_struct.append(np.mean(ts_struct_k))
        text_sim_adj.append(np.mean(ts_adj_k))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, overlap_struct_adj, "o-", color="#8e44ad", linewidth=2, markersize=6)
    ax1.set_xlabel("k", fontsize=12)
    ax1.set_ylabel("Overlap Fraction", fontsize=12)
    ax1.set_title("Structural vs Sequential Nearest: Overlap", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, text_sim_struct, "s-", color="#3498db", label="Structural nearest",
             linewidth=2, markersize=6)
    ax2.plot(k_values, text_sim_adj, "^-", color="#e67e22", label="Sequential adjacent",
             linewidth=2, markersize=6)
    ax2.set_xlabel("k", fontsize=12)
    ax2.set_ylabel("Mean Text Similarity", fontsize=12)
    ax2.set_title("Top-k Retrieval: Text Similarity", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary" / "topk_retrieval_plot.png", dpi=150)
    plt.close(fig)
    print("  Saved topk_retrieval_plot.png")


def generate_latex_table(tier1_results, tier2_results, partial_results,
                         tier3_results, topk_results, hmm_results, corr_results):
    """Generate a LaTeX table summarizing all results."""
    print("── Generating LaTeX table ──")

    rhos_t1 = [r["spearman_rho"] for r in tier1_results if not np.isnan(r["spearman_rho"])]
    rhos_t2 = [r["spearman_rho"] for r in tier2_results if not np.isnan(r["spearman_rho"])]
    rhos_partial = [r["partial_spearman_rho"] for r in partial_results
                    if not np.isnan(r["partial_spearman_rho"])]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Semantic Validation: Structure--Text Similarity Correlation}",
        r"\label{tab:semantic_validation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Mean & Median & Std & Frac.\ $p<.05$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Per-sample Spearman $\rho$ (structure vs.\ text)}} \\",
        f"TF-IDF cosine & {np.mean(rhos_t1):.3f} & {np.median(rhos_t1):.3f} & "
        f"{np.std(rhos_t1):.3f} & {np.mean([r['p_value'] < 0.05 for r in tier1_results]):.1%} \\\\",
        f"Embedding cosine & {np.mean(rhos_t2):.3f} & {np.median(rhos_t2):.3f} & "
        f"{np.std(rhos_t2):.3f} & {np.mean([r['p_value'] < 0.05 for r in tier2_results]):.1%} \\\\",
        f"Partial Spearman & {np.mean(rhos_partial):.3f} & {np.median(rhos_partial):.3f} & "
        f"{np.std(rhos_partial):.3f} & {np.mean([r['p_value'] < 0.05 for r in partial_results]):.1%} \\\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Cross-encoder (sampled pairs)}} \\",
        f"Spearman $\\rho$ & \\multicolumn{{4}}{{c}}{{{tier3_results['spearman_rho']:.3f} "
        f"($n$={tier3_results['n_pairs']}, $p$={tier3_results['p_value']:.1e})}} \\\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Top-5 retrieval: mean text similarity}} \\",
        f"Structural nearest & \\multicolumn{{4}}{{c}}{{{topk_results['structural_mean']:.4f}}} \\\\",
        f"Sequential adjacent & \\multicolumn{{4}}{{c}}{{{topk_results['adjacent_mean']:.4f}}} \\\\",
        f"Random baseline & \\multicolumn{{4}}{{c}}{{{topk_results['random_mean']:.4f}}} \\\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{HMM state stratification (mean $\rho$)}} \\",
    ]

    for state_name, display_name in [("explore-explore", "Explore--explore"),
                                      ("exploit-exploit", "Exploit--exploit"),
                                      ("cross-state", "Cross-state")]:
        if state_name in hmm_results:
            h = hmm_results[state_name]
            lines.append(
                f"{display_name} & {h['mean_rho']:.3f} & {h['median_rho']:.3f} & "
                f"{h['std_rho']:.3f} & {h['frac_p05']:.1%} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Correct vs.\ incorrect samples (embedding $\rho$)}} \\",
        f"Correct ($n$={corr_results['correct_n']}) & "
        f"\\multicolumn{{4}}{{c}}{{{corr_results['correct_mean']:.3f} $\\pm$ {corr_results['correct_std']:.3f}}} \\\\",
        f"Incorrect ($n$={corr_results['incorrect_n']}) & "
        f"\\multicolumn{{4}}{{c}}{{{corr_results['incorrect_mean']:.3f} $\\pm$ {corr_results['incorrect_std']:.3f}}} \\\\",
        f"Mann-Whitney $p$ & \\multicolumn{{4}}{{c}}{{{corr_results['mannwhitney_p']:.2e}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    with open(RESULTS_DIR / "summary" / "table_for_paper.tex", "w") as f:
        f.write(tex)
    print("  Saved table_for_paper.tex")


def generate_summary_json(tier1_results, tier2_results, partial_results,
                          tier3_results, topk_results, hmm_results, corr_results):
    """Generate comprehensive JSON summary."""
    rhos_t1 = [r["spearman_rho"] for r in tier1_results if not np.isnan(r["spearman_rho"])]
    rhos_t2 = [r["spearman_rho"] for r in tier2_results if not np.isnan(r["spearman_rho"])]
    rhos_partial = [r["partial_spearman_rho"] for r in partial_results
                    if not np.isnan(r["partial_spearman_rho"])]

    summary = {
        "dataset": "aime24",
        "n_samples": len(tier2_results),
        "tier1_tfidf": {
            "mean_rho": float(np.mean(rhos_t1)),
            "median_rho": float(np.median(rhos_t1)),
            "std_rho": float(np.std(rhos_t1)),
            "frac_p05": float(np.mean([r["p_value"] < 0.05 for r in tier1_results])),
        },
        "tier2_embedding": {
            "mean_rho": float(np.mean(rhos_t2)),
            "median_rho": float(np.median(rhos_t2)),
            "std_rho": float(np.std(rhos_t2)),
            "frac_p05": float(np.mean([r["p_value"] < 0.05 for r in tier2_results])),
        },
        "tier2_partial_spearman": {
            "mean_rho": float(np.mean(rhos_partial)),
            "median_rho": float(np.median(rhos_partial)),
            "std_rho": float(np.std(rhos_partial)),
            "frac_p05": float(np.mean([r["p_value"] < 0.05 for r in partial_results])),
        },
        "tier3_crossencoder": {
            "spearman_rho": tier3_results["spearman_rho"],
            "p_value": tier3_results["p_value"],
            "n_pairs": tier3_results["n_pairs"],
        },
        "topk_retrieval": topk_results,
        "hmm_state_analysis": hmm_results,
        "correctness_comparison": corr_results,
    }

    with open(RESULTS_DIR / "summary" / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Saved summary_statistics.json")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Semantic Validation Experiment")
    print("  Folding Structure Similarity vs Text Semantic Similarity")
    print("=" * 70)

    t_start = time.time()

    # 1. Load data
    samples, total_slices = load_all_samples()

    # 2. Tier 1: TF-IDF
    tier1_results = run_tier1_tfidf(samples)

    # 3. Tier 2: Embedding
    embeddings, sample_offsets = encode_all_slices(samples, total_slices)
    tier2_results = run_tier2_embedding(samples, embeddings, sample_offsets)

    # Free embeddings to save memory
    del embeddings

    # 4. Tier 3: Cross-encoder
    pairs_data = sample_pairs_for_tier3(samples)
    tier3_results = run_tier3_crossencoder(pairs_data)
    del pairs_data

    # 5. Controls
    partial_results = run_partial_spearman(samples)
    topk_results, struct_scores, adj_scores, rand_scores = run_topk_retrieval(samples)
    hmm_results = run_hmm_state_analysis(samples)
    corr_results = run_correctness_comparison(tier2_results)

    # 6. Figures
    print("\n── Generating Figures ──")
    generate_combined_binned_plot(samples, tier3_results)
    generate_correlation_histogram(tier2_results, partial_results)
    generate_position_control_plot(topk_results)
    generate_topk_plot(samples)

    # 7. Summary
    generate_latex_table(tier1_results, tier2_results, partial_results,
                         tier3_results, topk_results, hmm_results, corr_results)
    summary = generate_summary_json(tier1_results, tier2_results, partial_results,
                                     tier3_results, topk_results, hmm_results, corr_results)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Experiment complete in {elapsed:.1f}s")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")

    # Print key results
    print("\n── Key Results ──")
    print(f"  Tier 1 (TF-IDF) mean ρ:       {summary['tier1_tfidf']['mean_rho']:.4f}")
    print(f"  Tier 2 (Embedding) mean ρ:     {summary['tier2_embedding']['mean_rho']:.4f}")
    print(f"  Tier 2 (Partial) mean ρ:       {summary['tier2_partial_spearman']['mean_rho']:.4f}")
    print(f"  Tier 3 (Cross-encoder) ρ:      {summary['tier3_crossencoder']['spearman_rho']:.4f}")
    print(f"  Top-5 structural nearest sim:  {summary['topk_retrieval']['structural_mean']:.4f}")
    print(f"  Top-5 sequential adjacent sim: {summary['topk_retrieval']['adjacent_mean']:.4f}")
    print(f"  Top-5 random baseline sim:     {summary['topk_retrieval']['random_mean']:.4f}")


if __name__ == "__main__":
    main()
