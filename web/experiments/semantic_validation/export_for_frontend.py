#!/usr/bin/env python3
"""
Export semantic validation results for frontend consumption.

Generates:
  1. public/data/aime24/semantic_validation.json  (global summary)
  2. Injects alignment_rho, partial_rho into problems.index.json
  3. public/data/aime24/samples/p{pid}/s{sid}.neighbors.json  (per-slice top-5 neighbors)
  4. Copies combined_binned_plot.png to public/data/aime24/

Usage:
  python export_for_frontend.py            # generates 1-2 and 4 (no GPU needed)
  python export_for_frontend.py --neighbors  # also generates 3 (needs GPU + sentence-transformers)
"""

import os
import sys
import json
import csv
import shutil
import argparse
import base64
import time
from pathlib import Path

import numpy as np

# ── Paths ──
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "public" / "data" / "aime24"
INDEX_FILE = DATA_DIR / "problems.index.json"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


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


def load_per_sample_csvs():
    """Load tier2 embedding, partial spearman, and answer island CSVs as dicts keyed by (pid, sid)."""
    embedding = {}
    path = RESULTS_DIR / "tier2_embedding" / "per_sample_correlations.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["problem_id"]), int(row["sample_id"]))
            embedding[key] = float(row["spearman_rho"])

    partial = {}
    path = RESULTS_DIR / "controls" / "partial_spearman.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["problem_id"]), int(row["sample_id"]))
            partial[key] = float(row["partial_spearman_rho"])

    # Answer island analysis (optional)
    answer_island = {}
    ai_path = RESULTS_DIR / "controls" / "answer_island_analysis.csv"
    if ai_path.exists():
        with open(ai_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row["problem_id"]), int(row["sample_id"]))
                detected = row["graph_detected"] == "True"
                answer_island[key] = {
                    "detected": detected,
                    "t_start": int(row["graph_t_start"]) if detected else None,
                    "tail_length": int(row["graph_tail_length"]) if detected else None,
                    "attachment_score": round(float(row["attachment_score"]), 3) if detected else None,
                    "contains_answer": row["graph_contains_answer"] == "True" if detected else None,
                }

    return embedding, partial, answer_island


def export_global_summary():
    """Generate semantic_validation.json from summary_statistics.json."""
    with open(RESULTS_DIR / "summary" / "summary_statistics.json") as f:
        stats = json.load(f)

    summary = {
        "embedding_rho": {
            "mean": round(stats["tier2_embedding"]["mean_rho"], 3),
            "median": round(stats["tier2_embedding"]["median_rho"], 3),
            "std": round(stats["tier2_embedding"]["std_rho"], 3),
        },
        "partial_rho": {
            "mean": round(stats["tier2_partial_spearman"]["mean_rho"], 3),
            "median": round(stats["tier2_partial_spearman"]["median_rho"], 3),
            "std": round(stats["tier2_partial_spearman"]["std_rho"], 3),
        },
        "topk": {
            "structural": round(stats["topk_retrieval"]["structural_mean"], 3),
            "adjacent": round(stats["topk_retrieval"]["adjacent_mean"], 3),
            "random": round(stats["topk_retrieval"]["random_mean"], 3),
        },
        "n_samples": stats["n_samples"],
        "method": "all-MiniLM-L6-v2",
    }

    # Add Tier 4 source-model results if available
    if "tier4_source_model" in stats:
        t4 = stats["tier4_source_model"]
        summary["source_model_rho"] = {
            "mean": round(t4["mean_rho"], 3),
            "median": round(t4["median_rho"], 3),
            "std": round(t4["std_rho"], 3),
        }
    if "tier4_topk_retrieval" in stats:
        t4k = stats["tier4_topk_retrieval"]
        summary["source_topk"] = {
            "structural": round(t4k["structural_mean"], 3),
            "adjacent": round(t4k["adjacent_mean"], 3),
            "random": round(t4k["random_mean"], 3),
        }

    # Add shuffle control results if available
    shuffle_path = RESULTS_DIR / "controls" / "shuffle_summary.json"
    if shuffle_path.exists():
        with open(shuffle_path) as f:
            shuffle = json.load(f)
        summary["shuffle_controls"] = shuffle

    out_path = DATA_DIR / "semantic_validation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {out_path}")
    return summary


def inject_into_index(embedding_rho, partial_rho, answer_island=None):
    """Inject alignment_rho, partial_rho, and answer_island into problems.index.json samples."""
    with open(INDEX_FILE) as f:
        index = json.load(f)

    injected = 0
    missing = 0
    ai_injected = 0
    for prob in index["problems"]:
        pid = prob["problem_id"]
        for sample in prob["samples"]:
            sid = sample["sample_id"]
            key = (pid, sid)
            if key in embedding_rho:
                sample["alignment_rho"] = round(embedding_rho[key], 3)
                sample["partial_rho"] = round(partial_rho.get(key, 0), 3)
                injected += 1
            else:
                missing += 1
            # Answer island data
            if answer_island and key in answer_island:
                sample["answer_island"] = answer_island[key]
                ai_injected += 1

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"  Injected {injected} samples into problems.index.json ({missing} missing)")
    if answer_island:
        print(f"  Injected {ai_injected} answer_island entries")


def copy_binned_plot():
    """Copy combined_binned_plot.png to public data dir."""
    src = RESULTS_DIR / "summary" / "combined_binned_plot.png"
    dst = DATA_DIR / "semantic_validation_binned.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {dst}")
    else:
        print(f"  WARNING: {src} not found, skipping")


def export_neighbors(k=5):
    """
    Generate per-sample neighbors JSON files.
    Requires sentence-transformers + GPU.
    """
    # Set CUDA_VISIBLE_DEVICES before importing torch if you need a specific GPU
    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from sentence_transformers import SentenceTransformer

    print("\n── Loading sentence-transformers model ──")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open(INDEX_FILE) as f:
        index = json.load(f)

    total = sum(len(p["samples"]) for p in index["problems"])
    done = 0
    t0 = time.time()

    for prob in index["problems"]:
        pid = prob["problem_id"]
        prob_dir = SAMPLES_DIR / f"p{pid}"

        for sinfo in prob["samples"]:
            sid = sinfo["sample_id"]
            n = sinfo["n_slices"]

            # Load text
            text_path = prob_dir / f"s{sid}.text.json"
            with open(text_path) as f:
                text_data = json.load(f)

            full_text = text_data["full_text"]
            slice_texts = [
                full_text[item["char_start"]:item["char_end"]]
                for item in text_data["items"]
            ]

            if len(slice_texts) != n:
                print(f"  SKIP p{pid}/s{sid}: expected {n} slices, got {len(slice_texts)}")
                continue

            # Load structural similarity
            sim_path = prob_dir / f"s{sid}.sim.b64"
            with open(sim_path) as f:
                b64_str = f.read().strip()
            struct_sim = decode_sim_b64(b64_str, n)

            # Encode slice texts
            embeddings = model.encode(
                slice_texts,
                batch_size=512,
                device="cuda",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            text_sim = embeddings @ embeddings.T

            # Build per-slice neighbors
            slices = []
            for i in range(n):
                # Structural top-k (exclude self)
                ss = struct_sim[i].copy()
                ss[i] = -1
                topk_struct_idx = np.argsort(ss)[-k:][::-1]
                topk_struct_sims = text_sim[i, topk_struct_idx]

                # Sequential top-k (nearest by position, exclude self)
                dists = np.abs(np.arange(n) - i).astype(float)
                dists[i] = n + 1
                topk_seq_idx = np.argsort(dists)[:k]
                topk_seq_sims = text_sim[i, topk_seq_idx]

                slices.append({
                    "idx": i,
                    "structural_top5": topk_struct_idx.tolist(),
                    "structural_sims": [round(float(v), 3) for v in topk_struct_sims],
                    "sequential_top5": topk_seq_idx.tolist(),
                    "sequential_sims": [round(float(v), 3) for v in topk_seq_sims],
                    "structural_mean_text_sim": round(float(topk_struct_sims.mean()), 3),
                    "sequential_mean_text_sim": round(float(topk_seq_sims.mean()), 3),
                })

            out_path = prob_dir / f"s{sid}.neighbors.json"
            with open(out_path, "w") as f:
                json.dump({"slices": slices}, f, separators=(",", ":"))

            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Generated {done} neighbors files in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Export semantic validation for frontend")
    parser.add_argument("--neighbors", action="store_true",
                        help="Also generate per-slice neighbors (requires GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Export semantic validation results for frontend")
    print("=" * 60)

    # 1. Load per-sample CSVs
    print("\n── Loading per-sample correlations ──")
    embedding_rho, partial_rho, answer_island = load_per_sample_csvs()
    print(f"  Loaded {len(embedding_rho)} embedding, {len(partial_rho)} partial, {len(answer_island)} answer_island samples")

    # 2. Generate global summary
    print("\n── Generating global summary ──")
    export_global_summary()

    # 3. Inject into problems.index.json
    print("\n── Injecting into problems.index.json ──")
    inject_into_index(embedding_rho, partial_rho, answer_island)

    # 4. Copy binned plot
    print("\n── Copying binned plot ──")
    copy_binned_plot()

    # 5. Generate neighbors (optional, needs GPU)
    if args.neighbors:
        export_neighbors()
    else:
        print("\n── Skipping neighbors generation (use --neighbors to enable) ──")

    print("\n  Done!")


if __name__ == "__main__":
    main()
