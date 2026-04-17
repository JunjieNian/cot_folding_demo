#!/usr/bin/env python3
"""
Answer Island Analysis: Formalize terminal branch detection from structural similarity graphs.

Detects "answer islands" (terminal branches) using two methods:
  1. Graph-based: structural similarity attachment/cohesion analysis
  2. HMM-based: final contiguous exploit segment (matches frontend logic)

Then measures overlap with actual answer slices (\\boxed{} location).

Output:
  results/controls/answer_island_analysis.csv
  results/controls/answer_island_summary.json
  results/summary/answer_island_plot.png
"""

import os
import sys
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "public" / "data" / "aime24"
INDEX_FILE = DATA_DIR / "problems.index.json"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared data loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import load_all_samples


# ═══════════════════════════════════════════════════════════════════════════════
#  2a: Terminal Branch Detection (Graph-based)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_terminal_branch(struct_sim, min_tail_len=3, max_tail_frac=0.5):
    """
    Detect terminal weakly-connected branch from structural similarity matrix.

    For candidate t_start values (from n-min_tail_len down to n//2):
      attachment = mean(struct_sim[i>=t_start, j<t_start])   # tail-to-body connection
      cohesion   = mean(struct_sim[i,j>=t_start, i!=j])      # tail internal cohesion

    Select t_start that maximizes detachment_gap = cohesion - attachment.
    Detection requires: attachment < global_mean * 0.7 and cohesion > attachment.

    Returns dict with detection results.
    """
    n = struct_sim.shape[0]
    if n < min_tail_len + 2:
        return {"detected": False}

    # Global mean of off-diagonal similarities
    iu = np.triu_indices(n, k=1)
    global_mean = struct_sim[iu].mean()

    min_start = max(n // 2, n - int(n * max_tail_frac))
    max_start = n - min_tail_len

    if min_start > max_start:
        return {"detected": False}

    best_gap = -np.inf
    best_result = None

    for t_start in range(min_start, max_start + 1):
        tail_indices = np.arange(t_start, n)
        body_indices = np.arange(0, t_start)

        if len(body_indices) == 0 or len(tail_indices) < min_tail_len:
            continue

        # Attachment: mean similarity between tail and body
        attachment_block = struct_sim[np.ix_(tail_indices, body_indices)]
        attachment = attachment_block.mean()

        # Cohesion: mean similarity within tail (excluding diagonal)
        tail_block = struct_sim[np.ix_(tail_indices, tail_indices)]
        n_tail = len(tail_indices)
        if n_tail < 2:
            continue
        tail_iu = np.triu_indices(n_tail, k=1)
        cohesion = tail_block[tail_iu].mean() if len(tail_iu[0]) > 0 else 0

        gap = cohesion - attachment

        if gap > best_gap:
            best_gap = gap
            best_result = {
                "t_start": int(t_start),
                "tail_length": int(n - t_start),
                "attachment_score": float(attachment),
                "cohesion": float(cohesion),
                "detachment_gap": float(gap),
            }

    if best_result is None:
        return {"detected": False}

    # Detection criteria
    detected = (best_result["attachment_score"] < global_mean * 0.7 and
                best_result["cohesion"] > best_result["attachment_score"])

    best_result["detected"] = detected
    best_result["global_mean"] = float(global_mean)
    return best_result


# ═══════════════════════════════════════════════════════════════════════════════
#  2b: HMM-based Detection (matching frontend logic)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_hmm_answer_tail(hmm_states):
    """
    Detect answer tail using HMM states (matches frontend useAnswerIslandDetection).

    The last contiguous exploit (1) segment.
    """
    n = len(hmm_states)
    if n < 4:
        return {"detected": False}

    # Last state must be exploit
    if hmm_states[-1] != 1:
        return {"detected": False}

    # Walk backwards
    tail_start = n - 1
    while tail_start > 0 and hmm_states[tail_start - 1] == 1:
        tail_start -= 1

    tail_len = n - tail_start
    if tail_len < 2:
        return {"detected": False}

    return {
        "detected": True,
        "t_start": tail_start,
        "tail_length": tail_len,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2b: Answer Span Detection
# ═══════════════════════════════════════════════════════════════════════════════

def find_answer_slices(text_data):
    """
    Find slices containing \\boxed{...} in the full text.
    Returns set of slice indices.
    """
    full_text = text_data["full_text"]
    items = text_data["items"]

    # Find all \boxed{...} positions
    answer_slices = set()
    for match in re.finditer(r"\\boxed\{", full_text):
        pos = match.start()
        for idx, item in enumerate(items):
            if item["char_start"] <= pos < item["char_end"]:
                answer_slices.add(idx)
                break

    return answer_slices


def compute_overlap(tail_range, answer_slices):
    """Compute IoU between tail range and answer slices."""
    if not tail_range or not answer_slices:
        return 0.0
    intersection = tail_range & answer_slices
    union = tail_range | answer_slices
    return len(intersection) / len(union) if union else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(samples):
    """Run full answer island analysis on all samples."""
    print("\n── Running Answer Island Analysis ──")
    t0 = time.time()

    # Load text data for answer detection
    with open(INDEX_FILE) as f:
        index = json.load(f)

    # Build text data lookup
    text_lookup = {}
    for prob in index["problems"]:
        pid = prob["problem_id"]
        prob_dir = SAMPLES_DIR / f"p{pid}"
        for sinfo in prob["samples"]:
            sid = sinfo["sample_id"]
            text_path = prob_dir / f"s{sid}.text.json"
            with open(text_path) as f:
                text_lookup[(pid, sid)] = json.load(f)

    results = []
    for s in samples:
        pid = s["problem_id"]
        sid = s["sample_id"]
        n = s["n_slices"]

        # Graph-based detection
        graph = detect_terminal_branch(s["struct_sim"])

        # HMM-based detection
        hmm = detect_hmm_answer_tail(s["hmm_states"])

        # Answer slices
        text_data = text_lookup.get((pid, sid))
        answer_slices = find_answer_slices(text_data) if text_data else set()

        # Compute overlaps
        graph_tail_range = set(range(graph["t_start"], n)) if graph["detected"] else set()
        hmm_tail_range = set(range(hmm["t_start"], n)) if hmm["detected"] else set()

        graph_iou = compute_overlap(graph_tail_range, answer_slices)
        hmm_iou = compute_overlap(hmm_tail_range, answer_slices)

        graph_contains = bool(answer_slices & graph_tail_range) if graph["detected"] else False
        hmm_contains = bool(answer_slices & hmm_tail_range) if hmm["detected"] else False

        results.append({
            "problem_id": pid,
            "sample_id": sid,
            "n_slices": n,
            "is_correct": s["is_correct"],
            # Graph-based
            "graph_detected": graph["detected"],
            "graph_t_start": graph.get("t_start", -1),
            "graph_tail_length": graph.get("tail_length", 0),
            "graph_tail_fraction": graph.get("tail_length", 0) / n,
            "attachment_score": graph.get("attachment_score", 0),
            "cohesion": graph.get("cohesion", 0),
            "detachment_gap": graph.get("detachment_gap", 0),
            # HMM-based
            "hmm_detected": hmm["detected"],
            "hmm_t_start": hmm.get("t_start", -1),
            "hmm_tail_length": hmm.get("tail_length", 0),
            # Answer overlap
            "answer_slices_count": len(answer_slices),
            "graph_answer_iou": graph_iou,
            "hmm_answer_iou": hmm_iou,
            "graph_contains_answer": graph_contains,
            "hmm_contains_answer": hmm_contains,
        })

    elapsed = time.time() - t0
    print(f"  {len(results)} samples analyzed in {elapsed:.1f}s")
    return results


def generate_summary(results):
    """Generate summary statistics."""
    df = pd.DataFrame(results)

    graph_detected = df[df["graph_detected"]]
    hmm_detected = df[df["hmm_detected"]]
    correct = df[df["is_correct"]]
    correct_graph = correct[correct["graph_detected"]]

    # Agreement: both detect or both don't
    both_detected = (df["graph_detected"] == df["hmm_detected"]).mean()

    summary = {
        "n_samples": len(df),
        "graph_detection_rate": round(float(df["graph_detected"].mean()), 4),
        "hmm_detection_rate": round(float(df["hmm_detected"].mean()), 4),
        "graph_contains_answer_rate": round(
            float(graph_detected["graph_contains_answer"].mean()) if len(graph_detected) > 0 else 0, 4),
        "hmm_contains_answer_rate": round(
            float(hmm_detected["hmm_contains_answer"].mean()) if len(hmm_detected) > 0 else 0, 4),
        "graph_contains_answer_correct_rate": round(
            float(correct_graph["graph_contains_answer"].mean()) if len(correct_graph) > 0 else 0, 4),
        "agreement_rate": round(float(both_detected), 4),
        "mean_attachment": round(float(graph_detected["attachment_score"].mean()) if len(graph_detected) > 0 else 0, 4),
        "mean_cohesion": round(float(graph_detected["cohesion"].mean()) if len(graph_detected) > 0 else 0, 4),
        "mean_detachment_gap": round(float(graph_detected["detachment_gap"].mean()) if len(graph_detected) > 0 else 0, 4),
        "mean_tail_fraction": round(float(graph_detected["graph_tail_fraction"].mean()) if len(graph_detected) > 0 else 0, 4),
        "mean_graph_iou": round(float(graph_detected["graph_answer_iou"].mean()) if len(graph_detected) > 0 else 0, 4),
        "mean_hmm_iou": round(float(hmm_detected["hmm_answer_iou"].mean()) if len(hmm_detected) > 0 else 0, 4),
    }

    print("\n── Answer Island Summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def generate_plot(results):
    """
    Generate answer island visualization:
      Panel 1: Detection rates comparison (graph vs HMM)
      Panel 2: Attachment vs Cohesion scatter for detected samples
      Panel 3: Tail fraction distribution
    """
    print("\n── Generating answer island plot ──")
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Detection rates ──
    ax = axes[0]
    graph_rate = df["graph_detected"].mean()
    hmm_rate = df["hmm_detected"].mean()
    graph_answer = df[df["graph_detected"]]["graph_contains_answer"].mean() if df["graph_detected"].any() else 0
    hmm_answer = df[df["hmm_detected"]]["hmm_contains_answer"].mean() if df["hmm_detected"].any() else 0

    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [graph_rate, graph_answer], width,
                    label="Graph-based", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, [hmm_rate, hmm_answer], width,
                    label="HMM-based", color="#e67e22", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Detection rate", "Contains answer"])
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Terminal Branch Detection", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)

    # ── Panel 2: Attachment vs Cohesion scatter ──
    ax = axes[1]
    detected = df[df["graph_detected"]]
    not_detected = df[~df["graph_detected"]]

    if len(detected) > 0:
        ax.scatter(detected["attachment_score"], detected["cohesion"],
                   alpha=0.3, s=10, c="#2ecc71", label=f"Detected ({len(detected)})")
    if len(not_detected) > 0:
        ax.scatter(not_detected["attachment_score"], not_detected["cohesion"],
                   alpha=0.15, s=10, c="#e74c3c", label=f"Not detected ({len(not_detected)})")

    # Diagonal line (cohesion = attachment)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="cohesion = attachment")
    ax.set_xlabel("Attachment (tail → body)", fontsize=12)
    ax.set_ylabel("Cohesion (within tail)", fontsize=12)
    ax.set_title("Attachment vs Cohesion", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Tail fraction distribution ──
    ax = axes[2]
    graph_fracs = df[df["graph_detected"]]["graph_tail_fraction"]
    if len(graph_fracs) > 0:
        ax.hist(graph_fracs, bins=30, alpha=0.6, color="#3498db", label="Graph tail fraction")
    hmm_fracs = df[df["hmm_detected"]]["hmm_tail_length"] / df[df["hmm_detected"]]["n_slices"]
    if len(hmm_fracs) > 0:
        ax.hist(hmm_fracs, bins=30, alpha=0.6, color="#e67e22", label="HMM tail fraction")
    ax.set_xlabel("Tail fraction (of total slices)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Tail Fraction Distribution", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = RESULTS_DIR / "summary" / "answer_island_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print("=" * 70)
    print("  Answer Island Analysis: Terminal Branch Formalization")
    print("=" * 70)
    t_start = time.time()

    # 1. Load all samples
    samples, _ = load_all_samples()

    # 2. Run analysis
    results = run_analysis(samples)

    # 3. Save CSV
    out_dir = RESULTS_DIR / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "answer_island_analysis.csv", index=False)
    print(f"  Saved answer_island_analysis.csv")

    # 4. Summary
    summary = generate_summary(results)
    with open(out_dir / "answer_island_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved answer_island_summary.json")

    # 5. Plot
    generate_plot(results)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Answer island analysis complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
