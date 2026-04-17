#!/usr/bin/env python3
"""
Generate curated compare presets for the frontend Compare page.

Selects (correct_sample_id, incorrect_sample_id) pairs based on:
  - Alignment contrast (rho difference between correct/incorrect)
  - NFS score spread (structural quality)

Categories:
  - Strong positive: high alignment ρ, clear correct/incorrect structural difference
  - Weak/failure: low ρ or similar structures
  - Semantic advantage: embedding ρ >> TF-IDF ρ (meaning matters beyond keywords)

Output: public/data/aime24/compare_presets.json
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Paths ──
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "public" / "data" / "aime24"
INDEX_FILE = DATA_DIR / "problems.index.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results():
    """Load per-sample results from tier1, tier2 CSVs and compare data."""
    # Tier 2 (embedding)
    tier2 = {}
    with open(RESULTS_DIR / "tier2_embedding" / "per_sample_correlations.csv") as f:
        for row in csv.DictReader(f):
            key = (int(row["problem_id"]), int(row["sample_id"]))
            tier2[key] = float(row["spearman_rho"])

    # Tier 1 (TF-IDF)
    tier1 = {}
    with open(RESULTS_DIR / "tier1_tfidf" / "per_sample_correlations.csv") as f:
        for row in csv.DictReader(f):
            key = (int(row["problem_id"]), int(row["sample_id"]))
            tier1[key] = float(row["spearman_rho"])

    # Index for sample metadata
    with open(INDEX_FILE) as f:
        index = json.load(f)

    problems = {}
    for prob in index["problems"]:
        pid = prob["problem_id"]
        correct = []
        incorrect = []
        for s in prob["samples"]:
            sid = s["sample_id"]
            entry = {
                "sample_id": sid,
                "n_slices": s["n_slices"],
                "is_correct": s.get("is_correct", False),
                "nfs": s.get("nfs", 0),
                "tier2_rho": tier2.get((pid, sid), 0),
                "tier1_rho": tier1.get((pid, sid), 0),
            }
            if entry["is_correct"]:
                correct.append(entry)
            else:
                incorrect.append(entry)

        if correct and incorrect:  # Need both for Compare page
            problems[pid] = {"correct": correct, "incorrect": incorrect}

    return problems


def select_pair(correct_samples, incorrect_samples):
    """
    Select the best (correct_sid, incorrect_sid) pair.

    Strategy: pick the correct sample with highest tier2 ρ
    and the incorrect sample with lowest tier2 ρ (maximize alignment contrast).
    """
    # Sort correct by rho (descending) — best aligned correct
    correct_sorted = sorted(correct_samples, key=lambda s: s["tier2_rho"], reverse=True)
    # Sort incorrect by rho (ascending) — worst aligned incorrect
    incorrect_sorted = sorted(incorrect_samples, key=lambda s: s["tier2_rho"])

    return correct_sorted[0], incorrect_sorted[0]


def classify_problems(problems):
    """Classify problems into preset categories."""
    presets = []

    scored = []
    for pid, data in problems.items():
        c_rhos = [s["tier2_rho"] for s in data["correct"]]
        i_rhos = [s["tier2_rho"] for s in data["incorrect"]]
        mean_rho = np.mean(c_rhos + i_rhos)
        contrast = abs(np.mean(c_rhos) - np.mean(i_rhos))

        # Semantic advantage: how much embedding > tfidf
        c_t1 = [s["tier1_rho"] for s in data["correct"]]
        i_t1 = [s["tier1_rho"] for s in data["incorrect"]]
        t1_mean = np.mean(c_t1 + i_t1)
        semantic_gap = mean_rho - t1_mean

        scored.append({
            "pid": pid,
            "mean_rho": mean_rho,
            "contrast": contrast,
            "semantic_gap": semantic_gap,
            "n_correct": len(data["correct"]),
            "n_incorrect": len(data["incorrect"]),
        })

    # Sort by different criteria for each category
    # Strong positive: high rho + good balance
    strong = sorted(scored, key=lambda x: x["mean_rho"] * (1 + x["contrast"]), reverse=True)
    # Weak/failure: low rho
    weak = sorted(scored, key=lambda x: x["mean_rho"])
    # Semantic advantage: largest embedding-tfidf gap
    semantic = sorted(scored, key=lambda x: x["semantic_gap"], reverse=True)

    added = set()

    # Pick top 3 strong
    for s in strong[:5]:
        if s["pid"] not in added and len(presets) < 3:
            correct_s, incorrect_s = select_pair(
                problems[s["pid"]]["correct"],
                problems[s["pid"]]["incorrect"]
            )
            presets.append({
                "problem_id": s["pid"],
                "label": "Strong alignment contrast",
                "correct_sample_id": correct_s["sample_id"],
                "incorrect_sample_id": incorrect_s["sample_id"],
                "meta": {
                    "mean_rho": round(s["mean_rho"], 3),
                    "contrast": round(s["contrast"], 3),
                }
            })
            added.add(s["pid"])

    # Pick 2 weak
    for s in weak[:5]:
        if s["pid"] not in added and sum(1 for p in presets if "Weak" in p["label"]) < 2:
            correct_s, incorrect_s = select_pair(
                problems[s["pid"]]["correct"],
                problems[s["pid"]]["incorrect"]
            )
            presets.append({
                "problem_id": s["pid"],
                "label": "Weak alignment",
                "correct_sample_id": correct_s["sample_id"],
                "incorrect_sample_id": incorrect_s["sample_id"],
                "meta": {
                    "mean_rho": round(s["mean_rho"], 3),
                    "contrast": round(s["contrast"], 3),
                }
            })
            added.add(s["pid"])

    # Pick 2 semantic
    for s in semantic[:5]:
        if s["pid"] not in added and sum(1 for p in presets if "Semantic" in p["label"]) < 2:
            correct_s, incorrect_s = select_pair(
                problems[s["pid"]]["correct"],
                problems[s["pid"]]["incorrect"]
            )
            presets.append({
                "problem_id": s["pid"],
                "label": "Semantic > keyword",
                "correct_sample_id": correct_s["sample_id"],
                "incorrect_sample_id": incorrect_s["sample_id"],
                "meta": {
                    "mean_rho": round(s["mean_rho"], 3),
                    "semantic_gap": round(s["semantic_gap"], 3),
                }
            })
            added.add(s["pid"])

    return presets


def main():
    print("=" * 60)
    print("  Generate Compare Presets")
    print("=" * 60)

    # 1. Load data
    problems = load_results()
    print(f"\n  {len(problems)} problems with both correct/incorrect samples")

    # 2. Classify and select
    presets = classify_problems(problems)

    # 3. Output
    output = {"presets": presets}
    out_path = DATA_DIR / "compare_presets.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Generated {len(presets)} presets:")
    for p in presets:
        print(f"    P{p['problem_id']}: {p['label']} "
              f"(correct=S{p['correct_sample_id']}, incorrect=S{p['incorrect_sample_id']})")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
