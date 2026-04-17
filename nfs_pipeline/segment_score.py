#!/usr/bin/env python3
"""
Segment 级 Native Fold Score (NFS) + 与 slice 级基线的对比分析

读取 segment 级原语 → 计算 NFS → 外部验证 → 与 slice 级 NFS 对比

CLI: --method union|avg
输出: nfs_segment_union.json / nfs_segment_avg.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from project_paths import (
    REPO_ROOT,
    default_segment_batch_dir,
    list_available_benchmarks,
    resolve_cache_path,
)

# 默认值
BATCH_DIR = default_segment_batch_dir()
CACHE_ROOT = None
EVAL_REPORT = None
META_FILE = None

# slice 级基线
SLICE_NFS_FILE = REPO_ROOT / "batch_results" / "nfs_analysis.json"


# ═══════════════════════════════════════════════════════════════
#  NFS 计算（segment 级）
# ═══════════════════════════════════════════════════════════════

def compute_nfs(sample):
    """从 segment 级 primitives 单样本 dict 计算 B, H, D*, NFS*。

    V4 changes (adaptive-segment compatible):
    - H: structural coherence — mean similarity of non-core segments to
      core, weighted by normalized slice distance.  Replaces the old
      return-edge-only formula that collapsed to 0 on small graphs.
    - D*: continuous drift — uses return_ratio = sim_to_core / τ instead
      of binary is_drift, weighted by segment size and time position.
    """
    n_slices = sample["n_slices"]
    n_seg = sample["n_segments"]
    tau = sample["contact_threshold"]
    segments = sample.get("segments")  # list of {id, state, start, end, n_slices}

    # ── B: Backbone (unchanged) ──
    core = sample["core"]
    s_core = core["internal_similarity"]
    f_core = core["fraction_of_exploit"]
    B = s_core * f_core

    # ── H: Structural coherence to core ──
    # For each non-core segment, measure mean similarity to core × slice distance.
    # Uses return_edges when available as high-value contributions, plus a
    # baseline coherence term from drift branches.
    edges = sample["return_edges"]
    branches = sample["drift_branches"]

    h_contributions = []

    # Contribution from return edges (slice-gap weighted)
    for e in edges:
        s_e = e["similarity"]
        slice_gap = e.get("slice_gap", e["gap"])
        r_e = s_e * (slice_gap / max(n_slices - 1, 1))
        h_contributions.append(r_e)

    # Contribution from drift branches that partially return to core
    for b in branches:
        m_b = b["max_sim_to_core"]
        if m_b <= tau * 0.5:
            continue  # too dissimilar, skip
        # Slice distance from this segment to core
        seg_id = b["seg_id"]
        if segments is not None and "start" in b:
            # Use segment start/end metadata
            seg_start = b["start"]
            seg_end = b["end"]
        elif segments is not None and seg_id < len(segments):
            seg_start = segments[seg_id]["start"]
            seg_end = segments[seg_id]["end"]
        else:
            # Fallback: approximate from segment index
            seg_start = seg_id * max(n_slices // n_seg, 1)
            seg_end = seg_start + max(n_slices // n_seg, 1)

        # Nearest core segment distance (approximate via core indices)
        core_indices = core["indices"]
        if core_indices and segments is not None:
            min_dist = min(
                abs(seg_start - segments[c]["end"])
                for c in core_indices if c < len(segments)
            )
        else:
            min_dist = 1

        if min_dist < 2:
            continue  # adjacent to core, not informative

        weight = min_dist / max(n_slices, 1)
        h_contributions.append(m_b * weight)

    H = float(np.mean(h_contributions)) if h_contributions else 0.0

    # ── D*: Continuous drift with time & size weighting ──
    if branches:
        numerator = 0.0
        denominator = 0.0
        for b in branches:
            seg_id = b["seg_id"]
            seg_size = b.get("n_slices_in_seg", 1)
            # Time weight: later segments matter more
            if segments is not None and seg_id < len(segments):
                mid = (segments[seg_id]["start"] + segments[seg_id]["end"]) / 2.0
            else:
                mid = (seg_id + 0.5) / max(n_seg, 1) * n_slices
            t_b = mid / max(n_slices, 1)
            w_b = 1.0 + t_b  # weight 1~2
            # Continuous return ratio
            m_b = b["max_sim_to_core"]
            r_b = min(1.0, m_b / tau) if tau > 0 else 0.0
            numerator += w_b * seg_size * (1.0 - r_b)
            denominator += w_b * seg_size
        D0 = numerator / denominator if denominator > 0 else 0.0
    else:
        D0 = 0.0

    # G: final closure gate (unchanged)
    fc = sample.get("final_closure")
    if fc and fc.get("closure_coefficient") is not None:
        C = fc["closure_coefficient"]
    else:
        C = 1.0
    G = (1.0 + C) / 2.0

    # D* = 1 - G·(1-D0)
    D_star = 1.0 - G * (1.0 - D0)

    # NFS*
    product = B * H * (1.0 - D_star)
    NFS = 100.0 * (product ** (1.0 / 3.0)) if product > 0 else 0.0

    return {
        "B": float(B),
        "H": float(H),
        "D0": float(D0),
        "G": float(G),
        "D_star": float(D_star),
        "NFS": float(NFS),
    }


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_primitives(method):
    primitives_file = BATCH_DIR / f"primitives_segment_{method}.json"
    with open(primitives_file) as f:
        data = json.load(f)
    return data["samples"]


def load_correctness():
    if META_FILE is None or EVAL_REPORT is None:
        raise FileNotFoundError("Cache paths are not initialized; run main() with a valid --cache.")

    with open(META_FILE) as f:
        meta = json.load(f)

    global_to_run = {}
    for idx, s in enumerate(meta["samples"]):
        global_to_run[idx] = (s["problem_id"], s["run_index"])

    with open(EVAL_REPORT) as f:
        report = json.load(f)

    run_to_result = {}
    for problem in report["results"]:
        pid = problem["problem_id"]
        for run in problem["runs"]:
            run_to_result[(pid, run["run_index"])] = (
                run["is_correct"],
                run.get("extracted_answer", ""),
            )

    mapping = {}
    for global_idx, (pid, run_idx) in global_to_run.items():
        if (pid, run_idx) in run_to_result:
            mapping[(pid, global_idx)] = run_to_result[(pid, run_idx)]

    return mapping


def parse_answer(extracted_answer):
    if not extracted_answer:
        return ""
    try:
        parsed = eval(extracted_answer)
        if isinstance(parsed, list) and len(parsed) >= 2:
            return str(parsed[1]).strip()
        return str(parsed).strip()
    except:
        return str(extracted_answer).strip()


def load_slice_baseline():
    """加载 slice 级 NFS 作为基线对比。"""
    if not SLICE_NFS_FILE.exists():
        return None
    with open(SLICE_NFS_FILE) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  验证指标
# ═══════════════════════════════════════════════════════════════

def compute_auroc_auprc(scores, labels):
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    tp = fp = 0
    tpr_prev = fpr_prev = 0.0
    auroc = 0.0
    precisions = []
    recalls = []

    for i in range(len(sorted_labels)):
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)

    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2.0

    return float(auroc), float(auprc)


def selective_accuracy(scores, labels, fractions=(0.1, 0.2, 0.3, 0.5)):
    order = np.argsort(-scores)
    results = {}
    for frac in fractions:
        k = max(1, int(len(scores) * frac))
        top_labels = labels[order[:k]]
        results[f"top_{int(frac*100)}%"] = float(top_labels.mean())
    results["overall"] = float(labels.mean())
    return results


def per_problem_ranking(problem_scores):
    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []
    pairwise_correct = 0
    pairwise_total = 0

    for pid, runs in sorted(problem_scores.items()):
        runs_sorted = sorted(runs, key=lambda x: -x[0])
        labels = [r[1] for r in runs_sorted]

        hit_at_1.append(1.0 if labels[0] else 0.0)
        hit_at_3.append(1.0 if any(labels[:3]) else 0.0)
        hit_at_5.append(1.0 if any(labels[:5]) else 0.0)

        correct_scores = [r[0] for r in runs if r[1]]
        incorrect_scores = [r[0] for r in runs if not r[1]]
        for cs in correct_scores:
            for ics in incorrect_scores:
                if cs > ics:
                    pairwise_correct += 1
                elif cs == ics:
                    pairwise_correct += 0.5
                pairwise_total += 1

    return {
        "hit_at_1": float(np.mean(hit_at_1)),
        "hit_at_3": float(np.mean(hit_at_3)),
        "hit_at_5": float(np.mean(hit_at_5)),
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total > 0 else 0.5,
        "pairwise_total": pairwise_total,
        "n_problems": len(problem_scores),
    }


def voting_analysis(problem_scores):
    majority_correct = 0
    weighted_correct = 0
    top1_correct = 0
    n_problems = 0
    per_problem = []

    for pid, runs in sorted(problem_scores.items()):
        n_problems += 1
        answer_groups = defaultdict(list)
        for nfs, is_correct, answer in runs:
            answer_groups[answer].append((nfs, is_correct))

        majority_answer = max(answer_groups.keys(), key=lambda a: len(answer_groups[a]))
        majority_is_correct = answer_groups[majority_answer][0][1]

        answer_weights = {}
        for ans, group in answer_groups.items():
            answer_weights[ans] = sum(nfs for nfs, _ in group)
        weighted_answer = max(answer_weights.keys(), key=lambda a: answer_weights[a])
        weighted_is_correct = answer_groups[weighted_answer][0][1]

        best_run = max(runs, key=lambda x: x[0])
        top1_is_correct = best_run[1]

        majority_correct += majority_is_correct
        weighted_correct += weighted_is_correct
        top1_correct += top1_is_correct

        per_problem.append({
            "problem_id": pid,
            "majority_correct": majority_is_correct,
            "weighted_correct": weighted_is_correct,
            "top1_correct": top1_is_correct,
        })

    return {
        "majority_accuracy": majority_correct / n_problems if n_problems > 0 else 0.0,
        "weighted_accuracy": weighted_correct / n_problems if n_problems > 0 else 0.0,
        "top1_accuracy": top1_correct / n_problems if n_problems > 0 else 0.0,
        "n_problems": n_problems,
        "per_problem": per_problem,
    }


# ═══════════════════════════════════════════════════════════════
#  对比分析
# ═══════════════════════════════════════════════════════════════

def compare_with_slice_baseline(seg_nfs_arr, seg_labels, seg_auroc, seg_auprc,
                                 seg_sel_acc, seg_ranking, seg_voting, method):
    """与 slice 级 NFS 做对比。"""
    baseline = load_slice_baseline()
    if baseline is None:
        print("\n  [Comparison] Slice baseline not found, skipping comparison.")
        return None

    slice_auroc = baseline["discrimination"]["auroc"]
    slice_auprc = baseline["discrimination"]["auprc"]
    slice_sel_acc = baseline["selective_accuracy"]
    slice_ranking = baseline["ranking"]
    slice_voting = baseline["voting"]
    slice_nfs_mean = baseline["nfs_distribution"]["mean"]
    slice_nfs_std = baseline["nfs_distribution"]["std"]

    # Slice 级 correct/incorrect NFS
    slice_nfs_correct = baseline["discrimination"].get("nfs_correct_mean", 0)
    slice_nfs_incorrect = baseline["discrimination"].get("nfs_incorrect_mean", 0)

    seg_correct = seg_nfs_arr[seg_labels]
    seg_incorrect = seg_nfs_arr[~seg_labels]

    # Cohen's d for segment level
    if len(seg_correct) > 0 and len(seg_incorrect) > 0:
        diff = seg_correct.mean() - seg_incorrect.mean()
        pooled = np.sqrt((seg_correct.var() * len(seg_correct) +
                          seg_incorrect.var() * len(seg_incorrect)) /
                         (len(seg_correct) + len(seg_incorrect)))
        seg_cohens_d = diff / pooled if pooled > 0 else 0
    else:
        seg_cohens_d = 0

    # Slice Cohen's d (from baseline data)
    if slice_nfs_correct and slice_nfs_incorrect:
        slice_diff = slice_nfs_correct - slice_nfs_incorrect
        # approximate pooled std from overall
        slice_cohens_d = slice_diff / slice_nfs_std if slice_nfs_std > 0 else 0
    else:
        slice_cohens_d = 0

    comparison = {
        "method": method,
        "auroc": {
            "segment": seg_auroc,
            "slice": slice_auroc,
            "delta": seg_auroc - slice_auroc,
        },
        "auprc": {
            "segment": seg_auprc,
            "slice": slice_auprc,
            "delta": seg_auprc - slice_auprc,
        },
        "cohens_d": {
            "segment": seg_cohens_d,
            "slice": slice_cohens_d,
            "delta": seg_cohens_d - slice_cohens_d,
        },
        "selective_accuracy": {},
        "ranking": {
            "hit_at_1": {
                "segment": seg_ranking["hit_at_1"],
                "slice": slice_ranking["hit_at_1"],
                "delta": seg_ranking["hit_at_1"] - slice_ranking["hit_at_1"],
            },
            "pairwise": {
                "segment": seg_ranking["pairwise_accuracy"],
                "slice": slice_ranking["pairwise_accuracy"],
                "delta": seg_ranking["pairwise_accuracy"] - slice_ranking["pairwise_accuracy"],
            },
        },
        "voting": {
            "majority": {
                "segment": seg_voting["majority_accuracy"],
                "slice": slice_voting["majority_accuracy"],
                "delta": seg_voting["majority_accuracy"] - slice_voting["majority_accuracy"],
            },
            "weighted": {
                "segment": seg_voting["weighted_accuracy"],
                "slice": slice_voting["weighted_accuracy"],
                "delta": seg_voting["weighted_accuracy"] - slice_voting["weighted_accuracy"],
            },
        },
    }

    # Selective accuracy comparison
    for key in seg_sel_acc:
        if key in slice_sel_acc:
            comparison["selective_accuracy"][key] = {
                "segment": seg_sel_acc[key],
                "slice": slice_sel_acc[key],
                "delta": seg_sel_acc[key] - slice_sel_acc[key],
            }

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Segment ({method}) vs Slice Baseline")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Segment':>10} {'Slice':>10} {'Δ':>10}")
    print(f"  {'-'*55}")
    print(f"  {'AUROC':<25} {seg_auroc:10.4f} {slice_auroc:10.4f} {seg_auroc-slice_auroc:+10.4f}")
    print(f"  {'AUPRC':<25} {seg_auprc:10.4f} {slice_auprc:10.4f} {seg_auprc-slice_auprc:+10.4f}")
    print(f"  {'Cohen d':<25} {seg_cohens_d:10.3f} {slice_cohens_d:10.3f} {seg_cohens_d-slice_cohens_d:+10.3f}")

    for key in ["top_10%", "top_20%", "top_30%"]:
        if key in seg_sel_acc and key in slice_sel_acc:
            print(f"  {'SelAcc ' + key:<25} {seg_sel_acc[key]:10.4f} "
                  f"{slice_sel_acc[key]:10.4f} {seg_sel_acc[key]-slice_sel_acc[key]:+10.4f}")

    print(f"  {'Hit@1':<25} {seg_ranking['hit_at_1']:10.4f} "
          f"{slice_ranking['hit_at_1']:10.4f} {seg_ranking['hit_at_1']-slice_ranking['hit_at_1']:+10.4f}")
    print(f"  {'Pairwise':<25} {seg_ranking['pairwise_accuracy']:10.4f} "
          f"{slice_ranking['pairwise_accuracy']:10.4f} "
          f"{seg_ranking['pairwise_accuracy']-slice_ranking['pairwise_accuracy']:+10.4f}")
    print(f"  {'Majority Vote':<25} {seg_voting['majority_accuracy']:10.4f} "
          f"{slice_voting['majority_accuracy']:10.4f} "
          f"{seg_voting['majority_accuracy']-slice_voting['majority_accuracy']:+10.4f}")
    print(f"  {'Weighted Vote':<25} {seg_voting['weighted_accuracy']:10.4f} "
          f"{slice_voting['weighted_accuracy']:10.4f} "
          f"{seg_voting['weighted_accuracy']-slice_voting['weighted_accuracy']:+10.4f}")
    print(f"{'='*70}")

    return comparison


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def _do_analysis(method):
    """Core segment NFS analysis (uses module globals)."""
    print("=" * 70)
    print(f"Segment-level NFS Analysis (method={method})")
    print(f"  cache: {CACHE_ROOT}")
    print(f"  batch: {BATCH_DIR}")
    print("=" * 70)

    print("\n[1] Loading segment primitives and correctness labels...")
    samples = load_primitives(method)
    correctness = load_correctness()
    print(f"    Primitives: {len(samples)} samples")
    print(f"    Correctness: {len(correctness)} labels")

    print("\n[2] Computing NFS for each sample...")
    all_scores = []
    problem_scores = defaultdict(list)

    for s in samples:
        pid = s["problem_id"]
        sid = s["sample_id"]

        nfs_result = compute_nfs(s)
        # correctness keys use pid from meta.json (may be str or int)
        is_correct, answer_raw = correctness.get((pid, sid), (None, ""))
        if is_correct is None:
            # fallback: try int conversion for numeric problem IDs
            try:
                is_correct, answer_raw = correctness.get((int(pid), sid), (False, ""))
            except (ValueError, TypeError):
                is_correct, answer_raw = False, ""
        answer = parse_answer(answer_raw)

        record = {
            "problem_id": pid,
            "sample_id": sid,
            "n_slices": s["n_slices"],
            "n_segments": s["n_segments"],
            **nfs_result,
            "is_correct": is_correct,
            "answer": answer,
        }
        all_scores.append(record)
        problem_scores[pid].append((nfs_result["NFS"], is_correct, answer))

    print(f"    Computed: {len(all_scores)} samples")

    print("\n[3] NFS Distribution...")
    nfs_arr = np.array([r["NFS"] for r in all_scores])
    labels = np.array([r["is_correct"] for r in all_scores], dtype=bool)

    print(f"    NFS mean={nfs_arr.mean():.4f} std={nfs_arr.std():.4f} "
          f"min={nfs_arr.min():.4f} max={nfs_arr.max():.4f}")

    nfs_correct = nfs_arr[labels]
    nfs_incorrect = nfs_arr[~labels]
    print(f"    Correct   (n={labels.sum():4d}): mean={nfs_correct.mean():.4f}")
    print(f"    Incorrect (n={int((~labels).sum()):4d}): mean={nfs_incorrect.mean():.4f}")

    if len(nfs_correct) > 0 and len(nfs_incorrect) > 0:
        diff = nfs_correct.mean() - nfs_incorrect.mean()
        pooled = np.sqrt((nfs_correct.var() * len(nfs_correct) +
                          nfs_incorrect.var() * len(nfs_incorrect)) /
                         (len(nfs_correct) + len(nfs_incorrect)))
        d = diff / pooled if pooled > 0 else 0
        print(f"    Δ = {diff:+.4f},  Cohen's d = {d:.3f}")

    print("\n[4] Discrimination metrics...")
    auroc, auprc = compute_auroc_auprc(nfs_arr, labels)
    print(f"    AUROC:  {auroc:.4f}")
    print(f"    AUPRC:  {auprc:.4f}")
    print(f"    (baseline AUPRC = {labels.mean():.4f})")

    print("\n[5] Selective Accuracy...")
    sel_acc = selective_accuracy(nfs_arr, labels.astype(float))
    for k, v in sel_acc.items():
        print(f"    {k}: {v:.4f}")

    print("\n[6] Per-problem Ranking...")
    ranking = per_problem_ranking(problem_scores)
    print(f"    Hit@1:  {ranking['hit_at_1']:.4f}")
    print(f"    Hit@3:  {ranking['hit_at_3']:.4f}")
    print(f"    Hit@5:  {ranking['hit_at_5']:.4f}")
    print(f"    Pairwise: {ranking['pairwise_accuracy']:.4f}")

    print("\n[7] Voting Analysis...")
    voting = voting_analysis(problem_scores)
    print(f"    Majority:  {voting['majority_accuracy']:.4f}")
    print(f"    Weighted:  {voting['weighted_accuracy']:.4f}")
    print(f"    Top-1:     {voting['top1_accuracy']:.4f}")

    comparison = compare_with_slice_baseline(
        nfs_arr, labels, auroc, auprc, sel_acc, ranking, voting, method
    )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    output_file = BATCH_DIR / f"nfs_segment_{method}.json"
    output = {
        "method": method,
        "formula": "NFS* = 100 * (B * H * (1-D*))^(1/3)",
        "granularity": "segment",
        "nfs_distribution": {
            "mean": float(nfs_arr.mean()),
            "std": float(nfs_arr.std()),
            "min": float(nfs_arr.min()),
            "max": float(nfs_arr.max()),
            "p10": float(np.percentile(nfs_arr, 10)),
            "p50": float(np.percentile(nfs_arr, 50)),
            "p90": float(np.percentile(nfs_arr, 90)),
        },
        "discrimination": {
            "auroc": auroc,
            "auprc": auprc,
            "baseline_auprc": float(labels.mean()),
            "nfs_correct_mean": float(nfs_correct.mean()) if len(nfs_correct) > 0 else None,
            "nfs_incorrect_mean": float(nfs_incorrect.mean()) if len(nfs_incorrect) > 0 else None,
        },
        "selective_accuracy": sel_acc,
        "ranking": {k: v for k, v in ranking.items() if k != "per_problem"},
        "voting": {
            "majority_accuracy": voting["majority_accuracy"],
            "weighted_accuracy": voting["weighted_accuracy"],
            "top1_accuracy": voting["top1_accuracy"],
            "n_problems": voting["n_problems"],
        },
        "comparison_vs_slice": comparison,
        "samples": all_scores,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n    Results saved to: {output_file}")
    print("=" * 70)


def run(cache_path, batch_dir, method="union", slice_nfs=None):
    """Phase 6: segment-level NFS scoring (programmatic API).

    Results saved to batch_dir/nfs_segment_{method}.json.
    """
    global BATCH_DIR, CACHE_ROOT, EVAL_REPORT, META_FILE, SLICE_NFS_FILE
    CACHE_ROOT = Path(cache_path)
    EVAL_REPORT = CACHE_ROOT / "evaluation_report_compact.json"
    META_FILE = CACHE_ROOT / "meta.json"
    BATCH_DIR = Path(batch_dir)
    if slice_nfs is not None:
        SLICE_NFS_FILE = Path(slice_nfs)
    _do_analysis(method)


def main():
    parser = argparse.ArgumentParser(description="Segment-level NFS analysis")
    parser.add_argument("--method", type=str, default="union", choices=["union", "avg"],
                        help="Distance method: union or avg")
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径或简写")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="segment batch_results 目录")
    parser.add_argument("--slice-nfs", type=str, default=None,
                        help="slice 级 nfs_analysis.json 路径（用于对比）")
    args = parser.parse_args()

    global BATCH_DIR, CACHE_ROOT, EVAL_REPORT, META_FILE, SLICE_NFS_FILE
    try:
        CACHE_ROOT = resolve_cache_path(
            args.cache,
            default_benchmark="aime24",
            default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
            required=True,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        benchmarks = list_available_benchmarks()
        if benchmarks:
            print(f"可用: {benchmarks}")
        sys.exit(1)

    EVAL_REPORT = CACHE_ROOT / "evaluation_report_compact.json"
    META_FILE = CACHE_ROOT / "meta.json"

    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
    if args.slice_nfs:
        SLICE_NFS_FILE = Path(args.slice_nfs)

    _do_analysis(args.method)


if __name__ == "__main__":
    main()
