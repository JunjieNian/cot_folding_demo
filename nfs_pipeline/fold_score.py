#!/usr/bin/env python3
"""
Native Fold Score (NFS) — 基于结构原语的 CoT 折叠质量评分

定义：
  B = s_core · f_core              (Backbone: 主核致密度)
  H = mean_e[ (s_e - τ)/(1-τ) · g_e/(n-1) ]  (Hydrogen: 长程回返强度)
  D = Σ drift_slices / Σ explore_slices        (Drift: 未回并漂移比例)
  NFS = 100 · (B · H · (1-D))^(1/3)

外部验证：
  - AUROC / AUPRC
  - Selective Accuracy (top 10/20/30%)
  - Hit@1 / Hit@k (per-problem ranking)
  - Pairwise ranking accuracy
  - Fold-weighted voting vs majority voting

用法：
  python native_fold_score.py              # 全量分析
  python native_fold_score.py --problem_id 60  # 单题分析
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from project_paths import (
    REPO_ROOT,
    default_batch_dir_for_benchmark,
    list_available_benchmarks,
    resolve_cache_path,
)

# 默认值（可被 --cache / --batch-dir 覆盖）
BATCH_DIR = REPO_ROOT / "batch_results"
PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"
CACHE_ROOT = None
EVAL_REPORT = None
META_FILE = None


# ═══════════════════════════════════════════════════════════════
#  NFS 计算
# ═══════════════════════════════════════════════════════════════

def compute_nfs(sample):
    """从 primitives 单样本 dict 计算 B, H, D*, NFS*。

    D* = 1 - G·(1-D0)
      D0: 连续回核程度 + 时间加权的未收束漂移
      G:  final closure gate = (1+C)/2
    """
    n = sample["n_slices"]
    tau = sample["contact_threshold"]

    # ── B: Backbone ──
    core = sample["core"]
    s_core = core["internal_similarity"]
    f_core = core["fraction_of_exploit"]
    B = s_core * f_core

    # ── H: Hydrogen (return edge strength) ──
    edges = sample["return_edges"]
    if len(edges) > 0:
        contributions = []
        for e in edges:
            s_e = e["similarity"]
            g_e = e["gap"]
            r_e = ((s_e - tau) / (1.0 - tau + 1e-9)) * (g_e / (n - 1))
            contributions.append(r_e)
        H = float(np.mean(contributions))
    else:
        H = 0.0

    # ── D*: Unresolved drift with closure gate ──
    branches = sample["drift_branches"]
    if len(branches) > 0:
        # D0: continuous drift with time weighting
        numerator = 0.0
        denominator = 0.0
        for b in branches:
            length = b["length"]
            t_b = (b["start"] + b["end"]) / (2.0 * n)  # normalized midpoint
            w_b = 1.0 + t_b                              # time weight: 1~2
            m_b = b["max_sim_to_core"]
            r_b = min(1.0, m_b / tau) if tau > 0 else 0.0  # return-to-core ratio
            numerator += w_b * length * (1.0 - r_b)
            denominator += w_b * length
        D0 = numerator / denominator if denominator > 0 else 0.0
    else:
        D0 = 0.0

    # G: final closure gate
    fc = sample.get("final_closure")
    if fc and fc.get("closure_coefficient") is not None:
        C = fc["closure_coefficient"]
    else:
        C = 1.0  # no closure info → assume closed
    G = (1.0 + C) / 2.0  # G ∈ [0.5, 1.0]

    # D* = 1 - G·(1-D0)
    D_star = 1.0 - G * (1.0 - D0)

    # ── NFS* ──
    product = B * H * (1.0 - D_star)
    if product > 0:
        NFS = 100.0 * (product ** (1.0 / 3.0))
    else:
        NFS = 0.0

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

def load_primitives():
    """加载已提取的结构原语。"""
    with open(PRIMITIVES_FILE) as f:
        data = json.load(f)
    return data["samples"]


def load_correctness():
    """加载正确率标签。

    sample_id 在 batch_summary / primitives 中是全局索引 (0-1919)，
    需要通过 meta.json 映射到 (problem_id, run_index)，再查 eval report。

    Returns:
        {(problem_id, global_sample_id): (is_correct, extracted_answer)}
    """
    if META_FILE is None or EVAL_REPORT is None:
        raise FileNotFoundError("Cache paths are not initialized; run main() with a valid --cache.")

    # 1. 从 meta.json 建立 global_sample_id → (problem_id, run_index) 映射
    with open(META_FILE) as f:
        meta = json.load(f)

    global_to_run = {}  # global_idx → (problem_id, run_index)
    for idx, s in enumerate(meta["samples"]):
        global_to_run[idx] = (s["problem_id"], s["run_index"])

    # 2. 从 eval report 建立 (problem_id, run_index) → (is_correct, answer)
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

    # 3. 合并：(problem_id, global_sample_id) → (is_correct, answer)
    mapping = {}
    for global_idx, (pid, run_idx) in global_to_run.items():
        if (pid, run_idx) in run_to_result:
            mapping[(pid, global_idx)] = run_to_result[(pid, run_idx)]

    return mapping


def parse_answer(extracted_answer):
    """从 extracted_answer 字段提取可比较的答案字符串。"""
    # 格式通常是 "[204, '204']" 或类似
    if not extracted_answer:
        return ""
    try:
        parsed = eval(extracted_answer)  # safe: only from our own eval report
        if isinstance(parsed, list) and len(parsed) >= 2:
            return str(parsed[1]).strip()
        return str(parsed).strip()
    except:
        return str(extracted_answer).strip()


# ═══════════════════════════════════════════════════════════════
#  验证指标
# ═══════════════════════════════════════════════════════════════

def compute_auroc_auprc(scores, labels):
    """手动计算 AUROC 和 AUPRC（不依赖 sklearn）。"""
    # 按分数降序排列
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # AUROC via trapezoidal rule on ROC
    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auroc = 0.0

    # AUPRC
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

        precision = tp / (tp + fp)
        recall = tp / n_pos
        precisions.append(precision)
        recalls.append(recall)

    # AUPRC via trapezoidal on PR curve
    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0

    return float(auroc), float(auprc)


def selective_accuracy(scores, labels, fractions=(0.1, 0.2, 0.3, 0.5)):
    """按分数排序，取 top fraction，计算准确率。"""
    order = np.argsort(-scores)
    results = {}
    for frac in fractions:
        k = max(1, int(len(scores) * frac))
        top_labels = labels[order[:k]]
        results[f"top_{int(frac*100)}%"] = float(top_labels.mean())
    results["overall"] = float(labels.mean())
    return results


def per_problem_ranking(problem_scores):
    """对每道题内部做排序评测。

    Args:
        problem_scores: dict {problem_id: [(nfs, is_correct, answer), ...]}

    Returns:
        dict with hit@k, pairwise accuracy
    """
    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []
    pairwise_correct = 0
    pairwise_total = 0

    for pid, runs in sorted(problem_scores.items()):
        # 按 NFS 降序
        runs_sorted = sorted(runs, key=lambda x: -x[0])
        labels = [r[1] for r in runs_sorted]

        # Hit@k
        hit_at_1.append(1.0 if labels[0] else 0.0)
        hit_at_3.append(1.0 if any(labels[:3]) else 0.0)
        hit_at_5.append(1.0 if any(labels[:5]) else 0.0)

        # Pairwise: 对每一对 (correct, incorrect)，看 correct 的分数是否更高
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
    """Majority voting vs NFS-weighted voting vs Top-1 NFS。

    Args:
        problem_scores: dict {problem_id: [(nfs, is_correct, answer), ...]}
    """
    majority_correct = 0
    weighted_correct = 0
    top1_correct = 0
    n_problems = 0

    per_problem = []

    for pid, runs in sorted(problem_scores.items()):
        n_problems += 1

        # 按答案分组
        answer_groups = defaultdict(list)
        for nfs, is_correct, answer in runs:
            answer_groups[answer].append((nfs, is_correct))

        # Majority voting: 最多票数的答案
        majority_answer = max(answer_groups.keys(), key=lambda a: len(answer_groups[a]))
        majority_is_correct = answer_groups[majority_answer][0][1]

        # NFS-weighted voting: 最高 NFS 加权和的答案
        answer_weights = {}
        for ans, group in answer_groups.items():
            answer_weights[ans] = sum(nfs for nfs, _ in group)
        weighted_answer = max(answer_weights.keys(), key=lambda a: answer_weights[a])
        weighted_is_correct = answer_groups[weighted_answer][0][1]

        # Top-1 NFS: 最高 NFS 的那条 run
        best_run = max(runs, key=lambda x: x[0])
        top1_is_correct = best_run[1]

        majority_correct += majority_is_correct
        weighted_correct += weighted_is_correct
        top1_correct += top1_is_correct

        per_problem.append({
            "problem_id": pid,
            "majority_answer": majority_answer,
            "majority_correct": majority_is_correct,
            "weighted_answer": weighted_answer,
            "weighted_correct": weighted_is_correct,
            "top1_nfs": float(best_run[0]),
            "top1_correct": top1_is_correct,
            "n_distinct_answers": len(answer_groups),
            "majority_count": len(answer_groups[majority_answer]),
        })

    return {
        "majority_accuracy": majority_correct / n_problems if n_problems > 0 else 0.0,
        "weighted_accuracy": weighted_correct / n_problems if n_problems > 0 else 0.0,
        "top1_accuracy": top1_correct / n_problems if n_problems > 0 else 0.0,
        "n_problems": n_problems,
        "per_problem": per_problem,
    }


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def _do_analysis(problem_id=None):
    """Core analysis logic (uses module globals BATCH_DIR, CACHE_ROOT, etc.)."""
    print("=" * 70)
    print("Native Fold Score (NFS) Analysis")
    print(f"  cache: {CACHE_ROOT}")
    print(f"  batch: {BATCH_DIR}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] Loading primitives and correctness labels...")
    samples = load_primitives()
    correctness = load_correctness()
    print(f"    Primitives: {len(samples)} samples")
    print(f"    Correctness: {len(correctness)} labels")

    # 2. 计算 NFS
    print("\n[2] Computing NFS for each sample...")
    all_scores = []
    problem_scores = defaultdict(list)

    for s in samples:
        pid = s["problem_id"]
        sid = s["sample_id"]

        if problem_id is not None and str(pid) != str(problem_id):
            continue

        nfs_result = compute_nfs(s)
        is_correct, answer_raw = correctness.get((pid, sid), (False, ""))
        answer = parse_answer(answer_raw)

        record = {
            "problem_id": pid,
            "sample_id": sid,
            "n_slices": s["n_slices"],
            **nfs_result,
            "is_correct": is_correct,
            "answer": answer,
        }
        all_scores.append(record)
        problem_scores[pid].append((nfs_result["NFS"], is_correct, answer))

    print(f"    Computed: {len(all_scores)} samples")

    # 3. NFS 统计
    print("\n[3] NFS Distribution...")
    nfs_arr = np.array([r["NFS"] for r in all_scores])
    b_arr = np.array([r["B"] for r in all_scores])
    h_arr = np.array([r["H"] for r in all_scores])
    d0_arr = np.array([r["D0"] for r in all_scores])
    g_arr = np.array([r["G"] for r in all_scores])
    d_star_arr = np.array([r["D_star"] for r in all_scores])
    labels = np.array([r["is_correct"] for r in all_scores], dtype=bool)

    print(f"\n    {'Metric':<8} {'Mean':>8} {'Std':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
    print(f"    {'-'*48}")
    for name, arr in [("NFS", nfs_arr), ("B", b_arr), ("H", h_arr),
                      ("D0", d0_arr), ("G", g_arr), ("D*", d_star_arr)]:
        print(f"    {name:<8} {arr.mean():8.4f} {arr.std():8.4f} "
              f"{np.percentile(arr, 10):8.4f} {np.percentile(arr, 50):8.4f} "
              f"{np.percentile(arr, 90):8.4f}")

    nfs_correct = nfs_arr[labels]
    nfs_incorrect = nfs_arr[~labels]
    print(f"\n    Correct   (n={labels.sum():4d}): NFS mean={nfs_correct.mean():.4f} "
          f"± {nfs_correct.std():.4f}")
    print(f"    Incorrect (n={int((~labels).sum()):4d}): NFS mean={nfs_incorrect.mean():.4f} "
          f"± {nfs_incorrect.std():.4f}")
    if len(nfs_correct) > 0 and len(nfs_incorrect) > 0:
        diff = nfs_correct.mean() - nfs_incorrect.mean()
        pooled = np.sqrt((nfs_correct.var() * len(nfs_correct) +
                          nfs_incorrect.var() * len(nfs_incorrect)) /
                         (len(nfs_correct) + len(nfs_incorrect)))
        d = diff / pooled if pooled > 0 else 0
        print(f"    Δ = {diff:+.4f},  Cohen's d = {d:.3f}")

    # 4. AUROC / AUPRC
    print("\n[4] Discrimination metrics...")
    auroc, auprc = compute_auroc_auprc(nfs_arr, labels)
    print(f"    AUROC:  {auroc:.4f}")
    print(f"    AUPRC:  {auprc:.4f}")
    print(f"    (baseline AUPRC = {labels.mean():.4f})")

    for name, arr in [("B", b_arr), ("H", h_arr), ("1-D*", 1.0 - d_star_arr),
                       ("G", g_arr)]:
        sub_auroc, _ = compute_auroc_auprc(arr, labels)
        print(f"    AUROC({name}): {sub_auroc:.4f}")

    # 5. Selective Accuracy
    print("\n[5] Selective Accuracy...")
    sel_acc = selective_accuracy(nfs_arr, labels.astype(float))
    for k, v in sel_acc.items():
        print(f"    {k}: {v:.4f}")

    # 6. Per-problem ranking
    print("\n[6] Per-problem Ranking...")
    ranking = per_problem_ranking(problem_scores)
    print(f"    Hit@1:  {ranking['hit_at_1']:.4f}")
    print(f"    Hit@3:  {ranking['hit_at_3']:.4f}")
    print(f"    Hit@5:  {ranking['hit_at_5']:.4f}")
    print(f"    Pairwise accuracy: {ranking['pairwise_accuracy']:.4f} "
          f"({ranking['pairwise_total']} pairs)")

    # 7. Voting
    print("\n[7] Voting Analysis...")
    voting = voting_analysis(problem_scores)
    print(f"    Majority voting:      {voting['majority_accuracy']:.4f} "
          f"({int(voting['majority_accuracy'] * voting['n_problems'])}/{voting['n_problems']})")
    print(f"    NFS-weighted voting:  {voting['weighted_accuracy']:.4f} "
          f"({int(voting['weighted_accuracy'] * voting['n_problems'])}/{voting['n_problems']})")
    print(f"    Top-1 NFS:            {voting['top1_accuracy']:.4f} "
          f"({int(voting['top1_accuracy'] * voting['n_problems'])}/{voting['n_problems']})")

    # 8. Per-problem breakdown
    print(f"\n[8] Per-problem Breakdown...")
    print(f"    {'Prob':<6} {'Acc':>6} {'NFS_c':>8} {'NFS_i':>8} {'Δ':>8} "
          f"{'Maj':>5} {'Wgt':>5} {'Top1':>5}")
    print(f"    {'-'*55}")

    for pid in sorted(problem_scores.keys()):
        runs = problem_scores[pid]
        correct_nfs = [r[0] for r in runs if r[1]]
        incorrect_nfs = [r[0] for r in runs if not r[1]]
        acc = len(correct_nfs) / len(runs) if runs else 0

        nfs_c = np.mean(correct_nfs) if correct_nfs else 0
        nfs_i = np.mean(incorrect_nfs) if incorrect_nfs else 0
        delta = nfs_c - nfs_i if correct_nfs and incorrect_nfs else 0

        vp = next((v for v in voting["per_problem"] if v["problem_id"] == pid), None)
        maj = "Y" if vp and vp["majority_correct"] else "N"
        wgt = "Y" if vp and vp["weighted_correct"] else "N"
        t1 = "Y" if vp and vp["top1_correct"] else "N"

        print(f"    {pid:<6} {acc:6.3f} {nfs_c:8.4f} {nfs_i:8.4f} {delta:+8.4f} "
              f"{maj:>5} {wgt:>5} {t1:>5}")

    # 9. 保存结果
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    output_file = BATCH_DIR / "nfs_analysis.json"
    output = {
        "formula": "NFS* = 100 * (B * H * (1-D*))^(1/3)",
        "definitions": {
            "B": "s_core * f_core (backbone density)",
            "H": "mean[ (s_e - tau)/(1-tau) * g_e/(n-1) ] (hydrogen bond strength)",
            "D0": "Σ w_b·l_b·(1-r_b) / Σ w_b·l_b (continuous unresolved drift)",
            "G": "(1+C)/2, C=min(1, s_close/tau) (final closure gate)",
            "D_star": "1 - G·(1-D0) (drift with closure constraint)",
        },
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
        "samples": all_scores,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n    Results saved to: {output_file}")
    print("=" * 70)


def run(cache_path, batch_dir, problem_id=None):
    """Phase 3: NFS scoring + validation (programmatic API).

    Results saved to batch_dir/nfs_analysis.json.
    """
    global BATCH_DIR, PRIMITIVES_FILE, CACHE_ROOT, EVAL_REPORT, META_FILE
    CACHE_ROOT = Path(cache_path)
    EVAL_REPORT = CACHE_ROOT / "evaluation_report_compact.json"
    META_FILE = CACHE_ROOT / "meta.json"
    BATCH_DIR = Path(batch_dir)
    PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"
    _do_analysis(problem_id)


def main():
    parser = argparse.ArgumentParser(description="Native Fold Score analysis")
    parser.add_argument("--problem_id", type=str, help="Analyze single problem")
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径或简写 (如 'gpqa', 'aime24')")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="batch_results 目录 (默认: ./batch_results_<benchmark>)")
    args = parser.parse_args()

    global BATCH_DIR, PRIMITIVES_FILE, CACHE_ROOT, EVAL_REPORT, META_FILE
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

    if args.cache is not None:
        benchmark = CACHE_ROOT.parent.name
        BATCH_DIR = Path(args.batch_dir) if args.batch_dir else default_batch_dir_for_benchmark(benchmark)
        PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"
    elif args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
        PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"

    _do_analysis(args.problem_id)


if __name__ == "__main__":
    main()
