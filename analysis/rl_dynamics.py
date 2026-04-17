#!/usr/bin/env python3
"""
RL 结构-能力动力学分析 (Phase 7b)

长度去偏 + 结构-能力诊断 + Top-k 投票追踪。
读取已有 nfs_analysis.json + batch_summary.json，无需重跑 Phase 1-6。

用法:
  python -m analysis.rl_dynamics                         # 全部 checkpoint
  python -m analysis.rl_dynamics --checkpoint base       # 单 checkpoint 去偏诊断
  python -m analysis.rl_dynamics --peak-step 600         # 自定义 peak step
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from project_paths import (
    REPO_ROOT,
    default_rl_batch_dir,
    list_rl_checkpoints,
)

# Reuse AUROC from fold_score
from nfs_pipeline.fold_score import compute_auroc_auprc, parse_answer

# Accuracy reference (mirrors run_rl_pipeline.RL_ACCURACY)
RL_ACCURACY = {
    "base": 28.61, "step-100": 29.48, "step-200": 30.49,
    "step-300": 31.55, "step-400": 31.68, "step-500": 32.64,
    "step-600": 33.19, "step-700": 32.99, "step-800": 32.11,
    "step-900": 32.39, "step-1000": 32.01,
}


def _rl_step(checkpoint: str) -> int:
    return 0 if checkpoint == "base" else int(checkpoint.split("-")[1])


# ═══════════════════════════════════════════════════════════════
#  1. 数据加载层
# ═══════════════════════════════════════════════════════════════

def load_checkpoint_samples(checkpoint: str) -> list[dict]:
    """从 nfs_analysis.json 读取 per-sample records.

    每条: {problem_id, sample_id, n_slices, B, H, D0, G, D_star, NFS,
           is_correct, answer}
    """
    nfs_file = default_rl_batch_dir(checkpoint) / "nfs_analysis.json"
    if not nfs_file.exists():
        print(f"  WARNING: {nfs_file} not found, skipping {checkpoint}")
        return []
    with open(nfs_file) as f:
        data = json.load(f)
    return data.get("samples", [])


def load_checkpoint_batch_summary(checkpoint: str) -> dict[int, dict]:
    """从 batch_summary.json 取 {sample_id: {n_explore, n_exploit, ...}}.

    用于计算 exploit_fraction = n_exploit / n_slices.
    """
    summary_file = default_rl_batch_dir(checkpoint) / "batch_summary.json"
    if not summary_file.exists():
        return {}
    with open(summary_file) as f:
        data = json.load(f)
    lookup: dict[int, dict] = {}
    for prob_data in data.get("problems", {}).values():
        for s in prob_data.get("samples", []):
            lookup[s["sample_id"]] = s
    return lookup


def load_all_checkpoints(
    checkpoints: list[str] | None = None,
) -> dict[str, list[dict]]:
    """加载所有可用 checkpoint，合并 exploit_fraction 到每条 sample."""
    if checkpoints is None:
        checkpoints = list_rl_checkpoints()

    all_data: dict[str, list[dict]] = {}
    for ckpt in checkpoints:
        samples = load_checkpoint_samples(ckpt)
        if not samples:
            continue
        batch_lookup = load_checkpoint_batch_summary(ckpt)
        for s in samples:
            info = batch_lookup.get(s["sample_id"], {})
            n_exploit = info.get("n_exploit", 0)
            n_slices = s.get("n_slices", 1)
            s["exploit_fraction"] = n_exploit / max(n_slices, 1)
        all_data[ckpt] = samples
    return all_data


# ═══════════════════════════════════════════════════════════════
#  2. 长度去偏 (优先级 1)
# ═══════════════════════════════════════════════════════════════

def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Simple OLS: y = alpha + beta * x. Returns (alpha, beta)."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx < 1e-12:
        return float(y_mean), 0.0
    beta = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = float(y_mean - beta * x_mean)
    return alpha, beta


def fit_length_model(samples: list[dict]) -> tuple[float, float]:
    """单 checkpoint 内拟合 log(NFS+1) = α + β·log(n_slices), OLS."""
    log_l = np.array([math.log(max(s["n_slices"], 1)) for s in samples])
    log_nfs = np.array([math.log(s["NFS"] + 1) for s in samples])
    return _ols_fit(log_l, log_nfs)


def residualize_samples(samples: list[dict], alpha: float, beta: float) -> list[dict]:
    """对 NFS, B, H, D_star 各自拟合长度模型，添加 _res 字段。

    NFS_res = log(NFS+1) - (α_nfs + β_nfs·log(n_slices))
    同理 B_res, H_res, D_star_res.
    """
    log_l = np.array([math.log(max(s["n_slices"], 1)) for s in samples])

    # Fit per-metric models
    metrics = ["NFS", "B", "H", "D_star"]
    models: dict[str, tuple[float, float]] = {}
    for m in metrics:
        vals = np.array([math.log(s[m] + 1) for s in samples])
        models[m] = _ols_fit(log_l, vals)

    # NFS model is the one passed in (for consistency)
    models["NFS"] = (alpha, beta)

    for i, s in enumerate(samples):
        ll = log_l[i]
        for m in metrics:
            a, b = models[m]
            raw = math.log(s[m] + 1)
            s[f"{m}_res"] = raw - (a + b * ll)

    return samples


def _split_correct_incorrect(samples: list[dict]):
    correct = [s for s in samples if s.get("is_correct")]
    incorrect = [s for s in samples if not s.get("is_correct")]
    return correct, incorrect


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_debiased_discrimination(samples: list[dict]) -> dict:
    """计算 raw 和 residualized 版本的 AUROC 和 Δ 指标."""
    correct, incorrect = _split_correct_incorrect(samples)

    nfs_arr = np.array([s["NFS"] for s in samples])
    labels = np.array([s["is_correct"] for s in samples], dtype=bool)

    result: dict = {}

    # Raw AUROC
    auroc_raw, _ = compute_auroc_auprc(nfs_arr, labels)
    result["auroc_raw"] = float(auroc_raw)

    # Residualized AUROC (use NFS_res if available)
    if samples and "NFS_res" in samples[0]:
        nfs_res_arr = np.array([s["NFS_res"] for s in samples])
        auroc_res, _ = compute_auroc_auprc(nfs_res_arr, labels)
        result["auroc_res"] = float(auroc_res)
    else:
        result["auroc_res"] = None

    # Δ metrics (raw + residualized)
    for metric in ["B", "H", "D_star", "NFS"]:
        raw_c = _safe_mean([s[metric] for s in correct])
        raw_i = _safe_mean([s[metric] for s in incorrect])
        result[f"delta_{metric}"] = raw_c - raw_i

        res_key = f"{metric}_res"
        if samples and res_key in samples[0]:
            res_c = _safe_mean([s[res_key] for s in correct])
            res_i = _safe_mean([s[res_key] for s in incorrect])
            result[f"delta_{metric}_res"] = res_c - res_i
        else:
            result[f"delta_{metric}_res"] = None

    return result


# ═══════════════════════════════════════════════════════════════
#  3. 结构-能力动力学 (优先级 2)
# ═══════════════════════════════════════════════════════════════

def compute_checkpoint_metrics(checkpoint: str, samples: list[dict]) -> dict:
    """单 checkpoint 的完整指标向量."""
    correct, incorrect = _split_correct_incorrect(samples)
    n = len(samples)
    n_correct = len(correct)

    # Fit length model and residualize
    alpha, beta = fit_length_model(samples)
    residualize_samples(samples, alpha, beta)
    disc = compute_debiased_discrimination(samples)

    return {
        "name": checkpoint,
        "rl_step": _rl_step(checkpoint),
        "accuracy": RL_ACCURACY.get(checkpoint),
        "n_samples": n,
        "n_correct": n_correct,
        # Raw structural means
        "B_mean": _safe_mean([s["B"] for s in samples]),
        "H_mean": _safe_mean([s["H"] for s in samples]),
        "D_star_mean": _safe_mean([s["D_star"] for s in samples]),
        "NFS_mean": _safe_mean([s["NFS"] for s in samples]),
        # Raw deltas (correct - incorrect)
        "delta_B": disc["delta_B"],
        "delta_H": disc["delta_H"],
        "delta_D": disc["delta_D_star"],
        "delta_NFS": disc["delta_NFS"],
        # Residualized deltas
        "delta_B_res": disc["delta_B_res"],
        "delta_H_res": disc["delta_H_res"],
        "delta_D_res": disc["delta_D_star_res"],
        "delta_NFS_res": disc["delta_NFS_res"],
        # Length and exploitation
        "L": _safe_mean([s["n_slices"] for s in samples]),
        "R": _safe_mean([s.get("exploit_fraction", 0) for s in samples]),
        # Discrimination
        "auroc": disc["auroc_raw"],
        "auroc_res": disc["auroc_res"],
        # Length-NFS coupling
        "beta_nfs": beta,
        "alpha_nfs": alpha,
    }


def _pearson_r(x: list[float], y: list[float]) -> dict:
    """Pearson correlation with p-value approximation (t-test)."""
    n = len(x)
    if n < 3:
        return {"r": 0.0, "p": 1.0, "n": n}
    xa = np.array(x)
    ya = np.array(y)
    xm = xa - xa.mean()
    ym = ya - ya.mean()
    ss_xx = (xm ** 2).sum()
    ss_yy = (ym ** 2).sum()
    if ss_xx < 1e-12 or ss_yy < 1e-12:
        return {"r": 0.0, "p": 1.0, "n": n}
    r = float((xm * ym).sum() / math.sqrt(ss_xx * ss_yy))
    # t-test for significance
    if abs(r) >= 1.0 - 1e-12:
        p = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        # Approximate two-tailed p using normal for small n
        p = 2.0 * math.exp(-0.717 * abs(t_stat) - 0.416 * t_stat * t_stat)
        p = min(1.0, max(0.0, p))
    return {"r": round(r, 4), "p": round(p, 4), "n": n}


def pre_post_peak_correlation(
    trajectory: list[dict],
    peak_step: int = 600,
) -> dict:
    """分 pre-peak 和 post-peak 两段，计算 accuracy 与各指标的 Pearson r."""
    pre = [t for t in trajectory if t["rl_step"] <= peak_step]
    post = [t for t in trajectory if t["rl_step"] >= peak_step]

    metrics = [
        "delta_B", "delta_H", "delta_D", "delta_NFS",
        "delta_B_res", "delta_H_res", "delta_D_res", "delta_NFS_res",
        "L", "R", "auroc", "auroc_res",
    ]

    def _correlate(subset: list[dict]) -> dict:
        acc = [t["accuracy"] for t in subset if t["accuracy"] is not None]
        result = {}
        for m in metrics:
            vals = [t.get(m) for t in subset if t.get(m) is not None and t["accuracy"] is not None]
            if len(vals) == len(acc) and len(vals) >= 3:
                result[m] = _pearson_r(acc, vals)
            else:
                result[m] = {"r": None, "p": None, "n": len(vals)}
        return result

    pre_corr = _correlate(pre)
    post_corr = _correlate(post)

    # Find leading / degradation indicators
    def _best_metric(corr_dict: dict, positive: bool = True) -> str | None:
        best_m, best_r = None, -2.0
        for m, v in corr_dict.items():
            r = v.get("r")
            if r is None:
                continue
            score = abs(r) if positive else -r  # for degradation, want most negative
            if score > best_r:
                best_r = score
                best_m = m
        return best_m

    return {
        "pre_peak": pre_corr,
        "post_peak": post_corr,
        "leading_indicator_pre": _best_metric(pre_corr, positive=True),
        "degradation_indicator_post": _best_metric(post_corr, positive=False),
    }


def problem_level_tracking(all_data: dict[str, list[dict]]) -> dict:
    """Per-problem 跨 checkpoint 追踪: 分类 h_rises_first vs d_collapses_first."""
    checkpoints = list(all_data.keys())
    if len(checkpoints) < 3:
        return {"categorization": {"h_rises_first": [], "d_collapses_first": [], "other": []}}

    # Gather per-problem means across checkpoints
    problem_trajectories: dict[str, dict[str, dict]] = defaultdict(dict)
    for ckpt, samples in all_data.items():
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_problem[str(s["problem_id"])].append(s)
        for pid, psamples in by_problem.items():
            problem_trajectories[pid][ckpt] = {
                "H_mean": _safe_mean([s["H"] for s in psamples]),
                "D_star_mean": _safe_mean([s["D_star"] for s in psamples]),
                "NFS_mean": _safe_mean([s["NFS"] for s in psamples]),
            }

    h_rises_first = []
    d_collapses_first = []
    other = []

    for pid, ckpt_data in problem_trajectories.items():
        ordered = [ckpt_data.get(c) for c in checkpoints if c in ckpt_data]
        if len(ordered) < 3:
            other.append(pid)
            continue

        # Find first checkpoint where H increases > 10% or D_star increases > 10%
        h_first_rise = None
        d_first_rise = None
        base = ordered[0]
        for i, vals in enumerate(ordered[1:], 1):
            if h_first_rise is None and vals["H_mean"] > base["H_mean"] * 1.1:
                h_first_rise = i
            if d_first_rise is None and vals["D_star_mean"] > base["D_star_mean"] * 1.1:
                d_first_rise = i

        if h_first_rise is not None and (d_first_rise is None or h_first_rise <= d_first_rise):
            h_rises_first.append(pid)
        elif d_first_rise is not None:
            d_collapses_first.append(pid)
        else:
            other.append(pid)

    return {
        "categorization": {
            "h_rises_first": h_rises_first,
            "d_collapses_first": d_collapses_first,
            "other": other,
        },
        "n_problems_tracked": len(problem_trajectories),
    }


# ═══════════════════════════════════════════════════════════════
#  4. Top-k 过滤投票分析
# ═══════════════════════════════════════════════════════════════

def topk_vote_analysis(
    all_data: dict[str, list[dict]],
    ks: list[int] | None = None,
) -> dict:
    """对每个 checkpoint + 每个 k: 用 NFS 选 top-k runs，再 majority vote."""
    if ks is None:
        ks = [5, 10, 20, 32]

    per_checkpoint: dict[str, dict] = {}
    best_k_trajectory: list[dict] = []

    for ckpt, samples in all_data.items():
        # Group by problem
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_problem[str(s["problem_id"])].append(s)

        k_results: dict[str, float] = {}
        for k in ks:
            n_correct = 0
            n_problems = 0
            for pid, psamples in by_problem.items():
                n_problems += 1
                # Sort by NFS descending, take top-k
                sorted_s = sorted(psamples, key=lambda x: -x["NFS"])[:k]
                # Majority vote on answers
                votes: dict[str, int] = defaultdict(int)
                correct_answers: dict[str, bool] = {}
                for s in sorted_s:
                    ans = str(s.get("answer", ""))
                    votes[ans] += 1
                    correct_answers[ans] = s.get("is_correct", False)
                if votes:
                    winner = max(votes, key=votes.get)
                    if correct_answers.get(winner, False):
                        n_correct += 1

            acc = n_correct / max(n_problems, 1)
            k_results[f"k{k}"] = round(acc, 4)

        per_checkpoint[ckpt] = k_results

        # Find best k for this checkpoint
        best_k = max(ks, key=lambda k_val: k_results.get(f"k{k_val}", 0))
        best_k_trajectory.append({
            "name": ckpt,
            "rl_step": _rl_step(ckpt),
            "best_k": best_k,
            "best_acc": k_results[f"k{best_k}"],
        })

    return {
        "ks": ks,
        "per_checkpoint": per_checkpoint,
        "best_k_trajectory": best_k_trajectory,
    }


# ═══════════════════════════════════════════════════════════════
#  5. 主入口 + CLI
# ═══════════════════════════════════════════════════════════════

def run_rl_dynamics(
    checkpoints: list[str] | None = None,
    output_dir: Path | str | None = None,
    peak_step: int = 600,
) -> dict:
    """完整 RL 动力学分析流程.

    1. 加载所有 checkpoint 数据
    2. 每 checkpoint 计算指标 (含去偏)
    3. Pre/post-peak 相关性
    4. Problem-level 追踪
    5. Top-k vote 分析
    6. 保存 JSON + 打印摘要
    """
    print(f"\n{'='*80}")
    print("  Phase 7b: RL Structure–Capability Dynamics")
    print(f"{'='*80}")

    # 1. Load
    print("\n[1] Loading checkpoint data...")
    all_data = load_all_checkpoints(checkpoints)
    if not all_data:
        print("  ERROR: No checkpoint data found")
        return {}
    print(f"    Loaded {len(all_data)} checkpoints: {list(all_data.keys())}")
    for ckpt, samples in all_data.items():
        n_correct = sum(1 for s in samples if s.get("is_correct"))
        print(f"    {ckpt}: {len(samples)} samples ({n_correct} correct)")

    # 2. Per-checkpoint metrics
    print("\n[2] Computing per-checkpoint metrics (with length debiasing)...")
    trajectory: list[dict] = []
    length_debiasing: dict[str, dict] = {}

    for ckpt, samples in all_data.items():
        m = compute_checkpoint_metrics(ckpt, samples)
        trajectory.append(m)
        length_debiasing[ckpt] = {
            "alpha": m["alpha_nfs"],
            "beta": m["beta_nfs"],
            "auroc_raw": m["auroc"],
            "auroc_res": m["auroc_res"],
        }

    # Sort by RL step
    trajectory.sort(key=lambda t: t["rl_step"])

    # Print trajectory table
    print(f"\n    {'Ckpt':<12} {'Step':>5} {'Acc%':>6} {'AUROC':>7} {'AUR_r':>7} "
          f"{'ΔH':>7} {'ΔH_r':>7} {'ΔD':>7} {'ΔD_r':>7} {'β_NFS':>7}")
    print(f"    {'-'*78}")
    for t in trajectory:
        acc = t.get("accuracy") or 0
        auroc = t.get("auroc") or 0
        auroc_r = t.get("auroc_res") or 0
        dh = t.get("delta_H") or 0
        dh_r = t.get("delta_H_res") or 0
        dd = t.get("delta_D") or 0
        dd_r = t.get("delta_D_res") or 0
        beta = t.get("beta_nfs") or 0
        print(f"    {t['name']:<12} {t['rl_step']:>5} {acc:>6.2f} "
              f"{auroc:>7.4f} {auroc_r:>7.4f} "
              f"{dh:>+7.4f} {dh_r:>+7.4f} {dd:>+7.4f} {dd_r:>+7.4f} "
              f"{beta:>7.3f}")

    # 3. Pre/post-peak correlation
    print(f"\n[3] Pre/post-peak correlation (peak at step-{peak_step})...")
    peak_result = pre_post_peak_correlation(trajectory, peak_step)
    lead = peak_result.get("leading_indicator_pre")
    degrade = peak_result.get("degradation_indicator_post")
    print(f"    Leading indicator (pre-peak):  {lead}")
    print(f"    Degradation indicator (post):  {degrade}")

    for phase_name, phase_data in [("pre_peak", peak_result["pre_peak"]),
                                    ("post_peak", peak_result["post_peak"])]:
        print(f"\n    {phase_name}:")
        for m, v in sorted(phase_data.items()):
            r = v.get("r")
            if r is not None:
                print(f"      {m:<18} r={r:>+7.4f}  p={v.get('p', 1):.4f}  n={v.get('n', 0)}")

    # 4. Problem-level tracking
    print(f"\n[4] Problem-level tracking...")
    prob_tracking = problem_level_tracking(all_data)
    cat = prob_tracking.get("categorization", {})
    print(f"    h_rises_first:     {len(cat.get('h_rises_first', []))}")
    print(f"    d_collapses_first: {len(cat.get('d_collapses_first', []))}")
    print(f"    other:             {len(cat.get('other', []))}")

    # 5. Top-k vote analysis
    print(f"\n[5] Top-k vote analysis...")
    topk = topk_vote_analysis(all_data)
    for entry in topk.get("best_k_trajectory", []):
        ckpt_results = topk["per_checkpoint"].get(entry["name"], {})
        k_str = "  ".join(f"k{k}={ckpt_results.get(f'k{k}', 0):.3f}" for k in topk["ks"])
        print(f"    {entry['name']:<12} {k_str}  best_k={entry['best_k']}")

    # 6. Save output
    if output_dir is None:
        output_dir = REPO_ROOT / "batch_results_rl" / "cross_checkpoint"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strip alpha_nfs from trajectory (internal detail)
    for t in trajectory:
        t.pop("alpha_nfs", None)

    output = {
        "metadata": {
            "peak_checkpoint": f"step-{peak_step}",
            "peak_step": peak_step,
            "n_checkpoints": len(trajectory),
            "checkpoints": [t["name"] for t in trajectory],
        },
        "length_debiasing": {"per_checkpoint": length_debiasing},
        "trajectory": trajectory,
        "pre_post_peak": peak_result,
        "topk_vote": topk,
        "problem_tracking": prob_tracking,
    }

    output_file = output_dir / "rl_dynamics.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n    Results saved to: {output_file}")
    print(f"{'='*80}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="RL Structure–Capability Dynamics Analysis (Phase 7b)",
    )
    parser.add_argument(
        "--checkpoint", nargs="*", default=None,
        help="Checkpoint(s) to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: batch_results_rl/cross_checkpoint)",
    )
    parser.add_argument(
        "--peak-step", type=int, default=600,
        help="RL step considered peak accuracy (default: 600)",
    )
    args = parser.parse_args()

    run_rl_dynamics(
        checkpoints=args.checkpoint,
        output_dir=args.output_dir,
        peak_step=args.peak_step,
    )


if __name__ == "__main__":
    main()
