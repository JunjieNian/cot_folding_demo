#!/usr/bin/env python3
"""
批量分析 AIME24 全部样本的状态聚类特性

验证 Exploration vs Exploitation 聚类现象是否普遍存在
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
from tqdm import tqdm

from project_paths import REPO_ROOT

BATCH_DIR = REPO_ROOT / "batch_results"
SUMMARY_FILE = BATCH_DIR / "batch_summary.json"


def _sort_problem_keys(items):
    def key_fn(item):
        key = item[0] if isinstance(item, tuple) else item
        try:
            return (0, int(key))
        except (TypeError, ValueError):
            return (1, str(key))
    return sorted(items, key=key_fn)


def _dataset_label(batch_dir: Path) -> str:
    name = batch_dir.name
    if name == "batch_results":
        return "AIME24"
    if name.startswith("batch_results_"):
        return name.replace("batch_results_", "").upper()
    return name


def analyze_single_sample(dist_matrix, hmm_states):
    """分析单个样本的状态聚类特性

    Returns:
        dict with keys:
            - n_slices
            - n_explore, n_exploit
            - ee_mean, xx_mean, cx_mean (三类配对的平均距离)
            - within_mean, cross_mean
            - separation (cross - within)
            - cohens_d
            - p_value (Mann-Whitney U test)
    """
    n = len(hmm_states)

    # 分类配对
    explore_explore = []
    exploit_exploit = []
    cross_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_matrix[i, j]
            si, sj = hmm_states[i], hmm_states[j]

            if si == 0 and sj == 0:
                explore_explore.append(dist)
            elif si == 1 and sj == 1:
                exploit_exploit.append(dist)
            else:
                cross_pairs.append(dist)

    # 转为数组
    ee = np.array(explore_explore) if explore_explore else np.array([])
    xx = np.array(exploit_exploit) if exploit_exploit else np.array([])
    cx = np.array(cross_pairs) if cross_pairs else np.array([])

    # 防止空数组
    if len(ee) == 0 or len(xx) == 0 or len(cx) == 0:
        return None

    within_state = np.concatenate([ee, xx])

    # 统计量
    ee_mean = float(ee.mean())
    xx_mean = float(xx.mean())
    cx_mean = float(cx.mean())
    within_mean = float(within_state.mean())
    cross_mean = float(cx.mean())
    separation = cross_mean - within_mean

    # Cohen's d
    n1, n2 = len(within_state), len(cx)
    var1, var2 = within_state.var(), cx.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (within_state.mean() - cx.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Mann-Whitney U test
    try:
        _, p_value = stats.mannwhitneyu(within_state, cx, alternative='less')
    except:
        p_value = 1.0

    return {
        'n_slices': n,
        'n_explore': int((hmm_states == 0).sum()),
        'n_exploit': int((hmm_states == 1).sum()),
        'n_ee': len(ee),
        'n_xx': len(xx),
        'n_cx': len(cx),
        'ee_mean': ee_mean,
        'xx_mean': xx_mean,
        'cx_mean': cx_mean,
        'within_mean': within_mean,
        'cross_mean': cross_mean,
        'separation': separation,
        'separation_pct': separation / within_mean * 100 if within_mean > 0 else 0.0,
        'cohens_d': float(cohens_d),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'highly_significant': bool(p_value < 0.001),
    }


def batch_analyze():
    """批量分析所有样本"""

    # 加载 batch_summary 获取样本列表
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    print("=" * 80)
    print(f"Batch State Clustering Analysis — {_dataset_label(BATCH_DIR)} Dataset")
    print("=" * 80)
    print(f"Total problems: {summary['n_problems']}")
    print(f"Total samples: {summary['n_samples']}")
    print()

    # 收集所有样本的结果
    all_results = []
    problem_results = defaultdict(list)
    failed_samples = []

    # 遍历所有问题
    for problem_id, problem_data in _sort_problem_keys(summary['problems'].items()):
        problem_id_key = problem_id
        try:
            problem_id = int(problem_id)
        except (TypeError, ValueError):
            problem_id = problem_id_key
        samples = problem_data['samples']

        print(f"Processing Problem {problem_id} ({len(samples):2d} samples)...", end=" ", flush=True)

        problem_stats = []

        for sample_info in samples:
            sample_id = sample_info['sample_id']

            # 加载数据
            dist_file = BATCH_DIR / f"dist_p{problem_id}_s{sample_id}.npy"
            hmm_file = BATCH_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"

            if not dist_file.exists() or not hmm_file.exists():
                failed_samples.append((problem_id, sample_id))
                continue

            try:
                dist_matrix = np.load(dist_file)
                hmm_states = np.load(hmm_file)

                result = analyze_single_sample(dist_matrix, hmm_states)

                if result is not None:
                    result['problem_id'] = problem_id
                    result['sample_id'] = sample_id
                    all_results.append(result)
                    problem_stats.append(result)
                else:
                    failed_samples.append((problem_id, sample_id))
            except Exception as e:
                failed_samples.append((problem_id, sample_id))
                continue

        # 问题级别汇总
        if problem_stats:
            avg_sep = np.mean([r['separation'] for r in problem_stats])
            n_sig = sum(1 for r in problem_stats if r['significant'])
            print(f"✓ avg_sep={avg_sep:+.4f}, {n_sig}/{len(problem_stats)} significant")
            problem_results[problem_id] = problem_stats
        else:
            print("✗ No valid samples")

    print()
    print("=" * 80)
    print(f"Analysis complete: {len(all_results)}/{summary['n_samples']} samples")
    if failed_samples:
        print(f"Failed: {len(failed_samples)} samples")
    print("=" * 80)
    print()

    # ========== 全局统计 ==========
    separations = np.array([r['separation'] for r in all_results])
    separation_pcts = np.array([r['separation_pct'] for r in all_results])
    cohens_ds = np.array([r['cohens_d'] for r in all_results])
    p_values = np.array([r['p_value'] for r in all_results])

    n_significant = sum(1 for r in all_results if r['significant'])
    n_highly_significant = sum(1 for r in all_results if r['highly_significant'])

    within_means = np.array([r['within_mean'] for r in all_results])
    cross_means = np.array([r['cross_mean'] for r in all_results])
    ee_means = np.array([r['ee_mean'] for r in all_results])
    xx_means = np.array([r['xx_mean'] for r in all_results])

    print("GLOBAL STATISTICS (all samples)")
    print("=" * 80)
    print(f"\nSample count: {len(all_results)}")
    print(f"Significant samples (p < 0.05):   {n_significant:4d} ({n_significant/len(all_results)*100:.1f}%)")
    print(f"Highly significant (p < 0.001):   {n_highly_significant:4d} ({n_highly_significant/len(all_results)*100:.1f}%)")

    print(f"\nSeparation (Cross - Within):")
    print(f"  Mean:     {separations.mean():+.4f}  ({separation_pcts.mean():+.2f}%)")
    print(f"  Median:   {np.median(separations):+.4f}")
    print(f"  Std:      {separations.std():.4f}")
    print(f"  Min:      {separations.min():+.4f}")
    print(f"  Max:      {separations.max():+.4f}")
    print(f"  P10-P90:  [{np.percentile(separations, 10):+.4f}, {np.percentile(separations, 90):+.4f}]")

    print(f"\nEffect size (Cohen's d):")
    print(f"  Mean:     {cohens_ds.mean():.3f}")
    print(f"  Median:   {np.median(cohens_ds):.3f}")
    print(f"  Std:      {cohens_ds.std():.3f}")
    print(f"  P10-P90:  [{np.percentile(cohens_ds, 10):.3f}, {np.percentile(cohens_ds, 90):.3f}]")

    # 效应量分类
    d_abs = np.abs(cohens_ds)
    n_negligible = sum(d_abs < 0.2)
    n_small = sum((d_abs >= 0.2) & (d_abs < 0.5))
    n_medium = sum((d_abs >= 0.5) & (d_abs < 0.8))
    n_large = sum(d_abs >= 0.8)

    print(f"\n  Effect size distribution:")
    print(f"    Negligible (|d| < 0.2):  {n_negligible:4d} ({n_negligible/len(all_results)*100:.1f}%)")
    print(f"    Small (0.2 ≤ |d| < 0.5): {n_small:4d} ({n_small/len(all_results)*100:.1f}%)")
    print(f"    Medium (0.5 ≤ |d| < 0.8):{n_medium:4d} ({n_medium/len(all_results)*100:.1f}%)")
    print(f"    Large (|d| ≥ 0.8):       {n_large:4d} ({n_large/len(all_results)*100:.1f}%)")

    print(f"\nMean distances:")
    print(f"  Within-state (E-E + X-X): {within_means.mean():.4f} ± {within_means.std():.4f}")
    print(f"  Cross-state (E-X):        {cross_means.mean():.4f} ± {cross_means.std():.4f}")
    print(f"  Explore-Explore:          {ee_means.mean():.4f} ± {ee_means.std():.4f}")
    print(f"  Exploit-Exploit:          {xx_means.mean():.4f} ± {xx_means.std():.4f}")

    # 聚类一致性
    ee_vs_xx = ee_means - xx_means
    print(f"\nExplore-Explore vs Exploit-Exploit:")
    print(f"  E-E > X-X (exploration more diverse): {sum(ee_vs_xx > 0):4d} ({sum(ee_vs_xx > 0)/len(all_results)*100:.1f}%)")
    print(f"  E-E < X-X (exploitation more diverse): {sum(ee_vs_xx < 0):4d} ({sum(ee_vs_xx < 0)/len(all_results)*100:.1f}%)")
    print(f"  Mean difference (E-E - X-X):          {ee_vs_xx.mean():+.4f}")

    # ========== 问题级别统计 ==========
    print("\n" + "=" * 80)
    print("PER-PROBLEM STATISTICS")
    print("=" * 80)
    print(f"\n{'Prob':<6} {'N':<4} {'Sig%':<7} {'AvgSep':<10} {'AvgD':<10} {'E-E':<8} {'X-X':<8}")
    print("-" * 80)

    for problem_id in sorted(problem_results.keys()):
        pstats = problem_results[problem_id]
        n = len(pstats)
        n_sig = sum(1 for r in pstats if r['significant'])
        avg_sep = np.mean([r['separation'] for r in pstats])
        avg_d = np.mean([r['cohens_d'] for r in pstats])
        avg_ee = np.mean([r['ee_mean'] for r in pstats])
        avg_xx = np.mean([r['xx_mean'] for r in pstats])

        print(f"{problem_id:<6} {n:<4} {n_sig/n*100:<6.1f}% {avg_sep:+<10.4f} {avg_d:<10.3f} "
              f"{avg_ee:<8.4f} {avg_xx:<8.4f}")

    # ========== 结论 ==========
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if n_highly_significant / len(all_results) > 0.90:
        strength = "OVERWHELMING"
    elif n_highly_significant / len(all_results) > 0.75:
        strength = "VERY STRONG"
    elif n_significant / len(all_results) > 0.75:
        strength = "STRONG"
    elif n_significant / len(all_results) > 0.50:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"\n{strength} EVIDENCE for state-based clustering across {_dataset_label(BATCH_DIR)}:")
    print(f"\n  ✓ {n_highly_significant}/{len(all_results)} samples ({n_highly_significant/len(all_results)*100:.1f}%) "
          f"show highly significant clustering (p < 0.001)")
    print(f"  ✓ {n_significant}/{len(all_results)} samples ({n_significant/len(all_results)*100:.1f}%) "
          f"show significant clustering (p < 0.05)")

    print(f"\n  • Average separation: {separations.mean():+.4f} ({separation_pcts.mean():+.2f}%)")
    print(f"  • Average effect size: |d| = {np.abs(cohens_ds).mean():.3f}")

    if separations.mean() > 0.05 and np.abs(cohens_ds).mean() > 0.5:
        print(f"\n  → Exploration and Exploitation consistently form DISTINCT CLUSTERS")
        print(f"    in the neuron activation space across the entire {_dataset_label(BATCH_DIR)} dataset.")

    if ee_vs_xx.mean() > 0.01:
        print(f"\n  → Exploration slices are consistently MORE DIVERSE than Exploitation")
        print(f"    (E-E distance {ee_vs_xx.mean():.4f} higher on average)")
        print(f"    {sum(ee_vs_xx > 0)/len(all_results)*100:.1f}% of samples show this pattern.")

    # 保存详细结果
    output_file = BATCH_DIR / "clustering_analysis.json"
    output_data = {
        'summary': {
            'n_samples': len(all_results),
            'n_significant': int(n_significant),
            'n_highly_significant': int(n_highly_significant),
            'mean_separation': float(separations.mean()),
            'median_separation': float(np.median(separations)),
            'mean_separation_pct': float(separation_pcts.mean()),
            'mean_cohens_d': float(cohens_ds.mean()),
            'median_cohens_d': float(np.median(cohens_ds)),
            'mean_within': float(within_means.mean()),
            'mean_cross': float(cross_means.mean()),
            'mean_ee': float(ee_means.mean()),
            'mean_xx': float(xx_means.mean()),
            'conclusion': strength,
        },
        'samples': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Detailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch state clustering analysis")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="batch_results 目录 (默认: ./batch_results)")
    args = parser.parse_args()

    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
        SUMMARY_FILE = BATCH_DIR / "batch_summary.json"

    batch_analyze()
