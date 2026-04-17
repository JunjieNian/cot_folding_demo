#!/usr/bin/env python3
"""
分析 Exploration vs Exploitation 在距离空间中的聚类特性

验证假设：
- Explore-Explore 配对距离较小（更相似）
- Exploit-Exploit 配对距离较小（更相似）
- Cross（Explore-Exploit）配对距离较大（较不相似）
"""

import numpy as np
from pathlib import Path
import scipy.stats as stats

from project_paths import REPO_ROOT

BATCH_DIR = REPO_ROOT / "batch_results"


def analyze_state_clustering(problem_id: int, sample_id: int):
    """分析 HMM 状态在距离空间中的聚类"""

    # 加载数据
    dist_file = BATCH_DIR / f"dist_p{problem_id}_s{sample_id}.npy"
    hmm_file = BATCH_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"

    dist_matrix = np.load(dist_file)
    hmm_states = np.load(hmm_file)
    n = len(hmm_states)

    # 分类所有配对
    explore_explore = []  # 0-0
    exploit_exploit = []  # 1-1
    cross_pairs = []      # 0-1 or 1-0

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

    # 转为 numpy 数组
    ee = np.array(explore_explore)
    xx = np.array(exploit_exploit)
    cx = np.array(cross_pairs)

    print("=" * 80)
    print(f"State Clustering Analysis — Problem {problem_id}, Sample {sample_id}")
    print("=" * 80)
    print(f"Total slices: {n}")
    print(f"  Exploration: {(hmm_states == 0).sum()} slices")
    print(f"  Exploitation: {(hmm_states == 1).sum()} slices")
    print()

    print("=" * 80)
    print("PAIRWISE DISTANCE STATISTICS")
    print("=" * 80)

    # 统计每类配对数量
    print(f"\nNumber of pairs:")
    print(f"  Explore-Explore:  {len(ee):6d}  ({len(ee)/(n*(n-1)/2)*100:5.2f}%)")
    print(f"  Exploit-Exploit:  {len(xx):6d}  ({len(xx)/(n*(n-1)/2)*100:5.2f}%)")
    print(f"  Cross (E-X):      {len(cx):6d}  ({len(cx)/(n*(n-1)/2)*100:5.2f}%)")
    print(f"  Total:            {len(ee)+len(xx)+len(cx):6d}")

    # 统计每类的距离分布
    print(f"\n{'Pair Type':<20} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    def print_stats(name, arr):
        print(f"{name:<20} {arr.mean():<10.4f} {np.median(arr):<10.4f} "
              f"{arr.std():<10.4f} {arr.min():<10.4f} {arr.max():<10.4f}")

    print_stats("Explore-Explore", ee)
    print_stats("Exploit-Exploit", xx)
    print_stats("Cross (E-X)", cx)
    print_stats("All pairs", np.concatenate([ee, xx, cx]))

    # 计算效应量
    print("\n" + "=" * 80)
    print("EFFECT SIZE (mean difference)")
    print("=" * 80)

    # 核心发现：比较 within-state 和 cross-state
    within_state = np.concatenate([ee, xx])

    print(f"\nWithin-state (E-E + X-X) mean:  {within_state.mean():.4f}")
    print(f"Cross-state (E-X) mean:         {cx.mean():.4f}")
    print(f"Difference (Cross - Within):    {cx.mean() - within_state.mean():.4f}")
    print(f"  → Cross pairs are {(cx.mean() - within_state.mean()) / within_state.mean() * 100:+.2f}% "
          f"more distant")

    # 细粒度比较
    print(f"\nExplore-Explore mean:           {ee.mean():.4f}")
    print(f"Exploit-Exploit mean:           {xx.mean():.4f}")
    print(f"Difference (X-X - E-E):         {xx.mean() - ee.mean():.4f}")

    if abs(xx.mean() - ee.mean()) > 0.01:
        if xx.mean() > ee.mean():
            print(f"  → Exploitation slices are MORE diverse (higher within-group distance)")
        else:
            print(f"  → Exploration slices are MORE diverse (higher within-group distance)")

    # 统计显著性检验
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Mann-Whitney U Test)")
    print("=" * 80)

    # Test 1: Within-state vs Cross-state
    u_stat, p_val = stats.mannwhitneyu(within_state, cx, alternative='less')
    print(f"\nWithin-state vs Cross-state:")
    print(f"  U statistic: {u_stat:.2e}")
    print(f"  p-value:     {p_val:.2e}")
    if p_val < 0.001:
        print(f"  → HIGHLY SIGNIFICANT: Cross-state pairs are more distant (p < 0.001)")
    elif p_val < 0.05:
        print(f"  → SIGNIFICANT: Cross-state pairs are more distant (p < 0.05)")
    else:
        print(f"  → NOT significant (p >= 0.05)")

    # Test 2: Explore-Explore vs Exploit-Exploit
    u_stat2, p_val2 = stats.mannwhitneyu(ee, xx, alternative='two-sided')
    print(f"\nExplore-Explore vs Exploit-Exploit:")
    print(f"  U statistic: {u_stat2:.2e}")
    print(f"  p-value:     {p_val2:.2e}")
    if p_val2 < 0.001:
        print(f"  → HIGHLY SIGNIFICANT difference in within-group cohesion")
    elif p_val2 < 0.05:
        print(f"  → SIGNIFICANT difference")
    else:
        print(f"  → NOT significant (similar cohesion)")

    # Test 3: Each within-state vs cross-state
    u_stat3, p_val3 = stats.mannwhitneyu(ee, cx, alternative='less')
    u_stat4, p_val4 = stats.mannwhitneyu(xx, cx, alternative='less')

    print(f"\nExplore-Explore vs Cross:")
    print(f"  p-value: {p_val3:.2e}  {'(E-E < Cross)' if p_val3 < 0.05 else ''}")

    print(f"Exploit-Exploit vs Cross:")
    print(f"  p-value: {p_val4:.2e}  {'(X-X < Cross)' if p_val4 < 0.05 else ''}")

    # Cohen's d 效应量
    print("\n" + "=" * 80)
    print("COHEN'S d EFFECT SIZE")
    print("=" * 80)

    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (group1.mean() - group2.mean()) / pooled_std

    d_within_cross = cohens_d(within_state, cx)
    d_ee_xx = cohens_d(ee, xx)
    d_ee_cross = cohens_d(ee, cx)
    d_xx_cross = cohens_d(xx, cx)

    print(f"\nWithin-state vs Cross-state:  d = {d_within_cross:.3f}")
    print(f"  Interpretation: ", end="")
    if abs(d_within_cross) < 0.2:
        print("negligible")
    elif abs(d_within_cross) < 0.5:
        print("small")
    elif abs(d_within_cross) < 0.8:
        print("medium")
    else:
        print("large")

    print(f"\nExplore-Explore vs Exploit-Exploit:  d = {d_ee_xx:.3f}")
    print(f"Explore-Explore vs Cross:            d = {d_ee_cross:.3f}")
    print(f"Exploit-Exploit vs Cross:            d = {d_xx_cross:.3f}")

    # 百分位数分布
    print("\n" + "=" * 80)
    print("PERCENTILE DISTRIBUTION")
    print("=" * 80)

    percentiles = [10, 25, 50, 75, 90]
    print(f"\n{'Pair Type':<20} ", end="")
    for p in percentiles:
        print(f"P{p:<3d}", end="  ")
    print()
    print("-" * 80)

    def print_percentiles(name, arr):
        print(f"{name:<20} ", end="")
        for p in percentiles:
            print(f"{np.percentile(arr, p):.3f}  ", end="")
        print()

    print_percentiles("Explore-Explore", ee)
    print_percentiles("Exploit-Exploit", xx)
    print_percentiles("Cross (E-X)", cx)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # 综合判断
    if p_val < 0.001 and abs(d_within_cross) > 0.5:
        print("\n✓ STRONG EVIDENCE for state-based clustering:")
        print(f"  • Cross-state pairs are {(cx.mean() - within_state.mean()):.4f} more distant on average")
        print(f"  • Effect size: {abs(d_within_cross):.2f} (medium to large)")
        print(f"  • Statistical significance: p < 0.001")
        print(f"\n  → Exploration and Exploitation slices form DISTINCT CLUSTERS")
        print(f"    in the neuron activation space.")
    elif p_val < 0.05:
        print("\n✓ MODERATE EVIDENCE for state-based clustering")
    else:
        print("\n✗ NO SIGNIFICANT clustering by HMM state")

    print("=" * 80)


if __name__ == "__main__":
    analyze_state_clustering(problem_id=60, sample_id=0)
