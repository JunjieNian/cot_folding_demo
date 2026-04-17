#!/usr/bin/env python3
"""
COT 折叠图 (COT Folding Map)

类比蛋白质折叠结构图：
- 蛋白质折叠：1D 氨基酸序列在 3D 空间折叠，远距残基可能空间接近
- COT 折叠：1D token 序列在神经元激活空间"折叠"，远距 slices 可能激活模式接近

输入：HMM 分割结果 + Jaccard 距离矩阵
输出：4-panel 折叠图 (Contact Map | MDS 2D | Contact Density | Metrics)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.linalg import eigh

from project_paths import (
    default_batch_dir_for_benchmark,
    default_results_dir,
    ensure_project_imports,
    resolve_cache_path,
)

ensure_project_imports()

CACHE_PATH = resolve_cache_path(
    default_benchmark="aime24",
    default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
)
BATCH_DIR = default_batch_dir_for_benchmark("aime24")
RESULTS_DIR = default_results_dir()


def load_data(problem_id: int, sample_id: int):
    """加载距离矩阵、HMM 状态、entropy/confidence"""
    if CACHE_PATH is None:
        raise FileNotFoundError("AIME24 cache 未配置，请通过 --cache 指定。")

    # 距离矩阵和 HMM 状态
    dist_file = BATCH_DIR / f"dist_p{problem_id}_s{sample_id}.npy"
    hmm_file = BATCH_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"

    if not dist_file.exists():
        raise FileNotFoundError(f"距离矩阵文件不存在: {dist_file}")
    if not hmm_file.exists():
        raise FileNotFoundError(f"HMM 状态文件不存在: {hmm_file}")

    dist_matrix = np.load(dist_file)
    hmm_states = np.load(hmm_file)
    n_slices = len(hmm_states)

    # 计算 entropy/confidence
    from alignment import get_cache_reader
    reader = get_cache_reader(CACHE_PATH)

    token_start = reader.token_row_ptr[sample_id]
    token_end = reader.token_row_ptr[sample_id + 1]
    n_tokens = token_end - token_start

    tok_conf = reader.tok_conf[token_start:token_end]
    tok_neg_entropy = reader.tok_neg_entropy[token_start:token_end]

    entropy = np.zeros(n_slices, dtype=np.float32)
    confidence = np.zeros(n_slices, dtype=np.float32)

    for i in range(n_slices):
        start = i * 32
        end = min((i + 1) * 32, n_tokens)
        confidence[i] = tok_conf[start:end].mean()
        chunk_neg_entropy = tok_neg_entropy[start:end]
        entropy[i] = -chunk_neg_entropy.mean() + chunk_neg_entropy.std() * 0.1

    return dist_matrix, hmm_states, entropy, confidence


def classical_mds(dist_matrix, n_components=2):
    """Classical MDS (Torgerson MDS)

    双中心化距离矩阵 -> 特征值分解 -> 取前 n_components 特征向量
    """
    n = dist_matrix.shape[0]
    D_sq = dist_matrix ** 2

    # 中心化矩阵 H = I - (1/n) * 11^T
    H = np.eye(n) - np.ones((n, n)) / n

    # 双中心化: B = -0.5 * H * D^2 * H
    B = -0.5 * H @ D_sq @ H

    # 特征值分解（取最大的 n_components 个）
    eigenvalues, eigenvectors = eigh(B)

    # eigh 返回升序，取最后 n_components 个
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 截断负特征值为0
    eigenvalues = np.maximum(eigenvalues, 0)

    # 坐标 = V * sqrt(Lambda)
    coords = eigenvectors * np.sqrt(eigenvalues)

    # 计算 stress
    # Kruskal stress-1: sqrt(sum((d_ij - delta_ij)^2) / sum(d_ij^2))
    # delta_ij = MDS 空间中的欧氏距离
    mds_dist = np.zeros_like(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            mds_dist[i, j] = d
            mds_dist[j, i] = d

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    stress = np.sqrt(
        np.sum((dist_matrix[mask] - mds_dist[mask]) ** 2)
        / np.sum(dist_matrix[mask] ** 2)
    )

    return coords, stress, eigenvalues


def compute_folding_metrics(dist_matrix, hmm_states, coords, stress):
    """计算折叠指标"""
    n = len(hmm_states)
    similarity = 1.0 - dist_matrix

    # Contact 阈值：相似度 > 全局均值 + 1 std
    sim_upper = similarity[np.triu_indices(n, k=1)]
    contact_threshold = sim_upper.mean() + sim_upper.std()

    # Folding Degree: 远距接触占总接触比例
    # 远距定义：|i-j| > N/4
    long_range_threshold = n // 4
    total_contacts = 0
    long_range_contacts = 0

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] > contact_threshold:
                total_contacts += 1
                if abs(i - j) > long_range_threshold:
                    long_range_contacts += 1

    folding_degree = long_range_contacts / total_contacts if total_contacts > 0 else 0.0

    # Contact Order: 平均接触序列距离（相似度加权）
    weighted_dist_sum = 0.0
    weight_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] > contact_threshold:
                weighted_dist_sum += abs(i - j) * similarity[i, j]
                weight_sum += similarity[i, j]

    contact_order = (weighted_dist_sum / (weight_sum * n)) if weight_sum > 0 else 0.0

    # Radius of Gyration: MDS 坐标的回转半径
    centroid = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

    # HMM 统计
    n_explore = int((hmm_states == 0).sum())
    n_exploit = int((hmm_states == 1).sum())
    n_transitions = int((np.diff(hmm_states) != 0).sum())

    return {
        'n_slices': n,
        'n_explore': n_explore,
        'n_exploit': n_exploit,
        'n_transitions': n_transitions,
        'contact_threshold': float(contact_threshold),
        'total_contacts': total_contacts,
        'long_range_contacts': long_range_contacts,
        'folding_degree': float(folding_degree),
        'contact_order': float(contact_order),
        'radius_of_gyration': float(rg),
        'mds_stress': float(stress),
    }


def compute_contact_density(dist_matrix, hmm_states):
    """计算接触密度曲线：对每个序列间距 |i-j|，按 HMM 状态分类统计平均相似度"""
    n = dist_matrix.shape[0]
    similarity = 1.0 - dist_matrix

    max_gap = n - 1
    # 三类：Explore-Explore, Exploit-Exploit, Cross
    ee_sums = np.zeros(max_gap)   # explore-explore
    xx_sums = np.zeros(max_gap)   # exploit-exploit
    cx_sums = np.zeros(max_gap)   # cross
    ee_counts = np.zeros(max_gap, dtype=int)
    xx_counts = np.zeros(max_gap, dtype=int)
    cx_counts = np.zeros(max_gap, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            gap = j - i
            sim = similarity[i, j]
            si, sj = hmm_states[i], hmm_states[j]

            if si == 0 and sj == 0:  # both explore
                ee_sums[gap - 1] += sim
                ee_counts[gap - 1] += 1
            elif si == 1 and sj == 1:  # both exploit
                xx_sums[gap - 1] += sim
                xx_counts[gap - 1] += 1
            else:  # cross
                cx_sums[gap - 1] += sim
                cx_counts[gap - 1] += 1

    gaps = np.arange(1, max_gap + 1)
    ee_avg = np.divide(ee_sums, ee_counts, out=np.full(max_gap, np.nan), where=ee_counts > 0)
    xx_avg = np.divide(xx_sums, xx_counts, out=np.full(max_gap, np.nan), where=xx_counts > 0)
    cx_avg = np.divide(cx_sums, cx_counts, out=np.full(max_gap, np.nan), where=cx_counts > 0)

    return gaps, ee_avg, xx_avg, cx_avg


def plot_folding_map(dist_matrix, hmm_states, entropy, coords, stress, metrics,
                     problem_id, sample_id, output_path):
    """绘制 4-panel 折叠图"""
    n = len(hmm_states)
    similarity = 1.0 - dist_matrix

    # 颜色定义
    state_colors = {0: '#4285F4', 1: '#EA4335'}  # Explore=蓝, Exploit=红
    state_labels = {0: 'Exploration', 1: 'Exploitation'}
    node_colors = [state_colors[s] for s in hmm_states]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'COT Folding Map  —  Problem {problem_id}, Sample {sample_id}\n'
                 f'{n} slices  |  {metrics["n_explore"]} explore  |  '
                 f'{metrics["n_exploit"]} exploit  |  {metrics["n_transitions"]} transitions',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.30,
                           left=0.06, right=0.94, top=0.92, bottom=0.06)

    # ========== Panel 1: Contact Map (热力图) ==========
    ax1 = fig.add_subplot(gs[0, 0])

    im = ax1.imshow(similarity, cmap='RdYlBu_r', aspect='equal',
                    vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Contact Map (Similarity = 1 - Jaccard Distance)', fontsize=12, pad=10)
    ax1.set_xlabel('Slice Index')
    ax1.set_ylabel('Slice Index')

    # 色标
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Similarity', fontsize=10)

    # HMM 状态色带 (顶部和左侧)
    band_width = max(2, n // 40)
    for i in range(n):
        color = state_colors[hmm_states[i]]
        ax1.plot([i, i], [-band_width, -1], color=color, linewidth=1.5, solid_capstyle='butt')
        ax1.plot([-band_width, -1], [i, i], color=color, linewidth=1.5, solid_capstyle='butt')

    ax1.set_xlim(-band_width - 1, n - 0.5)
    ax1.set_ylim(n - 0.5, -band_width - 1)

    # ========== Panel 2: MDS 2D 折叠结构 ==========
    ax2 = fig.add_subplot(gs[0, 1])

    # 点大小按 entropy 缩放
    entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
    sizes = 30 + 120 * entropy_norm

    # 先画连线（按序列顺序）
    ax2.plot(coords[:, 0], coords[:, 1], color='#CCCCCC', linewidth=0.6, zorder=1, alpha=0.5)

    # 散点，按 HMM 状态着色
    for state_val in [0, 1]:
        mask = hmm_states == state_val
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    c=state_colors[state_val], s=sizes[mask],
                    label=state_labels[state_val],
                    edgecolors='white', linewidths=0.3, zorder=2, alpha=0.85)

    # 标注起点和终点
    ax2.annotate('START', coords[0], fontsize=9, fontweight='bold',
                 xytext=(8, 8), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', edgecolor='gray'))
    ax2.annotate('END', coords[-1], fontsize=9, fontweight='bold',
                 xytext=(8, -12), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCFFCC', edgecolor='gray'))

    ax2.set_title(f'MDS 2D Folding Structure (Stress = {stress:.4f})', fontsize=12, pad=10)
    ax2.set_xlabel('MDS Dimension 1')
    ax2.set_ylabel('MDS Dimension 2')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.2)

    # ========== Panel 3: Contact Density 曲线 ==========
    ax3 = fig.add_subplot(gs[1, 0])

    gaps, ee_avg, xx_avg, cx_avg = compute_contact_density(dist_matrix, hmm_states)

    # 使用滑动平均平滑曲线
    window = max(3, n // 20)
    kernel = np.ones(window) / window

    def smooth(arr):
        valid = ~np.isnan(arr)
        if valid.sum() < window:
            return arr
        out = arr.copy()
        smoothed = np.convolve(np.nan_to_num(arr, nan=0), kernel, mode='same')
        counts = np.convolve(valid.astype(float), kernel, mode='same')
        mask = counts > 0
        out[mask] = smoothed[mask] / counts[mask]
        out[~mask] = np.nan
        return out

    ee_smooth = smooth(ee_avg)
    xx_smooth = smooth(xx_avg)
    cx_smooth = smooth(cx_avg)

    ax3.plot(gaps, ee_smooth, color='#4285F4', linewidth=1.5, label='Explore-Explore', alpha=0.9)
    ax3.plot(gaps, xx_smooth, color='#EA4335', linewidth=1.5, label='Exploit-Exploit', alpha=0.9)
    ax3.plot(gaps, cx_smooth, color='#9E9E9E', linewidth=1.5, label='Cross', alpha=0.7)

    # 标注远距阈值线
    long_range_threshold = n // 4
    ax3.axvline(x=long_range_threshold, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax3.text(long_range_threshold + 2, ax3.get_ylim()[1] * 0.95 if ax3.get_ylim()[1] > 0 else 0.3,
             f'N/4={long_range_threshold}', fontsize=8, alpha=0.6)

    ax3.set_title('Contact Density by Sequence Gap', fontsize=12, pad=10)
    ax3.set_xlabel('Sequence Gap |i - j|')
    ax3.set_ylabel('Mean Similarity')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(1, n - 1)

    # ========== Panel 4: Folding Metrics (文本面板) ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # 构建指标文本
    lines = []
    lines.append(('FOLDING METRICS', '', True))
    lines.append(('', '', False))
    lines.append(('Folding Degree', f'{metrics["folding_degree"]:.4f}', False))
    lines.append(('  (long-range contacts / total contacts)', '', False))
    lines.append(('  long-range threshold |i-j| > N/4', f'= {n // 4}', False))
    lines.append(('  total contacts (sim > threshold)', f'{metrics["total_contacts"]}', False))
    lines.append(('  long-range contacts', f'{metrics["long_range_contacts"]}', False))
    lines.append(('', '', False))
    lines.append(('Contact Order', f'{metrics["contact_order"]:.4f}', False))
    lines.append(('  (weighted avg seq distance / N)', '', False))
    lines.append(('', '', False))
    lines.append(('Radius of Gyration', f'{metrics["radius_of_gyration"]:.4f}', False))
    lines.append(('  (spread in MDS space)', '', False))
    lines.append(('', '', False))
    lines.append(('MDS Stress', f'{metrics["mds_stress"]:.4f}', False))
    lines.append(('  (embedding distortion)', '', False))
    lines.append(('', '', False))
    lines.append(('Contact Threshold (sim)', f'{metrics["contact_threshold"]:.4f}', False))
    lines.append(('  (mean + 1 std of upper-tri)', '', False))
    lines.append(('', '', False))
    lines.append(('SEQUENCE STATS', '', True))
    lines.append(('', '', False))
    lines.append(('Total slices', f'{metrics["n_slices"]}', False))
    lines.append(('Exploration slices', f'{metrics["n_explore"]} ({metrics["n_explore"]/metrics["n_slices"]*100:.1f}%)', False))
    lines.append(('Exploitation slices', f'{metrics["n_exploit"]} ({metrics["n_exploit"]/metrics["n_slices"]*100:.1f}%)', False))
    lines.append(('State transitions', f'{metrics["n_transitions"]}', False))

    y = 0.95
    for label, value, is_header in lines:
        if is_header:
            ax4.text(0.05, y, label, transform=ax4.transAxes,
                     fontsize=13, fontweight='bold', fontfamily='monospace',
                     color='#333333')
        elif label == '':
            pass  # blank line
        elif label.startswith('  '):
            ax4.text(0.08, y, label, transform=ax4.transAxes,
                     fontsize=9, fontfamily='monospace', color='#888888')
            if value:
                ax4.text(0.75, y, value, transform=ax4.transAxes,
                         fontsize=9, fontfamily='monospace', color='#888888',
                         ha='left')
        else:
            ax4.text(0.08, y, label, transform=ax4.transAxes,
                     fontsize=11, fontfamily='monospace', color='#333333')
            if value:
                ax4.text(0.75, y, value, transform=ax4.transAxes,
                         fontsize=11, fontfamily='monospace', fontweight='bold',
                         color='#1A73E8', ha='left')
        y -= 0.035

    # 绘制背景框
    ax4.add_patch(plt.Rectangle((0.02, y - 0.02), 0.96, 0.95 - y + 0.04,
                                transform=ax4.transAxes,
                                facecolor='#F8F9FA', edgecolor='#DADCE0',
                                linewidth=1, zorder=-1, clip_on=False))

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    global CACHE_PATH, BATCH_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description='COT Folding Map')
    parser.add_argument('--problem_id', type=int, default=60, help='Problem ID')
    parser.add_argument('--sample_id', type=int, default=0, help='Sample ID')
    parser.add_argument('--cache', type=str, default=None, help='cache 目录路径或 benchmark 简写')
    parser.add_argument('--batch-dir', type=str, default=None, help='batch_results 目录')
    parser.add_argument('--results-dir', type=str, default=None, help='输出 results 目录')
    args = parser.parse_args()

    CACHE_PATH = resolve_cache_path(
        args.cache,
        default_benchmark="aime24",
        default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
        required=True,
    )
    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    problem_id = args.problem_id
    sample_id = args.sample_id

    print("=" * 70)
    print("COT Folding Map")
    print("=" * 70)
    print(f"Problem: {problem_id}, Sample: {sample_id}")

    # 1. 加载数据
    print("\n[1] Loading data...")
    dist_matrix, hmm_states, entropy, confidence = load_data(problem_id, sample_id)
    n = len(hmm_states)
    print(f"    Loaded: {n} slices, dist_matrix {dist_matrix.shape}")

    # 2. Classical MDS
    print("\n[2] Running Classical MDS...")
    coords, stress, eigenvalues = classical_mds(dist_matrix, n_components=2)
    print(f"    Stress: {stress:.4f}")
    print(f"    Top eigenvalues: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}")

    # 3. 计算折叠指标
    print("\n[3] Computing folding metrics...")
    metrics = compute_folding_metrics(dist_matrix, hmm_states, coords, stress)

    # 打印指标
    print(f"\n{'='*70}")
    print("FOLDING METRICS")
    print(f"{'='*70}")
    print(f"  Folding Degree:       {metrics['folding_degree']:.4f}")
    print(f"    (long-range contacts / total contacts)")
    print(f"    long-range: |i-j| > {n // 4},  threshold sim > {metrics['contact_threshold']:.4f}")
    print(f"    total contacts: {metrics['total_contacts']},  long-range: {metrics['long_range_contacts']}")
    print(f"  Contact Order:        {metrics['contact_order']:.4f}")
    print(f"  Radius of Gyration:   {metrics['radius_of_gyration']:.4f}")
    print(f"  MDS Stress:           {metrics['mds_stress']:.4f}")
    print(f"{'='*70}")
    print(f"  Slices: {metrics['n_slices']}  |  "
          f"Explore: {metrics['n_explore']}  |  "
          f"Exploit: {metrics['n_exploit']}  |  "
          f"Transitions: {metrics['n_transitions']}")
    print(f"{'='*70}")

    # 4. 绘图
    print("\n[4] Plotting folding map...")
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"folding_map_p{problem_id}_s{sample_id}.png"

    plot_folding_map(dist_matrix, hmm_states, entropy, coords, stress, metrics,
                     problem_id, sample_id, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
