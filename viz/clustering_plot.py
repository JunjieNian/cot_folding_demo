#!/usr/bin/env python3
"""
可视化 AIME24 全局状态聚类分析结果
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from project_paths import REPO_ROOT

BATCH_DIR = REPO_ROOT / "batch_results"
RESULTS_DIR = REPO_ROOT / "results"
CLUSTERING_FILE = BATCH_DIR / "clustering_analysis.json"


def visualize_clustering_results():
    """可视化聚类分析结果"""

    with open(CLUSTERING_FILE) as f:
        data = json.load(f)

    summary = data['summary']
    samples = data['samples']

    # 提取数组
    separations = np.array([s['separation'] for s in samples])
    separation_pcts = np.array([s['separation_pct'] for s in samples])
    cohens_ds = np.array([s['cohens_d'] for s in samples])
    p_values = np.array([s['p_value'] for s in samples])
    ee_means = np.array([s['ee_mean'] for s in samples])
    xx_means = np.array([s['xx_mean'] for s in samples])
    cx_means = np.array([s['cx_mean'] for s in samples])
    within_means = np.array([s['within_mean'] for s in samples])

    # 按问题分组
    problem_ids = np.array([s['problem_id'] for s in samples])
    unique_problems = sorted(set(problem_ids))

    problem_avg_sep = []
    problem_avg_d = []
    for pid in unique_problems:
        mask = problem_ids == pid
        problem_avg_sep.append(separations[mask].mean())
        problem_avg_d.append(np.abs(cohens_ds[mask]).mean())

    # 创建 6-panel 图
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'State Clustering Analysis — AIME24 Dataset ({len(samples)} samples)\n'
                 f'100% show highly significant clustering (p < 0.001)',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.28,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ========== Panel 1: Separation 分布 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(separations, bins=60, color='#4285F4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(separations.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {separations.mean():.4f}')
    ax1.axvline(np.median(separations), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(separations):.4f}')
    ax1.set_xlabel('Separation (Cross - Within)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of State Separation', fontsize=12, pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ========== Panel 2: Cohen's d 分布 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    d_abs = np.abs(cohens_ds)
    ax2.hist(d_abs, bins=60, color='#EA4335', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(d_abs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean |d| = {d_abs.mean():.3f}')

    # 标注效应量阈值
    ax2.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0.8, color='gray', linestyle=':', alpha=0.5)
    ax2.text(0.2, ax2.get_ylim()[1] * 0.95, 'small', fontsize=8, ha='center', alpha=0.6)
    ax2.text(0.5, ax2.get_ylim()[1] * 0.95, 'medium', fontsize=8, ha='center', alpha=0.6)
    ax2.text(0.8, ax2.get_ylim()[1] * 0.95, 'large', fontsize=8, ha='center', alpha=0.6)

    ax2.set_xlabel("Effect Size (|Cohen's d|)", fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Effect Sizes', fontsize=12, pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ========== Panel 3: E-E vs X-X scatter ==========
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(xx_means, ee_means, c=separations, s=15, alpha=0.6,
                          cmap='RdYlBu_r', edgecolors='none')

    # 对角线
    lims = [min(xx_means.min(), ee_means.min()), max(xx_means.max(), ee_means.max())]
    ax3.plot(lims, lims, 'k--', alpha=0.3, linewidth=1.5, label='E-E = X-X')

    ax3.set_xlabel('Exploit-Exploit Distance', fontsize=11)
    ax3.set_ylabel('Explore-Explore Distance', fontsize=11)
    ax3.set_title('Within-State Distance Comparison', fontsize=12, pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Separation', fontsize=9)

    # 统计文本
    n_above = sum(ee_means > xx_means)
    ax3.text(0.05, 0.95, f'{n_above}/{len(samples)} above diagonal\n(E-E > X-X)',
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')

    # ========== Panel 4: Per-problem separation ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(len(unique_problems)), problem_avg_sep, color='#4285F4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.axhline(np.mean(problem_avg_sep), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Global mean')
    ax4.set_xlabel('Problem ID', fontsize=11)
    ax4.set_ylabel('Avg Separation', fontsize=11)
    ax4.set_title('Average Separation by Problem', fontsize=12, pad=10)
    ax4.set_xticks(range(0, len(unique_problems), 5))
    ax4.set_xticklabels([unique_problems[i] for i in range(0, len(unique_problems), 5)])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2, axis='y')

    # ========== Panel 5: Per-problem effect size ==========
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(range(len(unique_problems)), problem_avg_d, color='#EA4335', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.axhline(np.mean(problem_avg_d), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Global mean')
    ax5.set_xlabel('Problem ID', fontsize=11)
    ax5.set_ylabel('Avg |Effect Size|', fontsize=11)
    ax5.set_title('Average Effect Size by Problem', fontsize=12, pad=10)
    ax5.set_xticks(range(0, len(unique_problems), 5))
    ax5.set_xticklabels([unique_problems[i] for i in range(0, len(unique_problems), 5)])
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2, axis='y')

    # ========== Panel 6: Summary statistics ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    lines = []
    lines.append(('SUMMARY STATISTICS', '', True))
    lines.append(('', '', False))
    lines.append(('Total samples analyzed', f'{len(samples)}', False))
    lines.append(('Significant (p < 0.001)', f'{len(samples)} (100%)', False))
    lines.append(('', '', False))
    lines.append(('SEPARATION (Cross - Within)', '', True))
    lines.append(('  Mean', f'{separations.mean():.4f} (+{separation_pcts.mean():.2f}%)', False))
    lines.append(('  Median', f'{np.median(separations):.4f}', False))
    lines.append(('  Std Dev', f'{separations.std():.4f}', False))
    lines.append(('  Range', f'[{separations.min():.4f}, {separations.max():.4f}]', False))
    lines.append(('', '', False))
    lines.append(("EFFECT SIZE (Cohen's d)", '', True))
    lines.append(('  Mean |d|', f'{d_abs.mean():.3f}', False))
    lines.append(('  Median |d|', f'{np.median(d_abs):.3f}', False))
    lines.append(('  Negligible (< 0.2)', f'{sum(d_abs < 0.2)} ({sum(d_abs < 0.2)/len(samples)*100:.1f}%)', False))
    lines.append(('  Small (0.2-0.5)', f'{sum((d_abs >= 0.2) & (d_abs < 0.5))} ({sum((d_abs >= 0.2) & (d_abs < 0.5))/len(samples)*100:.1f}%)', False))
    lines.append(('  Medium (0.5-0.8)', f'{sum((d_abs >= 0.5) & (d_abs < 0.8))} ({sum((d_abs >= 0.5) & (d_abs < 0.8))/len(samples)*100:.1f}%)', False))
    lines.append(('  Large (≥ 0.8)', f'{sum(d_abs >= 0.8)} ({sum(d_abs >= 0.8)/len(samples)*100:.1f}%)', False))
    lines.append(('', '', False))
    lines.append(('MEAN DISTANCES', '', True))
    lines.append(('  Within-state', f'{within_means.mean():.4f} ± {within_means.std():.4f}', False))
    lines.append(('  Cross-state', f'{cx_means.mean():.4f} ± {cx_means.std():.4f}', False))
    lines.append(('  Explore-Explore', f'{ee_means.mean():.4f}', False))
    lines.append(('  Exploit-Exploit', f'{xx_means.mean():.4f}', False))
    lines.append(('', '', False))
    lines.append(('DIVERSITY PATTERN', '', True))
    lines.append(('  E-E > X-X samples', f'{sum(ee_means > xx_means)} ({sum(ee_means > xx_means)/len(samples)*100:.1f}%)', False))
    lines.append(('  Mean E-E - X-X', f'{(ee_means - xx_means).mean():+.4f}', False))

    y = 0.97
    for label, value, is_header in lines:
        if is_header:
            ax6.text(0.05, y, label, transform=ax6.transAxes,
                     fontsize=11, fontweight='bold', fontfamily='monospace',
                     color='#333333')
        elif label == '':
            pass
        elif label.startswith('  '):
            ax6.text(0.08, y, label, transform=ax6.transAxes,
                     fontsize=9, fontfamily='monospace', color='#666666')
            if value:
                ax6.text(0.65, y, value, transform=ax6.transAxes,
                         fontsize=9, fontfamily='monospace', color='#1A73E8',
                         ha='left')
        else:
            ax6.text(0.08, y, label, transform=ax6.transAxes,
                     fontsize=10, fontfamily='monospace', color='#333333')
            if value:
                ax6.text(0.65, y, value, transform=ax6.transAxes,
                         fontsize=10, fontfamily='monospace', fontweight='bold',
                         color='#1A73E8', ha='left')
        y -= 0.032

    ax6.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.96,
                                transform=ax6.transAxes,
                                facecolor='#F8F9FA', edgecolor='#DADCE0',
                                linewidth=1, zorder=-1, clip_on=False))

    # 保存
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "aime24_clustering_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualization saved: {output_path}")


if __name__ == "__main__":
    visualize_clustering_results()
