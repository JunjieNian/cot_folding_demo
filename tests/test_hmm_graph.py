#!/usr/bin/env python3
"""
单 COT 内部图构建测试
基于 HMM 分割 + 完整 Jaccard 距离矩阵

测试目标：第一题的第一个 response (sample_id=0)
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
import resource

from project_paths import (
    default_results_dir,
    ensure_project_imports,
    resolve_cache_path,
)

ensure_project_imports()

from alignment import get_cache_reader
from nad.core.distance.engine import _jaccard_distance
from hmm_simple.core import _hmm_viterbi_2state


class MemoryTracker:
    """内存使用追踪器（使用 resource 模块）"""

    def __init__(self):
        self.baseline = self.current_mb()

    def current_mb(self):
        """当前内存使用 (MB)"""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def delta_mb(self):
        """相对于 baseline 的增量 (MB)"""
        return self.current_mb() - self.baseline

    def reset(self):
        """重置 baseline"""
        self.baseline = self.current_mb()


class IntraCOTGraphBuilder:
    """单 COT 内部图构建器"""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.reader = get_cache_reader(cache_path)
        print(f"✓ Cache loaded: {cache_path.name}")

    def extract_slice_keys(self, sample_id: int):
        """提取某个 sample 的所有 slice keys

        Returns:
            List[np.ndarray]: 每个 slice 的 keys 数组
        """
        # 获取该 sample 的 slices 范围
        rows_start = self.reader.rows_sample_row_ptr[sample_id]
        rows_end = self.reader.rows_sample_row_ptr[sample_id + 1]
        n_slices = rows_end - rows_start

        slice_keys_list = []
        for slice_idx in range(rows_start, rows_end):
            # 提取该 slice 的 keys
            key_start = self.reader.rows_row_ptr[slice_idx]
            key_end = self.reader.rows_row_ptr[slice_idx + 1]
            keys = self.reader.rows_keys[key_start:key_end]
            slice_keys_list.append(keys)

        return slice_keys_list

    def compute_entropy_confidence(self, sample_id: int, n_slices: int):
        """计算每个 slice 的熵和置信度

        Args:
            sample_id: sample ID
            n_slices: slice 数量

        Returns:
            entropy: np.ndarray of shape (n_slices,)
            confidence: np.ndarray of shape (n_slices,)
        """
        # 获取 token 范围
        token_start = self.reader.token_row_ptr[sample_id]
        token_end = self.reader.token_row_ptr[sample_id + 1]
        n_tokens = token_end - token_start

        # 获取 token-level 数据
        tok_conf = self.reader.tok_conf[token_start:token_end]
        tok_neg_entropy = self.reader.tok_neg_entropy[token_start:token_end]

        # 按 32-token 聚合
        entropy = np.zeros(n_slices, dtype=np.float32)
        confidence = np.zeros(n_slices, dtype=np.float32)

        for i in range(n_slices):
            start = i * 32
            end = min((i + 1) * 32, n_tokens)

            # 置信度：平均值
            confidence[i] = tok_conf[start:end].mean()

            # 熵：-tok_neg_entropy 的平均值 + 添加噪声（基于 std）
            chunk_neg_entropy = tok_neg_entropy[start:end]
            entropy[i] = -chunk_neg_entropy.mean() + chunk_neg_entropy.std() * 0.1

        return entropy, confidence

    def hmm_segment(self, entropy: np.ndarray, confidence: np.ndarray, p_stay: float = 0.9):
        """HMM 分割

        Returns:
            states: np.ndarray of shape (n_slices,), values in {0, 1}
                0 = Exploration (高熵)
                1 = Exploitation (低熵)
        """
        states = _hmm_viterbi_2state(entropy, p_stay=p_stay)
        return states

    def compute_distance_matrix(self, slice_keys_list):
        """计算完整的 Jaccard 距离矩阵

        Args:
            slice_keys_list: List of key arrays

        Returns:
            dist_matrix: np.ndarray of shape (n, n)
        """
        n = len(slice_keys_list)
        dist_matrix = np.zeros((n, n), dtype=np.float32)

        # 只计算上三角矩阵（对称）
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = _jaccard_distance(
                    slice_keys_list[i],
                    slice_keys_list[j],
                    assume_unique=True
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                n_pairs += 1

        return dist_matrix, n_pairs

    def build_graph(self, sample_id: int, p_stay: float = 0.9):
        """完整流程：构建单 COT 内部图

        Returns:
            dict with keys:
                - sample_id
                - n_slices
                - entropy, confidence
                - hmm_states
                - slice_keys_list
                - distance_matrix
                - stats
                - timing
        """
        mem = MemoryTracker()
        timing = {}

        # Step 1: 提取 slice keys
        print(f"\n[Step 1] Extracting slice keys...")
        t0 = time.perf_counter()
        slice_keys_list = self.extract_slice_keys(sample_id)
        n_slices = len(slice_keys_list)
        timing['extract_keys'] = time.perf_counter() - t0
        mem_after_keys = mem.delta_mb()
        print(f"  ✓ Extracted {n_slices} slices in {timing['extract_keys']:.3f}s")
        print(f"  Memory: +{mem_after_keys:.1f} MB")

        # Step 2: 计算熵和置信度
        print(f"\n[Step 2] Computing entropy and confidence...")
        t0 = time.perf_counter()
        entropy, confidence = self.compute_entropy_confidence(sample_id, n_slices)
        timing['compute_entropy_conf'] = time.perf_counter() - t0
        print(f"  ✓ Computed in {timing['compute_entropy_conf']:.3f}s")
        print(f"  Entropy: min={entropy.min():.3f}, max={entropy.max():.3f}, mean={entropy.mean():.3f}")
        print(f"  Confidence: min={confidence.min():.3f}, max={confidence.max():.3f}, mean={confidence.mean():.3f}")

        # Step 3: HMM 分割
        print(f"\n[Step 3] HMM segmentation...")
        t0 = time.perf_counter()
        hmm_states = self.hmm_segment(entropy, confidence, p_stay)
        timing['hmm_segment'] = time.perf_counter() - t0

        n_explore = int((hmm_states == 0).sum())
        n_exploit = int((hmm_states == 1).sum())
        n_transitions = int((np.diff(hmm_states) != 0).sum())

        print(f"  ✓ Segmented in {timing['hmm_segment']:.3f}s")
        print(f"  Exploration slices: {n_explore} ({n_explore/n_slices*100:.1f}%)")
        print(f"  Exploitation slices: {n_exploit} ({n_exploit/n_slices*100:.1f}%)")
        print(f"  State transitions: {n_transitions}")

        # Step 4: 计算距离矩阵
        print(f"\n[Step 4] Computing full Jaccard distance matrix...")
        n_pairs = n_slices * (n_slices - 1) // 2
        print(f"  Total pairs to compute: {n_pairs:,}")

        t0 = time.perf_counter()
        dist_matrix, computed_pairs = self.compute_distance_matrix(slice_keys_list)
        timing['compute_distances'] = time.perf_counter() - t0
        mem_after_dist = mem.delta_mb()

        print(f"  ✓ Computed {computed_pairs:,} distances in {timing['compute_distances']:.3f}s")
        print(f"  Average: {timing['compute_distances']/computed_pairs*1000:.3f} ms/pair")
        print(f"  Memory: +{mem_after_dist:.1f} MB (total: +{mem.delta_mb():.1f} MB)")

        # 统计
        dist_finite = dist_matrix[dist_matrix > 0]
        stats = {
            'n_slices': n_slices,
            'n_pairs': computed_pairs,
            'n_explore': n_explore,
            'n_exploit': n_exploit,
            'n_transitions': n_transitions,
            'distance_stats': {
                'min': float(dist_finite.min()) if dist_finite.size > 0 else 0.0,
                'max': float(dist_finite.max()) if dist_finite.size > 0 else 0.0,
                'mean': float(dist_finite.mean()) if dist_finite.size > 0 else 0.0,
                'median': float(np.median(dist_finite)) if dist_finite.size > 0 else 0.0,
            },
            'keys_stats': {
                'min_keys': int(min(len(k) for k in slice_keys_list)),
                'max_keys': int(max(len(k) for k in slice_keys_list)),
                'mean_keys': float(np.mean([len(k) for k in slice_keys_list])),
            }
        }

        timing['total'] = sum(timing.values())

        return {
            'sample_id': sample_id,
            'n_slices': n_slices,
            'entropy': entropy,
            'confidence': confidence,
            'hmm_states': hmm_states,
            'slice_keys_list': slice_keys_list,
            'distance_matrix': dist_matrix,
            'stats': stats,
            'timing': timing,
            'memory_mb': mem.delta_mb(),
        }


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="单 COT 内部图构建测试")
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径或 benchmark 简写")
    parser.add_argument("--sample-id", type=int, default=0,
                        help="测试的 sample_id")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果输出目录")
    args = parser.parse_args()

    print("=" * 80)
    print("单 COT 内部图构建测试")
    print("=" * 80)

    # 配置
    cache_path = resolve_cache_path(
        args.cache,
        default_benchmark="aime24",
        default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
        required=True,
    )
    sample_id = args.sample_id
    output_dir = Path(args.output_dir) if args.output_dir else default_results_dir()
    output_dir.mkdir(exist_ok=True)

    print(f"\nCache: {cache_path.name}")
    print(f"Sample ID: {sample_id}")
    print(f"Output: {output_dir}")

    # 构建图
    mem_tracker = MemoryTracker()
    overall_start = time.perf_counter()

    builder = IntraCOTGraphBuilder(cache_path)
    result = builder.build_graph(sample_id, p_stay=0.9)

    overall_time = time.perf_counter() - overall_start

    # 汇总报告
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    print(f"\n【总体性能】")
    print(f"  总耗时: {overall_time:.3f}s")
    print(f"  总内存: {result['memory_mb']:.1f} MB")

    print(f"\n【分步耗时】")
    for step, t in result['timing'].items():
        print(f"  {step:20s}: {t:.3f}s ({t/overall_time*100:.1f}%)")

    print(f"\n【图结构统计】")
    print(f"  节点数 (slices): {result['stats']['n_slices']}")
    print(f"  边数 (pairs): {result['stats']['n_pairs']:,}")
    print(f"  探索段: {result['stats']['n_explore']} ({result['stats']['n_explore']/result['stats']['n_slices']*100:.1f}%)")
    print(f"  利用段: {result['stats']['n_exploit']} ({result['stats']['n_exploit']/result['stats']['n_slices']*100:.1f}%)")
    print(f"  状态转移: {result['stats']['n_transitions']}")

    print(f"\n【距离统计】")
    for k, v in result['stats']['distance_stats'].items():
        print(f"  {k:8s}: {v:.4f}")

    print(f"\n【Keys 统计】")
    for k, v in result['stats']['keys_stats'].items():
        print(f"  {k:10s}: {v:.1f}")

    # 保存结果
    print(f"\n【保存结果】")

    # 1. 保存距离矩阵
    dist_file = output_dir / f"distance_matrix_sample{sample_id}.npy"
    np.save(dist_file, result['distance_matrix'])
    print(f"  ✓ Distance matrix: {dist_file}")

    # 2. 保存 HMM 状态
    states_file = output_dir / f"hmm_states_sample{sample_id}.npy"
    np.save(states_file, result['hmm_states'])
    print(f"  ✓ HMM states: {states_file}")

    # 3. 保存统计信息
    report = {
        'sample_id': sample_id,
        'cache_name': cache_path.name,
        'stats': result['stats'],
        'timing': result['timing'],
        'memory_mb': result['memory_mb'],
        'total_time_s': overall_time,
    }

    report_file = output_dir / f"report_sample{sample_id}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report: {report_file}")

    print("\n" + "=" * 80)
    print("✓ 测试完成！所有结果已保存。")
    print("=" * 80)


if __name__ == "__main__":
    main()
