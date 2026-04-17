#!/usr/bin/env python3
"""
批量处理 aime24 全部 30 题
为每个 sample 构建单 COT 内部图
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import resource

from project_paths import (
    default_batch_dir_for_benchmark,
    ensure_project_imports,
    list_available_benchmarks,
    resolve_cache_path,
)

ensure_project_imports()

from alignment import get_cache_reader
from nad.core.distance.engine import _jaccard_distance
from hmm_simple.core import _hmm_viterbi_2state


class MemoryTracker:
    """内存使用追踪器"""

    def __init__(self):
        self.baseline = self.current_mb()

    def current_mb(self):
        """当前内存使用 (MB)"""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def delta_mb(self):
        """相对于 baseline 的增量 (MB)"""
        return self.current_mb() - self.baseline


class BatchGraphBuilder:
    """批量图构建器"""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.reader = get_cache_reader(cache_path)

        # 加载 meta.json 获取 problem_id 映射
        meta_path = cache_path / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        # 构建 sample_id -> problem_id 映射
        self.sample_to_problem = {}
        for i, sample in enumerate(meta['samples']):
            self.sample_to_problem[i] = sample['problem_id']

        # 构建 problem_id -> [sample_ids] 映射
        self.problem_to_samples = defaultdict(list)
        for sample_id, problem_id in self.sample_to_problem.items():
            self.problem_to_samples[problem_id].append(sample_id)

        self.n_problems = len(self.problem_to_samples)
        self.n_samples = len(self.sample_to_problem)

        print(f"✓ Cache loaded: {cache_path.name}")
        print(f"  Problems: {self.n_problems}")
        print(f"  Total samples: {self.n_samples}")

    def extract_slice_keys(self, sample_id: int):
        """提取某个 sample 的所有 slice keys"""
        rows_start = self.reader.rows_sample_row_ptr[sample_id]
        rows_end = self.reader.rows_sample_row_ptr[sample_id + 1]

        slice_keys_list = []
        for slice_idx in range(rows_start, rows_end):
            key_start = self.reader.rows_row_ptr[slice_idx]
            key_end = self.reader.rows_row_ptr[slice_idx + 1]
            keys = self.reader.rows_keys[key_start:key_end]
            slice_keys_list.append(keys)

        return slice_keys_list

    def compute_entropy_confidence(self, sample_id: int, n_slices: int):
        """计算每个 slice 的熵和置信度"""
        token_start = self.reader.token_row_ptr[sample_id]
        token_end = self.reader.token_row_ptr[sample_id + 1]
        n_tokens = token_end - token_start

        tok_conf = self.reader.tok_conf[token_start:token_end]
        tok_neg_entropy = self.reader.tok_neg_entropy[token_start:token_end]

        entropy = np.zeros(n_slices, dtype=np.float32)
        confidence = np.zeros(n_slices, dtype=np.float32)

        for i in range(n_slices):
            start = i * 32
            end = min((i + 1) * 32, n_tokens)

            confidence[i] = tok_conf[start:end].mean()
            chunk_neg_entropy = tok_neg_entropy[start:end]
            entropy[i] = -chunk_neg_entropy.mean() + chunk_neg_entropy.std() * 0.1

        return entropy, confidence

    def hmm_segment(self, entropy: np.ndarray, confidence: np.ndarray, p_stay: float = 0.9):
        """HMM 分割"""
        states = _hmm_viterbi_2state(entropy, p_stay=p_stay)
        return states

    def compute_distance_matrix(self, slice_keys_list):
        """计算完整的 Jaccard 距离矩阵"""
        n = len(slice_keys_list)
        dist_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                dist = _jaccard_distance(
                    slice_keys_list[i],
                    slice_keys_list[j],
                    assume_unique=True
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def process_single_sample(self, sample_id: int, p_stay: float = 0.9):
        """处理单个 sample（简化版，不记录详细计时）"""
        # 提取 slice keys
        slice_keys_list = self.extract_slice_keys(sample_id)
        n_slices = len(slice_keys_list)

        if n_slices < 3:
            # 太短，跳过
            return None

        # 计算熵和置信度
        entropy, confidence = self.compute_entropy_confidence(sample_id, n_slices)

        # HMM 分割
        hmm_states = self.hmm_segment(entropy, confidence, p_stay)

        n_explore = int((hmm_states == 0).sum())
        n_exploit = int((hmm_states == 1).sum())
        n_transitions = int((np.diff(hmm_states) != 0).sum())

        # 计算距离矩阵
        dist_matrix = self.compute_distance_matrix(slice_keys_list)

        # 统计
        dist_finite = dist_matrix[dist_matrix > 0]

        return {
            'sample_id': sample_id,
            'problem_id': self.sample_to_problem[sample_id],
            'n_slices': n_slices,
            'n_explore': n_explore,
            'n_exploit': n_exploit,
            'n_transitions': n_transitions,
            'entropy': entropy,
            'confidence': confidence,
            'hmm_states': hmm_states,
            'distance_matrix': dist_matrix,
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

    def batch_process(self, output_dir: Path, p_stay: float = 0.9, save_matrices: bool = True):
        """批量处理所有 samples

        Args:
            output_dir: 输出目录
            p_stay: HMM 参数
            save_matrices: 是否保存距离矩阵（占空间）
        """
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n{'='*80}")
        print(f"开始批量处理")
        print(f"{'='*80}")
        print(f"输出目录: {output_dir}")
        print(f"HMM p_stay: {p_stay}")
        print(f"保存距离矩阵: {save_matrices}")

        mem_tracker = MemoryTracker()
        overall_start = time.perf_counter()

        # 按 problem 组织结果
        problem_results = {}
        sample_timings = []

        # 处理每个 problem
        for problem_idx, (problem_id, sample_ids) in enumerate(
            sorted(self.problem_to_samples.items()), start=1
        ):
            problem_start = time.perf_counter()
            print(f"\n[Problem {problem_idx}/{self.n_problems}] ID={problem_id}, Samples={len(sample_ids)}")

            problem_data = {
                'problem_id': problem_id,
                'samples': []
            }

            # 处理该 problem 的所有 samples
            for i, sample_id in enumerate(sample_ids, start=1):
                sample_start = time.perf_counter()

                try:
                    result = self.process_single_sample(sample_id, p_stay)

                    if result is None:
                        print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: SKIPPED (too short)")
                        continue

                    sample_time = time.perf_counter() - sample_start
                    sample_timings.append(sample_time)

                    # 保存距离矩阵
                    if save_matrices:
                        matrix_file = output_dir / f"dist_p{problem_id}_s{sample_id}.npy"
                        np.save(matrix_file, result['distance_matrix'])

                    # 保存 HMM 状态
                    states_file = output_dir / f"hmm_p{problem_id}_s{sample_id}.npy"
                    np.save(states_file, result['hmm_states'])

                    # 只保留统计信息到 JSON（不保存大数组）
                    sample_summary = {
                        'sample_id': sample_id,
                        'n_slices': result['n_slices'],
                        'n_explore': result['n_explore'],
                        'n_exploit': result['n_exploit'],
                        'n_transitions': result['n_transitions'],
                        'distance_stats': result['distance_stats'],
                        'keys_stats': result['keys_stats'],
                        'processing_time_s': sample_time,
                    }
                    problem_data['samples'].append(sample_summary)

                    # 进度显示（每 10 个显示一次）
                    if i % 10 == 0 or i == len(sample_ids):
                        avg_time = np.mean(sample_timings[-10:])
                        print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: "
                              f"{result['n_slices']:3d} slices, "
                              f"{sample_time:.3f}s (avg: {avg_time:.3f}s)")

                except Exception as e:
                    print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: ERROR - {e}")
                    continue

            problem_time = time.perf_counter() - problem_start
            problem_data['processing_time_s'] = problem_time
            problem_results[problem_id] = problem_data

            print(f"  ✓ Problem {problem_id} completed in {problem_time:.1f}s "
                  f"({len(problem_data['samples'])}/{len(sample_ids)} samples)")

        overall_time = time.perf_counter() - overall_start

        # 保存汇总报告
        summary = {
            'cache_name': self.cache_path.name,
            'n_problems': self.n_problems,
            'n_samples': self.n_samples,
            'p_stay': p_stay,
            'save_matrices': save_matrices,
            'total_time_s': overall_time,
            'avg_time_per_sample_s': np.mean(sample_timings),
            'median_time_per_sample_s': np.median(sample_timings),
            'memory_mb': mem_tracker.delta_mb(),
            'problems': problem_results,
        }

        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 打印汇总报告
        print(f"\n{'='*80}")
        print(f"批量处理完成！")
        print(f"{'='*80}")
        print(f"总耗时: {overall_time:.1f}s ({overall_time/60:.1f} min)")
        print(f"平均每 sample: {np.mean(sample_timings):.3f}s")
        print(f"中位数每 sample: {np.median(sample_timings):.3f}s")
        print(f"内存使用: {mem_tracker.delta_mb():.1f} MB")
        print(f"处理速率: {len(sample_timings)/overall_time:.1f} samples/s")

        # 统计信息
        total_slices = sum(
            s['n_slices']
            for p in problem_results.values()
            for s in p['samples']
        )
        total_edges = sum(
            s['n_slices'] * (s['n_slices'] - 1) // 2
            for p in problem_results.values()
            for s in p['samples']
        )

        print(f"\n【数据统计】")
        print(f"总 slices: {total_slices:,}")
        print(f"总边数: {total_edges:,}")
        print(f"成功处理: {len(sample_timings)}/{self.n_samples}")

        print(f"\n【输出文件】")
        print(f"  汇总报告: {summary_file}")
        if save_matrices:
            print(f"  距离矩阵: {len(sample_timings)} × dist_p*_s*.npy")
        print(f"  HMM 状态: {len(sample_timings)} × hmm_p*_s*.npy")

        print(f"\n{'='*80}")
        print(f"✓ 全部完成！")
        print(f"{'='*80}")

        return summary


def run(cache_path, output_dir, p_stay=0.9, save_matrices=True):
    """Phase 1: slice-level graph construction (programmatic API).

    Returns:
        dict: batch summary
    """
    cache_path, output_dir = Path(cache_path), Path(output_dir)
    builder = BatchGraphBuilder(cache_path)
    return builder.batch_process(output_dir, p_stay, save_matrices)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="批量处理 - 单 COT 内部图构建")
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径。也可用简写如 'gpqa', 'aime24' 等自动定位")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: ./batch_results_<benchmark>)")
    parser.add_argument("--p-stay", type=float, default=0.9, help="HMM p_stay 参数")
    parser.add_argument("--no-save-matrices", action="store_true", help="不保存距离矩阵")
    args = parser.parse_args()

    # 解析 cache 路径：支持简写
    try:
        cache_path = resolve_cache_path(
            args.cache,
            default_benchmark="aime24",
            default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
            required=True,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        benchmarks = list_available_benchmarks()
        if benchmarks:
            print(f"可用的 benchmarks: {benchmarks}")
        sys.exit(1)

    # 从 cache 路径推断 benchmark 名
    benchmark = cache_path.parent.name
    output_dir = Path(args.output) if args.output else default_batch_dir_for_benchmark(benchmark)
    save_matrices = not args.no_save_matrices

    print("="*80)
    print(f"{benchmark} 批量处理 - 单 COT 内部图构建")
    print("="*80)
    print(f"\nCache: {cache_path}")
    print(f"Output: {output_dir}")

    # 批量处理
    builder = BatchGraphBuilder(cache_path)
    summary = builder.batch_process(output_dir, args.p_stay, save_matrices)

    print(f"\n结果已保存至: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
