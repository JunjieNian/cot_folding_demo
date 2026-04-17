#!/usr/bin/env python3
"""
Segment 级图构建：以 HMM 连续探索/利用段（segment）为节点。

两种距离计算方式：
  A (union): 每个 segment 的 key 集合 = 其所有 slice key 的并集，直接 Jaccard
  B (avg):   读取已有 slice 级 dist.npy，取 segment 内 slice 对距离的均值

输出目录：batch_results_segment/
"""

import math
import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import resource

from project_paths import (
    default_batch_dir_for_benchmark,
    default_segment_batch_dir,
    ensure_project_imports,
    list_available_benchmarks,
    resolve_cache_path,
)

ensure_project_imports()

from alignment import get_cache_reader
from nad.core.distance.engine import _jaccard_distance
from hmm_simple.core import _hmm_viterbi_2state
from nfs_pipeline.primitives import find_contiguous_blocks


class MemoryTracker:
    """内存使用追踪器"""

    def __init__(self):
        self.baseline = self.current_mb()

    def current_mb(self):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def delta_mb(self):
        return self.current_mb() - self.baseline


class SegmentBatchBuilder:
    """Segment 级批量图构建器"""

    def __init__(self, cache_path: Path, slice_batch_dir: Path = None):
        self.cache_path = Path(cache_path)
        self.reader = get_cache_reader(cache_path)

        # 加载 meta.json
        meta_path = cache_path / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.sample_to_problem = {}
        for i, sample in enumerate(meta['samples']):
            self.sample_to_problem[i] = sample['problem_id']

        self.problem_to_samples = defaultdict(list)
        for sample_id, problem_id in self.sample_to_problem.items():
            self.problem_to_samples[problem_id].append(sample_id)

        self.n_problems = len(self.problem_to_samples)
        self.n_samples = len(self.sample_to_problem)

        # slice 级 batch_results 目录（用于方法 B）
        self.slice_batch_dir = slice_batch_dir

        print(f"✓ Cache loaded: {cache_path.name}")
        print(f"  Problems: {self.n_problems}")
        print(f"  Total samples: {self.n_samples}")
        if slice_batch_dir:
            print(f"  Slice batch dir: {slice_batch_dir}")

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

    def hmm_segment(self, entropy: np.ndarray, p_stay: float = 0.9):
        """HMM 分割"""
        return _hmm_viterbi_2state(entropy, p_stay=p_stay)

    def build_segments(self, hmm_states):
        """从 HMM 状态构建 segments（连续块列表）。

        Returns:
            list of dict: [{id, state, start, end, n_slices}, ...]
        """
        segments = []
        seg_id = 0
        n = len(hmm_states)
        i = 0
        while i < n:
            state = int(hmm_states[i])
            j = i
            while j < n and hmm_states[j] == state:
                j += 1
            segments.append({
                "id": seg_id,
                "state": state,
                "start": int(i),
                "end": int(j),
                "n_slices": j - i,
            })
            seg_id += 1
            i = j
        return segments

    def adaptive_split_segments(
        self,
        segments: list[dict],
        slice_keys_list: list,
        slice_dist_matrix=None,
        *,
        max_slices_per_segment: int = 8,
        variance_threshold: float = 0.15,
    ) -> list[dict]:
        """二次切分过粗 segment.

        1. SIZE: n_slices > max_slices_per_segment → 等分为 ceil(n/L_max) 段
        2. VARIANCE: 段内 self-similarity 方差 > threshold → 在最大不相似点二分

        每个子段继承父段 HMM state，添加 parent_segment_id 字段.
        """
        new_segments: list[dict] = []
        next_id = 0

        for seg in segments:
            parent_id = seg["id"]
            start, end = seg["start"], seg["end"]
            n = seg["n_slices"]

            # Determine sub-segments by size rule
            if n <= max_slices_per_segment:
                sub_ranges = [(start, end)]
            else:
                n_parts = math.ceil(n / max_slices_per_segment)
                part_size = math.ceil(n / n_parts)
                sub_ranges = []
                for k in range(n_parts):
                    s = start + k * part_size
                    e = min(start + (k + 1) * part_size, end)
                    if s < e:
                        sub_ranges.append((s, e))

            # Further split by variance if we have a distance matrix
            final_ranges: list[tuple[int, int]] = []
            for s, e in sub_ranges:
                if slice_dist_matrix is not None and (e - s) >= 4:
                    # Compute intra-segment variance
                    sub_mat = slice_dist_matrix[s:e, s:e]
                    upper = sub_mat[np.triu_indices(e - s, k=1)]
                    if len(upper) > 0 and float(np.var(upper)) > variance_threshold:
                        # Binary split at point of maximum dissimilarity
                        best_split, best_score = s + 2, -1.0
                        for sp in range(s + 2, e - 1):
                            left = slice_dist_matrix[s:sp, s:sp]
                            right = slice_dist_matrix[sp:e, sp:e]
                            cross = slice_dist_matrix[s:sp, sp:e]
                            score = float(cross.mean()) - 0.5 * (
                                float(left[np.triu_indices(sp - s, k=1)].mean())
                                + float(right[np.triu_indices(e - sp, k=1)].mean())
                            ) if left.shape[0] > 1 and right.shape[0] > 1 else 0.0
                            if score > best_score:
                                best_score = score
                                best_split = sp
                        final_ranges.append((s, best_split))
                        final_ranges.append((best_split, e))
                        continue
                final_ranges.append((s, e))

            for s, e in final_ranges:
                new_segments.append({
                    "id": next_id,
                    "state": seg["state"],
                    "start": int(s),
                    "end": int(e),
                    "n_slices": e - s,
                    "parent_segment_id": parent_id,
                })
                next_id += 1

        return new_segments

    def compute_segment_dist_union(self, slice_keys_list, segments):
        """方法 A：每个 segment 的 key 集合 = 其所有 slice key 的并集，直接 Jaccard。"""
        n_seg = len(segments)

        # 构建每个 segment 的 union key set
        seg_keys = []
        for seg in segments:
            union_keys = np.array([], dtype=slice_keys_list[0].dtype)
            for s_idx in range(seg["start"], seg["end"]):
                union_keys = np.union1d(union_keys, slice_keys_list[s_idx])
            seg_keys.append(union_keys)

        # 计算 segment 对距离
        dist_matrix = np.zeros((n_seg, n_seg), dtype=np.float32)
        for i in range(n_seg):
            for j in range(i + 1, n_seg):
                dist = _jaccard_distance(seg_keys[i], seg_keys[j], assume_unique=True)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def compute_segment_dist_avg(self, slice_dist_matrix, segments):
        """方法 B：取 segment 间 slice 对距离的均值。"""
        n_seg = len(segments)
        dist_matrix = np.zeros((n_seg, n_seg), dtype=np.float32)

        for i in range(n_seg):
            si = segments[i]
            rows_i = np.arange(si["start"], si["end"])
            for j in range(i + 1, n_seg):
                sj = segments[j]
                rows_j = np.arange(sj["start"], sj["end"])
                sub = slice_dist_matrix[np.ix_(rows_i, rows_j)]
                avg_dist = float(sub.mean())
                dist_matrix[i, j] = avg_dist
                dist_matrix[j, i] = avg_dist

        return dist_matrix

    def process_single_sample(
        self, sample_id: int, problem_id, p_stay: float = 0.9,
        adaptive: bool = False,
        max_slices_per_segment: int = 8,
        variance_threshold: float = 0.15,
    ):
        """处理单个 sample，返回 segment 级结果。"""
        # 提取 slice keys
        slice_keys_list = self.extract_slice_keys(sample_id)
        n_slices = len(slice_keys_list)

        if n_slices < 3:
            return None

        # 计算熵
        entropy, confidence = self.compute_entropy_confidence(sample_id, n_slices)

        # HMM 分割
        hmm_states = self.hmm_segment(entropy, p_stay)

        # 构建 segments
        segments = self.build_segments(hmm_states)
        n_segments_original = len(segments)

        if n_segments_original < 2:
            return None

        # Load slice distance matrix if available (needed for both avg and adaptive)
        slice_dist_matrix = None
        if self.slice_batch_dir is not None:
            slice_dist_file = self.slice_batch_dir / f"dist_p{problem_id}_s{sample_id}.npy"
            if slice_dist_file.exists():
                slice_dist_matrix = np.load(slice_dist_file)

        # Adaptive splitting
        if adaptive:
            segments = self.adaptive_split_segments(
                segments, slice_keys_list, slice_dist_matrix,
                max_slices_per_segment=max_slices_per_segment,
                variance_threshold=variance_threshold,
            )

        n_segments = len(segments)

        # 方法 A：union Jaccard
        dist_union = self.compute_segment_dist_union(slice_keys_list, segments)

        # 方法 B：avg slice distance
        dist_avg = None
        if slice_dist_matrix is not None:
            dist_avg = self.compute_segment_dist_avg(slice_dist_matrix, segments)

        # 距离统计
        def dist_stats(dm):
            d = dm[dm > 0]
            if d.size == 0:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
            return {
                "min": float(d.min()),
                "max": float(d.max()),
                "mean": float(d.mean()),
                "median": float(np.median(d)),
            }

        return {
            "sample_id": sample_id,
            "problem_id": problem_id,
            "n_slices": n_slices,
            "n_segments": n_segments,
            "n_segments_original": n_segments_original,
            "adaptive": adaptive,
            "segments": segments,
            "dist_union": dist_union,
            "dist_avg": dist_avg,
            "dist_union_stats": dist_stats(dist_union),
            "dist_avg_stats": dist_stats(dist_avg) if dist_avg is not None else None,
        }

    def batch_process(
        self, output_dir: Path, p_stay: float = 0.9,
        adaptive: bool = False,
        max_slices_per_segment: int = 8,
        variance_threshold: float = 0.15,
    ):
        """批量处理所有 samples"""
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n{'='*80}")
        print(f"Segment 级批量处理")
        print(f"{'='*80}")
        print(f"输出目录: {output_dir}")
        print(f"HMM p_stay: {p_stay}")
        if adaptive:
            print(f"Adaptive splitting: ON (max_slices={max_slices_per_segment}, "
                  f"var_threshold={variance_threshold})")

        mem_tracker = MemoryTracker()
        overall_start = time.perf_counter()

        problem_results = {}
        sample_timings = []

        for problem_idx, (problem_id, sample_ids) in enumerate(
            sorted(self.problem_to_samples.items()), start=1
        ):
            problem_start = time.perf_counter()
            print(f"\n[Problem {problem_idx}/{self.n_problems}] ID={problem_id}, "
                  f"Samples={len(sample_ids)}")

            problem_data = {
                "problem_id": problem_id,
                "samples": [],
            }

            for i, sample_id in enumerate(sample_ids, start=1):
                sample_start = time.perf_counter()

                try:
                    result = self.process_single_sample(
                        sample_id, problem_id, p_stay,
                        adaptive=adaptive,
                        max_slices_per_segment=max_slices_per_segment,
                        variance_threshold=variance_threshold,
                    )

                    if result is None:
                        print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: SKIPPED")
                        continue

                    sample_time = time.perf_counter() - sample_start
                    sample_timings.append(sample_time)

                    # 保存距离矩阵
                    np.save(
                        output_dir / f"seg_dist_union_p{problem_id}_s{sample_id}.npy",
                        result["dist_union"],
                    )
                    if result["dist_avg"] is not None:
                        np.save(
                            output_dir / f"seg_dist_avg_p{problem_id}_s{sample_id}.npy",
                            result["dist_avg"],
                        )

                    # 保存 segment 元数据
                    seg_meta = {
                        "n_segments": result["n_segments"],
                        "n_slices": result["n_slices"],
                        "segments": result["segments"],
                    }
                    if result.get("adaptive"):
                        seg_meta["adaptive"] = True
                        seg_meta["n_segments_original"] = result["n_segments_original"]
                    meta_file = output_dir / f"seg_meta_p{problem_id}_s{sample_id}.json"
                    with open(meta_file, "w") as f:
                        json.dump(seg_meta, f, indent=2)

                    # 汇总条目
                    sample_summary = {
                        "sample_id": sample_id,
                        "n_slices": result["n_slices"],
                        "n_segments": result["n_segments"],
                        "dist_union_stats": result["dist_union_stats"],
                        "dist_avg_stats": result["dist_avg_stats"],
                        "processing_time_s": sample_time,
                    }
                    problem_data["samples"].append(sample_summary)

                    if i % 10 == 0 or i == len(sample_ids):
                        avg_time = np.mean(sample_timings[-10:])
                        print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: "
                              f"{result['n_segments']:3d} segments, "
                              f"{sample_time:.3f}s (avg: {avg_time:.3f}s)")

                except Exception as e:
                    print(f"  [{i:2d}/{len(sample_ids)}] Sample {sample_id:4d}: ERROR - {e}")
                    continue

            problem_time = time.perf_counter() - problem_start
            problem_data["processing_time_s"] = problem_time
            problem_results[problem_id] = problem_data

            print(f"  ✓ Problem {problem_id} completed in {problem_time:.1f}s "
                  f"({len(problem_data['samples'])}/{len(sample_ids)} samples)")

        overall_time = time.perf_counter() - overall_start

        # 保存汇总
        summary = {
            "cache_name": self.cache_path.name,
            "n_problems": self.n_problems,
            "n_samples": self.n_samples,
            "p_stay": p_stay,
            "total_time_s": overall_time,
            "avg_time_per_sample_s": float(np.mean(sample_timings)) if sample_timings else 0,
            "median_time_per_sample_s": float(np.median(sample_timings)) if sample_timings else 0,
            "memory_mb": mem_tracker.delta_mb(),
            "problems": problem_results,
        }

        summary_file = output_dir / "segment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Segment 级批量处理完成！")
        print(f"{'='*80}")
        print(f"总耗时: {overall_time:.1f}s ({overall_time/60:.1f} min)")
        if sample_timings:
            print(f"平均每 sample: {np.mean(sample_timings):.3f}s")
            print(f"处理速率: {len(sample_timings)/overall_time:.1f} samples/s")
        print(f"内存使用: {mem_tracker.delta_mb():.1f} MB")
        print(f"成功处理: {len(sample_timings)}/{self.n_samples}")

        total_segments = sum(
            s["n_segments"]
            for p in problem_results.values()
            for s in p["samples"]
        )
        print(f"总 segments: {total_segments:,}")
        print(f"\n输出: {summary_file}")
        print(f"{'='*80}")

        return summary


def run(
    cache_path, output_dir, slice_batch_dir=None, p_stay=0.9,
    adaptive=False, max_slices_per_segment=8, variance_threshold=0.15,
):
    """Phase 4: segment-level graph construction (programmatic API).

    Args:
        cache_path: neuron cache directory
        output_dir: output directory for segment-level results
        slice_batch_dir: directory containing slice-level hmm_*.npy and dist_*.npy
        p_stay: HMM persistence parameter
        adaptive: enable adaptive segment splitting
        max_slices_per_segment: max slices before size-based split
        variance_threshold: intra-segment variance threshold for variance-based split
    """
    cache_path = Path(cache_path)
    output_dir = Path(output_dir)
    if slice_batch_dir is not None:
        slice_batch_dir = Path(slice_batch_dir)
        if not slice_batch_dir.is_dir():
            print(f"WARNING: slice_batch_dir {slice_batch_dir} not found, method B (avg) will be skipped")
            slice_batch_dir = None
    builder = SegmentBatchBuilder(cache_path, slice_batch_dir)
    return builder.batch_process(
        output_dir, p_stay,
        adaptive=adaptive,
        max_slices_per_segment=max_slices_per_segment,
        variance_threshold=variance_threshold,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Segment 级图构建")
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径或简写 (如 'aime24')")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: ./batch_results_segment)")
    parser.add_argument("--p-stay", type=float, default=0.9, help="HMM p_stay")
    parser.add_argument("--slice-batch-dir", type=str, default=None,
                        help="slice 级 batch_results 目录 (含 hmm_*.npy); 默认自动推断")
    parser.add_argument("--adaptive", action="store_true",
                        help="启用自适应 segment 二次切分")
    parser.add_argument("--max-slices-per-segment", type=int, default=8,
                        help="adaptive: 每 segment 最大 slice 数 (default: 8)")
    parser.add_argument("--variance-threshold", type=float, default=0.15,
                        help="adaptive: 段内距离方差阈值 (default: 0.15)")
    args = parser.parse_args()

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

    benchmark = cache_path.parent.name
    output_dir = Path(args.output) if args.output else default_segment_batch_dir()

    # slice 级 batch_results 目录（用于方法 B）
    if args.slice_batch_dir:
        slice_batch_dir = Path(args.slice_batch_dir)
        if not slice_batch_dir.is_dir():
            print(f"WARNING: --slice-batch-dir {slice_batch_dir} not found, method B (avg) will be skipped")
            slice_batch_dir = None
    else:
        slice_batch_dir = default_batch_dir_for_benchmark(benchmark)
        if not slice_batch_dir.is_dir():
            slice_batch_dir = default_batch_dir_for_benchmark("aime24")
        if not slice_batch_dir.is_dir():
            print(f"WARNING: slice batch dir not found, method B (avg) will be skipped")
            slice_batch_dir = None

    print("=" * 80)
    print(f"Segment 级图构建 — {benchmark}")
    print("=" * 80)
    print(f"Cache: {cache_path}")
    print(f"Output: {output_dir}")
    print(f"Slice batch dir: {slice_batch_dir}")

    builder = SegmentBatchBuilder(cache_path, slice_batch_dir)
    builder.batch_process(
        output_dir, args.p_stay,
        adaptive=args.adaptive,
        max_slices_per_segment=args.max_slices_per_segment,
        variance_threshold=args.variance_threshold,
    )

    print(f"\n结果已保存至: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
