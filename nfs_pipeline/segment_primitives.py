#!/usr/bin/env python3
"""
从 segment 级距离矩阵提取结构原语。

节点粒度：segment（HMM 连续探索/利用段），而非 slice。
  - Core: 贪心合并 exploit segments
  - Return Edge: gap 为 segment 间隔
  - Drift Branch: explore segment 与 core 的最大相似度
  - Final Closure: 最后一个 exploit segment 与 core

CLI: --batch --method union|avg
输出: primitives_segment_union.json / primitives_segment_avg.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

from project_paths import REPO_ROOT

BATCH_DIR = REPO_ROOT / "batch_results_segment"


# ─────────────────────────── 工具函数 ───────────────────────────

def compute_thresholds(dist_matrix, n_slices=None):
    """计算 contact_threshold 和 long_range_threshold。

    V4 changes:
    - contact_threshold has a floor of 0.35 to stabilize small matrices.
    - long_range_threshold is in slice units (n_slices // 4) when n_slices
      is provided, falling back to segment-index units (n_segments // 4).
    """
    n = dist_matrix.shape[0]
    similarity = 1.0 - dist_matrix
    sim_upper = similarity[np.triu_indices(n, k=1)]
    contact_threshold = float(sim_upper.mean() + sim_upper.std())
    contact_threshold = max(contact_threshold, 0.35)
    if n_slices is not None:
        long_range_threshold = max(1, n_slices // 4)
    else:
        long_range_threshold = max(1, n // 4)
    return contact_threshold, long_range_threshold


# ─────────────────────────── Core ───────────────────────────

def extract_core(similarity, seg_states, contact_threshold):
    """提取 Core：最大 exploit segment 致密块，贪心合并。

    seg_states: array of segment states (0=explore, 1=exploit)

    Returns:
        dict with: indices (segment ids), internal_similarity,
                   fraction_of_exploit, n_merged_blocks
    """
    n_seg = len(seg_states)
    exploit_ids = [i for i in range(n_seg) if seg_states[i] == 1]

    if not exploit_ids:
        return {
            "indices": [],
            "internal_similarity": 0.0,
            "fraction_of_exploit": 0.0,
            "n_merged_blocks": 0,
        }

    if len(exploit_ids) == 1:
        idx = exploit_ids
        return {
            "indices": idx,
            "internal_similarity": 1.0,
            "fraction_of_exploit": 1.0,
            "n_merged_blocks": 1,
        }

    # 找最大 exploit segment 作为种子（这里 "最大" 可以直接取第一个，
    # 但我们从 seg_meta 无法直接得到 slice 数，所以取 id 顺序中间的，
    # 或者简单取第一个）。实际上我们有 seg_meta，但这里只用 similarity 矩阵。
    # 用连续 exploit segments 的贪心合并逻辑：
    # 找连续 exploit segments 块
    exploit_blocks = _find_contiguous_ids(exploit_ids)

    # 找最大连续块作为种子
    sizes = [len(b) for b in exploit_blocks]
    seed_idx = int(np.argmax(sizes))
    merged = set(exploit_blocks[seed_idx])
    merged_block_ids = {seed_idx}

    # 贪心合并
    remaining = [i for i in range(len(exploit_blocks)) if i != seed_idx]
    changed = True
    while changed:
        changed = False
        next_remaining = []
        for bi in remaining:
            candidate = exploit_blocks[bi]
            avg_sim = _cross_similarity(similarity, list(merged), candidate)
            if avg_sim > contact_threshold:
                merged.update(candidate)
                merged_block_ids.add(bi)
                changed = True
            else:
                next_remaining.append(bi)
        remaining = next_remaining

    merged_list = sorted(merged)
    int_sim = _internal_similarity(similarity, merged_list)
    n_exploit = len(exploit_ids)

    return {
        "indices": merged_list,
        "internal_similarity": float(int_sim),
        "fraction_of_exploit": len(merged_list) / n_exploit if n_exploit > 0 else 0.0,
        "n_merged_blocks": len(merged_block_ids),
    }


def _find_contiguous_ids(ids):
    """将有序 id 列表分成连续子段。
    e.g. [1,2,3,5,6,9] -> [[1,2,3],[5,6],[9]]
    """
    if not ids:
        return []
    blocks = []
    current = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] == ids[i-1] + 1:
            current.append(ids[i])
        else:
            blocks.append(current)
            current = [ids[i]]
    blocks.append(current)
    return blocks


def _internal_similarity(similarity, indices):
    if len(indices) < 2:
        return 1.0
    idx = np.array(indices)
    sub = similarity[np.ix_(idx, idx)]
    triu = sub[np.triu_indices(len(idx), k=1)]
    return float(triu.mean()) if len(triu) > 0 else 1.0


def _cross_similarity(similarity, group_a, group_b):
    a = np.array(group_a)
    b = np.array(group_b)
    sub = similarity[np.ix_(a, b)]
    return float(sub.mean())


# ─────────────────────────── Return Edge ───────────────────────────

def extract_return_edges(similarity, seg_states, contact_threshold,
                         long_range_threshold, segments=None):
    """提取 Return Edge：长程回返接触（segment 级）。

    V4 changes:
    - When ``segments`` (list of dicts with start/end) is provided,
      gap is measured in **slice units** (start_j - end_i) and
      ``long_range_threshold`` is also in slice units.
    - Falls back to segment-index gap when segments is None.

    Returns:
        list of dict: {i, j, gap, slice_gap, similarity, type}
    """
    n = len(seg_states)
    is_exploit = np.array(seg_states) == 1

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Compute gap in slice or segment units
            if segments is not None:
                slice_gap = segments[j]["start"] - segments[i]["end"]
                if slice_gap < long_range_threshold:
                    continue
            else:
                seg_gap = j - i
                if seg_gap <= long_range_threshold:
                    continue
                slice_gap = seg_gap  # fallback

            sim = similarity[i, j]
            if sim <= contact_threshold:
                continue
            if not (is_exploit[i] or is_exploit[j]):
                continue

            si, sj = seg_states[i], seg_states[j]
            edge_type = "exploit-exploit" if (si == 1 and sj == 1) else "exploit-explore"
            edges.append({
                "i": int(i),
                "j": int(j),
                "gap": j - i,
                "slice_gap": int(slice_gap),
                "similarity": float(sim),
                "type": edge_type,
            })

    # Sort by slice gap descending
    edges.sort(key=lambda e: -e["slice_gap"])
    return edges


# ─────────────────────────── Drift Branch ───────────────────────────

def extract_drift_branches(similarity, seg_states, core_indices,
                           contact_threshold, segments=None):
    """提取 Drift Branch：explore segment 与 core 的最大相似度。

    V4: adds ``n_slices_in_seg`` and ``start``/``end`` fields when
    ``segments`` metadata is available, enabling continuous drift
    weighting in the NFS computation.
    """
    n = len(seg_states)
    explore_ids = [i for i in range(n) if seg_states[i] == 0]

    if not explore_ids or not core_indices:
        branches = []
        for sid in explore_ids:
            entry = {
                "seg_id": sid,
                "state": 0,
                "max_sim_to_core": 0.0,
                "is_drift": True,
                "position": _position_label(sid, n),
            }
            if segments is not None:
                entry["start"] = segments[sid]["start"]
                entry["end"] = segments[sid]["end"]
                entry["n_slices_in_seg"] = segments[sid]["n_slices"]
            return branches
        return branches

    core_arr = np.array(core_indices)
    branches = []
    for sid in explore_ids:
        sims_to_core = similarity[sid, core_arr]
        max_sim = float(sims_to_core.max())
        is_drift = max_sim <= contact_threshold
        entry = {
            "seg_id": sid,
            "state": 0,
            "max_sim_to_core": float(max_sim),
            "is_drift": is_drift,
            "position": _position_label(sid, n),
        }
        if segments is not None:
            entry["start"] = segments[sid]["start"]
            entry["end"] = segments[sid]["end"]
            entry["n_slices_in_seg"] = segments[sid]["n_slices"]
        branches.append(entry)

    return branches


def _position_label(seg_id, n_segments):
    ratio = seg_id / n_segments if n_segments > 0 else 0
    if ratio < 1.0 / 3.0:
        return "early"
    elif ratio < 2.0 / 3.0:
        return "middle"
    else:
        return "late"


# ─────────────────────────── Final Closure ───────────────────────────

def extract_final_closure(similarity, seg_states, core_indices, contact_threshold):
    """最后一个 exploit segment 与 core 的回连强度。"""
    n = len(seg_states)
    exploit_ids = [i for i in range(n) if seg_states[i] == 1]

    if not exploit_ids or not core_indices:
        return {
            "last_exploit_seg_id": None,
            "s_close": 0.0,
            "closure_coefficient": 0.0,
        }

    last_id = exploit_ids[-1]
    core_arr = np.array(core_indices)
    s_close = float(similarity[last_id, core_arr].max())
    C = min(1.0, s_close / contact_threshold) if contact_threshold > 0 else 0.0

    return {
        "last_exploit_seg_id": last_id,
        "s_close": float(s_close),
        "closure_coefficient": float(C),
    }


# ─────────────────────────── 单样本分析 ───────────────────────────

def analyze_single_sample(problem_id, sample_id, method="union"):
    """分析单个样本的 segment 级结构原语。"""
    dist_file = BATCH_DIR / f"seg_dist_{method}_p{problem_id}_s{sample_id}.npy"
    meta_file = BATCH_DIR / f"seg_meta_p{problem_id}_s{sample_id}.json"

    if not dist_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"数据文件不存在: p{problem_id}_s{sample_id} ({method})")

    dist_matrix = np.load(dist_file)
    with open(meta_file) as f:
        seg_meta = json.load(f)

    n_segments = seg_meta["n_segments"]
    n_slices = seg_meta["n_slices"]
    segments = seg_meta["segments"]
    seg_states = [s["state"] for s in segments]

    similarity = 1.0 - dist_matrix
    contact_threshold, long_range_threshold = compute_thresholds(
        dist_matrix, n_slices=n_slices,
    )

    # 1. Core
    core = extract_core(similarity, seg_states, contact_threshold)

    # 2. Return Edges (V4: slice-gap based)
    return_edges = extract_return_edges(similarity, seg_states,
                                         contact_threshold, long_range_threshold,
                                         segments=segments)

    # 3. Drift Branches (V4: with segment size info)
    drift_branches = extract_drift_branches(similarity, seg_states,
                                             core["indices"], contact_threshold,
                                             segments=segments)

    # 4. Final Closure
    final_closure = extract_final_closure(similarity, seg_states,
                                           core["indices"], contact_threshold)

    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "n_slices": n_slices,
        "n_segments": n_segments,
        "contact_threshold": contact_threshold,
        "long_range_threshold": long_range_threshold,
        "segments": segments,
        "core": core,
        "return_edges": return_edges,
        "drift_branches": drift_branches,
        "final_closure": final_closure,
    }


# ─────────────────────────── 批量分析 ───────────────────────────

def batch_analyze(method="union"):
    """批量分析所有样本。"""
    summary_file = BATCH_DIR / "segment_summary.json"
    with open(summary_file) as f:
        summary = json.load(f)

    print("=" * 70)
    print(f"Segment Primitives Extraction (method={method})")
    print("=" * 70)
    print(f"Total problems: {summary['n_problems']}")
    print(f"Total samples:  {summary['n_samples']}")
    print()

    all_results = []
    failed = []

    def _sort_key(x):
        try:
            return (0, int(x[0]))
        except ValueError:
            return (1, x[0])

    for pid_str, problem_data in sorted(summary["problems"].items(), key=_sort_key):
        samples = problem_data["samples"]
        print(f"Problem {pid_str} ({len(samples):2d} samples)...", end=" ", flush=True)

        n_ok = 0
        for sample_info in samples:
            sid = sample_info["sample_id"]
            try:
                result = analyze_single_sample(pid_str, sid, method)
                all_results.append(result)
                n_ok += 1
            except Exception as e:
                failed.append((pid_str, sid, str(e)))

        print(f"OK {n_ok}/{len(samples)}")

    print()
    print(f"Processed: {len(all_results)} / {summary['n_samples']}  (failed: {len(failed)})")
    if failed:
        print(f"  First failures: {failed[:5]}")
    print()

    # ── 汇总统计 ──
    core_sizes = []
    core_fractions = []
    core_int_sims = []
    core_n_merged = []
    re_counts = []
    re_gaps = []
    re_sims = []
    re_xx_count = 0
    re_total = 0
    drift_fractions = []
    n_all_productive = 0
    n_all_drift = 0
    closure_coefficients = []

    for r in all_results:
        core_sizes.append(len(r["core"]["indices"]))
        core_fractions.append(r["core"]["fraction_of_exploit"])
        core_int_sims.append(r["core"]["internal_similarity"])
        core_n_merged.append(r["core"]["n_merged_blocks"])

        edges = r["return_edges"]
        re_counts.append(len(edges))
        for e in edges:
            re_gaps.append(e["gap"])
            re_sims.append(e["similarity"])
            re_total += 1
            if e["type"] == "exploit-exploit":
                re_xx_count += 1

        branches = r["drift_branches"]
        n_drift = sum(1 for b in branches if b["is_drift"])
        n_total_branches = len(branches)
        if n_total_branches > 0:
            drift_fractions.append(n_drift / n_total_branches)
        else:
            drift_fractions.append(0.0)
        if n_total_branches > 0 and n_drift == 0:
            n_all_productive += 1
        if n_total_branches > 0 and n_drift == n_total_branches:
            n_all_drift += 1

        closure_coefficients.append(r["final_closure"]["closure_coefficient"])

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    output = {
        "method": method,
        "summary": {
            "n_samples": len(all_results),
            "core_stats": {
                "mean_size": safe_mean(core_sizes),
                "mean_fraction_of_exploit": safe_mean(core_fractions),
                "mean_internal_similarity": safe_mean(core_int_sims),
                "mean_n_merged_blocks": safe_mean(core_n_merged),
            },
            "return_edge_stats": {
                "mean_count": safe_mean(re_counts),
                "mean_gap": safe_mean(re_gaps),
                "mean_similarity": safe_mean(re_sims),
                "fraction_exploit_exploit": re_xx_count / re_total if re_total > 0 else 0.0,
            },
            "drift_stats": {
                "mean_drift_fraction": safe_mean(drift_fractions),
                "samples_all_productive": n_all_productive,
                "samples_all_drift": n_all_drift,
            },
            "closure_stats": {
                "mean_closure_coefficient": safe_mean(closure_coefficients),
            },
        },
        "samples": all_results,
    }

    output_file = BATCH_DIR / f"primitives_segment_{method}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # 打印汇总
    s = output["summary"]
    print("=" * 70)
    print(f"SUMMARY (method={method})")
    print("=" * 70)

    cs = s["core_stats"]
    print(f"\n  Core:")
    print(f"    mean size (segments):    {cs['mean_size']:.1f}")
    print(f"    mean fraction of exploit:{cs['mean_fraction_of_exploit']:.3f}")
    print(f"    mean internal similarity:{cs['mean_internal_similarity']:.4f}")
    print(f"    mean merged blocks:      {cs['mean_n_merged_blocks']:.2f}")

    rs = s["return_edge_stats"]
    print(f"\n  Return Edges:")
    print(f"    mean count per sample:   {rs['mean_count']:.1f}")
    print(f"    mean gap (segments):     {rs['mean_gap']:.1f}")
    print(f"    mean similarity:         {rs['mean_similarity']:.4f}")
    print(f"    exploit-exploit fraction: {rs['fraction_exploit_exploit']:.3f}")

    ds = s["drift_stats"]
    print(f"\n  Drift Branches:")
    print(f"    mean drift fraction:     {ds['mean_drift_fraction']:.3f}")
    print(f"    samples all productive:  {ds['samples_all_productive']}")
    print(f"    samples all drift:       {ds['samples_all_drift']}")

    cls = s["closure_stats"]
    print(f"\n  Final Closure:")
    print(f"    mean closure coefficient:{cls['mean_closure_coefficient']:.3f}")

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)


def run(batch_dir, method="union"):
    """Phase 5: extract segment-level structural primitives (programmatic API).

    Results saved to batch_dir/primitives_segment_{method}.json.
    """
    global BATCH_DIR
    BATCH_DIR = Path(batch_dir)
    batch_analyze(method)


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract segment-level structural primitives")
    parser.add_argument("--problem_id", type=str, help="Problem ID (single sample mode)")
    parser.add_argument("--sample_id", type=int, help="Sample ID (single sample mode)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: analyze all samples")
    parser.add_argument("--method", type=str, default="union", choices=["union", "avg"],
                        help="Distance method: union (exact) or avg (aggregated)")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="segment batch_results 目录")
    args = parser.parse_args()

    global BATCH_DIR
    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)

    if args.batch:
        batch_analyze(args.method)
    elif args.problem_id is not None and args.sample_id is not None:
        result = analyze_single_sample(args.problem_id, args.sample_id, args.method)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
