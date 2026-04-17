#!/usr/bin/env python3
"""
提取 CoT 折叠结构的 3 个结构原语：
  1. Core — 最大 exploit 致密块（贪心合并）
  2. Return Edge — 长程回返接触（远距高相似度 slice 对）
  3. Drift Branch — 无回并的探索支路

输入：batch_results/ 下的 dist_*.npy 和 hmm_*.npy
输出：batch_results/primitives_analysis.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from project_paths import REPO_ROOT

BATCH_DIR = REPO_ROOT / "batch_results"  # 默认值，可被 CLI 覆盖
SUMMARY_FILE = BATCH_DIR / "batch_summary.json"


# ─────────────────────────── 工具函数 ───────────────────────────

def find_contiguous_blocks(hmm_states, target_state):
    """找到 hmm_states 中连续等于 target_state 的段。

    Returns:
        list of (start, end) 左闭右开
    """
    blocks = []
    n = len(hmm_states)
    i = 0
    while i < n:
        if hmm_states[i] == target_state:
            j = i
            while j < n and hmm_states[j] == target_state:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def compute_thresholds(dist_matrix):
    """计算 contact_threshold 和 long_range_threshold（复用 cot_folding_map.py 逻辑）。"""
    n = dist_matrix.shape[0]
    similarity = 1.0 - dist_matrix
    sim_upper = similarity[np.triu_indices(n, k=1)]
    contact_threshold = float(sim_upper.mean() + sim_upper.std())
    long_range_threshold = n // 4
    return contact_threshold, long_range_threshold


# ─────────────────────────── Core ───────────────────────────

def extract_core(similarity, hmm_states, contact_threshold):
    """提取 Core：最大 exploit 致密块，贪心合并。

    Returns:
        dict with: indices (list[int]), internal_similarity (float),
                   fraction_of_exploit (float), n_merged_blocks (int)
    """
    exploit_blocks = find_contiguous_blocks(hmm_states, target_state=1)

    if not exploit_blocks:
        return {
            "indices": [],
            "internal_similarity": 0.0,
            "fraction_of_exploit": 0.0,
            "n_merged_blocks": 0,
        }

    # 每个块的索引集
    block_indices = [list(range(s, e)) for s, e in exploit_blocks]

    if len(block_indices) == 1:
        idx = block_indices[0]
        int_sim = _internal_similarity(similarity, idx)
        n_exploit = int((hmm_states == 1).sum())
        return {
            "indices": idx,
            "internal_similarity": float(int_sim),
            "fraction_of_exploit": len(idx) / n_exploit if n_exploit > 0 else 1.0,
            "n_merged_blocks": 1,
        }

    # 找最大块作为种子
    sizes = [len(b) for b in block_indices]
    seed_idx = int(np.argmax(sizes))
    merged = set(block_indices[seed_idx])
    merged_block_ids = {seed_idx}

    # 贪心合并
    remaining = [i for i in range(len(block_indices)) if i != seed_idx]
    changed = True
    while changed:
        changed = False
        next_remaining = []
        for bi in remaining:
            candidate = block_indices[bi]
            # 计算 candidate 与 merged 之间的平均相似度
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
    n_exploit = int((hmm_states == 1).sum())

    return {
        "indices": merged_list,
        "internal_similarity": float(int_sim),
        "fraction_of_exploit": len(merged_list) / n_exploit if n_exploit > 0 else 0.0,
        "n_merged_blocks": len(merged_block_ids),
    }


def _internal_similarity(similarity, indices):
    """计算索引集内部的平均相似度。"""
    if len(indices) < 2:
        return 1.0
    idx = np.array(indices)
    sub = similarity[np.ix_(idx, idx)]
    triu = sub[np.triu_indices(len(idx), k=1)]
    return float(triu.mean()) if len(triu) > 0 else 1.0


def _cross_similarity(similarity, group_a, group_b):
    """计算两组索引之间的平均相似度。"""
    a = np.array(group_a)
    b = np.array(group_b)
    sub = similarity[np.ix_(a, b)]
    return float(sub.mean())


# ─────────────────────────── Return Edge ───────────────────────────

def extract_return_edges(similarity, hmm_states, contact_threshold, long_range_threshold):
    """提取 Return Edge：长程回返接触。

    Returns:
        list of dict: {i, j, gap, similarity, type}
        type: "exploit-exploit" | "exploit-explore"
    """
    n = len(hmm_states)
    # 向量化
    ones = np.ones((n, n), dtype=bool)
    ii, jj = np.where(np.triu(ones, k=long_range_threshold + 1))

    sims = similarity[ii, jj]
    mask_contact = sims > contact_threshold

    # 至少一端为 exploit
    is_exploit = (hmm_states == 1)
    mask_exploit = is_exploit[ii] | is_exploit[jj]

    mask = mask_contact & mask_exploit
    ii_f, jj_f, sims_f = ii[mask], jj[mask], sims[mask]
    gaps = jj_f - ii_f

    # 按 gap 降序排列
    order = np.argsort(-gaps)
    ii_f, jj_f, sims_f, gaps = ii_f[order], jj_f[order], sims_f[order], gaps[order]

    edges = []
    for k in range(len(ii_f)):
        i, j = int(ii_f[k]), int(jj_f[k])
        si, sj = int(hmm_states[i]), int(hmm_states[j])
        if si == 1 and sj == 1:
            edge_type = "exploit-exploit"
        else:
            edge_type = "exploit-explore"
        edges.append({
            "i": i,
            "j": j,
            "gap": int(gaps[k]),
            "similarity": float(sims_f[k]),
            "type": edge_type,
        })

    return edges


# ─────────────────────────── Drift Branch ───────────────────────────

def extract_drift_branches(similarity, hmm_states, core_indices, contact_threshold):
    """提取 Drift Branch：无回并的探索支路。

    Returns:
        list of dict: {start, end, length, max_sim_to_core, is_drift, position}
        position: "early" | "middle" | "late"
    """
    n = len(hmm_states)
    explore_blocks = find_contiguous_blocks(hmm_states, target_state=0)

    if not explore_blocks or not core_indices:
        # 无 explore 块或无 core：全部标为 drift（如果有的话）
        branches = []
        for s, e in explore_blocks:
            mid = (s + e) / 2.0
            branches.append({
                "start": s,
                "end": e,
                "length": e - s,
                "max_sim_to_core": 0.0,
                "is_drift": True,
                "position": _position_label(mid, n),
            })
        return branches

    core_arr = np.array(core_indices)
    branches = []
    for s, e in explore_blocks:
        block_idx = np.arange(s, e)
        sub = similarity[np.ix_(block_idx, core_arr)]
        max_sim = float(sub.max())
        is_drift = max_sim <= contact_threshold
        mid = (s + e) / 2.0
        branches.append({
            "start": s,
            "end": e,
            "length": e - s,
            "max_sim_to_core": float(max_sim),
            "is_drift": is_drift,
            "position": _position_label(mid, n),
        })

    return branches


def _position_label(midpoint, n):
    """根据 midpoint / n 的比例分类位置。"""
    ratio = midpoint / n
    if ratio < 1.0 / 3.0:
        return "early"
    elif ratio < 2.0 / 3.0:
        return "middle"
    else:
        return "late"


# ─────────────────────────── Final Closure ───────────────────────────

def extract_final_closure(similarity, hmm_states, core_indices, contact_threshold):
    """提取 Final Closure：最后一个 exploit block 与 core 的回连强度。

    Returns:
        dict: {last_exploit_start, last_exploit_end, s_close, closure_coefficient}
    """
    exploit_blocks = find_contiguous_blocks(hmm_states, target_state=1)

    if not exploit_blocks or not core_indices:
        return {
            "last_exploit_start": None,
            "last_exploit_end": None,
            "s_close": 0.0,
            "closure_coefficient": 0.0,
        }

    # 最后一个 exploit block
    last_start, last_end = exploit_blocks[-1]
    last_idx = np.arange(last_start, last_end)
    core_arr = np.array(core_indices)

    # s_close = max similarity between last exploit block and core
    sub = similarity[np.ix_(last_idx, core_arr)]
    s_close = float(sub.max())

    # C = min(1, s_close / τ)
    C = min(1.0, s_close / contact_threshold) if contact_threshold > 0 else 0.0

    return {
        "last_exploit_start": last_start,
        "last_exploit_end": last_end,
        "s_close": float(s_close),
        "closure_coefficient": float(C),
    }


# ─────────────────────────── 单样本分析 ───────────────────────────

def analyze_single_sample(problem_id, sample_id):
    """完整分析单个样本的三个结构原语。"""
    dist_file = BATCH_DIR / f"dist_p{problem_id}_s{sample_id}.npy"
    hmm_file = BATCH_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"

    if not dist_file.exists() or not hmm_file.exists():
        raise FileNotFoundError(f"数据文件不存在: p{problem_id}_s{sample_id}")

    dist_matrix = np.load(dist_file)
    hmm_states = np.load(hmm_file)
    n = len(hmm_states)
    similarity = 1.0 - dist_matrix

    contact_threshold, long_range_threshold = compute_thresholds(dist_matrix)

    # 1. Core
    core = extract_core(similarity, hmm_states, contact_threshold)

    # 2. Return Edges
    return_edges = extract_return_edges(similarity, hmm_states,
                                        contact_threshold, long_range_threshold)

    # 3. Drift Branches
    drift_branches = extract_drift_branches(similarity, hmm_states,
                                            core["indices"], contact_threshold)

    # 4. Final Closure
    final_closure = extract_final_closure(similarity, hmm_states,
                                          core["indices"], contact_threshold)

    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "n_slices": n,
        "contact_threshold": contact_threshold,
        "long_range_threshold": long_range_threshold,
        "core": core,
        "return_edges": return_edges,
        "drift_branches": drift_branches,
        "final_closure": final_closure,
    }


# ─────────────────────────── 批量分析 ───────────────────────────

def batch_analyze():
    """批量分析所有 1920 个样本，输出汇总 JSON。"""
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    print("=" * 70)
    print("Structural Primitives Extraction")
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

    for problem_id_str, problem_data in sorted(summary["problems"].items(), key=_sort_key):
        problem_id = problem_id_str
        samples = problem_data["samples"]
        print(f"Problem {problem_id} ({len(samples):2d} samples)...", end=" ", flush=True)

        n_ok = 0
        for sample_info in samples:
            sample_id = sample_info["sample_id"]
            try:
                result = analyze_single_sample(problem_id, sample_id)
                all_results.append(result)
                n_ok += 1
            except Exception as e:
                failed.append((problem_id, sample_id, str(e)))

        print(f"OK {n_ok}/{len(samples)}")

    print()
    print(f"Processed: {len(all_results)} / {summary['n_samples']}  "
          f"(failed: {len(failed)})")
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

    drift_fractions = []       # drift 块数 / explore 块总数
    drift_slice_fractions = [] # drift slice 数 / explore slice 总数
    n_all_productive = 0
    n_all_drift = 0

    closure_coefficients = []

    for r in all_results:
        # Core
        core_sizes.append(len(r["core"]["indices"]))
        core_fractions.append(r["core"]["fraction_of_exploit"])
        core_int_sims.append(r["core"]["internal_similarity"])
        core_n_merged.append(r["core"]["n_merged_blocks"])

        # Return Edges
        edges = r["return_edges"]
        re_counts.append(len(edges))
        for e in edges:
            re_gaps.append(e["gap"])
            re_sims.append(e["similarity"])
            re_total += 1
            if e["type"] == "exploit-exploit":
                re_xx_count += 1

        # Drift Branches
        branches = r["drift_branches"]
        n_drift = sum(1 for b in branches if b["is_drift"])
        n_total_branches = len(branches)
        drift_slices = sum(b["length"] for b in branches if b["is_drift"])
        total_explore_slices = sum(b["length"] for b in branches)

        if n_total_branches > 0:
            drift_fractions.append(n_drift / n_total_branches)
        else:
            drift_fractions.append(0.0)

        if total_explore_slices > 0:
            drift_slice_fractions.append(drift_slices / total_explore_slices)
        else:
            drift_slice_fractions.append(0.0)

        if n_total_branches > 0 and n_drift == 0:
            n_all_productive += 1
        if n_total_branches > 0 and n_drift == n_total_branches:
            n_all_drift += 1

        # Final Closure
        closure_coefficients.append(r["final_closure"]["closure_coefficient"])

    # 构建 summary
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    output = {
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
                "mean_drift_slice_fraction": safe_mean(drift_slice_fractions),
                "samples_all_productive": n_all_productive,
                "samples_all_drift": n_all_drift,
            },
            "closure_stats": {
                "mean_closure_coefficient": safe_mean(closure_coefficients),
            },
        },
        "samples": all_results,
    }

    # 保存
    output_file = BATCH_DIR / "primitives_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # 打印汇总
    s = output["summary"]
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    cs = s["core_stats"]
    print(f"\n  Core:")
    print(f"    mean size:               {cs['mean_size']:.1f} slices")
    print(f"    mean fraction of exploit:{cs['mean_fraction_of_exploit']:.3f}")
    print(f"    mean internal similarity:{cs['mean_internal_similarity']:.4f}")
    print(f"    mean merged blocks:      {cs['mean_n_merged_blocks']:.2f}")

    rs = s["return_edge_stats"]
    print(f"\n  Return Edges:")
    print(f"    mean count per sample:   {rs['mean_count']:.1f}")
    print(f"    mean gap:                {rs['mean_gap']:.1f}")
    print(f"    mean similarity:         {rs['mean_similarity']:.4f}")
    print(f"    exploit-exploit fraction: {rs['fraction_exploit_exploit']:.3f}")

    ds = s["drift_stats"]
    print(f"\n  Drift Branches:")
    print(f"    mean drift fraction:     {ds['mean_drift_fraction']:.3f}")
    print(f"    mean drift slice fraction:{ds['mean_drift_slice_fraction']:.3f}")
    print(f"    samples all productive:  {ds['samples_all_productive']}")
    print(f"    samples all drift:       {ds['samples_all_drift']}")

    cls = s["closure_stats"]
    print(f"\n  Final Closure:")
    print(f"    mean closure coefficient:{cls['mean_closure_coefficient']:.3f}")

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)


def run(batch_dir):
    """Phase 2: extract slice-level structural primitives (programmatic API).

    Results saved to batch_dir/primitives_analysis.json.
    """
    global BATCH_DIR, SUMMARY_FILE
    BATCH_DIR = Path(batch_dir)
    SUMMARY_FILE = BATCH_DIR / "batch_summary.json"
    batch_analyze()


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract structural primitives from CoT folding")
    parser.add_argument("--problem_id", type=int, help="Problem ID (single sample mode)")
    parser.add_argument("--sample_id", type=int, help="Sample ID (single sample mode)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: analyze all samples")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="batch_results 目录 (默认: ./batch_results)")
    args = parser.parse_args()

    global BATCH_DIR, SUMMARY_FILE
    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
        SUMMARY_FILE = BATCH_DIR / "batch_summary.json"

    if args.batch:
        batch_analyze()
    elif args.problem_id is not None and args.sample_id is not None:
        result = analyze_single_sample(args.problem_id, args.sample_id)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
