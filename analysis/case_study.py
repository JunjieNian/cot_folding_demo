#!/usr/bin/env python3
"""
Case Study: 将三个结构原语 (Core / Return Edge / Drift Branch) 对应到 CoT 原文

验证结构标注在语义层面是否成立：
  - Core slices 是否对应"主线推理 / 构建解题骨架"
  - Return Edges 是否对应"回顾、修正、折回验证"
  - Drift Branches 是否对应"未收束的探索 / 偏离主线"

用法：
  python case_study_primitives.py --problem_id 61 --sample_id 76   # 高 NFS 正确样本
  python case_study_primitives.py --problem_id 61 --sample_id 94   # 低 NFS 错误样本
  python case_study_primitives.py --problem_id 61 --sample_id 127  # 高 NFS 错误样本
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from textwrap import fill

from project_paths import (
    default_batch_dir_for_benchmark,
    ensure_project_imports,
    resolve_cache_path,
)

ensure_project_imports()

CACHE_PATH = resolve_cache_path(
    default_benchmark="aime24",
    default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
)
BATCH_DIR = default_batch_dir_for_benchmark("aime24")
PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"
NFS_FILE = BATCH_DIR / "nfs_analysis.json"

SLICE_SIZE = 32  # tokens per slice


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def init_reader_and_tokenizer():
    """初始化 cache reader 和 tokenizer。"""
    if CACHE_PATH is None:
        raise FileNotFoundError("AIME24 cache 未配置，请通过 --cache 指定。")

    from alignment import get_cache_reader
    from alignment_lite.tokenizer import TokenizerWrapper

    reader = get_cache_reader(CACHE_PATH)
    with open(CACHE_PATH / "meta.json") as f:
        meta = json.load(f)
    tw = TokenizerWrapper.load(meta["model_path"])
    return reader, tw


def get_slice_text(reader, tokenizer, sample_id, slice_idx):
    """解码单个 slice 的文本。"""
    token_start = reader.token_row_ptr[sample_id]
    token_end = reader.token_row_ptr[sample_id + 1]
    tids = reader.token_ids[token_start:token_end]

    s = slice_idx * SLICE_SIZE
    e = min((slice_idx + 1) * SLICE_SIZE, len(tids))
    if s >= len(tids):
        return ""
    return tokenizer.decode(tids[s:e].tolist())


def get_slice_range_text(reader, tokenizer, sample_id, start_slice, end_slice):
    """解码连续 slice 范围的文本。"""
    token_start = reader.token_row_ptr[sample_id]
    token_end = reader.token_row_ptr[sample_id + 1]
    tids = reader.token_ids[token_start:token_end]

    s = start_slice * SLICE_SIZE
    e = min(end_slice * SLICE_SIZE, len(tids))
    if s >= len(tids):
        return ""
    return tokenizer.decode(tids[s:e].tolist())


def load_sample_primitives(problem_id, sample_id):
    """从 primitives_analysis.json 加载单样本数据。"""
    with open(PRIMITIVES_FILE) as f:
        data = json.load(f)
    for s in data["samples"]:
        if s["problem_id"] == problem_id and s["sample_id"] == sample_id:
            return s
    return None


def load_sample_nfs(problem_id, sample_id):
    """从 nfs_analysis.json 加载单样本 NFS。"""
    with open(NFS_FILE) as f:
        data = json.load(f)
    for s in data["samples"]:
        if s["problem_id"] == problem_id and s["sample_id"] == sample_id:
            return s
    return None


# ═══════════════════════════════════════════════════════════════
#  文本截断与格式化
# ═══════════════════════════════════════════════════════════════

def truncate(text, max_len=200):
    """截断文本并添加省略号。"""
    text = text.replace("\n", " ↵ ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def format_block_text(text, indent=6, width=90):
    """格式化长文本块。"""
    text = text.replace("\n", " ↵ ").strip()
    lines = fill(text, width=width - indent)
    prefix = " " * indent
    return "\n".join(prefix + line for line in lines.split("\n"))


# ═══════════════════════════════════════════════════════════════
#  Case Study 主逻辑
# ═══════════════════════════════════════════════════════════════

def run_case_study(problem_id, sample_id):
    """对单个样本做结构原语 → 原文对照分析。"""

    # 1. 加载所有数据
    print("Loading data...")
    reader, tokenizer = init_reader_and_tokenizer()
    primitives = load_sample_primitives(problem_id, sample_id)
    nfs_data = load_sample_nfs(problem_id, sample_id)

    if primitives is None:
        print(f"ERROR: primitives not found for p{problem_id}_s{sample_id}")
        return
    if nfs_data is None:
        print(f"ERROR: NFS data not found for p{problem_id}_s{sample_id}")
        return

    n = primitives["n_slices"]
    tau = primitives["contact_threshold"]
    hmm_file = BATCH_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"
    hmm_states = np.load(hmm_file)

    # ── 头部信息 ──
    print()
    print("=" * 90)
    print(f"  CASE STUDY: Problem {problem_id}, Sample {sample_id}")
    print("=" * 90)
    print(f"  Slices: {n}  |  Correct: {nfs_data['is_correct']}  |  Answer: {nfs_data.get('answer', '?')}")
    print(f"  NFS = {nfs_data['NFS']:.2f}  |  B = {nfs_data['B']:.4f}  |  "
          f"H = {nfs_data['H']:.4f}  |  D* = {nfs_data['D_star']:.4f}  |  G = {nfs_data['G']:.4f}")
    print(f"  Contact threshold τ = {tau:.4f}")
    print(f"  Explore: {int((hmm_states == 0).sum())}  |  Exploit: {int((hmm_states == 1).sum())}")
    print("=" * 90)

    # ── 2. Core 分析 ──
    core = primitives["core"]
    core_indices = core["indices"]

    print(f"\n{'─'*90}")
    print(f"  [B] CORE (Backbone) — {len(core_indices)} slices, "
          f"internal_sim={core['internal_similarity']:.4f}, "
          f"fraction_of_exploit={core['fraction_of_exploit']:.3f}, "
          f"merged_blocks={core['n_merged_blocks']}")
    print(f"{'─'*90}")

    if core_indices:
        # 找 core 的连续段
        core_segments = _find_segments(core_indices)
        for seg_start, seg_end in core_segments:
            seg_len = seg_end - seg_start
            text = get_slice_range_text(reader, tokenizer, sample_id, seg_start, seg_end)
            # 显示开头和结尾
            print(f"\n  Slices [{seg_start}:{seg_end}] ({seg_len} slices)")
            if len(text) > 400:
                head = truncate(text[:200], 200)
                tail = truncate(text[-200:], 200)
                print(f"      HEAD: {head}")
                print(f"      TAIL: {tail}")
            else:
                print(f"      TEXT: {truncate(text, 400)}")
    else:
        print("  (empty core)")

    # ── 3. Return Edge 分析 ──
    edges = primitives["return_edges"]
    print(f"\n{'─'*90}")
    print(f"  [H] RETURN EDGES — {len(edges)} edges total")
    print(f"{'─'*90}")

    if edges:
        # 取 top-10 strongest (by similarity * gap)
        scored_edges = []
        for e in edges:
            score = ((e["similarity"] - tau) / (1.0 - tau + 1e-9)) * (e["gap"] / (n - 1))
            scored_edges.append((score, e))
        scored_edges.sort(key=lambda x: -x[0])

        print(f"  Top-10 strongest return edges (by r_e contribution):\n")
        for rank, (score, e) in enumerate(scored_edges[:10], 1):
            i, j = e["i"], e["j"]
            text_i = get_slice_text(reader, tokenizer, sample_id, i)
            text_j = get_slice_text(reader, tokenizer, sample_id, j)
            state_i = "EXPLOIT" if hmm_states[i] == 1 else "EXPLORE"
            state_j = "EXPLOIT" if hmm_states[j] == 1 else "EXPLORE"

            print(f"  #{rank}  gap={e['gap']:3d}  sim={e['similarity']:.4f}  "
                  f"r_e={score:.4f}  type={e['type']}")
            print(f"    Slice {i:3d} [{state_i}]: {truncate(text_i, 80)}")
            print(f"    Slice {j:3d} [{state_j}]: {truncate(text_j, 80)}")
            print()
    else:
        print("  (no return edges)")

    # ── 4. Drift Branch 分析 ──
    branches = primitives["drift_branches"]
    drift_branches = [b for b in branches if b["is_drift"]]
    productive_branches = [b for b in branches if not b["is_drift"]]

    print(f"{'─'*90}")
    print(f"  [D] DRIFT BRANCHES — {len(drift_branches)} drift / "
          f"{len(productive_branches)} productive / {len(branches)} total explore blocks")
    print(f"{'─'*90}")

    # 显示所有 drift branches
    if drift_branches:
        print(f"\n  DRIFT (max_sim_to_core ≤ τ={tau:.4f}):\n")
        for b in drift_branches:
            text = get_slice_range_text(reader, tokenizer, sample_id,
                                        b["start"], b["end"])
            print(f"  Slices [{b['start']}:{b['end']}] ({b['length']} slices)  "
                  f"max_sim_to_core={b['max_sim_to_core']:.4f}  position={b['position']}")
            if len(text) > 300:
                print(f"      HEAD: {truncate(text[:150], 150)}")
                print(f"      TAIL: {truncate(text[-150:], 150)}")
            else:
                print(f"      TEXT: {truncate(text, 300)}")
            print()
    else:
        print("\n  (no drift branches — all explore blocks connect back to core)")

    # 显示 productive branches 的摘要
    if productive_branches:
        print(f"  PRODUCTIVE explore blocks (max_sim_to_core > τ):\n")
        for b in productive_branches[:5]:  # 只显示前 5 个
            text = get_slice_range_text(reader, tokenizer, sample_id,
                                        b["start"], b["end"])
            print(f"  Slices [{b['start']}:{b['end']}] ({b['length']} slices)  "
                  f"max_sim_to_core={b['max_sim_to_core']:.4f}  position={b['position']}")
            print(f"      TEXT: {truncate(text, 120)}")
        if len(productive_branches) > 5:
            print(f"  ... and {len(productive_branches) - 5} more productive blocks")
        print()

    # ── 5. Final Closure 分析 ──
    fc = primitives.get("final_closure", {})
    print(f"{'─'*90}")
    print(f"  [G] FINAL CLOSURE — C={fc.get('closure_coefficient', 0):.4f}, "
          f"s_close={fc.get('s_close', 0):.4f}")
    print(f"{'─'*90}")

    if fc.get("last_exploit_start") is not None:
        ls, le = fc["last_exploit_start"], fc["last_exploit_end"]
        text = get_slice_range_text(reader, tokenizer, sample_id, ls, le)
        print(f"\n  Last exploit block: slices [{ls}:{le}] ({le - ls} slices)")
        if len(text) > 300:
            print(f"      TAIL (last 200 chars): {truncate(text[-200:], 200)}")
        else:
            print(f"      TEXT: {truncate(text, 300)}")

    # ── 6. 序列鸟瞰图 ──
    print(f"\n{'─'*90}")
    print(f"  SEQUENCE OVERVIEW (E=explore, X=exploit, C=core, D=drift)")
    print(f"{'─'*90}\n")

    core_set = set(core_indices)
    drift_slices = set()
    for b in drift_branches:
        for si in range(b["start"], b["end"]):
            drift_slices.add(si)

    # 每行 80 个 slice
    line = []
    for i in range(n):
        if i in core_set:
            line.append("C")
        elif i in drift_slices:
            line.append("D")
        elif hmm_states[i] == 1:
            line.append("X")
        else:
            line.append("E")

    row_width = 80
    print(f"  {'':6s} {'0':^10s} {'20':^10s} {'40':^10s} {'60':^10s}")
    for row_start in range(0, n, row_width):
        row_end = min(row_start + row_width, n)
        segment = "".join(line[row_start:row_end])
        print(f"  {row_start:4d}: {segment}")

    print(f"\n  Legend: C=Core(exploit), X=non-core exploit, E=productive explore, D=drift explore")
    print("=" * 90)


def _find_segments(indices):
    """将排序的索引列表分成连续段 [(start, end), ...]。"""
    if not indices:
        return []
    segments = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i != prev + 1:
            segments.append((start, prev + 1))
            start = i
        prev = i
    segments.append((start, prev + 1))
    return segments


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    global CACHE_PATH, BATCH_DIR, PRIMITIVES_FILE, NFS_FILE

    parser = argparse.ArgumentParser(description="Case study: primitives → original text")
    parser.add_argument("--problem_id", type=int, required=True)
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--cache", type=str, default=None,
                        help="cache 目录路径或 benchmark 简写")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="primitives/nfs 所在 batch_results 目录")
    args = parser.parse_args()

    CACHE_PATH = resolve_cache_path(
        args.cache,
        default_benchmark="aime24",
        default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
        required=True,
    )
    if args.batch_dir:
        BATCH_DIR = Path(args.batch_dir)
        PRIMITIVES_FILE = BATCH_DIR / "primitives_analysis.json"
        NFS_FILE = BATCH_DIR / "nfs_analysis.json"

    run_case_study(args.problem_id, args.sample_id)


if __name__ == "__main__":
    main()
