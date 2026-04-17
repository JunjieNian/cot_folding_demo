#!/usr/bin/env python3
"""
Phase 2.5: 基于稀疏图的经典图结构特征提取（无向图 + 有向图）

从单条 CoT 的 JA 距离矩阵构建稀疏图（阈值：similarity > mean + std），
同时构建无向图（nx.Graph）和有向图（nx.DiGraph, 时序方向 i→j where i<j），
提取经典图结构特征并叠加 HMM 标签做交叉分析。

输入：batch_results/ 下的 dist_p{pid}_s{sid}.npy 和 hmm_p{pid}_s{sid}.npy
输出：batch_results_graph/graph_features.json
"""

import sys
import time
import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import REPO_ROOT, default_graph_batch_dir

INPUT_DIR = REPO_ROOT / "batch_results"          # dist/hmm .npy 所在目录
OUTPUT_DIR = default_graph_batch_dir()            # graph_features.json 输出目录
SUMMARY_FILE = INPUT_DIR / "batch_summary.json"


# ─────────────────────────── 阈值计算 ───────────────────────────

def compute_threshold(dist_matrix):
    """计算稀疏图阈值 τ = mean(sim_upper_tri) + std(sim_upper_tri)。

    复用 primitives.py:compute_thresholds 逻辑。
    """
    n = dist_matrix.shape[0]
    similarity = 1.0 - dist_matrix
    sim_upper = similarity[np.triu_indices(n, k=1)]
    return float(sim_upper.mean() + sim_upper.std())


# ─────────────────────────── 稀疏图构建 ───────────────────────────

def build_sparse_graphs(dist_matrix, hmm_states):
    """从距离矩阵构建无向图和有向图。

    Args:
        dist_matrix: N×N JA 距离矩阵 (.npy)
        hmm_states:  N 维 HMM 状态向量 (0=explore, 1=exploit)

    Returns:
        G:         nx.Graph  — 无向图 (edge iff sim > τ)
        D:         nx.DiGraph — 有向图 (arc i→j, i<j, iff sim > τ)
        threshold: float
    """
    n = dist_matrix.shape[0]
    similarity = 1.0 - dist_matrix
    threshold = compute_threshold(dist_matrix)

    G = nx.Graph()
    D = nx.DiGraph()

    # 添加所有节点（含 HMM 状态属性）
    for i in range(n):
        state = int(hmm_states[i])
        G.add_node(i, hmm_state=state)
        D.add_node(i, hmm_state=state)

    # 向量化找出上三角中超过阈值的 pair
    ii, jj = np.triu_indices(n, k=1)
    sims = similarity[ii, jj]
    mask = sims > threshold
    sel = np.where(mask)[0]

    edges = [(int(ii[k]), int(jj[k]), {"similarity": float(sims[k])}) for k in sel]
    G.add_edges_from(edges)
    D.add_edges_from(edges)  # i < j (上三角保证时序方向)

    return G, D, threshold


# ─────────────────────── 高性能工具函数 ───────────────────────────

CLIQUE_TIMEOUT_S = 3    # find_cliques 超时秒数（超时后报告已知下界）


def _find_clique_number(G, timeout_s=CLIQUE_TIMEOUT_S):
    """计算最大团大小，超时返回已知最优下界。

    Returns:
        (clique_number, exact): exact=True 时为精确值，False 时为下界。
    """
    if G.number_of_nodes() == 0:
        return 0, True
    t0 = time.time()
    best = 0
    for clique in nx.find_cliques(G):
        size = len(clique)
        if size > best:
            best = size
        if time.time() - t0 > timeout_s:
            return best, False
    return best, True


def _sp_diameter_and_avg(G, cc_nodes):
    """用 scipy 编译后端计算最大连通分量的直径和平均最短路。

    比 nx.diameter + nx.average_shortest_path_length 快 ~100 倍。
    """
    cc_list = sorted(cc_nodes)
    n_cc = len(cc_list)
    if n_cc < 2:
        return 0, 0.0

    idx_map = {v: i for i, v in enumerate(cc_list)}
    rows, cols = [], []
    for u, v in G.subgraph(cc_list).edges():
        i, j = idx_map[u], idx_map[v]
        rows.extend([i, j])
        cols.extend([j, i])
    adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                     shape=(n_cc, n_cc))

    sp = shortest_path(adj, method='D', unweighted=True)
    finite = sp[sp != np.inf]
    diameter = int(finite.max())
    # 排除对角线 (self-loop = 0)
    off_diag = sp[np.triu_indices(n_cc, k=1)]
    off_diag = off_diag[off_diag != np.inf]
    avg_sp = float(off_diag.mean()) if len(off_diag) > 0 else 0.0
    return diameter, avg_sp


# ─────────────────────────── 无向图特征 ───────────────────────────

def extract_undirected_features(G, hmm_states):
    """提取无向图的经典图结构特征。"""
    n = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # ── 连通分量 ──
    components = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    largest_ratio = comp_sizes[0] / n if n > 0 else 0.0

    # ── 割点 (Articulation Points) ──
    art_points = sorted(nx.articulation_points(G))
    art_explore = sum(1 for p in art_points if hmm_states[p] == 0)
    art_exploit = sum(1 for p in art_points if hmm_states[p] == 1)

    # ── 桥边 (Bridges) ──
    bridge_list = list(nx.bridges(G))

    # ── 双连通分量 ──
    biconn = list(nx.biconnected_components(G))
    biconn_sizes = sorted([len(c) for c in biconn], reverse=True) if biconn else []

    # ── 度分布 ──
    degrees = np.array([d for _, d in G.degree()])

    # ── 聚类系数 ──
    global_clustering = nx.transitivity(G)            # 3·triangles / triads
    mean_local_clustering = nx.average_clustering(G)  # mean of per-node CC

    # ── k-core 分解 ──
    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values()) if core_numbers else 0
    core_nodes = [v for v, k in core_numbers.items() if k == max_k]

    # ── 最大团 (Clique Number) — 带超时 ──
    clique_number, clique_exact = _find_clique_number(G)

    # ── 密度 ──
    density = nx.density(G)

    # ── 直径 & 平均最短路（最大连通分量，scipy 加速）──
    largest_cc = max(components, key=len)
    diameter, avg_sp = _sp_diameter_and_avg(G, largest_cc)

    return {
        "n_edges": n_edges,
        "density": round(float(density), 6),
        "connected_components": {
            "count": len(components),
            "sizes": comp_sizes,
            "largest_ratio": round(float(largest_ratio), 4),
        },
        "articulation_points": {
            "count": len(art_points),
            "indices": art_points,
            "explore_count": art_explore,
            "exploit_count": art_exploit,
        },
        "bridges": {
            "count": len(bridge_list),
            "edges": [[int(a), int(b)] for a, b in bridge_list],
        },
        "biconnected_components": {
            "count": len(biconn),
            "sizes": biconn_sizes,
        },
        "degree_stats": {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": round(float(degrees.mean()), 2),
            "std": round(float(degrees.std()), 2),
        },
        "clustering": {
            "global": round(float(global_clustering), 4),
            "mean_local": round(float(mean_local_clustering), 4),
        },
        "k_core": {
            "max_k": int(max_k),
            "core_size": len(core_nodes),
        },
        "clique_number": int(clique_number),
        "clique_exact": clique_exact,
        "diameter": int(diameter),
        "avg_shortest_path": round(float(avg_sp), 2),
    }


# ─────────────────────────── 有向图特征 ───────────────────────────

def extract_directed_features(D, hmm_states):
    """提取有向图的经典图结构特征。

    注意：D 是 DAG（所有弧 i→j 满足 i<j），因此 SCC 均为单点。
    """
    n = D.number_of_nodes()

    # ── 强联通分量 (SCC) ──
    sccs = list(nx.strongly_connected_components(D))
    scc_sizes = sorted([len(c) for c in sccs], reverse=True)
    scc_largest = scc_sizes[0] / n if n > 0 and scc_sizes else 0.0

    # ── 弱联通分量 (WCC) ──
    wccs = list(nx.weakly_connected_components(D))
    wcc_sizes = sorted([len(c) for c in wccs], reverse=True)

    # ── DAG 最长路径长度 ──
    if D.number_of_edges() > 0:
        dag_depth = nx.dag_longest_path_length(D)
    else:
        dag_depth = 0

    # ── 入度 / 出度分布 ──
    in_deg = np.array([d for _, d in D.in_degree()])
    out_deg = np.array([d for _, d in D.out_degree()])

    # ── 可达性：从首 slice 出发可达的节点比例 ──
    if n > 0 and 0 in D:
        reachable = nx.descendants(D, 0)
        reachable.add(0)
        reachability = len(reachable) / n
    else:
        reachability = 0.0

    return {
        "n_arcs": D.number_of_edges(),
        "scc": {
            "count": len(sccs),
            "sizes": scc_sizes[:20],       # DAG 中全为 1，截取前 20
            "largest_ratio": round(float(scc_largest), 4),
        },
        "wcc": {
            "count": len(wccs),
            "sizes": wcc_sizes,
        },
        "dag_depth": int(dag_depth),
        "in_degree_stats": {
            "min": int(in_deg.min()),
            "max": int(in_deg.max()),
            "mean": round(float(in_deg.mean()), 2),
            "std": round(float(in_deg.std()), 2),
        },
        "out_degree_stats": {
            "min": int(out_deg.min()),
            "max": int(out_deg.max()),
            "mean": round(float(out_deg.mean()), 2),
            "std": round(float(out_deg.std()), 2),
        },
        "reachability_from_first": round(float(reachability), 4),
    }


# ─────────────────────────── HMM 交叉分析 ───────────────────────────

def extract_cross_state_features(G, D, hmm_states):
    """按 HMM 状态统计无向图边和有向图弧的分布。"""
    # ── 无向图边分类 ──
    n_explore_edges = 0   # 两端均为 explore (0-0)
    n_exploit_edges = 0   # 两端均为 exploit (1-1)
    n_cross_edges = 0     # 跨状态 (0-1)

    for u, v in G.edges():
        su, sv = hmm_states[u], hmm_states[v]
        if su == 0 and sv == 0:
            n_explore_edges += 1
        elif su == 1 and sv == 1:
            n_exploit_edges += 1
        else:
            n_cross_edges += 1

    total_edges = G.number_of_edges()
    cross_ratio = n_cross_edges / total_edges if total_edges > 0 else 0.0

    # ── 有向图弧的状态转移方向 ──
    d_explore_to_exploit = 0   # 弧 explore→exploit
    d_exploit_to_explore = 0   # 弧 exploit→explore

    for u, v in D.edges():
        su, sv = hmm_states[u], hmm_states[v]
        if su == 0 and sv == 1:
            d_explore_to_exploit += 1
        elif su == 1 and sv == 0:
            d_exploit_to_explore += 1

    return {
        "n_explore_edges": n_explore_edges,
        "n_exploit_edges": n_exploit_edges,
        "n_cross_edges": n_cross_edges,
        "cross_ratio": round(float(cross_ratio), 4),
        "directed_explore_to_exploit": d_explore_to_exploit,
        "directed_exploit_to_explore": d_exploit_to_explore,
    }


# ─────────────────────────── 单样本分析 ───────────────────────────

def analyze_single_sample(problem_id, sample_id):
    """完整分析单个样本的图结构特征。"""
    dist_file = INPUT_DIR / f"dist_p{problem_id}_s{sample_id}.npy"
    hmm_file = INPUT_DIR / f"hmm_p{problem_id}_s{sample_id}.npy"

    if not dist_file.exists() or not hmm_file.exists():
        raise FileNotFoundError(f"数据文件不存在: p{problem_id}_s{sample_id}")

    dist_matrix = np.load(dist_file)
    hmm_states = np.load(hmm_file)
    n = len(hmm_states)

    G, D, threshold = build_sparse_graphs(dist_matrix, hmm_states)

    undirected = extract_undirected_features(G, hmm_states)
    directed = extract_directed_features(D, hmm_states)
    cross_state = extract_cross_state_features(G, D, hmm_states)

    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "n_slices": n,
        "threshold": round(float(threshold), 4),
        "undirected": undirected,
        "directed": directed,
        "cross_state": cross_state,
    }


# ─────────────────────────── 批量分析 ───────────────────────────

def batch_analyze():
    """批量分析所有样本，输出 graph_features.json。"""
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    print("=" * 70)
    print("Graph Structural Features Extraction  (Phase 2.5)")
    print("=" * 70)
    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Problems: {summary['n_problems']},  Samples: {summary['n_samples']}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    all_results = []
    failed = []

    def _sort_key(x):
        try:
            return (0, int(x[0]))
        except ValueError:
            return (1, x[0])

    for pid_str, pdata in sorted(summary["problems"].items(), key=_sort_key):
        samples = pdata["samples"]
        print(f"Problem {pid_str} ({len(samples):2d} samples)...", end=" ", flush=True)

        n_ok = 0
        for sinfo in samples:
            sid = sinfo["sample_id"]
            try:
                result = analyze_single_sample(pid_str, sid)
                all_results.append(result)
                n_ok += 1
            except Exception as e:
                failed.append((pid_str, sid, str(e)))

        print(f"OK {n_ok}/{len(samples)}")

    print()
    print(f"Processed: {len(all_results)} / {summary['n_samples']}  "
          f"(failed: {len(failed)})")

    if failed:
        print(f"\nFailed samples ({len(failed)}):")
        for pid, sid, err in failed[:10]:
            print(f"  p{pid}_s{sid}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # ── 汇总统计 ──
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    # 无向图
    densities     = [r["undirected"]["density"] for r in all_results]
    n_comp        = [r["undirected"]["connected_components"]["count"] for r in all_results]
    largest_ratio = [r["undirected"]["connected_components"]["largest_ratio"] for r in all_results]
    art_counts    = [r["undirected"]["articulation_points"]["count"] for r in all_results]
    bridge_counts = [r["undirected"]["bridges"]["count"] for r in all_results]
    clusterings   = [r["undirected"]["clustering"]["global"] for r in all_results]
    max_ks        = [r["undirected"]["k_core"]["max_k"] for r in all_results]
    clique_nums   = [r["undirected"]["clique_number"] for r in all_results]
    diameters     = [r["undirected"]["diameter"] for r in all_results]
    avg_sps       = [r["undirected"]["avg_shortest_path"] for r in all_results]

    # 有向图
    n_sccs        = [r["directed"]["scc"]["count"] for r in all_results]
    dag_depths    = [r["directed"]["dag_depth"] for r in all_results]
    reachabilities = [r["directed"]["reachability_from_first"] for r in all_results]

    # 交叉分析
    cross_ratios  = [r["cross_state"]["cross_ratio"] for r in all_results]

    output = {
        "summary": {
            "n_samples": len(all_results),
            "undirected": {
                "mean_density":             round(safe_mean(densities), 6),
                "mean_n_components":        round(safe_mean(n_comp), 2),
                "mean_largest_ratio":       round(safe_mean(largest_ratio), 4),
                "mean_articulation_points": round(safe_mean(art_counts), 2),
                "mean_bridges":             round(safe_mean(bridge_counts), 2),
                "mean_clustering":          round(safe_mean(clusterings), 4),
                "mean_max_k_core":          round(safe_mean(max_ks), 2),
                "mean_clique_number":       round(safe_mean(clique_nums), 2),
                "mean_diameter":            round(safe_mean(diameters), 2),
                "mean_avg_shortest_path":   round(safe_mean(avg_sps), 2),
            },
            "directed": {
                "mean_n_scc":           round(safe_mean(n_sccs), 2),
                "mean_dag_depth":       round(safe_mean(dag_depths), 2),
                "mean_reachability":    round(safe_mean(reachabilities), 4),
            },
            "cross_state": {
                "mean_cross_ratio": round(safe_mean(cross_ratios), 4),
            },
        },
        "samples": all_results,
    }

    output_file = OUTPUT_DIR / "graph_features.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # ── 打印汇总 ──
    s = output["summary"]
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    u = s["undirected"]
    print(f"\n  Undirected Graph:")
    print(f"    mean density:            {u['mean_density']:.6f}")
    print(f"    mean components:         {u['mean_n_components']:.2f}")
    print(f"    mean largest ratio:      {u['mean_largest_ratio']:.4f}")
    print(f"    mean articulation pts:   {u['mean_articulation_points']:.2f}")
    print(f"    mean bridges:            {u['mean_bridges']:.2f}")
    print(f"    mean clustering:         {u['mean_clustering']:.4f}")
    print(f"    mean max k-core:         {u['mean_max_k_core']:.2f}")
    print(f"    mean clique number:      {u['mean_clique_number']:.2f}")
    print(f"    mean diameter:           {u['mean_diameter']:.2f}")
    print(f"    mean avg shortest path:  {u['mean_avg_shortest_path']:.2f}")

    d = s["directed"]
    print(f"\n  Directed Graph (DAG, arcs i->j where i<j):")
    print(f"    mean SCC count:          {d['mean_n_scc']:.2f}")
    print(f"    mean DAG depth:          {d['mean_dag_depth']:.2f}")
    print(f"    mean reachability:       {d['mean_reachability']:.4f}")

    cs = s["cross_state"]
    print(f"\n  Cross-State (HMM):")
    print(f"    mean cross ratio:        {cs['mean_cross_ratio']:.4f}")

    print(f"\n  Results saved to: {output_file}")
    print("=" * 70)


def run(batch_dir, output_dir=None):
    """Phase 2.5: extract graph structural features (programmatic API).

    Args:
        batch_dir:  输入目录（含 dist/hmm .npy 及 batch_summary.json）
        output_dir: 输出目录（默认 batch_results_graph/）
    """
    global INPUT_DIR, OUTPUT_DIR, SUMMARY_FILE
    INPUT_DIR = Path(batch_dir)
    SUMMARY_FILE = INPUT_DIR / "batch_summary.json"
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
    batch_analyze()


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5: Extract graph structural features from CoT folding"
    )
    parser.add_argument("--problem_id", type=str,
                        help="Problem ID (single sample mode)")
    parser.add_argument("--sample_id", type=int,
                        help="Sample ID (single sample mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: analyze all samples")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="Input batch_results directory (default: ./batch_results)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./batch_results_graph)")
    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_DIR, SUMMARY_FILE
    if args.batch_dir:
        INPUT_DIR = Path(args.batch_dir)
        SUMMARY_FILE = INPUT_DIR / "batch_summary.json"
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    if args.batch:
        batch_analyze()
    elif args.problem_id is not None and args.sample_id is not None:
        result = analyze_single_sample(args.problem_id, args.sample_id)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
