#!/usr/bin/env python3
"""
KNN 图数据加载模块 (严格模式)

提供统一的数据加载 API，消除各脚本中的重复代码。

Usage:
    from alignment import load_all_problems_data

    problems = load_all_problems_data(cache_path, sep_up=8)
    for prob in problems:
        knn_indices = prob["knn_indices"]
        responses = prob["responses"]
        ...

严格模式: 所有函数不允许 fallback，缺少必要数据时直接报错。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .response_mapping import merge_responses


def load_knn_data(
    viz_dir: Path,
    sep_up: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载单个问题的 KNN 数据 (严格模式)。

    Args:
        viz_dir: viz 问题子目录 (如 viz/0/)
        sep_up: 上采样因子 (1/2/4/8/16)

    Returns:
        (knn_indices, knn_dists, slice_info)

    Raises:
        FileNotFoundError: 缺少 KNN 文件
    """
    from alignment_lite.core import load_knn_data as _load
    return _load(viz_dir, sep_up)


def load_problem_metadata(viz_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    加载单个问题的 metadata.json。

    Args:
        viz_dir: viz 问题子目录

    Returns:
        (problem_id, responses)

    Raises:
        FileNotFoundError: 缺少 metadata.json
    """
    metadata_path = viz_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"缺少 metadata.json: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    problem_id = str(metadata.get("problem_id", ""))
    responses = metadata.get("responses", [])

    return problem_id, responses


def load_problem_data(
    cache_path: Path,
    viz_dir: Path,
    problem_idx: int,
    sep_up: int = 8,
    eval_indexed: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    加载并合并单个问题的完整数据。

    参考 verify_viz_eval_mapping.py 的实现方式:
    - 用 problem_id 字符串匹配 eval report (不是用数字索引)
    - 用 run_index (1-based) 字典查找 eval_runs

    Args:
        cache_path: cache 目录路径
        viz_dir: viz 问题子目录
        problem_idx: 问题索引 (仅用于返回值，不用于匹配)
        sep_up: 上采样因子
        eval_indexed: 预加载的评估结果 (可选，用于批量优化)
            格式: {problem_id: {run_index: run_data}}

    Returns:
        {
            "viz_idx": int,
            "problem_id": str,
            "knn_indices": ndarray,
            "knn_dists": ndarray,
            "slice_info": ndarray,
            "responses": List[Dict],
            "n_correct": int,
            "n_responses": int,
        }
    """
    from alignment_lite.core import load_knn_data as _knn, load_eval_by_pid as _eval

    # 加载 KNN 数据
    knn_indices, knn_dists, slice_info = _knn(viz_dir, sep_up)

    # 加载 metadata (获取 problem_id 字符串)
    problem_id, responses = load_problem_metadata(viz_dir)

    # 加载/复用 eval_indexed (按 problem_id 字符串索引)
    if eval_indexed is None:
        eval_indexed = _eval(cache_path)

    # 用 problem_id 字符串查找 eval_runs
    eval_runs_by_idx = eval_indexed.get(str(problem_id), {})

    if not eval_runs_by_idx:
        raise ValueError(
            f"problem_id='{problem_id}' 在 evaluation_report 中不存在。"
            f"可用的 problem_id: {list(eval_indexed.keys())[:5]}..."
        )

    # 合并 eval 数据 (带 validation 包装)
    responses = merge_responses(
        responses, eval_runs_by_idx,
        cache_path=cache_path,
        problem_id=problem_id,
        viz_dir=viz_dir,
        sep_up=sep_up,
    )

    return {
        "viz_idx": problem_idx,
        "problem_id": problem_id,
        "knn_indices": knn_indices,
        "knn_dists": knn_dists,
        "slice_info": slice_info,
        "responses": responses,
        "n_correct": sum(1 for r in responses if r.get("is_correct")),
        "n_responses": len(responses),
    }


def load_all_problems_data(
    cache_path: Path,
    sep_up: int = 8,
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    加载 cache 的所有问题数据 (严格模式)。

    这是主要的公开 API，替代 run_batch_analysis.py 和
    analyze_knn_answer_alignment.py 中的重复实现。

    参考 verify_viz_eval_mapping.py 的实现方式:
    - 用 problem_id 字符串匹配 eval report (不是用数字索引)
    - 用 run_index (1-based) 字典查找 eval_runs

    Args:
        cache_path: cache 目录路径
        sep_up: KNN 图上采样因子 (1/2/4/8/16)
        max_problems: 最多加载的问题数 (None = 全部)

    Returns:
        问题数据列表，每个元素包含:
        - viz_idx, problem_id
        - knn_indices, knn_dists, slice_info
        - responses (已合并 eval 数据)
        - n_correct, n_responses

    Raises:
        FileNotFoundError: viz 目录不存在
        FileNotFoundError: 缺少 evaluation_report_compact.json
        FileNotFoundError: 缺少指定 sep_up 的 knn 文件

    Example:
        from alignment import load_all_problems_data

        problems = load_all_problems_data(cache_path, sep_up=8)
        for prob in problems:
            print(f"Problem {prob['problem_id']}: {prob['n_correct']}/{prob['n_responses']}")
    """
    from alignment_lite.core import load_problems
    return load_problems(cache_path, sep_up=sep_up, max_problems=max_problems)
