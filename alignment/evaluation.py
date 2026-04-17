#!/usr/bin/env python3
"""Evaluation report loading utilities.

This module provides functions to load and access evaluation results
from cache directories.

Usage:
    from alignment.evaluation import load_evaluation_results

    eval_results = load_evaluation_results(cache_path)
    # Returns: {idx: {"is_correct": bool, "runs": [...], ...}, ...}
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_full_evaluation_report(cache_dir: Path) -> Dict[str, Any]:
    """Load full evaluation_report.json (contains generated_text/CoT content).

    完整版 evaluation_report.json 包含 generated_text 字段，即完整的 CoT 推理内容。
    文件较大 (~100MB)，包含所有 run 的完整文本。

    Args:
        cache_dir: Path to cache directory

    Returns:
        Complete evaluation report dictionary:
        {
            "test_info": {...},
            "results": [
                {
                    "problem_id": "...",
                    "runs": [
                        {
                            "run_index": 1,
                            "generated_text": "<think>...</think>\\boxed{...}",
                            "is_correct": True,
                            ...
                        },
                        ...
                    ]
                },
                ...
            ]
        }

    Raises:
        FileNotFoundError: If evaluation_report.json is missing
    """
    cache_dir = Path(cache_dir)
    report_path = cache_dir / "evaluation_report.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation_report.json (full version): {report_path}"
        )

    return _load_json(report_path)


def get_run_data(
    full_report: Dict[str, Any], problem_id: Any, run_index: int
) -> Optional[Dict[str, Any]]:
    """Get complete run data for a specific problem and run.

    Args:
        full_report: Full evaluation report from load_full_evaluation_report()
        problem_id: Problem ID (string or int, will be compared flexibly)
        run_index: Run index (1-based)

    Returns:
        Complete run dictionary with all fields:
        {
            "run_index": int,
            "actual_prompt": str,
            "inference_time": float,
            "input_tokens": int,
            "output_tokens": int,
            "tokens_per_second": float,
            "time_per_token_ms": float,
            "stop_reason": str or None,
            "finish_reason": str,
            "generated_text": str,  # CoT content
            "extracted_answer": str,
            "is_correct": bool,
        }
        Returns None if not found.
    """
    problem_id_str = str(problem_id)
    for problem in full_report.get("results", []):
        if str(problem.get("problem_id", "")) == problem_id_str:
            for run in problem.get("runs", []):
                if run.get("run_index") == run_index:
                    return run
    return None


def get_generated_text(
    full_report: Dict[str, Any], problem_id: Any, run_index: int
) -> str:
    """Get generated_text (CoT content) for a specific problem and run.

    Args:
        full_report: Full evaluation report from load_full_evaluation_report()
        problem_id: Problem ID (string or int, will be compared flexibly)
        run_index: Run index (1-based)

    Returns:
        Generated text (CoT content), or empty string if not found
    """
    run = get_run_data(full_report, problem_id, run_index)
    return run.get("generated_text", "") if run else ""


def load_evaluation_results(cache_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load evaluation_report_compact.json and build idx -> entry mapping.

    Args:
        cache_dir: Path to cache directory

    Returns:
        Dictionary mapping index to evaluation entry:
        {
            0: {"problem_id": "...", "is_correct": True, "runs": [...], ...},
            1: {...},
            ...
        }

    Raises:
        FileNotFoundError: If evaluation_report_compact.json is missing
    """
    cache_dir = Path(cache_dir)
    report_path = cache_dir / "evaluation_report_compact.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation_report_compact.json: {report_path}"
        )

    data = _load_json(report_path)
    results = data.get("results", [])
    mapping: Dict[int, Dict[str, Any]] = {}

    if isinstance(results, list):
        for idx, entry in enumerate(results):
            mapping[int(idx)] = entry
    elif isinstance(results, dict):
        for key, entry in results.items():
            try:
                idx = int(key)
            except ValueError:
                warnings.warn(
                    f"evaluation results 中存在无效 key: {key!r} (无法转换为整数，已跳过)"
                )
                continue
            mapping[idx] = entry

    return mapping


def load_evaluation_results_by_problem_id(
    cache_dir: Path,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load evaluation_report_compact.json and build problem_id -> {run_index: run} mapping.

    参考 verify_viz_eval_mapping.py 的实现方式:
    - 用 problem_id 字符串作为第一层 key
    - 用 run_index (1-based) 作为第二层 key

    Args:
        cache_dir: Path to cache directory

    Returns:
        Dictionary mapping problem_id to run_index -> run mapping:
        {
            "HumanEval/102": {1: {...}, 2: {...}, ...},
            "HumanEval/103": {1: {...}, 2: {...}, ...},
            ...
        }

    Raises:
        FileNotFoundError: If evaluation_report_compact.json is missing
    """
    from alignment_lite.core import load_eval_by_pid
    return load_eval_by_pid(cache_dir)


def get_runs_for_problem_by_id(
    eval_indexed: Dict[str, Dict[int, Dict[str, Any]]],
    problem_id: str,
) -> Dict[int, Dict[str, Any]]:
    """Get run_index -> run mapping for a problem by problem_id string.

    参考 verify_viz_eval_mapping.py line 112:
    eval_runs = eval_indexed.get(str(problem_id), {})

    Args:
        eval_indexed: Mapping from load_evaluation_results_by_problem_id()
        problem_id: Problem ID string (e.g., "HumanEval/102")

    Returns:
        Dictionary mapping run_index to run data: {1: {...}, 2: {...}, ...}
    """
    return eval_indexed.get(str(problem_id), {})


def get_runs_for_problem(
    eval_results: Dict[int, Dict[str, Any]], problem_idx: int
) -> list[Dict[str, Any]]:
    """Get list of runs for a problem from evaluation results.

    Args:
        eval_results: Mapping from load_evaluation_results()
        problem_idx: Problem index

    Returns:
        List of run dictionaries with run_index, is_correct, etc.
    """
    entry = eval_results.get(problem_idx, {})
    return entry.get("runs", [])


def resolve_n_runs(eval_results: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Resolve the number of runs per problem from evaluation results.

    Args:
        eval_results: Mapping from load_evaluation_results()

    Returns:
        Number of runs per problem, or None if not found
    """
    for entry in eval_results.values():
        n_runs = entry.get("n_runs")
        if n_runs:
            return int(n_runs)
    return None


def get_correct_indices(
    eval_results: Dict[int, Dict[str, Any]], problem_idx: int
) -> list[int]:
    """Get list of correct run indices for a problem.

    Args:
        eval_results: Mapping from load_evaluation_results()
        problem_idx: Problem index

    Returns:
        List of run_index values for correct responses
    """
    runs = get_runs_for_problem(eval_results, problem_idx)
    return [
        int(run.get("run_index", idx))
        for idx, run in enumerate(runs)
        if run.get("is_correct", False)
    ]


def compute_problem_accuracy(
    eval_results: Dict[int, Dict[str, Any]], problem_idx: int
) -> float:
    """Compute accuracy for a problem (fraction of correct runs).

    Args:
        eval_results: Mapping from load_evaluation_results()
        problem_idx: Problem index

    Returns:
        Accuracy as fraction [0.0, 1.0]
    """
    runs = get_runs_for_problem(eval_results, problem_idx)
    if not runs:
        return 0.0
    n_correct = sum(1 for run in runs if run.get("is_correct", False))
    return n_correct / len(runs)


def get_problem_id_from_eval_report(cache_path: Path, problem_idx: int) -> str:
    """从 evaluation_report_compact.json 获取指定索引的 problem_id。

    严格模式: 只使用 evaluation_report_compact.json，不 fallback。

    Args:
        cache_path: Cache 目录路径
        problem_idx: 问题索引 (0-based)

    Returns:
        problem_id 字符串

    Raises:
        FileNotFoundError: evaluation_report_compact.json 不存在
        ValueError: problem_idx 超出范围或解析失败
    """
    cache_path = Path(cache_path)

    report_path = cache_path / "evaluation_report_compact.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"evaluation_report_compact.json 不存在: {report_path}"
        )

    try:
        report = _load_json(report_path)
    except Exception as e:
        raise ValueError(f"Failed to parse evaluation report {report_path}: {e}")

    results = report.get("results")
    if not isinstance(results, list) or problem_idx >= len(results):
        raise ValueError(
            f"problem_idx {problem_idx} out of range "
            f"(results has {len(results) if isinstance(results, list) else 0} entries)"
        )

    return str(results[problem_idx].get("problem_id", problem_idx))
