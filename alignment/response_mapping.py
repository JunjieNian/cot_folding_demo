#!/usr/bin/env python3
"""Response mapping utilities for building sample_id mappings.

This module provides strict mapping between response indices and global sample_ids.
No fallback logic - if required data is missing, raises an error.

Usage:
    from alignment.response_mapping import build_response_mapping, merge_responses

    # Merge viz responses with eval runs (with automatic validation)
    merged = merge_responses(
        metadata_responses, eval_runs,
        cache_path=cache_path, problem_id=problem_id, viz_dir=viz_dir
    )

    # Build mapping (requires eval_runs)
    sample_ids, run_indices = build_response_mapping(
        merged, problem_id, meta_mapping, eval_runs
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def build_response_mapping(
    responses: List[Dict[str, Any]],
    problem_id: str,
    meta_mapping: Dict[str, Dict[str, Any]],
    eval_runs: List[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    """Build response to sample_id mapping using eval_runs (strict mode).

    严格模式: 必须提供 eval_runs，不允许 fallback。

    Args:
        responses: List of response dictionaries from viz metadata
        problem_id: Problem identifier
        meta_mapping: Mapping from load_meta_mapping()
        eval_runs: List of eval run dictionaries with run_index field (required)

    Returns:
        Tuple of (response_sample_ids, response_run_indices)
        - response_sample_ids: List of global sample_ids for each response
        - response_run_indices: List of local run_indices for each response

    Raises:
        ValueError: If responses is empty, eval_runs is empty, or mapping fails

    Example:
        >>> sample_ids, run_indices = build_response_mapping(
        ...     responses, "problem_1", meta_mapping, eval_runs
        ... )
    """
    if not responses:
        raise ValueError(f"[{problem_id}] responses 为空")

    if not eval_runs:
        raise ValueError(f"[{problem_id}] eval_runs 为空，严格模式下必须提供")

    if len(responses) != len(eval_runs):
        raise ValueError(
            f"[{problem_id}] responses 与 eval_runs 数量不匹配: "
            f"responses={len(responses)}, eval_runs={len(eval_runs)}"
        )

    mapping_entry = meta_mapping.get(str(problem_id))
    if not mapping_entry:
        raise ValueError(f"[{problem_id}] 在 meta_mapping 中不存在")

    run_to_sample = mapping_entry.get("run_to_sample", {})
    if not run_to_sample:
        raise ValueError(f"[{problem_id}] run_to_sample 映射为空")

    response_sample_ids: List[int] = []
    response_run_indices: List[int] = []

    for idx, run in enumerate(eval_runs):
        run_index = run.get("run_index")
        if run_index is None:
            raise ValueError(
                f"[{problem_id}] eval_runs[{idx}] 缺少 run_index 字段"
            )

        run_index = int(run_index)
        sample_id = run_to_sample.get(run_index)

        if sample_id is None:
            raise ValueError(
                f"[{problem_id}] run_index={run_index} 在 run_to_sample 中不存在"
            )

        response_run_indices.append(run_index)
        response_sample_ids.append(int(sample_id))

    return response_sample_ids, response_run_indices


def _warn_field_mismatches(
    metadata_responses: List[Dict[str, Any]],
    eval_runs_by_idx: Dict[int, Dict[str, Any]],
    problem_id: Union[str, int],
) -> None:
    """检查 viz 与 eval 字段不一致，发出汇总 warning。

    检查字段: is_correct, finish_reason
    """
    mismatch_counts = {"is_correct": 0, "finish_reason": 0}
    mismatch_samples: List[Dict[str, Any]] = []

    for idx in range(len(metadata_responses)):
        resp = metadata_responses[idx]
        run_index = idx + 1
        run = eval_runs_by_idx.get(run_index)
        if run is None:
            continue

        viz_is_correct = resp.get("is_correct")
        eval_is_correct = run.get("is_correct")
        viz_finish = resp.get("finish_reason")
        eval_finish = run.get("finish_reason")

        if viz_is_correct is not None and viz_is_correct != eval_is_correct:
            mismatch_counts["is_correct"] += 1
            if len(mismatch_samples) < 3:
                mismatch_samples.append({
                    "field": "is_correct",
                    "run_index": run_index,
                    "viz": viz_is_correct,
                    "eval": eval_is_correct,
                })

        if viz_finish is not None and viz_finish != eval_finish:
            mismatch_counts["finish_reason"] += 1
            if len(mismatch_samples) < 3:
                mismatch_samples.append({
                    "field": "finish_reason",
                    "run_index": run_index,
                    "viz": viz_finish,
                    "eval": eval_finish,
                })

    total_mismatches = mismatch_counts["is_correct"] + mismatch_counts["finish_reason"]
    if total_mismatches > 0:
        sample_details = "; ".join(
            f"{s['field']}[run_index={s['run_index']}]: viz={s['viz']}, eval={s['eval']}"
            for s in mismatch_samples
        )
        warnings.warn(
            f"[problem_id={problem_id}] viz metadata 与 eval report 存在不一致: "
            f"is_correct={mismatch_counts['is_correct']}, "
            f"finish_reason={mismatch_counts['finish_reason']} 处不匹配。"
            f"已使用 eval report 的值 (权威来源)。"
            f"示例: {sample_details}",
            UserWarning,
            stacklevel=2,
        )


def merge_responses(
    metadata_responses: List[Dict[str, Any]],
    eval_runs_by_idx: Dict[int, Dict[str, Any]],
    *,
    cache_path: Union[str, Path],
    problem_id: Union[str, int],
    viz_dir: Optional[Union[str, Path]] = None,
    sep_up: int = 8,
    validate: bool = True,
    warn_on_mismatch: bool = True,
) -> List[Dict[str, Any]]:
    """Merge visualization metadata responses with evaluation runs.

    参考 verify_viz_eval_mapping.py 的实现方式:
    - eval_runs_by_idx 是 {run_index: run} 字典格式
    - 用 1-based run_index 进行精确匹配

    This function combines information from two sources:
    - metadata_responses: From viz metadata (contains knn/slice info, GLOBAL run_index)
    - eval_runs_by_idx: From evaluation report, indexed by run_index (1-based)

    重要: metadata_responses 的 run_index 是 GLOBAL sample_id，
         eval_runs 的 run_index 是 LOCAL per-problem index。
         此函数保留 viz metadata 的 GLOBAL sample_id，不会覆盖！

    强制验证: cache_path 和 problem_id 是必需参数，用于验证 sample_id 正确性。

    Args:
        metadata_responses: List of response dicts from viz metadata
        eval_runs_by_idx: Dict mapping run_index (1-based) to run data
            格式: {1: {...}, 2: {...}, ...}
        cache_path: NAD cache 路径 (必需，用于验证)
        problem_id: 当前问题 ID (必需，用于验证)
        viz_dir: viz 子目录路径 (用于长度验证)
        sep_up: slice_info 的 upsampling 因子
        validate: 是否启用验证 (默认 True)
        warn_on_mismatch: 当 viz metadata 与 eval report 字段值不一致时发出警告 (默认 True)
            检查字段: is_correct, finish_reason
            警告表示数据源可能不同步，但会使用 eval report 的值 (权威来源)

    Returns:
        Merged list of response dictionaries

    Raises:
        ValueError: 如果数量不匹配或验证失败 (sample_id 归属错误或长度不匹配)

    Example:
        >>> merged = merge_responses(
        ...     metadata, eval_runs_by_idx,
        ...     cache_path=cache_path, problem_id="HumanEval/102", viz_dir=viz_dir
        ... )
    """
    if warn_on_mismatch:
        _warn_field_mismatches(metadata_responses, eval_runs_by_idx, problem_id)

    from alignment_lite.core import merge_responses as _merge_lite
    merged = _merge_lite(metadata_responses, eval_runs_by_idx, str(problem_id))

    if validate:
        from .verification import validate_response_sample_ids

        validate_response_sample_ids(
            merged,
            cache_path,
            problem_id,
            viz_dir=viz_dir,
            sep_up=sep_up,
            strict=True,
        )

    return merged
