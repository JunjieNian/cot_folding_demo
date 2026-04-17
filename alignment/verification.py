#!/usr/bin/env python3
"""交叉验证模块：通过 is_correct/finish_reason 验证 viz-eval 对齐。

此模块提供 viz metadata 与 evaluation report 的交叉验证功能。

核心原理:
  - is_correct 和 finish_reason 是与 response 内容强相关的字段
  - 如果 viz 与 eval 的这些字段不匹配，说明索引映射可能有错误

用法:
    from alignment.verification import validate_viz_eval_alignment

    result = validate_viz_eval_alignment(viz_dir, eval_results)
    if not result.is_valid:
        raise ValueError(result.message)

来源: sc_slice_compress.py:326-418 (_validate_viz_eval_alignment)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class VerificationResult:
    """验证结果"""

    is_valid: bool  # 是否通过验证
    total_checked: int  # 检查的总数
    matched: int  # 匹配的数量
    match_rate: float  # 匹配率 (0.0 - 1.0)
    mismatches: List[str] = field(default_factory=list)  # 不匹配详情
    message: str = ""  # 人类可读的验证消息


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_viz_indices(viz_dir: Path) -> List[int]:
    """Discover numeric subdirectories in viz_dir."""
    indices = []
    if not viz_dir.exists():
        return indices
    for entry in viz_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            indices.append(int(entry.name))
    return sorted(indices)


def validate_viz_eval_alignment(
    viz_dir: Path,
    eval_results: List[Dict[str, Any]],
    strict: bool = True,
) -> VerificationResult:
    """
    验证 viz metadata 与 eval 的对齐关系。

    检查项:
    1. 每个 viz problem_id 必须在 eval 中存在
    2. 每个 viz 的 responses 数量必须与 eval 的 runs 数量一致
    3. 每个 response 的 is_correct 必须与 eval 一致
    4. 每个 response 的 finish_reason 必须与 eval 一致 (严格匹配)

    Args:
        viz_dir: viz 数据目录 (包含 0/, 1/, 2/, ... 子目录)
        eval_results: evaluation_report_compact.json 的 results 数组
        strict: 严格模式，任何不匹配都标记为失败

    Returns:
        VerificationResult 包含验证结果和详情

    Raises:
        ValueError: 如果 strict=True 且验证失败
    """
    viz_dir = Path(viz_dir)

    # 构建 eval problem_id 索引 (关键: 用 problem_id 匹配, 不是位置!)
    eval_by_pid: Dict[str, Dict[str, Any]] = {
        str(e.get("problem_id", "")): e for e in eval_results
    }

    viz_indices = _discover_viz_indices(viz_dir)
    mismatches: List[str] = []
    total_checked = 0
    matched = 0

    for viz_idx in viz_indices:
        metadata_path = viz_dir / str(viz_idx) / "metadata.json"
        if not metadata_path.exists():
            warnings.warn(
                f"viz/{viz_idx}: metadata.json 不存在，跳过该问题验证",
                UserWarning, stacklevel=2,
            )
            continue

        metadata = _load_json(metadata_path)
        viz_pid = str(metadata.get("problem_id", ""))
        viz_responses = metadata.get("responses", [])

        if not viz_pid:
            mismatches.append(f"viz/{viz_idx}: problem_id 为空")
            continue

        # 检查 1: viz_pid 在 eval 中存在
        if viz_pid not in eval_by_pid:
            mismatches.append(
                f"viz/{viz_idx}: problem_id='{viz_pid}' 在 eval 中不存在"
            )
            continue

        eval_entry = eval_by_pid[viz_pid]
        eval_runs = eval_entry.get("runs", [])

        # 检查 2: run 数量一致
        if len(viz_responses) != len(eval_runs):
            mismatches.append(
                f"viz/{viz_idx}: problem_id='{viz_pid}' run_count 不一致 "
                f"(viz={len(viz_responses)}, eval={len(eval_runs)})"
            )
            continue  # 数量不一致则跳过字段检查

        # 检查 3-4: 每个 response 的关键字段与 eval 一致
        # 构建 run_index -> run 的字典 (参考 verify_viz_eval_mapping.py)
        # 不依赖数组顺序，而是通过 run_index 字段进行精确匹配
        eval_runs_by_idx: Dict[int, Dict[str, Any]] = {
            run["run_index"]: run for run in eval_runs if "run_index" in run
        }

        for resp_idx, viz_resp in enumerate(viz_responses):
            total_checked += 1
            resp_matched = True  # 当前 response 是否完全匹配

            # 1-based mapping: viz_responses[i] -> eval_runs[run_index = i+1]
            run_index = resp_idx + 1
            eval_run = eval_runs_by_idx.get(run_index)

            if eval_run is None:
                mismatches.append(
                    f"viz/{viz_idx}: problem_id='{viz_pid}' response[{resp_idx}] "
                    f"run_index={run_index} 在 eval_runs 中不存在"
                )
                continue

            # 检查 is_correct (严格匹配)
            viz_is_correct = bool(viz_resp.get("is_correct", False))
            eval_is_correct = bool(eval_run.get("is_correct", False))

            if viz_is_correct != eval_is_correct:
                mismatches.append(
                    f"viz/{viz_idx}: problem_id='{viz_pid}' response[{resp_idx}] "
                    f"is_correct 不一致 (viz={viz_is_correct}, eval={eval_is_correct})"
                )
                resp_matched = False

            # 检查 finish_reason (严格匹配: 空值视为不匹配，计入 match_rate)
            viz_finish = str(viz_resp.get("finish_reason", "")).strip().lower()
            eval_finish = str(eval_run.get("finish_reason", "")).strip().lower()

            if viz_finish != eval_finish:
                mismatches.append(
                    f"viz/{viz_idx}: problem_id='{viz_pid}' response[{resp_idx}] "
                    f"finish_reason 不一致 (viz='{viz_finish}', eval='{eval_finish}')"
                )
                resp_matched = False

            if resp_matched:
                matched += 1

    # 计算结果
    match_rate = matched / total_checked if total_checked > 0 else 1.0
    is_valid = len(mismatches) == 0 if strict else match_rate >= 0.95

    if len(mismatches) == 0:
        message = f"验证通过: {matched}/{total_checked} responses 完全匹配"
    else:
        error_lines = [f"验证失败: 发现 {len(mismatches)} 个问题"]
        error_lines.extend(f"  - {m}" for m in mismatches[:20])
        if len(mismatches) > 20:
            error_lines.append(f"  ... 还有 {len(mismatches) - 20} 个问题")
        message = "\n".join(error_lines)

    result = VerificationResult(
        is_valid=is_valid,
        total_checked=total_checked,
        matched=matched,
        match_rate=match_rate,
        mismatches=mismatches,
        message=message,
    )

    if strict and not is_valid:
        raise ValueError(message)

    return result


def validate_response_mapping(
    responses: List[Dict[str, Any]],
    eval_runs: List[Dict[str, Any]],
    problem_id: str,
) -> VerificationResult:
    """
    验证单个 problem 的 response 与 eval_runs 的对齐。

    这是 validate_viz_eval_alignment 的简化版本，用于单个 problem 的验证。
    使用严格模式: is_correct 和 finish_reason 都必须完全匹配。

    Args:
        responses: viz metadata 中的 responses 列表
        eval_runs: evaluation report 中的 runs 列表
        problem_id: 问题 ID

    Returns:
        VerificationResult 包含验证结果
    """
    mismatches: List[str] = []
    total_checked = 0
    matched = 0

    if len(responses) != len(eval_runs):
        return VerificationResult(
            is_valid=False,
            total_checked=0,
            matched=0,
            match_rate=0.0,
            mismatches=[
                f"[{problem_id}] run_count 不一致: "
                f"responses={len(responses)}, eval_runs={len(eval_runs)}"
            ],
            message=f"[{problem_id}] run 数量不一致",
        )

    # 构建 run_index -> run 的字典 (参考 verify_viz_eval_mapping.py)
    # 不依赖数组顺序，而是通过 run_index 字段进行精确匹配
    eval_runs_by_idx: Dict[int, Dict[str, Any]] = {
        run["run_index"]: run for run in eval_runs if "run_index" in run
    }

    for resp_idx, resp in enumerate(responses):
        total_checked += 1
        resp_matched = True  # 当前 response 是否完全匹配

        # 1-based mapping: responses[i] -> eval_runs[run_index = i+1]
        run_index = resp_idx + 1
        eval_run = eval_runs_by_idx.get(run_index)

        if eval_run is None:
            mismatches.append(
                f"[{problem_id}] response[{resp_idx}] run_index={run_index} "
                f"在 eval_runs 中不存在"
            )
            continue

        # 检查 is_correct (严格匹配)
        viz_is_correct = bool(resp.get("is_correct", False))
        eval_is_correct = bool(eval_run.get("is_correct", False))

        if viz_is_correct != eval_is_correct:
            mismatches.append(
                f"[{problem_id}] response[{resp_idx}] is_correct: "
                f"viz={viz_is_correct}, eval={eval_is_correct}"
            )
            resp_matched = False

        # 检查 finish_reason (严格匹配: 空值视为不匹配)
        viz_finish = str(resp.get("finish_reason", "")).strip().lower()
        eval_finish = str(eval_run.get("finish_reason", "")).strip().lower()

        if viz_finish != eval_finish:
            mismatches.append(
                f"[{problem_id}] response[{resp_idx}] finish_reason: "
                f"viz='{viz_finish}', eval='{eval_finish}'"
            )
            resp_matched = False

        if resp_matched:
            matched += 1

    match_rate = matched / total_checked if total_checked > 0 else 1.0
    is_valid = len(mismatches) == 0

    if is_valid:
        message = f"[{problem_id}] 验证通过: {matched}/{total_checked} 匹配"
    else:
        message = f"[{problem_id}] 验证失败: {len(mismatches)} 处不匹配"

    return VerificationResult(
        is_valid=is_valid,
        total_checked=total_checked,
        matched=matched,
        match_rate=match_rate,
        mismatches=mismatches,
        message=message,
    )


def validate_response_sample_ids(
    responses: List[Dict[str, Any]],
    cache_path: Union[str, Path],
    problem_id: Union[str, int],
    viz_dir: Optional[Union[str, Path]] = None,
    sep_up: int = 8,
    strict: bool = True,
) -> VerificationResult:
    """
    端到端验证: 验证 responses 的 run_index (sample_id) 是否正确。

    双重验证逻辑:
    1. problem_id 归属检查: sample_id 对应的 problem_id 是否与当前 problem 一致
    2. token 长度范围检查: 用 slice_info 的 slice 数量计算期望的 token 范围

    原理:
      - viz metadata 的 run_index 应该是 GLOBAL sample_id (用于索引 token_row_ptr)
      - 如果 merge_responses 错误地覆盖了 run_index (用 LOCAL index)
      - 这两个检查都能捕获到错误

    Args:
        responses: viz metadata 的 responses 列表，每个元素应包含:
            - run_index: GLOBAL sample_id
        cache_path: NAD cache 路径
        problem_id: 当前处理的 problem ID
        viz_dir: viz 子目录路径 (包含 metadata.json 和 slice_info)，用于长度验证
        sep_up: slice_info 的 upsampling 因子 (默认 8)
        strict: 严格模式，任何错误都抛出异常

    Returns:
        VerificationResult 包含验证结果

    Raises:
        ValueError: 如果 strict=True 且验证失败

    Example:
        >>> responses = [{"run_index": 64}, ...]
        >>> validate_response_sample_ids(responses, cache_path, problem_id="61")
        VerificationResult(is_valid=True, ...)

        >>> # 如果 run_index 被错误替换为 LOCAL index
        >>> wrong_responses = [{"run_index": 1}, ...]
        >>> validate_response_sample_ids(wrong_responses, cache_path, problem_id="61")
        # ValueError: sample_id=1 属于 problem 60, 不是 61
    """
    cache_path = Path(cache_path)
    problem_id_str = str(problem_id)

    errors: List[str] = []
    total_checked = 0
    passed = 0

    # 1. 加载 token_row_ptr (用于长度验证)
    token_row_ptr = None
    token_row_ptr_path = cache_path / "token_data" / "token_row_ptr.int64"
    if token_row_ptr_path.exists():
        try:
            token_row_ptr = np.memmap(token_row_ptr_path, dtype=np.int64, mode="r")
        except Exception as e:
            warnings.warn(
                f"token_row_ptr 加载失败，跳过长度验证: {e}",
                UserWarning, stacklevel=2,
            )

    # 2. 加载 cache meta.json (用于归属验证)
    samples: List[Dict[str, Any]] = []
    meta_path = cache_path / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                samples = json.load(f).get("samples", [])
        except Exception as e:
            warnings.warn(
                f"meta.json 加载失败，跳过归属验证: {e}",
                UserWarning, stacklevel=2,
            )

    # 3. 加载 slice_info 和 config_details (用于长度验证)
    resp_slice_counts: Dict[int, int] = {}
    effective_pos_size = 32 * sep_up  # 默认值

    if viz_dir is not None:
        viz_dir = Path(viz_dir)
        slice_info_path = viz_dir / f"sep_up{sep_up}x_slice_info.npy"
        metadata_path = viz_dir / "metadata.json"

        # 读取 slice_info
        if slice_info_path.exists():
            try:
                slice_info = np.load(slice_info_path)
                # 统计每个 response 的 slice 数量
                from collections import Counter
                resp_slice_counts = dict(Counter(slice_info[:, 0]))
            except Exception as e:
                warnings.warn(
                    f"slice_info 加载失败: {e}",
                    UserWarning, stacklevel=2,
                )

        # 读取 config_details 获取 effective_pos_size
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    viz_meta = json.load(f)
                config = viz_meta.get("config_details", {}).get(f"sep_up{sep_up}x", {})
                effective_pos_size = config.get("effective_pos_size", 32 * sep_up)
            except Exception as e:
                warnings.warn(
                    f"config_details 加载失败，使用默认 effective_pos_size={32 * sep_up}: {e}",
                    UserWarning, stacklevel=2,
                )

    # 如果两个数据源都不可用，跳过验证
    if not samples and token_row_ptr is None:
        warnings.warn(
            f"[{problem_id_str}] 验证数据全部不可用 (meta.json 和 token_row_ptr 均加载失败)，"
            f"sample_id 验证被跳过",
            UserWarning, stacklevel=2,
        )
        return VerificationResult(
            is_valid=True,
            total_checked=0,
            matched=0,
            match_rate=1.0,
            mismatches=[],
            message="无法加载验证数据，跳过 sample_id 验证",
        )

    for i, resp in enumerate(responses):
        sample_id = resp.get("sample_id", resp.get("run_index"))

        total_checked += 1
        resp_passed = True

        # 检查 0: run_index 存在性
        if sample_id is None:
            errors.append(f"response[{i}]: run_index 为 None")
            resp_passed = False
            continue

        # 检查 1: 范围检查 (使用 token_row_ptr)
        if token_row_ptr is not None:
            if sample_id < 0 or sample_id >= len(token_row_ptr) - 1:
                errors.append(
                    f"response[{i}]: sample_id={sample_id} 超出 token_row_ptr 范围 "
                    f"[0, {len(token_row_ptr) - 1})"
                )
                resp_passed = False
                continue

        # 检查 2: problem_id 归属 (最可靠的验证)
        if samples and sample_id < len(samples):
            actual_problem = str(samples[sample_id].get("problem_id", ""))
            if actual_problem != problem_id_str:
                errors.append(
                    f"[归属错误] response[{i}]: sample_id={sample_id} "
                    f"属于 problem {actual_problem}, 不是 {problem_id_str}"
                )
                resp_passed = False
                continue  # 归属错误，跳过长度验证

        # 检查 3: token 长度范围 (用 slice_info 的 slice 数量)
        if token_row_ptr is not None and resp_slice_counts and i in resp_slice_counts:
            actual_len = int(token_row_ptr[sample_id + 1] - token_row_ptr[sample_id])
            n_slices = resp_slice_counts[i]

            # 最后一个 slice 可能不满，所以是范围检查
            min_expected = (n_slices - 1) * effective_pos_size + 1
            max_expected = n_slices * effective_pos_size

            if not (min_expected <= actual_len <= max_expected):
                errors.append(
                    f"[长度错误] response[{i}]: sample_id={sample_id}, "
                    f"actual_tokens={actual_len}, "
                    f"expected=[{min_expected}, {max_expected}] "
                    f"(n_slices={n_slices}, effective_pos_size={effective_pos_size})"
                )
                resp_passed = False

        if resp_passed:
            passed += 1

    # 计算结果
    match_rate = passed / total_checked if total_checked > 0 else 1.0
    is_valid = len(errors) == 0

    if is_valid:
        message = f"sample_id 验证通过: {passed}/{total_checked} responses 有效"
    else:
        error_lines = [
            f"sample_id 验证失败! 发现 {len(errors)} 个错误 "
            f"(problem_id={problem_id_str}):"
        ]
        error_lines.extend(f"  {e}" for e in errors[:10])
        if len(errors) > 10:
            error_lines.append(f"  ... 还有 {len(errors) - 10} 个错误")
        error_lines.append("")
        error_lines.append("可能原因: GLOBAL sample_id 被错误替换为 LOCAL index")
        message = "\n".join(error_lines)

    result = VerificationResult(
        is_valid=is_valid,
        total_checked=total_checked,
        matched=passed,
        match_rate=match_rate,
        mismatches=errors,
        message=message,
    )

    if strict and not is_valid:
        raise ValueError(message)

    return result
