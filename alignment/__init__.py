#!/usr/bin/env python3
"""Alignment utilities for sample_id/run_index mapping.

This package provides utilities for checking and aligning sample_id/run_index
mappings between cache token data and visualization metadata.

严格模式: 所有函数不允许 fallback，缺少必要数据时直接报错。

Key concepts:
- sample_id: GLOBAL position in cache arrays (token_row_ptr, tok_conf)
- run_index: LOCAL position within a problem's runs

For example, with 64 runs per problem:
- Problem 0: sample_ids = [0-63], run_indices = [0-63]
- Problem 1: sample_ids = [64-127], run_indices = [0-63]
- Problem 10: sample_ids = [640-703], run_indices = [0-63]

Usage:
    from alignment import (
        load_meta_mapping,
        build_response_mapping,
        merge_responses,
        load_evaluation_results,
        validate_viz_eval_alignment,
    )

    # Load mappings (strict: raises if files missing)
    meta_mapping = load_meta_mapping(cache_path)
    eval_results = load_evaluation_results(cache_path)

    # Get eval runs for a problem
    eval_runs = get_runs_for_problem(eval_results, problem_idx)

    # Merge responses with eval runs
    merged = merge_responses(metadata_responses, eval_runs)

    # Build sample_id mapping (strict: requires eval_runs)
    sample_ids, run_indices = build_response_mapping(
        merged, problem_id, meta_mapping, eval_runs
    )

    # Validate alignment (strict: is_correct AND finish_reason must match)
    result = validate_viz_eval_alignment(viz_dir, eval_results.values())
"""

from .meta_mapping import (
    load_meta_mapping,
    get_sample_ids_for_problem,
    run_index_to_sample_id,
    sample_id_to_run_index,
)
from .response_mapping import (
    build_response_mapping,
    merge_responses,
)
from .evaluation import (
    load_evaluation_results,
    load_evaluation_results_by_problem_id,
    load_full_evaluation_report,
    get_run_data,
    get_generated_text,
    get_runs_for_problem,
    get_runs_for_problem_by_id,
    resolve_n_runs,
    get_correct_indices,
    compute_problem_accuracy,
    get_problem_id_from_eval_report,
)
from .verification import (
    validate_viz_eval_alignment,
    validate_response_mapping,
    validate_response_sample_ids,
    VerificationResult,
)
from .data_loading import (
    load_knn_data,
    load_problem_metadata,
    load_problem_data,
    load_all_problems_data,
)
from .viz_path import (
    resolve_viz_dir,
)
from .cache_reader import (
    CacheReader,
    TokenView,
    get_cache_reader,
)
from .server_adapter import (
    VizDataAdapter,
)
from .tokenizer import (
    TokenizerWrapper,
)
from .token_text import (
    TokenTextService,
)

__all__ = [
    # meta_mapping
    "load_meta_mapping",
    "get_sample_ids_for_problem",
    "run_index_to_sample_id",
    "sample_id_to_run_index",
    # response_mapping
    "build_response_mapping",
    "merge_responses",
    # evaluation
    "load_evaluation_results",
    "load_evaluation_results_by_problem_id",
    "load_full_evaluation_report",
    "get_run_data",
    "get_generated_text",
    "get_runs_for_problem",
    "get_runs_for_problem_by_id",
    "resolve_n_runs",
    "get_correct_indices",
    "compute_problem_accuracy",
    "get_problem_id_from_eval_report",
    # verification
    "validate_viz_eval_alignment",
    "validate_response_mapping",
    "validate_response_sample_ids",
    "VerificationResult",
    # data_loading
    "load_knn_data",
    "load_problem_metadata",
    "load_problem_data",
    "load_all_problems_data",
    # viz_path
    "resolve_viz_dir",
    # cache_reader
    "CacheReader",
    "TokenView",
    "get_cache_reader",
    # server_adapter
    "VizDataAdapter",
    # tokenizer
    "TokenizerWrapper",
    # token_text
    "TokenTextService",
]
