#!/usr/bin/env python3
"""Meta mapping utilities for sample_id/run_index alignment.

This module provides functions to build bidirectional mappings between
sample_id (global position in cache) and run_index (local position within problem).

Usage:
    from alignment.meta_mapping import load_meta_mapping

    meta_mapping = load_meta_mapping(cache_path)
    # Returns:
    # {
    #     "problem_id": {
    #         "sample_ids": [0, 1, 2, ...],
    #         "run_indices": [0, 1, 2, ...],
    #         "run_to_sample": {0: 0, 1: 1, ...},
    #         "sample_to_run": {0: 0, 1: 1, ...},
    #     },
    #     ...
    # }
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta_mapping(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build sample_id/run_index mapping from cache/meta.json.

    严格模式: meta.json 必须存在，否则抛出异常。

    Args:
        cache_dir: Path to cache directory containing meta.json

    Returns:
        Dictionary mapping problem_id to mapping info:
        {
            "problem_id": {
                "sample_ids": [global_sample_id_1, ...],
                "run_indices": [local_run_index_1, ...],
                "run_to_sample": {run_index: sample_id, ...},
                "sample_to_run": {sample_id: run_index, ...},
            }
        }

    Raises:
        FileNotFoundError: If meta.json does not exist

    Notes:
        - sample_id is the GLOBAL position in token_row_ptr/tok_conf arrays
        - run_index is the LOCAL position within a problem's runs
        - For problem 0 with 64 runs: sample_ids = [0-63], run_indices = [0-63]
        - For problem 1 with 64 runs: sample_ids = [64-127], run_indices = [0-63]
    """
    from alignment_lite.core import load_meta_mapping as _load
    return _load(cache_dir)


def get_sample_ids_for_problem(
    meta_mapping: Dict[str, Dict[str, Any]], problem_id: str
) -> list[int]:
    """Get list of global sample_ids for a problem.

    Args:
        meta_mapping: Mapping from load_meta_mapping()
        problem_id: Problem identifier

    Returns:
        List of global sample_ids for the problem
    """
    entry = meta_mapping.get(str(problem_id), {})
    return entry.get("sample_ids", [])


def run_index_to_sample_id(
    meta_mapping: Dict[str, Dict[str, Any]], problem_id: str, run_index: int
) -> int:
    """Convert local run_index to global sample_id.

    Args:
        meta_mapping: Mapping from load_meta_mapping()
        problem_id: Problem identifier
        run_index: Local run index within the problem

    Returns:
        Global sample_id, or -1 if not found
    """
    entry = meta_mapping.get(str(problem_id), {})
    run_to_sample = entry.get("run_to_sample", {})
    return run_to_sample.get(int(run_index), -1)


def sample_id_to_run_index(
    meta_mapping: Dict[str, Dict[str, Any]], problem_id: str, sample_id: int
) -> int:
    """Convert global sample_id to local run_index.

    Args:
        meta_mapping: Mapping from load_meta_mapping()
        problem_id: Problem identifier
        sample_id: Global sample id in the cache

    Returns:
        Local run_index, or -1 if not found
    """
    entry = meta_mapping.get(str(problem_id), {})
    sample_to_run = entry.get("sample_to_run", {})
    return sample_to_run.get(int(sample_id), -1)
