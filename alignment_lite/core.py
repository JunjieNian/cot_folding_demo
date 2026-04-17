#!/usr/bin/env python3
"""alignment_lite: minimal alignment from first principles.

The only operation: JOIN eval_report with viz_metadata on problem_id,
then overlay eval fields onto viz responses using 1-based run_index mapping.

    responses[i].is_correct   = eval_runs[i+1].is_correct
    responses[i].sample_id    = responses[i].run_index   (preserve global ID)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from project_paths import resolve_nad_core_path, resolve_vizcache_root

# ---------------------------------------------------------------------------
# NAD core re-exports (cache reader)
# ---------------------------------------------------------------------------

_NAD_CORE_PATH = resolve_nad_core_path(required=False)
if _NAD_CORE_PATH and str(_NAD_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_NAD_CORE_PATH))

try:
    from nad.core.views.reader import CacheReader, TokenView  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "无法导入 nad 核心库，请设置 INTRA_COT_NAD_ROOT 或 NAD_CORE_PATH。"
    ) from exc


# ---------------------------------------------------------------------------
# 1. load_eval_by_pid
# ---------------------------------------------------------------------------

def load_eval_by_pid(cache_path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load evaluation_report_compact.json -> {problem_id: {run_index: run}}."""
    report_path = Path(cache_path) / "evaluation_report_compact.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing evaluation_report_compact.json: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    eval_indexed: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for problem in data.get("results", []):
        pid = str(problem.get("problem_id", ""))
        eval_indexed[pid] = {
            run["run_index"]: run
            for run in problem.get("runs", [])
            if "run_index" in run
        }
    return eval_indexed


# ---------------------------------------------------------------------------
# 2. resolve_viz_dir
# ---------------------------------------------------------------------------

_DEFAULT_VIZCACHE_ROOT = resolve_vizcache_root(required=False)


def resolve_viz_dir(
    cache_path: Path, viz_root: Optional[str] = None
) -> Path:
    """Infer viz directory from cache path.

    Layout: {vizcache_root}/{model}/{dataset}/{cache_name}/viz/
    """
    cache_path = Path(cache_path)
    cache_name = cache_path.name
    dataset = cache_path.parent.name
    model = cache_path.parent.parent.name

    candidate_roots = [Path(viz_root)] if viz_root else []
    if _DEFAULT_VIZCACHE_ROOT:
        candidate_roots.append(_DEFAULT_VIZCACHE_ROOT)

    tried = []
    for root in candidate_roots:
        viz_dir = root / model / dataset / cache_name / "viz"
        tried.append(str(viz_dir))
        if viz_dir.exists():
            return viz_dir

    raise FileNotFoundError(
        "viz dir not found.\n"
        + "\n".join(f"  - {path}" for path in tried)
        + "\nExpected layout: {viz_root}/{model}/{dataset}/{cache_name}/viz/"
    )


# ---------------------------------------------------------------------------
# 3. load_knn_data
# ---------------------------------------------------------------------------

def load_knn_data(
    viz_dir: Path, sep_up: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load knn_indices, knn_dists, slice_info for one problem sub-dir."""
    prefix = f"sep_up{sep_up}x"
    knn_indices_path = viz_dir / f"{prefix}_knn_indices.npy"

    if not knn_indices_path.exists():
        available = [
            int(f.name.split("x")[0].replace("sep_up", ""))
            for f in viz_dir.glob("sep_up*_knn_indices.npy")
        ]
        raise FileNotFoundError(
            f"Missing knn file for sep_up={sep_up}: {knn_indices_path}\n"
            f"Available sep_up values: {available}"
        )

    knn_indices = np.load(knn_indices_path)
    knn_dists = np.load(viz_dir / f"{prefix}_knn_dists.npy")
    slice_info = np.load(viz_dir / f"{prefix}_slice_info.npy")
    return knn_indices, knn_dists, slice_info


# ---------------------------------------------------------------------------
# 4. merge_responses
# ---------------------------------------------------------------------------

def merge_responses(
    metadata_responses: List[Dict[str, Any]],
    eval_runs_by_idx: Dict[int, Dict[str, Any]],
    problem_id: str,
) -> List[Dict[str, Any]]:
    """JOIN viz responses with eval runs via 1-based run_index.

    For each response at position i:
      - eval run = eval_runs_by_idx[i + 1]
      - overlay: is_correct, finish_reason, stop_reason, extracted_answer
      - preserve: run_index as sample_id (global ID)
    """
    if len(metadata_responses) != len(eval_runs_by_idx):
        raise ValueError(
            f"[{problem_id}] count mismatch: "
            f"metadata_responses={len(metadata_responses)}, "
            f"eval_runs={len(eval_runs_by_idx)}"
        )

    merged: List[Dict[str, Any]] = []
    for idx in range(len(metadata_responses)):
        resp = dict(metadata_responses[idx])
        run_index = idx + 1  # 1-based
        run = eval_runs_by_idx.get(run_index)
        if run is None:
            raise ValueError(
                f"[{problem_id}] run_index={run_index} not in eval_runs. "
                f"Available: {sorted(eval_runs_by_idx.keys())[:10]}..."
            )

        # overlay eval fields (eval is authoritative)
        resp["finish_reason"] = run.get("finish_reason", resp.get("finish_reason", ""))
        resp["stop_reason"] = run.get("stop_reason", resp.get("stop_reason", ""))
        resp["is_correct"] = run.get("is_correct", resp.get("is_correct", False))
        resp["extracted_answer"] = run.get(
            "extracted_answer", resp.get("extracted_answer", None)
        )

        # preserve global sample_id from viz metadata's run_index
        resp["sample_id"] = resp.get("run_index")

        merged.append(resp)
    return merged


# ---------------------------------------------------------------------------
# 5. load_problems  (main entry point)
# ---------------------------------------------------------------------------

def load_problems(
    cache_path: Path,
    sep_up: int = 8,
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load all problems for a cache, merging eval + viz + knn data.

    Returns list of dicts, each with:
        viz_idx, problem_id, knn_indices, knn_dists, slice_info,
        responses (merged), n_correct, n_responses
    """
    cache_path = Path(cache_path)

    # step 1: eval index (load once)
    eval_indexed = load_eval_by_pid(cache_path)

    # step 2: viz root
    viz_root = resolve_viz_dir(cache_path)

    # step 3: enumerate problem sub-dirs
    viz_indices = sorted(
        int(d.name) for d in viz_root.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not viz_indices:
        raise FileNotFoundError(f"No problem dirs in: {viz_root}")

    problems: List[Dict[str, Any]] = []
    for viz_idx in viz_indices:
        viz_dir = viz_root / str(viz_idx)

        # load metadata
        metadata_path = viz_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        problem_id = str(metadata.get("problem_id", ""))
        responses = metadata.get("responses", [])

        # lookup eval runs
        eval_runs = eval_indexed.get(problem_id)
        if not eval_runs:
            raise ValueError(
                f"problem_id='{problem_id}' not in eval report. "
                f"Available: {list(eval_indexed.keys())[:5]}..."
            )

        # merge
        merged = merge_responses(responses, eval_runs, problem_id)

        # load knn
        knn_indices, knn_dists, slice_info = load_knn_data(viz_dir, sep_up)

        problems.append({
            "viz_idx": viz_idx,
            "problem_id": problem_id,
            "knn_indices": knn_indices,
            "knn_dists": knn_dists,
            "slice_info": slice_info,
            "responses": merged,
            "n_correct": sum(1 for r in merged if r.get("is_correct")),
            "n_responses": len(merged),
        })

        if max_problems and len(problems) >= max_problems:
            break

    return problems


# ---------------------------------------------------------------------------
# Auxiliary: load_meta_mapping
# ---------------------------------------------------------------------------

def load_meta_mapping(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build sample_id / run_index mapping from cache/meta.json.

    Returns {problem_id: {sample_ids, run_indices, run_to_sample, sample_to_run}}.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    mapping: Dict[str, Dict[str, Any]] = {}
    for sample_id, sample in enumerate(meta.get("samples", [])):
        pid = str(sample.get("problem_id", ""))
        run_index = int(sample.get("run_index", sample_id))
        entry = mapping.setdefault(pid, {"sample_ids": [], "run_indices": []})
        entry["sample_ids"].append(int(sample_id))
        entry["run_indices"].append(run_index)

    for entry in mapping.values():
        entry["run_to_sample"] = {
            int(ri): int(si)
            for si, ri in zip(entry["sample_ids"], entry["run_indices"])
        }
        entry["sample_to_run"] = {
            int(si): int(ri)
            for si, ri in zip(entry["sample_ids"], entry["run_indices"])
        }

    return mapping


# ---------------------------------------------------------------------------
# Auxiliary: get_cache_reader
# ---------------------------------------------------------------------------

def get_cache_reader(cache_path: Path) -> CacheReader:
    """Get a CacheReader instance (strict: requires tok_conf)."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_path}")
    reader = CacheReader(str(cache_path))
    if reader.tok_conf is None:
        raise ValueError(f"cache missing tok_conf data: {cache_path}")
    return reader


# ---------------------------------------------------------------------------
# Auxiliary: discover_available_sep_ups
# ---------------------------------------------------------------------------

def discover_available_sep_ups(cache_path: Path) -> List[int]:
    """Scan viz/0/ directory and return all available sep_up values (ascending).

    Looks for files matching ``sep_up<N>x_coords.npy`` in the first problem
    sub-directory of the viz cache.
    """
    viz_root = resolve_viz_dir(cache_path)
    first_problem = viz_root / "0"
    if not first_problem.exists():
        return [8]
    values = sorted({
        int(f.stem.split("x")[0].replace("sep_up", ""))
        for f in first_problem.glob("sep_up*_coords.npy")
    })
    return values or [8]
