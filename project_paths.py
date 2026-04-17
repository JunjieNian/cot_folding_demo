#!/usr/bin/env python3
"""Project-wide path resolution helpers.

All paths are resolved from environment variables. No hardcoded absolute paths.

Required environment variables (set the ones you need):

    INTRA_COT_NAD_ROOT      Path to NAD_Next library root
    INTRA_COT_CACHE_BASE    Path to neuron activation cache
                            (e.g. /path/to/cache/DeepSeek-R1-0528-Qwen3-8B)
    INTRA_COT_RL_CACHE_ROOT Path to RL checkpoint neuron cache
    INTRA_COT_VIZCACHE_ROOT Path to vizcache output directory
    INTRA_COT_MODEL_SEARCH_ROOTS
                            Colon-separated list of model weight directories

See README.md for setup instructions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent

# ── Environment variable names ──

ENV_NAD_ROOTS = (
    "INTRA_COT_NAD_ROOT",
    "NAD_CORE_PATH",
    "NAD_NEXT_PATH",
)
ENV_CACHE_BASES = (
    "INTRA_COT_CACHE_BASE",
    "MUI_CACHE_BASE",
)
ENV_VIZCACHE_ROOTS = (
    "INTRA_COT_VIZCACHE_ROOT",
    "VIZCACHE_ROOT",
)
ENV_MODEL_SEARCH_ROOTS = "INTRA_COT_MODEL_SEARCH_ROOTS"
ENV_RL_CACHE_ROOT = "INTRA_COT_RL_CACHE_ROOT"

# ── Default cache directory names per benchmark ──

DEFAULT_CACHE_NAMES = {
    "aime24": "cache_neuron_output_1_act_no_rms_20250902_025610",
    "gpqa": "cache_neuron_output_1_act_no_rms_20251126_111853",
}

# ── RL checkpoint registry ──

RL_CHECKPOINTS = {
    "base":      "Qwen3-4B-Base_base",
    "step-100":  "Qwen3-4B-Base_math7500-step-100",
    "step-200":  "Qwen3-4B-Base_math7500-step-200",
    "step-300":  "Qwen3-4B-Base_math7500-step-300",
    "step-400":  "Qwen3-4B-Base_math7500-step-400",
    "step-500":  "Qwen3-4B-Base_math7500-step-500",
    "step-600":  "Qwen3-4B-Base_math7500-step-600",
    "step-700":  "Qwen3-4B-Base_math7500-step-700",
    "step-800":  "Qwen3-4B-Base_math7500-step-800",
    "step-900":  "Qwen3-4B-Base_math7500-step-900",
    "step-1000": "Qwen3-4B-Base_math7500-step-1000",
}
RL_DATASET = "variable-reasoning-mini"


# ── Internal helpers ──

def _iter_env_paths(var_names: Iterable[str]) -> Iterator[Path]:
    """Yield Path objects from colon-separated environment variables."""
    for name in var_names:
        raw = os.environ.get(name)
        if not raw:
            continue
        for part in raw.split(os.pathsep):
            if part.strip():
                yield Path(part.strip()).expanduser()


def _iter_unique_existing(paths: Iterable[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            yield resolved


# ── Batch output directories ──

def default_batch_dir_for_benchmark(benchmark: str, repo_root: Path = REPO_ROOT) -> Path:
    if benchmark == "aime24":
        return repo_root / "batch_results"
    return repo_root / f"batch_results_{benchmark}"


def default_graph_batch_dir(repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "batch_results_graph"


def default_segment_batch_dir(repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "batch_results_segment"


def default_results_dir(repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "results"


# ── NAD core library ──

def iter_nad_core_candidates() -> Iterator[Path]:
    """Yield candidate NAD_Next root directories (env vars + local fallbacks)."""
    yield from _iter_env_paths(ENV_NAD_ROOTS)
    # Local fallbacks relative to this repo
    yield REPO_ROOT / "NAD_Next"
    yield REPO_ROOT.parent / "NAD_Next"


def resolve_nad_core_path(required: bool = False) -> Path | None:
    for candidate in _iter_unique_existing(iter_nad_core_candidates()):
        if (candidate / "nad" / "core").exists():
            return candidate
    if required:
        raise FileNotFoundError(
            "NAD_Next root not found. "
            "Set INTRA_COT_NAD_ROOT to your NAD_Next directory."
        )
    return None


def ensure_project_imports(include_nad: bool = True) -> Path | None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    nad_root = resolve_nad_core_path(required=False) if include_nad else None
    if nad_root and str(nad_root) not in sys.path:
        sys.path.insert(0, str(nad_root))
    return nad_root


# ── RL checkpoint cache ──

def resolve_rl_cache_path(checkpoint: str, rl_root: Path | None = None) -> Path:
    """Resolve the neuron-cache directory for an RL checkpoint.

    Returns e.g. <rl_root>/<model_dir>/<dataset>/<cache_dir>
    """
    env_root = os.environ.get(ENV_RL_CACHE_ROOT, "").strip()
    root = rl_root or (Path(env_root) if env_root else None)
    if root is None:
        raise FileNotFoundError(
            "RL cache root not configured. "
            "Set INTRA_COT_RL_CACHE_ROOT to the RL neuron cache directory."
        )
    if checkpoint not in RL_CHECKPOINTS:
        raise ValueError(f"Unknown RL checkpoint: {checkpoint!r}. "
                         f"Available: {list(RL_CHECKPOINTS)}")
    model_dir = root / RL_CHECKPOINTS[checkpoint] / RL_DATASET
    if not model_dir.is_dir():
        raise FileNotFoundError(f"RL dataset dir not found: {model_dir}")
    candidates = sorted(model_dir.glob("cache_*"), key=lambda p: p.name, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No cache_* dirs under {model_dir}")
    return candidates[0]


def default_rl_batch_dir(checkpoint: str, repo_root: Path = REPO_ROOT) -> Path:
    """Return batch_results_rl/<checkpoint> output directory."""
    return repo_root / "batch_results_rl" / checkpoint


def list_rl_checkpoints() -> list[str]:
    """Return checkpoint short-names in training order."""
    order = ["base"] + [f"step-{i}" for i in range(100, 1001, 100)]
    return [c for c in order if c in RL_CHECKPOINTS]


# ── Neuron activation cache ──

def iter_cache_base_candidates() -> Iterator[Path]:
    """Yield candidate cache base directories from environment variables."""
    yield from _iter_env_paths(ENV_CACHE_BASES)


def resolve_cache_base(required: bool = False) -> Path | None:
    for candidate in _iter_unique_existing(iter_cache_base_candidates()):
        if candidate.is_dir():
            return candidate
    if required:
        raise FileNotFoundError(
            "Cache base not found. "
            "Set INTRA_COT_CACHE_BASE to your neuron activation cache directory."
        )
    return None


def list_available_benchmarks(cache_base: Path | None = None) -> list[str]:
    base = cache_base or resolve_cache_base(required=False)
    if base is None or not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _pick_cache_candidate(dataset_dir: Path, preferred_name: str | None) -> Path | None:
    if preferred_name:
        preferred = dataset_dir / preferred_name
        if preferred.is_dir():
            return preferred
    candidates = sorted(
        (p for p in dataset_dir.glob("cache_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_cache_path(
    cache_arg: str | None = None,
    *,
    default_benchmark: str = "aime24",
    default_cache_name: str | None = None,
    cache_base: Path | None = None,
    required: bool = False,
) -> Path | None:
    """Resolve a cache directory from explicit path or benchmark shorthand."""
    base = cache_base or resolve_cache_base(required=required)

    if cache_arg:
        explicit = Path(cache_arg).expanduser()
        if explicit.is_dir():
            return explicit
        if base is not None:
            dataset_dir = base / cache_arg
            if dataset_dir.is_dir():
                selected = _pick_cache_candidate(
                    dataset_dir,
                    DEFAULT_CACHE_NAMES.get(cache_arg),
                )
                if selected:
                    return selected
            fallback = base / cache_arg
            if fallback.is_dir():
                return fallback
        if required:
            raise FileNotFoundError(f"Cache directory not found: {cache_arg}")
        return None

    if base is None:
        if required:
            raise FileNotFoundError(
                "Cache base not found and no --cache was provided. "
                "Set INTRA_COT_CACHE_BASE or pass --cache /path/to/cache."
            )
        return None

    dataset_dir = base / default_benchmark
    selected = _pick_cache_candidate(
        dataset_dir,
        default_cache_name or DEFAULT_CACHE_NAMES.get(default_benchmark),
    )
    if selected:
        return selected
    if required:
        raise FileNotFoundError(
            f"No cache directories found under benchmark: {dataset_dir}"
        )
    return None


# ── Vizcache ──

def iter_vizcache_root_candidates() -> Iterator[Path]:
    """Yield candidate vizcache root directories from environment variables."""
    yield from _iter_env_paths(ENV_VIZCACHE_ROOTS)


def resolve_vizcache_root(required: bool = False) -> Path | None:
    for candidate in _iter_unique_existing(iter_vizcache_root_candidates()):
        if candidate.is_dir():
            return candidate
    if required:
        raise FileNotFoundError(
            "Vizcache root not found. "
            "Set INTRA_COT_VIZCACHE_ROOT to your vizcache output directory."
        )
    return None


# ── Model weights ──

def model_search_roots() -> list[str]:
    """Return a list of directories to search for tokenizer/model weights.

    Reads from INTRA_COT_MODEL_SEARCH_ROOTS (colon-separated).
    """
    unique: list[str] = []
    seen: set[str] = set()
    for part in os.environ.get(ENV_MODEL_SEARCH_ROOTS, "").split(os.pathsep):
        if part.strip():
            value = str(Path(part.strip()).expanduser())
            if value not in seen:
                seen.add(value)
                unique.append(value)
    if not unique:
        raise FileNotFoundError(
            "No model search roots configured. "
            "Set INTRA_COT_MODEL_SEARCH_ROOTS to a colon-separated list of "
            "directories containing model weights (e.g. /path/to/models)."
        )
    return unique
