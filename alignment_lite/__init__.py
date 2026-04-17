"""alignment_lite: minimal alignment from first principles.

Usage:
    from alignment_lite import load_problems, load_meta_mapping, get_cache_reader
    from alignment_lite import VizDataAdapter, TokenTextService, TokenizerWrapper

    problems = load_problems(cache_path, sep_up=8)
"""

from .core import (
    load_problems,
    load_eval_by_pid,
    resolve_viz_dir,
    load_knn_data,
    merge_responses,
    load_meta_mapping,
    get_cache_reader,
    CacheReader,
    TokenView,
)
from .tokenizer import TokenizerWrapper
from .token_text import TokenTextService
from .server_adapter import VizDataAdapter

__all__ = [
    "load_problems",
    "load_eval_by_pid",
    "resolve_viz_dir",
    "load_knn_data",
    "merge_responses",
    "load_meta_mapping",
    "get_cache_reader",
    "CacheReader",
    "TokenView",
    "TokenizerWrapper",
    "TokenTextService",
    "VizDataAdapter",
]
