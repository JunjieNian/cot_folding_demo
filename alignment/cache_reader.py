#!/usr/bin/env python3
"""
Cache Token 数据读取模块

提供 CacheReader 的统一封装，用于访问 cache 中的 token 级别数据。

Usage:
    from alignment import get_cache_reader, CacheReader

    reader = get_cache_reader(cache_path)
    token_view = reader.get_token_view(sample_id)
    tok_conf = token_view.tok_conf

严格模式: 所有函数不允许 fallback，缺少必要数据时直接报错。
"""

from pathlib import Path
from typing import Union

# 从内部模块导入 (封装 NAD 核心库依赖)
from ._nad_compat import CacheReader, TokenView

__all__ = ["CacheReader", "TokenView", "get_cache_reader"]


def get_cache_reader(cache_path: Union[str, Path]) -> CacheReader:
    """
    获取 CacheReader 实例 (严格模式)。

    Args:
        cache_path: cache 目录路径

    Returns:
        CacheReader 实例

    Raises:
        FileNotFoundError: cache 目录不存在
        ValueError: cache 缺少必要的 token 数据 (tok_conf)

    Example:
        from alignment import get_cache_reader

        reader = get_cache_reader(cache_path)
        token_view = reader.get_token_view(sample_id)
        tok_conf = token_view.tok_conf
    """
    from alignment_lite.core import get_cache_reader as _get
    return _get(cache_path)
