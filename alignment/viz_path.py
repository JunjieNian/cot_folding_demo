#!/usr/bin/env python3
"""
Viz 目录路径解析模块

提供 viz 目录路径解析功能，用于定位 KNN 图和 metadata 文件。
"""

from pathlib import Path
from typing import Optional


def resolve_viz_dir(cache_path: Path, viz_root: Optional[str] = None) -> Path:
    """
    查找 viz 目录：默认从 vizcache_output 查找

    Args:
        cache_path: cache 目录路径，格式 .../MODEL/DATASET/CACHE_NAME
        viz_root: 可选的 viz 根目录

    Returns:
        viz 目录路径

    Raises:
        FileNotFoundError: viz 目录不存在
    """
    from alignment_lite.core import resolve_viz_dir as _resolve
    return _resolve(cache_path, viz_root)
