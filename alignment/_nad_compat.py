#!/usr/bin/env python3
"""NAD 核心库兼容层 (alignment 内部模块)

提供统一的 NAD 核心库导入接口，避免在多个文件中重复设置 sys.path。

注意: 这是 alignment 包的内部模块，外部代码不应直接导入。
请使用 alignment 包的公开 API:
    from alignment import CacheReader, get_cache_reader

用法 (仅限 alignment 包内部):
    from ._nad_compat import CacheReader, TokenView, DeepConfSelector
"""

import sys
from pathlib import Path

from project_paths import resolve_nad_core_path

NAD_CORE_PATH = resolve_nad_core_path(required=False)

if NAD_CORE_PATH and str(NAD_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(NAD_CORE_PATH))

try:
    from nad.core.views.reader import CacheReader, TokenView
    from nad.core.selectors.impl import DeepConfSelector
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "无法导入 nad 核心库，请设置 INTRA_COT_NAD_ROOT 或 NAD_CORE_PATH。"
    ) from exc

__all__ = ["CacheReader", "TokenView", "DeepConfSelector", "NAD_CORE_PATH"]
