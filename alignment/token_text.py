#!/usr/bin/env python3
"""
Token 文本提取服务

参考 sc_slice_text_extractor.py 的实现。

职责:
1. 从 NAD Cache 读取 token 数据 (使用 CacheReader)
2. 根据 sep_up 计算位置映射
3. 返回解码后的文本

Usage:
    from alignment.token_text import TokenTextService

    service = TokenTextService(cache_path)
    if service.initialize():
        result = service.get_token_text(problem_id=60, response_idx=0, position=10, sep_up=8)

NOTE: Implementation delegated to alignment_lite.token_text
"""

from alignment_lite.token_text import TokenTextService  # noqa: F401
