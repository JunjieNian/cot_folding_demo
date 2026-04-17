#!/usr/bin/env python3
"""
Tokenizer 加载和 Token 解码工具

参考 sc_slice_text_extractor.py 的实现。

职责:
1. Tokenizer 加载 (AutoTokenizer with trust_remote_code)
2. Token ID → 文本解码
3. 模型路径自动搜索 (递归搜索 + tokenizer 文件验证)

Usage:
    from alignment.tokenizer import TokenizerWrapper

    # Load tokenizer from model path
    tokenizer = TokenizerWrapper.load("/path/to/model")
    if tokenizer:
        text = tokenizer.decode([1, 2, 3])

NOTE: Implementation delegated to alignment_lite.tokenizer
"""

from alignment_lite.tokenizer import TokenizerWrapper  # noqa: F401
from alignment_lite.tokenizer import MODEL_SEARCH_ROOTS  # noqa: F401
