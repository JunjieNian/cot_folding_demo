#!/usr/bin/env python3
"""
Tokenizer 加载和 Token 解码工具

参考 sc_slice_text_extractor.py 的实现。

职责:
1. Tokenizer 加载 (AutoTokenizer with trust_remote_code)
2. Token ID → 文本解码
3. 模型路径自动搜索 (递归搜索 + tokenizer 文件验证)

Usage:
    from alignment_lite.tokenizer import TokenizerWrapper

    # Load tokenizer from model path
    tokenizer = TokenizerWrapper.load("/path/to/model")
    if tokenizer:
        text = tokenizer.decode([1, 2, 3])
"""

from pathlib import Path
from typing import List, Optional
from project_paths import model_search_roots

# Tokenizer 后端
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# 模型搜索根目录
MODEL_SEARCH_ROOTS = model_search_roots()


class TokenizerWrapper:
    """
    统一的 Tokenizer API 包装器

    使用 AutoTokenizer 加载，支持 trust_remote_code。
    """

    def __init__(self, tokenizer):
        """
        初始化 TokenizerWrapper

        Args:
            tokenizer: AutoTokenizer 对象
        """
        self._tokenizer = tokenizer

    def decode(self, token_ids: List[int]) -> str:
        """
        解码 token ID 列表为文本

        Args:
            token_ids: Token ID 列表

        Returns:
            解码后的文本字符串
        """
        if self._tokenizer is None:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    @classmethod
    def load(cls, model_path: str) -> Optional["TokenizerWrapper"]:
        """
        从模型路径加载 Tokenizer

        Args:
            model_path: 模型目录路径或模型名称

        Returns:
            TokenizerWrapper 实例，加载失败返回 None
        """
        resolved_path = cls._find_model_path(model_path)
        if not resolved_path:
            print(f"[Tokenizer] ⚠️ Model path not found: {model_path}")
            return None
        return cls._load_tokenizer(resolved_path)

    @staticmethod
    def _find_model_path(original_path: Optional[str]) -> Optional[str]:
        """
        搜索模型路径 (参考 sc_slice_text_extractor.py)

        搜索策略:
        1. 如果原始路径存在且包含 tokenizer 文件，直接返回
        2. 否则在 MODEL_SEARCH_ROOTS 下递归搜索同名目录
        3. 验证目录包含 tokenizer.json 或 tokenizer_config.json

        Args:
            original_path: 原始模型路径

        Returns:
            找到的模型路径，找不到返回 None
        """
        if not original_path:
            return None

        original = Path(original_path)

        # 1. 检查原始路径
        if original.exists() and original.is_dir():
            if (original / "tokenizer.json").exists() or (original / "tokenizer_config.json").exists():
                return str(original)

        # 2. 提取模型名称，在搜索目录中递归查找
        model_name = original.name
        if not model_name:
            return None

        print(f"[Tokenizer] Searching for '{model_name}' in search roots...")

        for search_root in MODEL_SEARCH_ROOTS:
            search_path = Path(search_root)
            if not search_path.exists():
                continue

            # 递归搜索
            for path in search_path.rglob(model_name):
                if path.is_dir():
                    # 验证包含 tokenizer 文件
                    if (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists():
                        print(f"[Tokenizer] ✓ Found at: {path}")
                        return str(path)

        return None

    @classmethod
    def _load_tokenizer(cls, model_path: str) -> Optional["TokenizerWrapper"]:
        """
        加载 Tokenizer (参考 sc_slice_text_extractor.py)

        使用 AutoTokenizer.from_pretrained with trust_remote_code=True
        支持 Qwen 等需要自定义代码的模型。

        Args:
            model_path: 模型目录路径

        Returns:
            TokenizerWrapper 实例，加载失败返回 None
        """
        if AutoTokenizer is None:
            print(f"[Tokenizer] ⚠️ transformers not installed")
            return None

        print(f"[Tokenizer] Loading from: {model_path}")

        try:
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"[Tokenizer] ✓ Loaded successfully")
            return cls(tok)
        except Exception as exc:
            print(f"[Tokenizer] ⚠️ Failed to load: {exc}")
            return None
