#!/usr/bin/env python3
"""
Token 文本提取服务

参考 sc_slice_text_extractor.py 的实现。

职责:
1. 从 NAD Cache 读取 token 数据 (使用 CacheReader)
2. 根据 sep_up 计算位置映射
3. 返回解码后的文本

Usage:
    from alignment_lite.token_text import TokenTextService

    service = TokenTextService(cache_path)
    if service.initialize():
        result = service.get_token_text(problem_id=60, response_idx=0, position=10, sep_up=8)
"""

from pathlib import Path
from threading import RLock
from typing import Dict, Any, Optional, List
import json

from .tokenizer import TokenizerWrapper
from .core import CacheReader


class TokenTextService:
    """
    Token 文本提取服务

    参考 sc_slice_text_extractor.py，使用 CacheReader 读取 token 数据。
    使用延迟初始化，首次调用 get_token_text 时才加载数据。
    """

    POS_SIZE = 32  # 每个 NAD slice 的 token 数

    def __init__(self, nad_cache_path: Path):
        """
        初始化 TokenTextService

        Args:
            nad_cache_path: NAD cache 目录路径
        """
        self.cache_path = Path(nad_cache_path)
        self._meta: Optional[Dict] = None
        self._cache_reader: Optional[CacheReader] = None
        self._tokenizer: Optional[TokenizerWrapper] = None
        self._initialized = False
        self._init_lock = RLock()

    def initialize(self) -> bool:
        """
        延迟初始化 (线程安全, 双重检查锁定)

        加载 meta.json、CacheReader 和 Tokenizer。

        Returns:
            初始化是否成功
        """
        if self._initialized:
            return self._tokenizer is not None

        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return self._tokenizer is not None

            # 1. 加载 meta.json
            meta_path = self.cache_path / "meta.json"
            if not meta_path.exists():
                print(f"[TokenText] ⚠️ meta.json not found: {meta_path}")
                self._initialized = True
                return False

            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"[TokenText] ✓ Loaded meta.json")

            # 2. 初始化 CacheReader (参考 sc_slice_text_extractor.py)
            try:
                self._cache_reader = CacheReader(str(self.cache_path))
                print(f"[TokenText] ✓ CacheReader initialized")
            except Exception as exc:
                print(f"[TokenText] ⚠️ Failed to initialize CacheReader: {exc}")
                self._initialized = True
                return False

            # 3. 加载 Tokenizer
            model_path = self._meta.get("model_path")
            if not model_path:
                print(f"[TokenText] ⚠️ No model_path in meta.json")
                self._initialized = True
                return False

            self._tokenizer = TokenizerWrapper.load(model_path)
            if self._tokenizer:
                print(f"[TokenText] ✓ Tokenizer ready")
            else:
                print(f"[TokenText] ⚠️ Tokenizer load failed")

            self._initialized = True
            return self._tokenizer is not None

    @property
    def is_ready(self) -> bool:
        """检查服务是否就绪"""
        return self._initialized and self._tokenizer is not None

    def validate_sample_id(self, sample_id: int) -> bool:
        """
        验证 sample_id 是否有效

        Args:
            sample_id: GLOBAL sample_id (来自 viz metadata 的 run_index)

        Returns:
            sample_id 是否有效
        """
        if self._meta is None:
            return False

        samples = self._meta.get("samples", [])
        return 0 <= sample_id < len(samples)

    def get_token_text(
        self,
        sample_id: int,
        position: int,
        sep_up: int = 8
    ) -> Dict[str, Any]:
        """
        获取指定位置的 token 文本 (参考 sc_slice_text_extractor.py)

        Args:
            sample_id: GLOBAL sample_id (来自 viz metadata 的 run_index)
            position: 位置索引 (1-based, 对应 UMAP 坐标位置)
            sep_up: 上采样因子

        Returns:
            包含 token 文本和元数据的字典
        """
        if not self.initialize():
            return {"success": False, "error": "Service not initialized"}

        if not self.validate_sample_id(sample_id):
            return {
                "success": False,
                "error": f"Invalid sample_id: {sample_id}"
            }

        if position < 1:
            return {
                "success": False,
                "error": f"Invalid position: {position}. Position is 1-based and must be >= 1."
            }

        try:
            # 使用 CacheReader.get_token_view() 获取 token 数据
            # 参考 sc_slice_text_extractor.py: tv = cache.get_token_view(sample_id)
            tv = self._cache_reader.get_token_view(sample_id)
            token_ids = tv.token_ids

            if token_ids is None or len(token_ids) == 0:
                return {
                    "success": True,
                    "sample_id": sample_id,
                    "position": position,
                    "token_ids": [],
                    "token_text": "",
                    "tokens": [],
                    "n_tokens": 0,
                    "current_position_start": 0,
                    "current_position_tokens": [],
                    "current_position_text": "",
                    "current_position_n_tokens": 0,
                    "prefix_text": "",
                }

            # 计算 token 范围 (保持与原有逻辑一致)
            # effective_pos_size = POS_SIZE * sep_up (与 sc_slice_text_extractor.py 相同)
            effective_pos_size = self.POS_SIZE * sep_up

            # 当前位置的 token 范围
            current_position_start = (position - 1) * effective_pos_size
            token_end = min(position * effective_pos_size, len(token_ids))

            # 获取从开始到当前位置的所有 token
            # 转换为 Python int 以确保 JSON 可序列化
            all_token_ids = [int(tid) for tid in token_ids[:token_end]]

            # 当前位置的 token
            current_token_ids = [int(tid) for tid in token_ids[current_position_start:token_end]]

            # 解码
            full_text = self._tokenizer.decode(all_token_ids)
            individual_tokens = [self._tokenizer.decode([tid]) for tid in all_token_ids]

            current_position_tokens = [self._tokenizer.decode([tid]) for tid in current_token_ids]
            current_position_text = self._tokenizer.decode(current_token_ids)
            prefix_text = self._tokenizer.decode(all_token_ids[:current_position_start])

            return {
                "success": True,
                "sample_id": sample_id,
                "position": position,
                "token_ids": all_token_ids,
                "token_text": full_text,
                "tokens": individual_tokens,
                "n_tokens": len(all_token_ids),
                "current_position_start": current_position_start,
                "current_position_tokens": current_position_tokens,
                "current_position_text": current_position_text,
                "current_position_n_tokens": len(current_position_tokens),
                "prefix_text": prefix_text,
            }

        except Exception as exc:
            return {"success": False, "error": str(exc)}
