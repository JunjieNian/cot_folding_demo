#!/usr/bin/env python3
"""
服务器适配器模块

将 alignment_lite 核心库的数据转换为 knn_viz_server 前端期望的格式。
不修改核心库的返回值，而是在适配层进行数据增强。

设计原则:
1. 核心数据来自 alignment_lite.load_problems() (不修改返回值)
2. 坐标数据在适配层单独加载 (核心库不负责)
3. 动态构建 index 数据

Usage:
    from alignment_lite import VizDataAdapter

    adapter = VizDataAdapter(cache_path, sep_up=8)

    # 获取问题列表
    problems = adapter.get_problems_list()

    # 获取轨迹数据
    trajectory = adapter.get_trajectory_data(viz_idx=0)
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import math

import numpy as np

from .core import load_problems as load_all_problems_data
from .core import resolve_viz_dir
from .token_text import TokenTextService


class VizDataAdapter:
    """
    将 alignment 数据适配为可视化服务器格式。

    设计原则:
    1. 核心数据来自 alignment_lite.load_problems() (不修改)
    2. 坐标数据在适配层单独加载 (核心库不负责)
    3. 动态构建 index 数据
    """

    def __init__(self, cache_path: Path, sep_up: int = 8):
        """
        初始化适配器。

        Args:
            cache_path: NAD cache 目录路径
            sep_up: KNN 图上采样因子 (1/2/4/8/16)
        """
        self.cache_path = Path(cache_path)
        self.sep_up = sep_up
        self.viz_root = resolve_viz_dir(cache_path)

        # 加载核心数据 (来自 alignment_lite 库，不修改其返回值)
        self._core_data = load_all_problems_data(cache_path, sep_up=sep_up)

        # 加载坐标数据 (适配层负责，核心库不加载)
        self._load_coords()

        # 构建 index (适配层负责)
        self._build_index()

        # Token 文本服务 (延迟初始化)
        self._token_text_service: Optional[TokenTextService] = None

        # Token row pointer (延迟加载，用于计算 per-response max_text_position)
        self._trp: Optional[np.ndarray] = None
        self._trp_loaded: bool = False

    def _load_coords(self):
        """加载坐标数据到每个问题 (核心库不负责这部分)"""
        prefix = f"sep_up{self.sep_up}x"
        for prob in self._core_data:
            viz_dir = self.viz_root / str(prob["viz_idx"])
            coords_path = viz_dir / f"{prefix}_coords.npy"
            if coords_path.exists():
                prob["coords"] = np.load(coords_path)
            else:
                warnings.warn(
                    f"viz/{prob['viz_idx']}: 坐标文件不存在 ({coords_path.name})，coords=None",
                    UserWarning, stacklevel=2,
                )
                prob["coords"] = None

    def _build_index(self):
        """构建前端期望的 index 结构"""
        self._index = {
            "problems": {},
            "pos_size": 32,  # 默认值
            "sep_up": self.sep_up,
            "max_positions": 0,
            "embedding_method": "umap",
        }

        for prob in self._core_data:
            coords = prob.get("coords")
            n_positions = coords.shape[0] if coords is not None else 0

            self._index["problems"][str(prob["viz_idx"])] = {
                "problem_id": prob["problem_id"],
                "n_responses": prob["n_responses"],
                "coord_position_count": n_positions,
                "n_correct": prob["n_correct"],
            }
            self._index["max_positions"] = max(
                self._index["max_positions"],
                n_positions
            )

    def _load_token_row_ptr(self) -> Optional[np.ndarray]:
        """延迟加载 token_row_ptr.int64 用于计算 per-response token 总数"""
        if not self._trp_loaded:
            self._trp_loaded = True
            trp_path = self.cache_path / "token_data" / "token_row_ptr.int64"
            if trp_path.exists():
                self._trp = np.memmap(trp_path, dtype=np.int64, mode="r")
        return self._trp

    def _get_max_text_position(self, sample_id: int) -> int:
        """
        计算指定 sample 在当前 sep_up 下的最大文本 position 数。

        Returns:
            ceil(total_tokens / (32 * sep_up))

        Raises:
            RuntimeError: token_row_ptr 不可用或 sample_id 越界时
        """
        trp = self._load_token_row_ptr()
        if trp is None:
            raise RuntimeError(
                f"token_row_ptr.int64 not found in {self.cache_path / 'token_data'}. "
                "Cannot compute max_text_position."
            )
        if sample_id < 0 or sample_id >= len(trp) - 1:
            raise RuntimeError(
                f"sample_id {sample_id} out of range [0, {len(trp) - 2}] "
                f"in token_row_ptr (length {len(trp)})"
            )
        total_tokens = int(trp[sample_id + 1] - trp[sample_id])
        if total_tokens <= 0:
            return 0
        effective_pos_size = 32 * self.sep_up
        return math.ceil(total_tokens / effective_pos_size)

    def _safe_max_text_position(self, sample_id, fallback: int) -> int:
        """Compute max text position for *sample_id*, returning *fallback* on error."""
        if sample_id is None:
            return fallback
        try:
            return self._get_max_text_position(sample_id)
        except RuntimeError:
            return fallback

    # =========== 公开 API ===========

    def get_index(self) -> Dict[str, Any]:
        """返回前端期望的 index 结构"""
        return self._index

    def get_problems_list(self) -> List[Dict[str, Any]]:
        """
        返回问题列表 (用于 /api/problems)

        Returns:
            List of problem info dicts with keys:
            - problem_id: str
            - viz_idx: int
            - n_responses: int
            - n_positions: int
            - n_correct: int
        """
        return [
            {
                "problem_id": prob["problem_id"],
                "viz_idx": prob["viz_idx"],
                "n_responses": prob["n_responses"],
                "coord_position_count": prob["coords"].shape[0] if prob.get("coords") is not None else 0,
                "n_correct": prob["n_correct"],
            }
            for prob in self._core_data
        ]

    def get_problem_data(self, viz_idx: int) -> Optional[Dict[str, Any]]:
        """
        返回单个问题的完整数据

        Args:
            viz_idx: 问题的 viz 索引

        Returns:
            Problem data dict or None if not found
        """
        for prob in self._core_data:
            if prob["viz_idx"] == viz_idx:
                return prob
        return None

    def get_problem_metadata(self, viz_idx: int) -> Optional[Dict[str, Any]]:
        """
        返回问题元数据 (用于 /api/problem/<id>)

        Args:
            viz_idx: 问题的 viz 索引

        Returns:
            Metadata dict compatible with frontend expectations
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None:
            return None

        coords = prob.get("coords")
        coord_position_count = coords.shape[0] if coords is not None else 0

        return {
            "problem_id": prob["problem_id"],
            "n_responses": prob["n_responses"],
            "coord_position_count": coord_position_count,
            "pos_size": 32,
            "mode": "non-overlapping",
            "embedding_method": "umap",
            "sep_up": self.sep_up,
            "effective_pos_size": 32 * self.sep_up,
            "upsampling": self.sep_up,
            "responses": [
                {
                    "sample_id": resp.get("sample_id", resp.get("run_index", i)),
                    "run_index": i + 1,
                    "is_correct": resp.get("is_correct", False),
                    "text_position_count": self._safe_max_text_position(
                        resp.get("sample_id", resp.get("run_index")),
                        coord_position_count,
                    ),
                    "finish_reason": resp.get("finish_reason", "unknown"),
                }
                for i, resp in enumerate(prob["responses"])
            ]
        }

    def get_trajectory_data(self, viz_idx: int, pos_end: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        返回轨迹数据 (用于 /api/trajectory)

        格式转换:
        - coords: [n_positions, n_responses, 2]
        - responses: 每个 response 的元数据

        Args:
            viz_idx: 问题的 viz 索引
            pos_end: 最大位置 (可选)

        Returns:
            Trajectory data dict or None if not found
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None:
            return None

        coords = prob.get("coords")
        if coords is None:
            return None

        n_positions, n_responses, _ = coords.shape

        # 如果指定了 pos_end，截取坐标 (defensive clamping)
        if pos_end is not None:
            pos_end = max(0, min(pos_end, n_positions))
            coords = coords[:pos_end]
            n_positions = pos_end

        trajectories = []
        for resp_idx, resp in enumerate(prob["responses"]):
            sample_id = resp.get("sample_id", resp.get("run_index"))
            max_text_pos = self._safe_max_text_position(sample_id, n_positions)
            traj = {
                "response_idx": resp_idx,
                "sample_id": sample_id if sample_id is not None else resp_idx,
                "run_index": resp_idx + 1,
                "is_correct": resp.get("is_correct", False),
                "finish_reason": resp.get("finish_reason", "unknown"),
                "coords": coords[:, resp_idx, :].tolist(),
                "prefix_sizes": [],
                "text_position_count": max_text_pos,
            }
            trajectories.append(traj)

        return {
            "trajectories": trajectories,
            "coord_position_count": n_positions,
            "n_responses": n_responses,
            "total_positions": prob["coords"].shape[0] if prob.get("coords") is not None else 0,
            "embedding_method": "umap",
            "pos_size": 32,
            "sep_up": self.sep_up,
            "upsampling": self.sep_up,
            "effective_pos_size": 32 * self.sep_up,
            "mode": "non-overlapping",
        }

    def get_tsne_data(
        self,
        viz_idx: int,
        pos_start: int = 0,
        pos_end: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        返回散点图数据 (用于 /api/tsne)

        Args:
            viz_idx: 问题的 viz 索引
            pos_start: 起始位置
            pos_end: 结束位置 (不包含)

        Returns:
            Scatter plot data dict or None if not found
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None:
            return None

        coords = prob.get("coords")
        if coords is None:
            return None

        total_positions, n_responses, _ = coords.shape

        # Defensive clamping
        pos_start = max(0, min(pos_start, total_positions))
        if pos_end is None or pos_end > total_positions:
            pos_end = total_positions
        pos_end = max(pos_start, pos_end)

        # 提取指定范围的坐标
        coords_slice = coords[pos_start:pos_end]
        n_positions = coords_slice.shape[0]

        # 预计算每个 response 的有效位置上限（跳过 padding 点）
        text_pos_limits = []
        for resp in prob["responses"]:
            sample_id = resp.get("sample_id", resp.get("run_index"))
            text_pos_limits.append(
                self._safe_max_text_position(sample_id, total_positions)
            )

        # 展平为散点列表（跳过超出 text_position_count 的 padding 点）
        flat_coords = []
        labels = []
        for pos_idx in range(n_positions):
            actual_pos = pos_start + pos_idx
            for resp_idx, resp in enumerate(prob["responses"]):
                if actual_pos >= text_pos_limits[resp_idx]:
                    continue
                flat_coords.append(coords_slice[pos_idx, resp_idx, :].tolist())
                labels.append({
                    "pos": actual_pos,
                    "response_idx": resp_idx,
                    "sample_id": resp.get("sample_id", resp.get("run_index", resp_idx)),
                    "run_index": resp_idx + 1,
                    "is_correct": resp.get("is_correct", False),
                })

        return {
            "coords": flat_coords,
            "labels": labels,
            "coord_position_count": n_positions,
            "n_responses": n_responses,
            "pos_start": pos_start,
            "pos_end": pos_end,
            "total_positions": total_positions,
            "sep_up": self.sep_up,
            "effective_pos_size": 32 * self.sep_up,
        }

    def get_knn_data(self, viz_idx: int) -> Optional[Dict[str, Any]]:
        """
        返回 KNN 图数据 (新 API: /api/knn)

        Args:
            viz_idx: 问题的 viz 索引

        Returns:
            KNN data dict or None if not found
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None:
            return None

        knn_indices = prob.get("knn_indices")
        knn_dists = prob.get("knn_dists")
        slice_info = prob.get("slice_info")

        if knn_indices is None:
            return None

        return {
            "knn_indices": knn_indices.tolist(),
            "knn_dists": knn_dists.tolist() if knn_dists is not None else None,
            "slice_info": slice_info.tolist() if slice_info is not None else None,
            "n_slices": knn_indices.shape[0],
            "n_neighbors": knn_indices.shape[1] if len(knn_indices.shape) > 1 else 0,
        }

    def get_coords(self, viz_idx: int) -> Optional[Dict[str, Any]]:
        """
        返回坐标数据 (新 API: /api/coords)

        Args:
            viz_idx: 问题的 viz 索引

        Returns:
            Coordinates data dict or None if not found
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None or prob.get("coords") is None:
            return None

        coords = prob["coords"]
        return {
            "coords": coords.tolist(),
            "n_slices": coords.shape[0],
            "n_responses": coords.shape[1],
        }

    def get_responses(self, viz_idx: int) -> Optional[Dict[str, Any]]:
        """
        返回响应列表 (新 API: /api/responses)

        Args:
            viz_idx: 问题的 viz 索引

        Returns:
            Responses data dict or None if not found
        """
        prob = self.get_problem_data(viz_idx)
        if prob is None:
            return None

        coords = prob.get("coords")
        coord_position_count = coords.shape[0] if coords is not None else 0

        return {
            "responses": [
                {
                    "sample_id": resp.get("sample_id", resp.get("run_index", i)),
                    "run_index": i + 1,
                    "is_correct": resp.get("is_correct", False),
                    "finish_reason": resp.get("finish_reason", "unknown"),
                    "text_position_count": self._safe_max_text_position(
                        resp.get("sample_id", resp.get("run_index")),
                        coord_position_count,
                    ),
                }
                for i, resp in enumerate(prob["responses"])
            ],
            "n_correct": prob["n_correct"],
            "n_responses": prob["n_responses"],
        }

    # =========== Token Text Service ===========

    def init_token_text_service(self, nad_cache_path: Optional[Path] = None) -> bool:
        """
        初始化 Token 文本服务

        Args:
            nad_cache_path: NAD cache 目录路径 (默认使用 self.cache_path)

        Returns:
            初始化是否成功
        """
        cache_path = nad_cache_path or self.cache_path
        self._token_text_service = TokenTextService(cache_path)
        return self._token_text_service.initialize()

    @property
    def token_text_ready(self) -> bool:
        """检查 Token 文本服务是否就绪"""
        return self._token_text_service is not None and self._token_text_service.is_ready

    def get_token_text(
        self,
        sample_id: int,
        position: int
    ) -> Dict[str, Any]:
        """
        获取 token 文本

        Args:
            sample_id: GLOBAL sample_id (来自 viz metadata 的 run_index)
            position: 位置索引

        Returns:
            Token 文本数据字典 (通过 TokenTextService)
        """
        if self._token_text_service is None:
            return {"success": False, "error": "Token text service not initialized"}
        return self._token_text_service.get_token_text(
            sample_id, position, sep_up=self.sep_up
        )

    def get_available_sep_ups(self) -> List[int]:
        """Return sorted list of available sep_up values for this cache."""
        from .core import discover_available_sep_ups
        return discover_available_sep_ups(self.cache_path)
