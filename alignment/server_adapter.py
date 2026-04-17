#!/usr/bin/env python3
"""
服务器适配器模块

将 alignment 核心库的数据转换为 knn_viz_server 前端期望的格式。
不修改核心库的返回值，而是在适配层进行数据增强。

设计原则:
1. 核心数据来自 alignment.load_all_problems_data() (不修改返回值)
2. 坐标数据在适配层单独加载 (核心库不负责)
3. 动态构建 index 数据

Usage:
    from alignment import VizDataAdapter

    adapter = VizDataAdapter(cache_path, sep_up=8)

    # 获取问题列表
    problems = adapter.get_problems_list()

    # 获取轨迹数据
    trajectory = adapter.get_trajectory_data(viz_idx=0)

NOTE: Implementation delegated to alignment_lite.server_adapter
"""

from alignment_lite.server_adapter import VizDataAdapter  # noqa: F401
