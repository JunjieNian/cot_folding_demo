# Alignment 包使用指南

**创建日期**: 2026-02-02
**更新日期**: 2026-02-05 (修复 problem_id 匹配方式)
**目的**: 统一数据加载接口，消除重复代码，确保严格模式

---

## 强制规范

> **所有新脚本必须使用 `alignment` 包进行数据加载，禁止直接读取文件。**

原因：
1. **单一来源**: 数据加载逻辑集中维护，避免多处重复
2. **严格模式**: 缺少必要文件时直接报错，不允许静默跳过
3. **数据一致性**: 自动合并 metadata 和 evaluation 数据
4. **减少错误**: 避免 sample_id/run_index 映射错误

---

## 2026-02-05 重大修复: Problem_id 匹配方式

发现旧的 alignment 包使用 **index-based 匹配**（viz/N → eval_results[N]），
但 viz 目录按**字符串排序**，eval_report 按**数字排序**，导致 4 个数据集错误匹配：

| 数据集 | 错误匹配数量 | 示例 |
|--------|-------------|------|
| brumo25 | 28/30 | viz/2 → `brumo25-10` (不是 `brumo25-2`) |
| gpqa | 196/198 | viz/3 → `gpqa-100` (不是 `gpqa-3`) |
| hmmt25 | 28/30 | viz/2 → `hmmt25-10` (不是 `hmmt25-2`) |
| livecodebench_v5 | 165/167 | 类似问题 |

**修复方案**: 参考 `verify_viz_eval_mapping.py`，改用 **problem_id 字符串匹配**：
```python
# 旧方式 (错误): index-based
eval_runs = eval_results[problem_idx]

# 新方式 (正确): problem_id 字符串匹配
eval_indexed = load_evaluation_results_by_problem_id(cache_path)
eval_runs = get_runs_for_problem_by_id(eval_indexed, problem_id)
```

### 修复后准确率变化 (k=1, neighbor_k=8)

| 数据集 | 旧 (错误匹配) | 新 (正确匹配) | 变化 |
|--------|--------------|--------------|------|
| brumo25 | 76.7% | 80.0% | **+3.3%** |
| gpqa | 61.1% | 65.2% | **+4.0%** |
| hmmt25 | 43.3% | 60.0% | **+16.7%** |
| livecodebench | 62.3% | 59.3% | -3.0% |
| **平均** | 60.9% | 66.1% | **+5.3%** |

---

## 快速开始

### 最简用法 (推荐)

```python
from pathlib import Path
from alignment import load_all_problems_data

cache_path = Path("/path/to/cache")
problems = load_all_problems_data(cache_path, sep_up=8)

for prob in problems:
    print(f"Problem {prob['problem_id']}:")
    print(f"  KNN shape: {prob['knn_indices'].shape}")
    print(f"  Responses: {prob['n_responses']}")
    print(f"  Correct: {prob['n_correct']}")
```

### 限制加载数量 (调试用)

```python
# 只加载前 3 个问题
problems = load_all_problems_data(cache_path, sep_up=8, max_problems=3)
```

---

## API 参考

### 主要函数

#### `load_all_problems_data(cache_path, sep_up=8, max_problems=None)`

加载 cache 的所有问题数据。**这是最常用的 API**。

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_path` | `Path` | 必需 | cache 目录路径 |
| `sep_up` | `int` | `8` | KNN 图上采样因子 (1/2/4/8/16) |
| `max_problems` | `int` | `None` | 最多加载的问题数 (None = 全部) |

**返回**: `List[Dict[str, Any]]`

每个问题包含:
```python
{
    "viz_idx": int,           # viz 目录索引 (0, 1, 2, ...)
    "problem_id": str,        # 问题 ID
    "knn_indices": ndarray,   # KNN 邻居索引 [n_slices, k]
    "knn_dists": ndarray,     # KNN 距离 [n_slices, k]
    "slice_info": ndarray,    # slice 信息 [n_slices, 2] (run_id, position)
    "responses": List[Dict],  # 响应列表 (已合并 eval 数据)
    "n_correct": int,         # 正确响应数
    "n_responses": int,       # 总响应数
}
```

**异常**:
- `FileNotFoundError`: viz 目录不存在
- `FileNotFoundError`: 缺少 `evaluation_report_compact.json`
- `FileNotFoundError`: 缺少指定 `sep_up` 的 KNN 文件

---

### 辅助函数

#### `load_problem_data(cache_path, viz_dir, problem_idx, sep_up=8, eval_results=None)`

加载单个问题的完整数据。适用于需要精细控制的场景。

```python
from alignment import load_problem_data, load_evaluation_results

# 预加载 eval_results (批量时复用)
eval_results = load_evaluation_results(cache_path)

# 加载单个问题
prob = load_problem_data(
    cache_path=cache_path,
    viz_dir=cache_path / "viz" / "0",
    problem_idx=0,
    sep_up=8,
    eval_results=eval_results,
)
```

#### `load_knn_data(viz_dir, sep_up=8)`

只加载 KNN 数据 (不含 metadata)。

```python
from alignment import load_knn_data

knn_indices, knn_dists, slice_info = load_knn_data(viz_dir, sep_up=8)
```

#### `load_problem_metadata(viz_dir)`

只加载 metadata.json。

```python
from alignment import load_problem_metadata

problem_id, responses = load_problem_metadata(viz_dir)
```

#### `resolve_viz_dir(cache_path, viz_root=None)`

解析 viz 目录路径。

```python
from alignment import resolve_viz_dir

# 默认从 vizcache_output 查找
viz_dir = resolve_viz_dir(cache_path)

# 指定自定义 viz 根目录
viz_dir = resolve_viz_dir(cache_path, viz_root="/custom/viz/root")
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_path` | `Path` | 必需 | cache 目录路径 (.../MODEL/DATASET/CACHE_NAME) |
| `viz_root` | `str` | `None` | 自定义 viz 根目录 (默认: vizcache_output) |

**返回**: `Path` - viz 目录路径

**异常**: `FileNotFoundError` - viz 目录不存在

---

### Token 数据读取

#### `get_cache_reader(cache_path)`

获取 CacheReader 实例，用于访问 cache 中的 token 级别数据。**推荐使用此函数**。

```python
from alignment import get_cache_reader

reader = get_cache_reader(cache_path)
token_view = reader.get_token_view(sample_id)
tok_conf = token_view.tok_conf
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `cache_path` | `Path` | cache 目录路径 |

**返回**: `CacheReader` 实例

**异常**:
- `FileNotFoundError`: cache 目录不存在
- `ValueError`: cache 缺少 tok_conf 数据

#### `CacheReader` 和 `TokenView`

直接使用 NAD CacheReader 类。通常使用 `get_cache_reader()` 更方便。

```python
from alignment import CacheReader

reader = CacheReader(str(cache_path))
token_view = reader.get_token_view(sample_id)

# TokenView 属性
tok_conf = token_view.tok_conf      # token 置信度 (np.ndarray)
token_ids = token_view.token_ids    # token IDs (np.ndarray)
```

---

## 完整导出列表

```python
from alignment import (
    # 数据加载 (主要 API)
    load_all_problems_data,  # 加载所有问题 ★ 推荐
    load_problem_data,       # 加载单个问题
    load_knn_data,           # 只加载 KNN 数据
    load_problem_metadata,   # 只加载 metadata

    # 路径解析
    resolve_viz_dir,         # 解析 viz 目录路径

    # Token 数据读取 (CacheReader)
    CacheReader,             # NAD CacheReader 类
    TokenView,               # Token 视图类
    get_cache_reader,        # 获取 CacheReader 实例 ★ 推荐

    # 元数据映射
    load_meta_mapping,
    get_sample_ids_for_problem,
    run_index_to_sample_id,
    sample_id_to_run_index,

    # 响应映射
    build_response_mapping,
    merge_responses,

    # 评估数据
    load_evaluation_results,                # 旧接口 (index-based)
    load_evaluation_results_by_problem_id,  # 新接口 (problem_id 匹配) ★ 推荐
    get_runs_for_problem,                   # 旧接口
    get_runs_for_problem_by_id,             # 新接口 (problem_id 匹配) ★ 推荐
    resolve_n_runs,
    get_correct_indices,
    compute_problem_accuracy,
    get_problem_id_from_eval_report,

    # 验证
    validate_viz_eval_alignment,
    validate_response_mapping,
    validate_response_sample_ids,
    VerificationResult,
)
```

---

## 数据流架构

```
cache_path
    │
    ├── evaluation_report_compact.json ──────┐
    │                                        │
    ├── token_data/                          │
    │   ├── token_row_ptr.int64             │
    │   └── tok_conf.float32 ───────────────┼──► get_cache_reader()
    │                                        │         │
    └── viz/                                 │         ▼
        ├── 0/                               │    CacheReader.get_token_view()
        │   ├── metadata.json ───────────────┼──► merge_responses()
        │   ├── sep_up8x_knn_indices.npy     │         │
        │   ├── sep_up8x_knn_dists.npy       │         ▼
        │   └── sep_up8x_slice_info.npy      │    responses (已合并)
        │                                    │         │
        ├── 1/                               │         │
        │   └── ...                          │         │
        └── ...                              │         ▼
                                             │
                                             └──► load_all_problems_data()
                                                       │
                                                       ▼
                                                 List[Dict] 问题数据
```

---

## 严格模式说明

所有函数都运行在 **严格模式**：

| 情况 | 行为 |
|------|------|
| viz 目录不存在 | `raise FileNotFoundError` |
| 缺少 evaluation_report_compact.json | `raise FileNotFoundError` |
| 缺少指定 sep_up 的 KNN 文件 | `raise FileNotFoundError` (附带可用 sep_up 列表) |
| metadata.json 不存在 | `raise FileNotFoundError` |

**不允许 fallback**。如果数据不完整，必须先修复数据，而非静默跳过。

---

## 迁移指南

### 旧代码 (禁止)

```python
# ❌ 禁止直接读取文件
import json
import numpy as np

viz_dir = cache_path / "viz" / "0"
knn_indices = np.load(viz_dir / "sep_up8x_knn_indices.npy")
with open(viz_dir / "metadata.json") as f:
    metadata = json.load(f)
responses = metadata["responses"]
```

### 新代码 (推荐)

```python
# ✓ 使用 alignment 包
from alignment import load_all_problems_data

problems = load_all_problems_data(cache_path, sep_up=8)
prob = problems[0]
knn_indices = prob["knn_indices"]
responses = prob["responses"]  # 已合并 eval 数据
```

---

## 使用示例

### 示例 1: 批量分析脚本

```python
#!/usr/bin/env python3
"""批量分析脚本模板"""

from pathlib import Path
from alignment import load_all_problems_data

def analyze_problem(prob):
    """分析单个问题"""
    knn_indices = prob["knn_indices"]
    responses = prob["responses"]

    # 统计正确邻居
    n_correct = prob["n_correct"]
    n_total = prob["n_responses"]

    return {
        "problem_id": prob["problem_id"],
        "accuracy": n_correct / n_total if n_total > 0 else 0,
    }

def main():
    cache_path = Path("/path/to/cache")

    # 一行加载所有数据
    problems = load_all_problems_data(cache_path, sep_up=8)

    results = []
    for prob in problems:
        result = analyze_problem(prob)
        results.append(result)
        print(f"{result['problem_id']}: {result['accuracy']:.1%}")

if __name__ == "__main__":
    main()
```

### 示例 2: 调试单个问题

```python
from pathlib import Path
from alignment import load_all_problems_data

cache_path = Path("/path/to/cache")

# 只加载 1 个问题用于调试
problems = load_all_problems_data(cache_path, sep_up=8, max_problems=1)
prob = problems[0]

print(f"Problem ID: {prob['problem_id']}")
print(f"KNN shape: {prob['knn_indices'].shape}")
print(f"Responses: {prob['n_responses']}")
print(f"Correct: {prob['n_correct']}")

# 查看第一个 response
resp = prob["responses"][0]
print(f"First response: is_correct={resp.get('is_correct')}, answer={resp.get('extracted_answer')}")
```

### 示例 3: 遍历所有 cache

```python
from pathlib import Path
from alignment import load_all_problems_data

cache_list = [
    Path("/path/to/cache1"),
    Path("/path/to/cache2"),
]

for cache_path in cache_list:
    try:
        problems = load_all_problems_data(cache_path, sep_up=8)
        total_correct = sum(p["n_correct"] for p in problems)
        total_responses = sum(p["n_responses"] for p in problems)
        print(f"{cache_path.name}: {total_correct}/{total_responses}")
    except FileNotFoundError as e:
        print(f"{cache_path.name}: ERROR - {e}")
```

---

## 相关文件

```
sc_selector_minimal/
├── alignment/
│   ├── __init__.py           # 包导出 (公开 API)
│   ├── _nad_compat.py        # NAD 核心库兼容层 (内部模块)
│   ├── cache_reader.py       # Token 数据读取 (CacheReader) ★
│   ├── data_loading.py       # KNN/Metadata 加载 ★
│   ├── viz_path.py           # viz 路径解析
│   ├── evaluation.py         # 评估数据加载
│   ├── meta_mapping.py       # sample_id 映射
│   ├── response_mapping.py   # 响应合并
│   ├── verification.py       # 数据验证
│   └── README.md             # 本文档
├── sc_representative_selector.py # 使用 alignment 包
├── sc_slice_text_extractor.py    # 使用 alignment 包
├── run_batch_analysis.py         # 使用 alignment 包
└── scripts/
    └── analyze_knn_answer_alignment.py  # 使用 alignment 包
```

**说明**:
- `_nad_compat.py` 以下划线开头，表示是内部模块
- 外部代码不应直接导入 `_nad_compat`，应使用 `alignment` 公开 API

---

## 常见问题

### Q: 为什么 responses 里没有 is_correct 字段？

A: 检查 `evaluation_report_compact.json` 是否存在。alignment 包会自动合并 eval 数据到 responses。

### Q: 如何获取不同 sep_up 的数据？

A: 修改 `sep_up` 参数：

```python
problems_sep4 = load_all_problems_data(cache_path, sep_up=4)
problems_sep8 = load_all_problems_data(cache_path, sep_up=8)
problems_sep16 = load_all_problems_data(cache_path, sep_up=16)
```

### Q: 报错 "缺少 sep_up=X 的 knn 文件"？

A: 错误信息会列出可用的 sep_up 值，选择其中一个。

### Q: 如何只获取部分字段？

A: 使用底层函数：

```python
from alignment import load_knn_data, load_problem_metadata

# 只要 KNN 数据
knn_indices, knn_dists, slice_info = load_knn_data(viz_dir, sep_up=8)

# 只要 metadata
problem_id, responses = load_problem_metadata(viz_dir)
```

### Q: 如何读取 token 置信度 (tok_conf)？

A: 使用 `get_cache_reader()`:

```python
from alignment import get_cache_reader

reader = get_cache_reader(cache_path)
token_view = reader.get_token_view(sample_id)
tok_conf = token_view.tok_conf  # np.ndarray
```

### Q: 报错 "cache 缺少 tok_conf 数据"？

A: cache 目录缺少 `token_data/tok_conf.float32` 文件。请确保 cache 包含完整的 token 数据。
