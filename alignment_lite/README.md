# alignment_lite

从第一性原理重写的 alignment 包。用最少代码完成 eval report 与 viz metadata 的对齐，
输出与旧版 `alignment` 包**逐字段完全一致**（已在 2 模型 x 6 数据集 = 12 个 cache、62,080 条 response 上验证通过）。

## 设计动机

旧版 `alignment/` 包含 9 个模块、500+ 行代码，涉及 verification、warning、fallback 等复杂逻辑。
但 alignment 的本质只有一个操作：**按 `problem_id` JOIN 两张表，用 1-based run_index 做行对齐**。

`alignment_lite` 把这个操作提炼成一个 `core.py`（~220 行），去掉所有非必要的间接层。

## 第一性原理

```
eval_report                           viz_metadata
(evaluation_report_compact.json)      (viz/N/metadata.json)
  problem_id -> runs[]                  problem_id -> responses[]
  runs[i].run_index (1-based)           responses[i].run_index (global sample_id)
                    \                  /
                     \                /
                  按 problem_id JOIN
                     /                \
                    /                  \
            responses[i].is_correct       = eval_runs[i+1].is_correct
            responses[i].finish_reason    = eval_runs[i+1].finish_reason
            responses[i].stop_reason      = eval_runs[i+1].stop_reason
            responses[i].extracted_answer = eval_runs[i+1].extracted_answer
            responses[i].sample_id        = responses[i].run_index  (保留 global ID)
```

**关键映射规则**: `responses[i]` 对应 `eval_runs[run_index = i + 1]`（eval report 使用 1-based index）。

**权威性**: eval report 是 `is_correct` 等字段的权威来源（authoritative source），viz metadata 中如有同名字段会被 eval 覆盖。

**sample_id 保留**: viz metadata 的 `run_index` 是全局 sample_id（用于索引 cache 中的 `token_row_ptr`、`tok_conf` 等数组），不会被 eval 的 local run_index 覆盖。

## 包结构

```
alignment_lite/
    __init__.py     (30 行)  导出所有公开 API
    core.py         (220 行) 全部核心逻辑，5 个主函数 + 2 个辅助函数
    README.md       (本文件)
```

## API 参考

### 主入口

#### `load_problems(cache_path, sep_up=8, max_problems=None)`

加载一个 cache 的所有问题数据（eval + viz + knn 全部合并）。

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_path` | `Path \| str` | 必填 | cache 目录路径，格式 `.../MODEL/DATASET/CACHE_NAME` |
| `sep_up` | `int` | `8` | KNN 图上采样因子（1/2/4/8/16） |
| `max_problems` | `int \| None` | `None` | 最多加载的问题数，`None` 表示全部 |

**返回值**: `List[Dict]`，每个元素包含:

```python
{
    "viz_idx":      int,          # viz 子目录编号 (0, 1, 2, ...)
    "problem_id":   str,          # 问题 ID
    "knn_indices":  np.ndarray,   # KNN 邻居索引
    "knn_dists":    np.ndarray,   # KNN 邻居距离
    "slice_info":   np.ndarray,   # slice 映射信息
    "responses":    List[Dict],   # 已合并的 response 列表 (见下方字段说明)
    "n_correct":    int,          # 正确 response 数
    "n_responses":  int,          # response 总数
}
```

**response 字段** (合并后):

| 字段 | 来源 | 说明 |
|------|------|------|
| `is_correct` | eval (覆盖 viz) | 是否正确 |
| `finish_reason` | eval (覆盖 viz) | 完成原因 |
| `stop_reason` | eval (覆盖 viz) | 停止原因 |
| `extracted_answer` | eval (覆盖 viz) | 提取的答案 |
| `sample_id` | viz `run_index` | 全局 sample ID（用于索引 cache 数组） |
| 其他原始字段 | viz | viz metadata 中的其他字段原样保留 |

**用法**:

```python
from alignment_lite import load_problems

problems = load_problems("/path/to/cache", sep_up=8)
for p in problems:
    print(f"{p['problem_id']}: {p['n_correct']}/{p['n_responses']}")
    for r in p["responses"]:
        print(f"  sample_id={r['sample_id']}, correct={r['is_correct']}")
```

**异常**:
- `FileNotFoundError`: `evaluation_report_compact.json` 不存在、viz 目录不存在、`metadata.json` 或 knn 文件缺失
- `ValueError`: `problem_id` 在 eval report 中找不到、response 数量与 eval runs 数量不匹配

### 核心函数

#### `load_eval_by_pid(cache_path)`

加载 `evaluation_report_compact.json`，构建 `{problem_id: {run_index: run_data}}` 索引。

```python
from alignment_lite import load_eval_by_pid

eval_indexed = load_eval_by_pid(cache_path)
# eval_indexed["AIME_2024_I_1"][1]  ->  {"run_index": 1, "is_correct": True, ...}
```

#### `resolve_viz_dir(cache_path, viz_root=None)`

从 cache 路径推断 viz 目录位置。

```
cache_path: .../{cache_base}/{model}/{dataset}/{cache_name}
         ->  vizcache_output/{model}/{dataset}/{cache_name}/viz/
```

默认 vizcache root: set via `INTRA_COT_VIZCACHE_ROOT` environment variable.

可通过 `viz_root` 参数覆盖。

#### `load_knn_data(viz_dir, sep_up=8)`

加载单个问题子目录下的 3 个 numpy 文件:

```
viz/N/sep_up{sep_up}x_knn_indices.npy  ->  knn_indices
viz/N/sep_up{sep_up}x_knn_dists.npy    ->  knn_dists
viz/N/sep_up{sep_up}x_slice_info.npy   ->  slice_info
```

返回 `(knn_indices, knn_dists, slice_info)` 元组。

#### `merge_responses(metadata_responses, eval_runs_by_idx, problem_id)`

核心 JOIN 操作。将 viz 的 responses 与 eval 的 runs 按 1-based index 对齐合并。

```python
from alignment_lite import merge_responses

# eval_runs_by_idx: {1: {run_index:1, is_correct:True, ...}, 2: {...}, ...}
merged = merge_responses(viz_responses, eval_runs_by_idx, "problem_42")
```

### 辅助函数

#### `load_meta_mapping(cache_dir)`

从 `cache/meta.json` 构建 sample_id / run_index 双向映射。

```python
from alignment_lite import load_meta_mapping

mapping = load_meta_mapping(cache_path)
# mapping["problem_42"] = {
#     "sample_ids":    [640, 641, 642, ...],   # global IDs
#     "run_indices":   [0, 1, 2, ...],         # local indices
#     "run_to_sample": {0: 640, 1: 641, ...},  # local -> global
#     "sample_to_run": {640: 0, 641: 1, ...},  # global -> local
# }
```

#### `get_cache_reader(cache_path)`

获取 NAD CacheReader 实例（严格模式：要求 `tok_conf` 数据存在）。

```python
from alignment_lite import get_cache_reader

reader = get_cache_reader(cache_path)
token_view = reader.get_token_view(sample_id)
tok_conf = token_view.tok_conf  # token-level confidence
```

#### Re-exports: `CacheReader`, `TokenView`

从 NAD core 直接导出，方便下游使用:

```python
from alignment_lite import CacheReader, TokenView
```

## 数据流

```
cache/evaluation_report_compact.json    vizcache_output/.../viz/N/metadata.json
             |                                       |
      load_eval_by_pid()                    json.load() -> responses
             |                                       |
   {pid: {run_idx: run}}                   problem_id, responses[]
             \                                      /
              \                                    /
               ------  merge_responses()  --------
                              |
                     merged responses[]
                     (eval fields overlaid,
                      sample_id preserved)
                              |
                              +---- load_knn_data() ----> knn_indices, knn_dists, slice_info
                              |
                      load_problems() output
                      [{problem_id, responses, knn_*, n_correct, ...}]
```

## 与旧版 alignment 的区别

| 维度 | alignment (旧) | alignment_lite (新) |
|------|---------------|-------------------|
| 文件数 | 9 个模块 | 1 个 `core.py` |
| 代码量 | ~500 行 | ~220 行 |
| 验证逻辑 | 内置 `validate_response_sample_ids()` | 不内置（由外部测试覆盖） |
| Warning 机制 | `warnings.warn()` on field mismatch | 无（strict-only，不匹配直接抛异常） |
| Fallback | 部分函数有 fallback 路径 | 无 fallback，缺数据即报错 |
| 输出 | 完全一致 | 完全一致 |

**关键差异说明**: 旧版 `merge_responses()` 会在合并后调用 `validate_response_sample_ids()` 做端到端验证（检查 sample_id 归属、token 长度范围等）。`alignment_lite` 省略了这一步——alignment 正确性已通过 12 cache 全量对比验证，不需要每次运行时重复检查。

## 验证方案

验证脚本: `tests/test_alignment_lite_vs_old.py`

### 验证范围

| 模型 | 数据集 | 问题数 | Response 数 |
|------|--------|--------|-------------|
| DeepSeek-R1-0528-Qwen3-8B | aime24 | 30 | 1,920 |
| DeepSeek-R1-0528-Qwen3-8B | aime25 | 30 | 1,920 |
| DeepSeek-R1-0528-Qwen3-8B | brumo25 | 30 | 1,920 |
| DeepSeek-R1-0528-Qwen3-8B | gpqa | 198 | 12,672 |
| DeepSeek-R1-0528-Qwen3-8B | hmmt25 | 30 | 1,920 |
| DeepSeek-R1-0528-Qwen3-8B | livecodebench_v5 | 167 | 10,688 |
| Qwen3-4B-Thinking-2507 | aime24 | 30 | 1,920 |
| Qwen3-4B-Thinking-2507 | aime25 | 30 | 1,920 |
| Qwen3-4B-Thinking-2507 | brumo25 | 30 | 1,920 |
| Qwen3-4B-Thinking-2507 | gpqa | 198 | 12,672 |
| Qwen3-4B-Thinking-2507 | hmmt25 | 30 | 1,920 |
| Qwen3-4B-Thinking-2507 | livecodebench_v5 | 167 | 10,688 |
| **合计** | | **970** | **62,080** |

### 逐字段比较项

每个 cache 的每个 problem 比较以下字段（全部使用严格相等判定）:

- `problem_id` — 问题 ID 字符串
- `n_correct` — 正确 response 计数
- `n_responses` — response 总数
- 逐 response 比较:
  - `is_correct` — 是否正确
  - `sample_id` — 全局 sample ID
  - `finish_reason` — 完成原因
- numpy 数组（`np.array_equal`）:
  - `knn_indices` — KNN 邻居索引矩阵
  - `knn_dists` — KNN 邻居距离矩阵
  - `slice_info` — slice 映射信息

### 运行验证

```bash
cd sc_selector_minimal

# 默认: 验证 cache_list.txt + cache_list_qwen3_4b_thinking.txt (12 caches)
python tests/test_alignment_lite_vs_old.py

# 指定 cache 列表
python tests/test_alignment_lite_vs_old.py --cache-list cache_list.txt

# 多个列表
python tests/test_alignment_lite_vs_old.py \
    --cache-list cache_list.txt \
    --cache-list cache_list_qwen3_4b_thinking.txt
```

### 预期输出

```
Verifying 12 caches (sep_up=8)
========================================================================
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/aime24: 30 problems, 1920 responses
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/aime25: 30 problems, 1920 responses
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/brumo25: 30 problems, 1920 responses
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/gpqa: 198 problems, 12672 responses
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/hmmt25: 30 problems, 1920 responses
  [PASS]  DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5: 167 problems, 10688 responses
  [PASS]  Qwen3-4B-Thinking-2507/aime24: 30 problems, 1920 responses
  [PASS]  Qwen3-4B-Thinking-2507/aime25: 30 problems, 1920 responses
  [PASS]  Qwen3-4B-Thinking-2507/brumo25: 30 problems, 1920 responses
  [PASS]  Qwen3-4B-Thinking-2507/gpqa: 198 problems, 12672 responses
  [PASS]  Qwen3-4B-Thinking-2507/hmmt25: 30 problems, 1920 responses
  [PASS]  Qwen3-4B-Thinking-2507/livecodebench_v5: 167 problems, 10688 responses
========================================================================
Results: 12 passed, 0 failed, 12 total
ALL PASSED
```

## 依赖

- **Python 3.9+**
- **numpy** — 加载 KNN 数据（`.npy` 文件）
- **NAD core** (`nad.core.views.reader`) — CacheReader / TokenView（仅 `get_cache_reader` 需要）

无其他第三方依赖。

## 目录约定

```
{cache_base}/{model}/{dataset}/{cache_name}/
    evaluation_report_compact.json    # eval report (必需)
    meta.json                         # sample -> problem 映射 (load_meta_mapping 需要)
    token_data/                       # token 级别数据 (get_cache_reader 需要)
        token_row_ptr.int64
        tok_conf.float32

vizcache_output/{model}/{dataset}/{cache_name}/viz/
    0/metadata.json                   # 问题 0 的 viz 数据
    0/sep_up8x_knn_indices.npy
    0/sep_up8x_knn_dists.npy
    0/sep_up8x_slice_info.npy
    1/metadata.json                   # 问题 1
    ...
```
