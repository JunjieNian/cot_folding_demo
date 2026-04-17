# RL Checkpoint NFS Pipeline

追踪 Qwen3-4B-Base 在 RL 训练过程中 CoT 折叠结构（NFS 各指标）的演变趋势。

## 数据来源

NAD RL Cache: Set via `INTRA_COT_RL_CACHE_ROOT` environment variable.

| Checkpoint | 模型目录 | RL Step | Accuracy |
|-----------|---------|---------|----------|
| base | Qwen3-4B-Base_base | 0 | 28.61% |
| step-100 | Qwen3-4B-Base_math7500-step-100 | 100 | 29.48% |
| step-200 | Qwen3-4B-Base_math7500-step-200 | 200 | 30.49% |
| step-300 | Qwen3-4B-Base_math7500-step-300 | 300 | 31.55% |
| step-400 | Qwen3-4B-Base_math7500-step-400 | 400 | 31.68% |
| step-500 | Qwen3-4B-Base_math7500-step-500 | 500 | 32.64% |
| step-600 | Qwen3-4B-Base_math7500-step-600 | 600 | 33.19% |
| step-700 | Qwen3-4B-Base_math7500-step-700 | 700 | 32.99% |
| step-800 | Qwen3-4B-Base_math7500-step-800 | 800 | 32.11% |
| step-900 | Qwen3-4B-Base_math7500-step-900 | 900 | 32.39% |
| step-1000 | Qwen3-4B-Base_math7500-step-1000 | 1000 | 32.01% |

- 数据集: `variable-reasoning-mini`
- 每个 checkpoint: **100 problems x 64 runs = 6400 samples**
- Problem ID: 32 字符 hex UUID + deepscaler 前缀

---

## Pipeline 架构

```
Per checkpoint (Phase 1-6):

  NAD Cache
      │
      ▼
  Phase 1: graph_builder.run()           ← nfs_pipeline/graph_builder.py
      ├─ 提取 slice keys (每 32 tokens)
      ├─ HMM Viterbi 分割 → Exploration(0) / Exploitation(1)
      └─ Jaccard 距离矩阵 O(n²)
      │  输出: dist_*.npy, hmm_*.npy, batch_summary.json
      ▼
  Phase 2: primitives.run()              ← nfs_pipeline/primitives.py
      ├─ Core (最大 exploit 致密块)
      ├─ Return Edge (长程回返接触)
      ├─ Drift Branch (未回并探索支路)
      └─ Final Closure (尾部回连)
      │  输出: primitives_analysis.json
      ▼
  Phase 3: fold_score.run()              ← nfs_pipeline/fold_score.py
      ├─ B = s_core · f_core (Backbone)
      ├─ H = mean(return strength) (Hydrogen)
      ├─ D* = 1 - G·(1-D0) (Drift)
      └─ NFS = 100·(B·H·(1-D*))^(1/3)
      │  输出: nfs_analysis.json (含 AUROC, Hit@k, Voting)
      ▼
  Phase 4: segment_graph.run()           ← nfs_pipeline/segment_graph.py
      ├─ HMM 连续段 → segment 节点
      ├─ 方法 A (union): segment key 并集的 Jaccard
      └─ 方法 B (avg): slice 距离矩阵的块均值
      │  输出: seg_dist_{union,avg}_*.npy, seg_meta_*.json, segment_summary.json
      ▼
  Phase 5: segment_primitives.run()      ← nfs_pipeline/segment_primitives.py
      │  输出: primitives_segment_{union,avg}.json
      ▼
  Phase 6: segment_score.run()           ← nfs_pipeline/segment_score.py
      │  输出: nfs_segment_{union,avg}.json (含 slice 对比)

Cross-checkpoint (Phase 7):
  Phase 7: aggregate_trajectory()
      │  输出: cross_checkpoint/nfs_trajectory.json
```

---

## 用法

### 全量运行（11 个 checkpoint）

```bash
python run_rl_pipeline.py
```

### 指定 checkpoint

```bash
python run_rl_pipeline.py --checkpoint base step-600
```

### 只跑 slice 级 (Phase 1-3)

```bash
python run_rl_pipeline.py --phase 1 2 3
```

### 只跑 segment 级 (Phase 4-6)

```bash
python run_rl_pipeline.py --phase 4 5 6
```

### 只做跨 checkpoint 汇总

```bash
python run_rl_pipeline.py --aggregate-only
```

### 节省磁盘（不保存距离矩阵）

```bash
python run_rl_pipeline.py --no-save-matrices
```

### Dry-run（打印调用不执行）

```bash
python run_rl_pipeline.py --dry-run
```

### Python 编程调用

```python
from nfs_pipeline import graph_builder, primitives, fold_score
from nfs_pipeline import segment_graph, segment_primitives, segment_score
from project_paths import resolve_rl_cache_path, default_rl_batch_dir

cache = resolve_rl_cache_path("base")
out = default_rl_batch_dir("base")

graph_builder.run(cache, out)                        # Phase 1
primitives.run(out)                                  # Phase 2
fold_score.run(cache, out)                           # Phase 3
segment_graph.run(cache, out, slice_batch_dir=out)   # Phase 4
segment_primitives.run(out, method="union")          # Phase 5
segment_score.run(cache, out, method="union",        # Phase 6
                  slice_nfs=out / "nfs_analysis.json")
```

---

## 输出目录结构

```
batch_results_rl/
├── base/                               # Qwen3-4B-Base (28.6% acc)
│   ├── dist_p{pid}_s{sid}.npy         # slice 级距离矩阵
│   ├── hmm_p{pid}_s{sid}.npy          # HMM 状态向量
│   ├── batch_summary.json             # Phase 1 汇总
│   ├── primitives_analysis.json       # Phase 2 结构原语
│   ├── nfs_analysis.json              # Phase 3 NFS 打分
│   ├── seg_dist_union_p{pid}_s{sid}.npy  # segment 距离 (union)
│   ├── seg_dist_avg_p{pid}_s{sid}.npy    # segment 距离 (avg)
│   ├── seg_meta_p{pid}_s{sid}.json       # segment 元信息
│   ├── segment_summary.json           # Phase 4 汇总
│   ├── primitives_segment_union.json  # Phase 5 segment 原语
│   ├── primitives_segment_avg.json
│   ├── nfs_segment_union.json         # Phase 6 segment NFS
│   └── nfs_segment_avg.json
├── step-100/ ~ step-1000/              # 同上结构
└── cross_checkpoint/
    └── nfs_trajectory.json            # Phase 7: 11 checkpoints 指标汇总
```

---

## 各 Phase 耗时参考 (base checkpoint)

| Phase | 描述 | 耗时 | 输出大小 |
|-------|------|------|---------|
| 1 | Slice graph construction | ~11 min | 6376 npy pairs |
| 2 | Slice primitives | ~15s | 475 MB json |
| 3 | Slice NFS scoring | <1s | 2.2 MB json |
| 4 | Segment graph construction | ~81s | 11972 npy + 5986 json |
| 5 | Segment primitives (x2) | ~15s | 10.6 MB json |
| 6 | Segment NFS (x2) | <1s | 4.0 MB json |
| **Total** | | **~14 min** | **~1.5 GB** |

Phase 1 是绝对瓶颈（O(n^2) Jaccard 距离），后续 RL checkpoint 的 CoT 更长，单 checkpoint 可达 20-25 min。

---

## Base Checkpoint 指标基线

| 指标 | Slice | Segment-Union | Segment-Avg |
|------|-------|---------------|-------------|
| NFS mean | 19.93 | 3.12 | 2.91 |
| AUROC | **0.554** | 0.456 | 0.462 |
| AUPRC | 0.323 | 0.268 | 0.279 |
| Hit@1 | 0.260 | **0.320** | 0.300 |
| Pairwise | 0.495 | 0.494 | 0.495 |
| Majority Vote | 0.420 | 0.420 | 0.420 |
| Weighted Vote | **0.390** | 0.340 | 0.320 |

**NFS 组分 (slice):** B=0.234, H=0.048, D*=0.086

**观察:**
- Slice AUROC=0.554 > 0.5, base 模型已有微弱区分信号
- Segment 粒度过粗（median=3 segments），区分力不足
- 正确样本更短（mean 22 vs 71 slices），base 模型正确时倾向简短回答

---

## nfs_trajectory.json 格式

```json
{
  "checkpoints": [
    {
      "name": "base",
      "rl_step": 0,
      "accuracy": 28.61,
      "slice": {
        "nfs_mean": 19.93,
        "auroc": 0.554,
        "auprc": 0.323,
        "hit_at_1": 0.260,
        "B_mean": 0.234,
        "H_mean": 0.048,
        "D_star_mean": 0.086,
        ...
      },
      "segment_union": { ... },
      "segment_avg": { ... }
    },
    { "name": "step-100", "rl_step": 100, ... },
    ...
  ]
}
```

---

## 注意事项

- **Problem ID**: 此数据集混合 32 字符 hex UUID 和 `deepscaler_*` 前缀，代码已全面兼容 str 类型
- **Accuracy 趋势**: base 28.6% → step-600 峰值 33.2% → step-1000 回落 32.0%
- **磁盘**: 全量 11 checkpoints 预计 ~15 GB（含距离矩阵），`--no-save-matrices` 可降至 ~3 GB
- **segment_summary.json**: segment 级汇总使用 `segment_summary.json`（非 `batch_summary.json`），避免与 slice 级冲突
- **直接调用**: 所有 nfs_pipeline 模块均暴露 `run()` 函数，支持 Python 编程调用，无需 subprocess
