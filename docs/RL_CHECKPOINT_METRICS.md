# RL Checkpoint Ranking — Label-Free 打分方法相关性分析

Ground truth best checkpoint: **step-600** (accuracy = 33.19%)

本分析仅包含 **label-free** 方法，即不使用任何 `is_correct` 标签、纯粹基于模型输出特征的聚合统计量。

---

## 什么是 Label-Free Checkpoint 打分？

每个 RL checkpoint 下有 ~6400 个 sample，每个 sample 有 NFS 分数及其子分量（B, H, D\*），还有 confidence、entropy 等特征。
**Label-free 方法不知道哪些 sample 做对了**，只能对所有 sample 的特征做统计聚合（均值、标准差等），得到一个 checkpoint 级别的分数。

目标：这个分数的排序与真实 accuracy 排序尽量一致（Spearman ρ 越高越好）。

### 三种粒度

| 粒度 | 含义 |
|------|------|
| **slice** | 最细粒度：每个 token slice 独立算 NFS，再对所有 sample 取均值 |
| **seg_union** | 每个 sample 整条 CoT 合并为一个 segment 算 NFS |
| **seg_avg** | 每个 sample 的多个 segment 分别算 NFS 再平均 |

### NFS 子分量

| 分量 | 含义 | 训练趋势 |
|------|------|---------|
| **B** (Backbone) | 基础得分 | 随训练递减（模型越强 NFS 基础分越低） |
| **H** (Hydrogen) | 信息熵分量 | 随训练递减（模型越自信 entropy 越低） |
| **D\*** (Drift) | 漂移/不稳定性 | 随训练递减（推理越稳定 drift 越小） |
| **NFS** = B + H + D\* | 综合分 | slice 粒度随训练递减；segment 粒度随训练递增（方向翻转） |

### 其他特征

| 特征 | 含义 |
|------|------|
| **Mean_Confidence** | 所有 sample 的平均 token confidence（baseline 方法） |
| **L** | 平均生成长度（token 数） |
| **R** | 平均正确/错误 sample 比例的 proxy |
| **beta_nfs** | NFS ~ length 线性回归斜率（NFS 对长度的依赖程度） |
| **NFS_cv** | NFS 变异系数 = std / |mean|（打分的离散程度） |
| **B/(B+D\*)** | Backbone 在 B+D\* 中的占比 |
| **H/B** | Entropy 与 Backbone 的比值 |

> 注：带 `-` 前缀的表示取负（翻转方向），因为原始值与 accuracy 负相关。

---

## 完整结果（按 Spearman ρ 降序）

| Rank | Method | Spearman ρ | Pearson r | Kendall τ | Top-1 | Predicted Best |
|------|--------|-----------|-----------|-----------|-------|----------------|
| 1 | -beta_nfs (dynamics) | +0.8273 | +0.8868 | +0.6727 |  | step-900 |
| 2 | -H_mean (slice) | +0.8091 | +0.7598 | +0.6727 |  | step-700 |
| 3 | -B_mean (slice) | +0.8000 | +0.8867 | +0.6727 |  | step-900 |
| 4 | H/B (slice) | +0.8000 | +0.8717 | +0.6727 |  | step-900 |
| 5 | -NFS_mean (slice) | +0.7909 | +0.8211 | +0.6364 |  | step-900 |
| 6 | L (dynamics) | +0.7818 | +0.8815 | +0.6000 |  | step-900 |
| 7 | nfs_mean (seg_union) | +0.7636 | +0.8936 | +0.6364 |  | step-900 |
| 8 | H_mean (seg_union) | +0.7636 | +0.8722 | +0.6364 |  | step-900 |
| 9 | D_star_mean (seg_union) | +0.7636 | +0.8656 | +0.6364 |  | step-900 |
| 10 | D_star_mean (seg_avg) | +0.7636 | +0.8287 | +0.6364 |  | step-900 |
| 11 | -B_mean (seg_union) | +0.7636 | +0.8791 | +0.6364 |  | step-900 |
| 12 | -B_mean (seg_avg) | +0.7636 | +0.8861 | +0.6364 |  | step-900 |
| 13 | H/B (seg_union) | +0.7636 | +0.7513 | +0.6364 |  | step-900 |
| 14 | 1-D* (slice) | +0.7545 | +0.8854 | +0.6000 |  | step-900 |
| 15 | -D* (slice) | +0.7545 | +0.8854 | +0.6000 |  | step-900 |
| 16 | nfs_mean (seg_avg) | +0.7455 | +0.8953 | +0.6000 |  | step-1000 |
| 17 | H/B (seg_avg) | +0.7455 | +0.7234 | +0.6000 |  | step-1000 |
| 18 | Mean_Confidence (baseline) | +0.7364 | +0.8577 | +0.6000 |  | step-900 |
| 19 | H_mean (seg_avg) | +0.7182 | +0.8341 | +0.5636 |  | step-1000 |
| 20 | R (dynamics) | +0.6727 | +0.6426 | +0.4909 |  | step-900 |
| 21 | B/(B+D*) (slice) | +0.2545 | +0.5994 | +0.1636 |  | step-800 |
| 22 | nfs_std (seg_union) | -0.3091 | -0.1042 | -0.1636 |  | step-300 |
| 23 | nfs_std (seg_avg) | -0.5091 | -0.4493 | -0.3455 |  | step-300 |
| 24 | NFS_cv (slice) | -0.5273 | -0.6874 | -0.2727 |  | base |
| 25 | -H_mean (seg_avg) | -0.7182 | -0.8341 | -0.5636 |  | base |
| 26 | -Mean_Confidence | -0.7364 | -0.8577 | -0.6000 |  | base |
| 27 | nfs_std (slice) | -0.7455 | -0.8429 | -0.6000 |  | base |
| 28 | -NFS_mean (seg_avg) | -0.7455 | -0.8953 | -0.6000 |  | base |
| 29 | D_star_mean (slice) | -0.7545 | -0.8854 | -0.6000 |  | base |
| 30 | B_mean (seg_union) | -0.7636 | -0.8791 | -0.6364 |  | base |
| 31 | B_mean (seg_avg) | -0.7636 | -0.8861 | -0.6364 |  | base |
| 32 | -NFS_mean (seg_union) | -0.7636 | -0.8936 | -0.6364 |  | base |
| 33 | -H_mean (seg_union) | -0.7636 | -0.8722 | -0.6364 |  | base |
| 34 | 1-D* (seg_union) | -0.7636 | -0.8656 | -0.6364 |  | base |
| 35 | -D* (seg_union) | -0.7636 | -0.8656 | -0.6364 |  | base |
| 36 | 1-D* (seg_avg) | -0.7636 | -0.8287 | -0.6364 |  | base |
| 37 | -D* (seg_avg) | -0.7636 | -0.8287 | -0.6364 |  | base |
| 38 | B/(B+D*) (seg_union) | -0.7636 | -0.8343 | -0.6364 |  | base |
| 39 | NFS_cv (seg_union) | -0.7636 | -0.9355 | -0.6364 |  | base |
| 40 | B/(B+D*) (seg_avg) | -0.7636 | -0.8217 | -0.6364 |  | base |
| 41 | NFS_cv (seg_avg) | -0.7636 | -0.9312 | -0.6364 |  | base |
| 42 | nfs_mean (slice) | -0.7909 | -0.8211 | -0.6364 |  | step-100 |
| 43 | B_mean (slice) | -0.8000 | -0.8867 | -0.6727 |  | base |
| 44 | H_mean (slice) | -0.8091 | -0.7598 | -0.6727 |  | base |
| 45 | beta_nfs (dynamics) | -0.8273 | -0.8868 | -0.6727 |  | base |

---

## 关键发现

### Top 5 Label-Free 方法

1. **-beta_nfs (dynamics)** — ρ = +0.8273, r = +0.8868
2. **-H_mean (slice)** — ρ = +0.8091, r = +0.7598
3. **-B_mean (slice)** — ρ = +0.8000, r = +0.8867
4. **H/B (slice)** — ρ = +0.8000, r = +0.8717
5. **-NFS_mean (slice)** — ρ = +0.7909, r = +0.8211

### 能正确预测 Top-1 (step-600) 的方法

无 label-free 方法能正确预测 Top-1。

### 粒度对比：同一特征在不同粒度下的表现

| 特征 | slice | seg_union | seg_avg |
|------|-------|-----------|---------|
| NFS_mean (正) | +0.7909 | +0.7636 | +0.7455 |
| B_mean (负) | +0.8000 | +0.7636 | +0.7636 |
| H_mean (负) | +0.8091 | -0.7636 | -0.7182 |
| D* (负) | +0.7545 | -0.7636 | -0.7636 |

**注意 NFS_mean 的方向翻转**：slice 粒度下 NFS_mean 与 accuracy 负相关（ρ = -0.7909），
需要取负才有正相关；而 seg_union/seg_avg 粒度下是正相关（ρ = 0.7636 / 0.7455），不需要翻转。

### vs Baseline (Mean Confidence, ρ = +0.7364)

共 **17** 个 label-free 方法超过 baseline：

- -beta_nfs (dynamics) (ρ = +0.8273)
- -H_mean (slice) (ρ = +0.8091)
- -B_mean (slice) (ρ = +0.8000)
- H/B (slice) (ρ = +0.8000)
- -NFS_mean (slice) (ρ = +0.7909)
- L (dynamics) (ρ = +0.7818)
- nfs_mean (seg_union) (ρ = +0.7636)
- H_mean (seg_union) (ρ = +0.7636)
- D_star_mean (seg_union) (ρ = +0.7636)
- D_star_mean (seg_avg) (ρ = +0.7636)
- -B_mean (seg_union) (ρ = +0.7636)
- -B_mean (seg_avg) (ρ = +0.7636)
- H/B (seg_union) (ρ = +0.7636)
- 1-D* (slice) (ρ = +0.7545)
- -D* (slice) (ρ = +0.7545)
- nfs_mean (seg_avg) (ρ = +0.7455)
- H/B (seg_avg) (ρ = +0.7455)

---

## 结论

1. **最优 label-free 方法**是取 NFS 子分量的 checkpoint 均值，尤其是 `-H_mean (slice)` 和 `-NFS_mean (slice)`，ρ 约 0.79–0.81。
2. **Segment 粒度的 NFS 均值**（seg_union/seg_avg）直接正相关，ρ ≈ 0.74–0.76，与 baseline Mean_Confidence 相当。
3. **Slice 粒度 NFS 方向翻转**：slice 下 NFS_mean 随训练递减（B, H, D\* 都在缩小），需要取负。Segment 下 NFS_mean 随训练递增（segment-level aggregation 改变了趋势）。
4. **没有 label-free 方法能正确预测 Top-1 (step-600)**，大多预测 step-700 或 step-900。这是因为 step-600→700→800 accuracy 变化很小（33.19→32.99→32.11），label-free 统计量难以区分这种微小的非单调波动。
5. **对比 label-dependent 方法**：使用标签的 AUPRC (seg_union) 能达到 ρ = 0.89 且 Top-1 正确，说明标签信息对精确排序至关重要。

