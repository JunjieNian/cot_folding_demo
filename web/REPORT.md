# COT Folding Map 技术文档

> 本文档完整说明了 COT Folding Map 网站的每一个可视化、指标、分数的计算方式与含义，以及数据存储架构和已知性能问题。

---

## 一、项目概述

COT Folding Map 是一个交互式可视化面板，用于分析大语言模型在数学推理（AIME24 竞赛题）中的 Chain-of-Thought 推理轨迹。

核心思想：将模型的推理过程类比为蛋白质折叠——推理序列中的 token 被分割成 **slices**（每 32 个 token 为一个 slice），通过距离矩阵、MDS 降维、HMM 状态分类等方法，分析推理的结构特征。

---

## 二、数据存储架构

### 2.1 文件结构

```
public/data/aime24/
├── app.json                          # 应用配置（数据集名、版本号）
├── overview.json                     # 批次级别统计（全部题目的聚合指标）
├── problems.index.json               # 题目索引（30 道题，每题 64 个 sample）
├── compare/p{pid}.json              # 每道题的结构对比数据（正确 vs 错误）
└── samples/p{pid}/
    ├── s{sid}.bundle.json           # 核心数据包（folding + clustering + flow + functional）
    ├── s{sid}.text.json             # 完整推理文本 + slice 的字符边界
    └── s{sid}.sim.b64              # 相似度矩阵（base64 编码的上三角压缩）
```

### 2.2 数据来源管线

```
FoldingEngine（Python 后端，基于 nfs_pipeline 模块）
    │
    │  export_static_aime24.py
    │  （读取 FoldingEngine 的内存对象，导出为静态 JSON）
    ▼
public/data/aime24/  （静态文件）
    │
    │  split_similarity.py
    │  （将 bundle 中的 similarity_b64 拆分为独立的 .sim.b64 文件）
    ▼
deploy_oss.py --gzip  （预压缩 → .gzip_cache/）
deploy_oss.py --upload （上传到阿里云 OSS + CDN）
    │
    ▼
前端 React 应用通过 HTTP 请求加载 JSON
```

### 2.3 数据包格式详解

#### `s{sid}.bundle.json` 顶层结构

```json
{
  "problem_id": 60,
  "sample_id": 0,
  "folding": { ... },       // 折叠分析（MDS 坐标、指标、相位等）
  "clustering": { ... },     // 聚类统计（explore/exploit 的分离度）
  "flow": { ... },           // 信息流分析（动脉/静脉/毛细/旁路）
  "functional": { ... }      // 功能分解（核心区、漂移区、回流边等）
}
```

#### 相似度矩阵的存储方式

相似度矩阵是对称的 n×n 矩阵，对角线为 1。为节省空间，只存储**上三角部分**，共 n(n-1)/2 个值。

编码方式：
1. 后端将上三角的 float 值量化为 uint8（乘以 255 取整）
2. 将 uint8 字节序列进行 base64 编码
3. 写入 `.sim.b64` 文件

前端解码方式：
```javascript
function decodeSimilarityB64(b64, n) {
  const bin = atob(b64);          // base64 → 原始字节
  const arr = new Float32Array(n * n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    arr[i * n + i] = 1;           // 对角线 = 1（自相似）
    for (let j = i + 1; j < n; j++, k++) {
      const v = bin.charCodeAt(k) / 255;   // uint8 → [0,1]
      arr[i * n + j] = v;         // 上三角
      arr[j * n + i] = v;         // 对称填充下三角
    }
  }
  return arr;   // 完整 n×n 矩阵
}
```

精度：约 1/255 ≈ 0.004。

#### 文本数据的存储方式

`s{sid}.text.json` 存储完整推理文本和每个 slice 的字符边界：

```json
{
  "full_text": "<think>Let me solve...",
  "items": [
    { "slice_idx": 0, "token_start": 0, "token_end": 32, "char_start": 0, "char_end": 150 },
    { "slice_idx": 1, "token_start": 32, "token_end": 64, "char_start": 150, "char_end": 298 },
    ...
  ]
}
```

前端通过 `char_start/char_end` 从 `full_text` 中切出对应文本，实现点击 slice → 高亮原文。

---

## 三、核心概念

### 3.1 Slice（切片）

推理文本被按 **每 32 个 token** 分割为 slice。每个 slice 是分析的最小单元。

例如：一个 925 token 的推理轨迹 → 29 个 slice（最后一个 slice 可能不足 32 token）。

### 3.2 距离矩阵与相似度矩阵

- **距离矩阵** D(i,j)：由外部计算（NAD 模块），衡量 slice i 和 slice j 在表示空间中的距离
- **相似度矩阵**：`similarity[i,j] = 1 - distance[i,j]`，值域 [0,1]
  - 1 = 完全相似
  - 0 = 完全不同

### 3.3 HMM 状态（Explore / Exploit）

每个 slice 被分类为两种状态之一：

| 状态 | 编码 | 含义 | 颜色 |
|------|------|------|------|
| **Explore**（探索） | 0 | 模型在尝试新的解题方向 | 蓝色 #5B8DEF |
| **Exploit**（利用） | 1 | 模型在沿已有方向深入推进 | 红色 #E05A47 |

状态由外部 HMM 模型从 token 级别的熵和置信度序列中推断得到。

### 3.4 Phase（相位）

Phase 是 **相邻且状态相同的 slice 组成的最大连续块**。

例如：HMM 状态序列 `[0,0,0,1,1,0,0,1]` → 4 个 phase：
- Phase 0: slice 0-2, explore
- Phase 1: slice 3-4, exploit
- Phase 2: slice 5-6, explore
- Phase 3: slice 7, exploit

### 3.5 Contact（接触）

两个 slice (i,j) 的相似度超过阈值时，称为一个 **contact**（接触对）。

接触阈值：`τ = μ + σ`（上三角相似度的均值 + 标准差）。

---

## 四、每个指标的计算方式与含义

### 4.1 接触类指标

#### Contact Threshold（接触阈值）

```
τ = mean(sim_upper_triangle) + std(sim_upper_triangle)
```

含义：均值 + 1 倍标准差，只有相似度显著高于平均水平的 slice 对才被视为"接触"。

#### Total Contacts（总接触数）

```
total_contacts = count{ (i,j) : i < j, similarity[i,j] > τ }
```

含义：所有超过阈值的 slice 对数量。

#### Long-Range Contacts（长程接触数）

```
long_range_contacts = count{ (i,j) : i < j, |j - i| > n/4, similarity[i,j] > τ }
```

含义：序列距离超过总长度 1/4 的接触对数量。长程接触说明推理中"远处"的内容在结构上高度相关。

#### Folding Degree（折叠度）

```
folding_degree = long_range_contacts / total_contacts
```

- 值域：[0, 1]
- 含义：长程接触占总接触的比例。越高说明推理的"折叠"越紧密——远处的推理片段彼此关联。
- 类比：蛋白质中高折叠度意味着三级结构更紧凑。

#### Contact Order（接触序）

```
contact_order = Σ(gap_ij × sim_ij) / (Σ(sim_ij) × n)
```

其中求和遍历所有接触对 (i,j)，`gap_ij = |j - i|`。

- 值域：[0, ~0.5]
- 含义：接触对的加权平均序列距离，归一化后表示折叠的"跨度"。值越大，说明远距离的接触越强。

### 4.2 MDS 类指标

#### MDS 坐标（经典 MDS / Torgerson 方法）

将 n×n 距离矩阵降维到 2D/3D，使低维空间中的距离尽可能保持原始距离关系。

算法：
```
1. D_sq = D ⊙ D              （逐元素平方）
2. H = I - (1/n)·11ᵀ          （中心化矩阵）
3. B = -0.5 · H · D_sq · H    （双重中心化）
4. 特征分解 B = QΛQᵀ
5. 取前 k 个最大特征值 λ₁,...,λₖ 及其特征向量
6. 坐标 X = Q · diag(√λ₁,...,√λₖ)
```

结果：每个 slice 在 2D/3D 空间中有一个坐标点，空间中的距离近似反映原始的语义距离。

#### MDS Stress（Kruskal Stress-1）

```
stress = √( Σ(d_orig - d_mds)² / Σ(d_orig²) )
```

其中 `d_orig` 是原始距离，`d_mds` 是 MDS 空间中的距离，求和遍历所有 n(n-1)/2 个 pair。

- 值域：[0, 1+]
- stress < 0.1：优秀（2D 完美还原距离结构）
- stress 0.1-0.2：良好
- stress > 0.3：较差（高维结构无法用 2D 充分表达）
- 含义：MDS 嵌入的质量。值越低，2D 图越能准确反映实际的推理结构。

#### Radius of Gyration（回转半径）

```
centroid = mean(coords)
R_g = √( mean( ||coords_i - centroid||² ) )
```

- 含义：MDS 空间中所有 slice 到质心的 RMS 距离，衡量结构的"紧凑程度"。
- 小 → 推理集中，主题统一
- 大 → 推理分散，方向多样

### 4.3 Per-Slice 指标

#### Entropy（熵）

```
对每个 slice（32 个 token）:
  entropy = -mean(tok_neg_entropy) + 0.1 × std(tok_neg_entropy)
```

`tok_neg_entropy` 是 token 级别的负熵值（来自 NAD 缓存/激活分析）。

- 含义：该 slice 输出的不确定性/变异性。
- 高熵 → 模型在该处犹豫不决，可能在探索多种方向
- 低熵 → 模型输出稳定，方向明确

#### Confidence（置信度）

```
对每个 slice:
  confidence = mean(tok_conf)
```

`tok_conf` 是 token 级别的置信度分数。

- 含义：模型对自身输出的自信程度。
- 高置信度 → 模型确信当前方向正确
- 低置信度 → 模型不确定

### 4.4 NFS 原语、标签与评分——完整管线

NFS（Neural Folding Score）分析的核心思路是：先从折叠结构中提取三大**结构原语**，再由原语派生出每个 slice 的**有效性标签**和**有效性分数**，最终将三大原语各自量化为一个分量，几何平均得到 NFS 总分。

完整管线如下图：

```
          距离矩阵 + HMM 状态
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
 ① Core      ② Return    ③ Drift        ← 三大结构原语提取
 (贪心合并     Edges      Branches          (§4.4.1)
  exploit 块)  (长程高      (explore 块
               相似 pair)    vs 核心区)
     │           │           │
     │     Final Closure ────┘            ← 辅助原语 (§4.4.1d)
     │           │
     ▼           ▼
  标签分配（优先级覆盖）                   ← §4.4.2
  core > closure > return_site > drift
  > productive_explore/exploit
  > explore/exploit
     │
     ▼
  有效性评分（per-slice 0-1）              ← §4.4.3
     │
     ▼
  NFS 四分量 (B, H, D₀, G)               ← §4.4.4
     │
     ▼
  NFS = 100 × (B × H × (1-D*))^(1/3)    ← §4.4.5
```

#### 4.4.1 三大结构原语提取

> 源码位置：`nfs_pipeline/primitives.py`

三大原语是从折叠结构中机械化提取的 **slice 集合**，无需人工标注。

##### a. Core（核心区） —— 解题骨干

**物理直觉**：蛋白质折叠中的"疏水核心"——结构最紧凑、内部凝聚力最强的区域。

**算法**：贪心合并
1. 找到所有 exploit 连续块
2. 以最长的 exploit 块为种子
3. 逐步合并与种子平均相似度 > τ 的其他 exploit 块
4. 直到没有新块可以合并

**输出**：
- `core.indices[]`：属于核心区的 slice 索引集合
- `internal_similarity`：核心区内部的平均相似度（衡量核心的"密度"）
- `fraction_of_exploit`：核心区占所有 exploit slice 的比例（衡量核心的"覆盖率"）

**→ 对应标签**：核心区内的 slice 被标记为 **`core`**（绿色）

**→ 对应 NFS 分量**：**B（Backbone，骨架分）**
```
B = internal_similarity × fraction_of_exploit
```

##### b. Return Edges（回流边） —— 长程结构连接

**物理直觉**：蛋白质中的"氢键"——在一维序列上相距很远的两个位点，在三维空间中折叠到一起形成的非共价键。

**提取条件**：slice 对 (i, j) 满足：
- 序列距离 |j - i| > n/4（长程）
- 相似度 > τ（显著相似）
- 至少一端是 exploit 状态

按相似度降序取 top 50 条。

**每条边的属性**：
- `i, j`：两端 slice 索引
- `gap`：序列距离
- `similarity`：相似度值
- `type`：catalytic（一端 explore 一端 exploit）或 structural（两端同状态）

**→ 对应标签**：回流边的两端 slice 被标记为 **`return_site`**（橙色）

**→ 对应 NFS 分量**：**H（Hydrogen，回流分）**
```
对每条回流边 e:
  r_e = ((sim_e - τ) / (1 - τ)) × (gap_e / (n - 1))

H = mean(r_e)
```
每条边的贡献 = 超出阈值的相似度强度 × 归一化序列跨度。H 越大，说明推理轨迹形成了越强的长程结构折叠。

##### c. Drift Branches（漂移分支） —— 无效探索

**物理直觉**：蛋白质折叠过程中的"错误折叠中间体"——走入了死胡同，无法回归正确构象。

**算法**：对每个 explore 连续块：
- 计算与核心区的最大相似度 `max_sim_to_core`
- 若 `max_sim_to_core ≤ τ` → 标记为 drift（与核心区完全脱节）
- 按时间位置标记：early / middle / late

**输出**：
- `drift_branches[]`：每个漂移分支的 `{start, end, length, is_drift, max_sim_to_core, position}`

**→ 对应标签**：漂移分支内的 slice 被标记为 **`drift`**（红色）

**→ 对应 NFS 分量**：**D₀（基础漂移分）**
```
对每个漂移分支 b:
  t_b = (start_b + end_b) / (2n)      （归一化时间位置，0~1）
  w_b = 1.0 + t_b                      （时间权重，越晚越重，1~2）
  r_b = min(1, max_sim_to_core / τ)    （回归核心的程度，0~1）
  贡献 = w_b × length_b × (1 - r_b)

D₀ = Σ贡献 / Σ(w_b × length_b)
```
越晚出现的漂移惩罚越重（因为在推理尾部漂移意味着"快到终点时走错路"）。

##### d. Final Closure（最终收束） —— 辅助原语

**物理直觉**：蛋白质折叠的"最终构象锁定"——最后一段 exploit 是否成功回归核心区。

```
s_close = 最后 exploit 块与核心区的最大相似度
closure_coefficient = min(1, s_close / τ)
```

**→ 对应标签**：最后一个 exploit 块内的 slice 被标记为 **`closure`**（浅绿色）

**→ 对应 NFS 分量**：**G（Closure Gate，收束门控）**
```
G = (1 + closure_coefficient) / 2      ∈ [0.5, 1.0]
```
G 调节漂移惩罚——好的收束可以部分弥补之前的漂移：
```
D* = 1 - G × (1 - D₀)
```

#### 4.4.2 有效性标签分配

> 源码位置：`nfs_pipeline/primitives.py (label assignment logic)` L415-433

标签由三大原语的集合成员关系决定，按**优先级从高到低**逐一覆盖：

```python
for i in range(n):
    if i in drift_set:                              → "drift"
    elif i in core_indices:                         → "core"
    elif i in closure_set:                          → "closure"
    elif i in return_endpoints:                     → "return_site"
    elif hmm_states[i] == 0 and scores[i] > 0.4:   → "productive_explore"
    elif hmm_states[i] == 1 and scores[i] > 0.5:   → "productive_exploit"
    elif hmm_states[i] == 0:                        → "explore"
    else:                                           → "exploit"
```

**标签与原语的对应关系一览**：

| 标签 | 来源原语 | 判定条件 | 可视化颜色 | 含义 |
|------|----------|----------|------------|------|
| **core** | Core 提取 | slice ∈ core.indices | 绿 `rgba(76,175,80,0.25)` | 解题骨干：exploit 中与种子块高度一致的片段 |
| **closure** | Final Closure | slice ∈ 最后 exploit 块 | 浅绿 `rgba(76,175,80,0.20)` | 最终收束区，锁定结论 |
| **return_site** | Return Edges | slice 是某条回流边的端点 | 橙 `rgba(255,152,0,0.18)` | 长程折叠的连接枢纽 |
| **drift** | Drift Branches | slice ∈ is_drift 的 explore 块 | 红 `rgba(244,67,54,0.25)` | 与核心脱节的无效探索 |
| **productive_explore** | 无直接原语 | explore 且 score > 0.4 | 浅灰 `rgba(158,158,158,0.12)` | 虽在探索但有贡献 |
| **productive_exploit** | 无直接原语 | exploit 且 score > 0.5 | 浅灰 `rgba(158,158,158,0.12)` | 虽非核心但有贡献 |
| **explore** | HMM 默认 | explore 状态兜底 | 极浅灰 `rgba(158,158,158,0.08)` | 普通探索 |
| **exploit** | HMM 默认 | exploit 状态兜底 | 极浅灰 `rgba(158,158,158,0.08)` | 普通利用 |

注意：前四个标签直接对应三大原语 + 辅助原语的集合划分；后四个是原语"剩余区域"的兜底分类（依据 HMM 状态 + 有效性分数阈值）。

#### 4.4.3 有效性评分（Per-Slice, 0-1）

> 源码位置：`nfs_pipeline/primitives.py` (effectiveness scoring logic)

每个 slice 的有效性分数由其所属原语决定基础分，再根据与核心区/回流边的关联度上浮：

| 角色 | 分数公式 | 基础分 | 上浮因素 |
|------|----------|--------|----------|
| **core** | `0.7 + 0.3 × core_sim_norm` | 0.7 | 与核心区越相似分越高 |
| **closure** | `0.6 + 0.4 × closure_coeff` | 0.6 | 收束质量越好分越高 |
| **return_site** | `0.5 + 0.3 × return_norm + 0.2 × core_sim_norm` | 0.5 | 回流强度 + 核心相似度 |
| **drift** | `0.1 + 0.15 × core_sim_norm` | 0.1 | 天花板很低，即使与核心略有关联也不超 0.25 |
| **其他** | `0.2 + 0.5 × core_sim_norm + 0.3 × return_norm` | 0.2 | 与核心/回流的关联 |

其中：
- `core_sim_norm[i]`：slice i 与核心区所有 slice 的平均相似度（归一化到 [0,1]）
- `return_norm[i]`：slice i 作为回流边端点的回流强度（归一化）
- `closure_coeff`：Final Closure 的 closure_coefficient

**Productive 标签的判定阈值**基于此分数：`productive_explore` 要求 score > 0.4，`productive_exploit` 要求 score > 0.5。

**Productive Fraction（产出率）**：
```
productive_fraction = count(label ∈ {core, closure, productive_explore, productive_exploit, return_site}) / n
```
含义：推理序列中"有效推理"占比，越高说明推理效率越好。

**Circling Regions（原地打转区域）**：连续多个 drift 标签的 slice 构成 circling region，表示模型在该处反复尝试但没有实质进展。

#### 4.4.4 NFS 四分量汇总

三大原语 + 辅助原语各自量化为一个 [0,1] 范围的分量：

| 分量 | 全称 | 来源原语 | 公式 | 物理类比 | 含义 |
|------|------|----------|------|----------|------|
| **B** | Backbone | Core | `s_core × f_core` | 疏水核心密度 | 核心推理骨架的密度 × 覆盖率 |
| **H** | Hydrogen | Return Edges | `mean(r_e)` | 氢键强度 | 长程回流连接的平均强度 |
| **D₀** | Drift | Drift Branches | 时间加权漂移比 | 错误折叠比例 | 未解决漂移的严重程度 |
| **G** | Gate | Final Closure | `(1+C)/2` | 最终构象锁定 | 收束质量，调节 D₀ → D* |

```
D* = 1 - G × (1 - D₀)     （门控后漂移：好收束可弥补漂移）
```

#### 4.4.5 NFS 总公式

```
NFS = 100 × (B × H × (1 - D*))^(1/3)
```

三个因子的几何平均意味着**任何一个维度为零都会使 NFS 归零**——推理必须同时满足"核心强、回流丰富、漂移少"才能获得高分。

**NFS 解读**：
- NFS 高 → B 大（核心密且广）、H 大（回流边多且强）、D* 小（漂移少或收束好）
- NFS 低 → 三个因子中至少一个薄弱
- 典型范围：0-20

**典型案例**（P70）：
| 样本 | 正误 | NFS | B | H | D* | 语义特征 |
|------|------|-----|---|---|-----|----------|
| S693 | 正确 | 18.22 | 0.18 | 0.03 | 0.00 | 96 个 core slice 构成紧密骨架，50 条 return edges，仅 1 个 drift slice |
| S684 | 错误 | 5.99 | 0.05 | 0.00 | 0.32 | 核心稀疏，几乎无回流，大量漂移且收束差 |

### 4.6 聚类统计

#### Separation（分离度）

```
within_mean = mean(所有 explore-explore 对和 exploit-exploit 对的距离)
cross_mean = mean(所有 explore-exploit 混合对的距离)

separation = cross_mean - within_mean
```

- 正值 → 同类 slice 之间距离小于异类 → 聚类分离良好
- 负值 → 同类与异类无明显区分
- 含义：explore 和 exploit 两种状态在距离空间中的区分度。分离度越高，说明模型的两种推理模式差异越显著。

#### Cohen's d（效应量）

```
pooled_std = √( ((n₁-1)var₁ + (n₂-1)var₂) / (n₁+n₂-2) )
d = (mean_within - mean_cross) / pooled_std
```

| |d| 范围 | 效应大小 |
|---------|----------|
| < 0.2 | 可忽略 |
| 0.2-0.5 | 小效应 |
| 0.5-0.8 | 中等效应 |
| ≥ 0.8 | 大效应 |

#### p-value

使用 Mann-Whitney U 检验，判断 within 和 cross 距离分布是否有显著差异。p < 0.05 为显著，p < 0.001 为高度显著。

### 4.7 Flow（信息流）指标

将推理过程中相邻 slice 间的变化类比为"血液循环"。

#### 流类型分类

对每对相邻 slice (i → i+1)：
```
d_ent = entropy[i+1] - entropy[i]      （熵变化）
d_conf = confidence[i+1] - confidence[i] （置信度变化）

归一化：
d_ent_norm = d_ent / std(d_entropy)
d_conf_norm = d_conf / std(d_confidence)
```

| 类型 | 条件 | 含义 | 颜色 |
|------|------|------|------|
| **Capillary**（毛细） | \|d_ent\| < 0.3 且 \|d_conf\| < 0.3 | 平稳推进 | 黄 #FFD54F |
| **Shunt**（旁路） | \|d_ent\| < 0.3 且 d_conf > 0.3 | 未探索就获得信心 | 灰 #9E9E9E |
| **Arterial**（动脉） | d_ent > 0 或 d_conf < -0.3 | 发散/探索 | 橙 #FF8A65 |
| **Venous**（静脉） | d_ent < 0 或 d_conf > 0.3 | 收敛/确认 | 紫 #7E57C2 |

#### Flow Magnitude（流量级）

```
flow_magnitude = √(d_ent_norm² + d_conf_norm²)
```

含义：该步的变化幅度。流量级越大，说明该步发生了较大的推理状态转变。

#### Flux Vectors（通量向量）

对每个 slice i，计算来自远处 slice（gap > 3）的加权拉力方向：
```
weights[i,j] = similarity[i,j] × |j - i|      （仅 |j-i| > 3 的 j）
flux[i] = Σ_j (weights[i,j] × (coords[j] - coords[i])) / Σ_j weights[i,j]
```

幅度上限为 0.15。含义：展示推理结构中的"吸引子"——哪些远处区域在结构上拉动当前 slice。

#### Congestion Count（拥堵数）

连续 3 个以上的高熵 explore slice（且无状态切换）构成一次"拥堵"。含义：模型在 explore 中卡住的次数。

### 4.8 Functional Decomposition（功能分解）

> 功能分解是 §4.4.1 三大原语提取的**上层封装**，同时包含一些额外的衍生指标。原语提取算法的完整说明参见 §4.4.1a-d，此处仅列出衍生指标。

#### 与 NFS 原语的关系

| 功能分解模块 | NFS 原语 | NFS 分量 | 可视化标签 |
|-------------|----------|----------|-----------|
| Core 提取 | Core（骨干区） | B = s_core × f_core | `core` 绿色 |
| Return Edges | Return Edges（回流边） | H = mean(r_e) | `return_site` 橙色 |
| Drift Branches | Drift（漂移分支） | D₀ → D* | `drift` 红色 |
| Final Closure | Closure（最终收束） | G = (1+C)/2 | `closure` 浅绿色 |

#### Catalytic Fraction（催化比例）

回流边按两端 HMM 状态分为两类：
- **catalytic**（催化型）：一端 explore 一端 exploit → 跨状态连接
- **structural**（结构型）：两端同状态

```
catalytic_fraction = n_catalytic / n_return
```

含义：催化型回流边越多，说明 explore 阶段的探索成果被后续 exploit 实际利用。

#### Functional Score（功能分数）

```
functional_score = catalytic_fraction × closure_coefficient × (1 - drift_fraction)
```

含义：综合衡量推理的"功能质量"——跨状态连接多、收束好、漂移少 → 分数高。注意此分数与 NFS 互补但不相同：NFS 侧重结构密度（B）和长程强度（H），而 Functional Score 侧重跨状态连接的比例。

---

## 五、每个视图的可视化说明

### 5.1 Arc Diagram（弧线图，主视图）

**上部 70%：折叠结构**

- **骨架线**：按 MDS 坐标连接所有 slice 的折线，颜色按 HMM 状态着色（蓝=explore，红=exploit）
- **节点**：每个 slice 一个圆点
  - 大小：按 entropy 归一化值缩放（`4 + 13 × entropy_norm`）
  - 颜色：取决于 colorMode
    - entropy 模式：浅棕→深棕（#FFF7EC → #8C2D04）
    - confidence 模式：浅蓝→深蓝（#EFF3FF → #08519C）
    - effectiveness 模式：红→橙→绿（#D32F2F → #2E7D32）
    - state 模式：蓝/红离散色
- **金色弧线（Bonds）**：长程接触对（gap > n/4 且 sim > τ），取 top 50 条，透明度与相似度成正比
- **起止标记**：绿色 N（起点）、橙色 C（终点）

**下部 30%：注释轨道**

从上到下依次是：
1. **HMM 状态轨道**：蓝/红色条带，每个 slice 一段
2. **Entropy 轨道**：橙色面积图，高度 = 归一化熵值
3. **Confidence 轨道**：蓝色面积图，高度 = 归一化置信度
4. **Flow 类型轨道**（启用时）：按流类型着色的条带
5. **Functional 角色轨道**（启用时）：按功能角色着色的条带
6. **Effectiveness 标签轨道**（启用时）：按有效性标签着色

### 5.2 Contact Map（接触图）

n×n 热力图，显示所有 slice 对之间的相似度。

- 色阶：RdYlBu（蓝=高相似度，红=低相似度）
- 轴边缘的彩色条带：按 HMM 状态着色（蓝=explore，红=exploit）
- 点击任意位置可跳转到对应 slice 的文本

### 5.3 MDS 2D Plot（MDS 散点图）

- 灰色折线连接按序列顺序排列的 slice
- 蓝色/红色点代表 explore/exploit 状态
- 点大小：`6 + 16 × entropy_norm`
- 标题显示 Stress 值

### 5.4 Contact Density（接触密度曲线）

X 轴：序列间距 |j - i|
Y 轴：平均相似度

三条曲线：
- **E-E**（蓝）：explore-explore 对
- **X-X**（红）：exploit-exploit 对
- **Cross**（灰）：混合对

经过滑动窗口平滑（窗口大小 = max(3, n/20)）。

虚线标注 N/4 位置（长程阈值）。

含义：如果某个间距处曲线值高，说明推理在该距离处存在"折回"——远处的内容与当前内容相关。

### 5.5 Metrics Panel（指标面板）

显示本 sample 的所有数值指标（见第四章各指标的定义）。

### 5.6 Phase View（相位视图）

Phase 级别的折叠结构：
- 每个 Phase 是一个大节点，大小 = `10 + √length × 4`
- 连接线颜色按 effectiveness 着色（红→绿）
- 下方注释轨道显示每个 phase 的状态、长度、熵、置信度

### 5.7 Structural Comparison（结构对比）

对比同一道题下正确 vs 错误 sample 的统计差异：

1. **Exploit 时间曲线**：将推理过程分为 10 个等分区间，展示 exploit 占比的变化（含误差带）
2. **长度分布**：正确/错误 sample 的推理长度直方图
3. **NFS 分布**：正确/错误 sample 的 NFS 分数直方图
4. **Effect Size**：各指标的 Cohen's d 值水平条形图
5. **个体 Sample 散点图**：X=长度，Y=最终 exploit 比例
6. **汇总表**：各指标的均值对比

### 5.8 Comparison View（并排对比）

2×2 网格，左侧正确 sample 右侧错误 sample，各展示一个弧线图 + 相位图。

### 5.9 Batch Overview（批次概览）

全部题目的聚合统计：
- 问题数、样本总数、总处理时间
- 聚类分离度和 Cohen's d 的分布直方图
- 正确/错误 sample 的 NFS 分布对比
- 每道题的详情表（样本数、平均/最小/最大 slice 数）

### 5.10 3D 视图（Arc 3D / Phase 3D）

基于 3D MDS 坐标的 Plotly 三维可视化，着色方式与 2D 版相同。

---

## 六、前端数据加载流程

```
1. 应用启动
   → 加载 problems.index.json（一次性，~370KB 明文）
   → 渲染题目列表

2. 选择题目
   → 从已加载的 index 中提取 samples 列表（无额外请求）

3. 选择 sample
   → 加载 s{sid}.bundle.json（核心数据包，~30-250KB）
   → 预取相邻 sample 的 bundle（fire-and-forget）
   → 清空旧的 similarity / text 数据

4. 切换到 Detail 视图
   → 懒加载 s{sid}.sim.b64（相似度矩阵，~19-524KB）
   → 解码为 Float32Array
   → 渲染 ContactMap 热力图

5. 点击 slice
   → 加载 s{sid}.text.json（首次点击时，~30-180KB）
   → 后续点击复用已加载的 text bundle
   → 从 full_text 中切出对应文本并高亮
```

### 缓存机制

- **LRU 内存缓存**：最多 24 个条目，避免重复请求
- **In-flight 去重**：同一 URL 的并发请求共享同一个 Promise，不重复下载
- **HTTP 缓存**：数据文件 max-age=30 天，hashed assets max-age=1 年
- **Prefetch**：加载当前 sample 后，fire-and-forget 预取相邻 sample 的 bundle

---

## 七、已知性能问题与卡顿原因

### 7.1 数据量较大

当前部署的是**完整的 30 道题数据集**（非 lite 版）：

- `problems.index.json`：~370 KB 明文
- 每道题 64 个 sample，共 1920 个 sample
- 大 sample（如 p88）的 bundle 约 250KB、sim.b64 约 524KB

首次加载和切 sample 时，需要下载较大的 JSON 文件。在慢网络下体感明显。

### 7.2 Plotly.js 体积大

Plotly.js 打包后约 **1.7 MB**（gzip 后 ~572 KB），是前端最大的依赖。

- 首次加载需要下载和解析这个大包
- 每次渲染图表（尤其 n=925 的大热力图）有较高的 CPU 开销
- 多个 Plotly 图表同时渲染时（如 Detail 视图的 4 个面板），开销叠加

### 7.3 大矩阵的 JavaScript 处理

对于 n=925 的 sample：
- 相似度矩阵：925 × 925 = 855,625 个浮点数
- base64 解码 + Float32Array 填充：需要处理 427,350 个元素
- ContactMap 重建 2D 数组：925 行 × 925 列的嵌套数组
- 长程接触计算：遍历所有 n(n-1)/2 ≈ 427K 个 pair

这些操作在 JavaScript 主线程执行，会造成短暂的界面卡顿。

### 7.4 FoldingArcDiagram 的重渲染

- `similarity` 从 `null` 变为 `Float32Array` 时，触发 `geo` useMemo 重新计算（包含 O(n²) 的接触筛选循环）
- entropy/confidence 的归一化（`Math.min(...arr)` 等展开操作）未被 memo，每次父组件 re-render 都会执行

### 7.5 CDN 地理因素

OSS 部署在阿里云上海节点。海外用户或非上海地区用户访问时，网络延迟较高，JSON 文件的下载时间会放大上述所有问题。

---

## 八、部署与压缩

### deploy_oss.py 的工作流程

1. `--gzip`：将 HTML/JS/CSS/JSON/SVG/B64 文件预压缩，存入 `.gzip_cache/`
2. `--upload`：上传到阿里云 OSS bucket `cot-folding-demo`
   - 预压缩文件带 `Content-Encoding: gzip` 头
   - 数据文件设置 30 天缓存
   - 带 hash 的 assets 设置 1 年缓存
   - `index.html` 设置 `no-cache`

### 典型压缩效果

| 文件类型 | 明文大小 | gzip 后 | 压缩率 |
|----------|----------|---------|--------|
| bundle.json (p88) | ~250 KB | ~58 KB | 77% |
| sim.b64 (p88) | ~524 KB | ~331 KB | 37% |
| text.json (p88) | ~183 KB | ~33 KB | 82% |
| problems.index.json | ~373 KB | ~40 KB | 89% |
| plotly chunk (JS) | 1,724 KB | 572 KB | 67% |

---

## 九、颜色方案速查

| 元素 | 颜色 | 色值 |
|------|------|------|
| Explore 状态 | 蓝 | #5B8DEF |
| Exploit 状态 | 红 | #E05A47 |
| 长程接触弧线 | 金 | rgba(218,165,32) |
| 核心区边框 | 绿 | #4CAF50 / #2E7D32 |
| 收束区边框 | 橙 | #FF9800 |
| 漂移区边框 | 红 | #F44336 / #D32F2F |
| 回流点边框 | 紫 | #AB47BC |
| 动脉流 (Arterial) | 橙 | #FF8A65 |
| 静脉流 (Venous) | 紫 | #7E57C2 |
| 毛细流 (Capillary) | 黄 | #FFD54F |
| 旁路流 (Shunt) | 灰 | #9E9E9E |

---

## 十、关键文件路径速查

| 文件 | 作用 |
|------|------|
| `src/App.jsx` | 应用主布局，视图切换 |
| `src/hooks/useFoldingState.js` | 核心状态管理 Hook |
| `src/api.js` | API 层，LRU 缓存 + in-flight 去重 |
| `src/components/FoldingArcDiagram.jsx` | 弧线图主可视化 |
| `src/components/ContactMap.jsx` | 接触热力图 |
| `src/components/ContactDensity.jsx` | 接触密度曲线 |
| `src/components/MDSPlot.jsx` | MDS 2D 散点图 |
| `src/components/MetricsPanel.jsx` | 指标面板 |
| `src/components/PhaseView.jsx` | 相位视图 |
| `src/components/StructuralComparison.jsx` | 结构对比 |
| `src/components/ComparisonView.jsx` | 并排对比 |
| `src/components/BatchOverview.jsx` | 批次概览 |
| `src/components/FoldingView3D.jsx` | 3D 折叠视图 |
| `backend/export_static_aime24.py` | 数据导出脚本 |
| `backend/split_similarity.py` | 相似度拆分脚本 |
| `deploy_oss.py` | OSS 部署脚本 |
