# Semantic Validation：Tier 1-4 实验说明

本文档介绍 `experiments/semantic_validation` 中 Tier 1 到 Tier 4 分别做了什么实验、用了哪些方法，以及主要结论。

## 0. 统一实验框架（四个 Tier 共享）

- **数据集**：`aime24`，共 `30` 道题、`1920` 条样本（约 `88` 万 slices）。
- **基本对象**：每条推理被切成多个 slice（时间序列片段）。
- **结构相似度（固定参照）**：来自缓存的神经元结构信号，存为 `s*.sim.b64`（解码为 `n×n` 相似度矩阵）。
- **核心检验问题**：  
  对每个样本，比较“结构相似度矩阵”与“文本语义相似度矩阵”的一致性。  
  一致性指标主要用 **Spearman ρ**（取上三角 pair）。

---

## 1. Tier 1：TF-IDF 基线实验（最弱语义基线）

### 用到的方法

- `TfidfVectorizer(max_features=5000, sublinear_tf=True)`
- slice 两两文本余弦相似度（TF-IDF cosine）
- 每个样本内：`Spearman(struct_sim, tfidf_text_sim)`

### 做了什么实验

对每条样本：
1. 将所有 slice 文本做 TF-IDF 向量化；
2. 得到 slice×slice 的文本相似度矩阵；
3. 与结构相似度矩阵逐对比较（上三角）；
4. 输出每条样本的相关系数和显著性。

### 输出文件

- `results/tier1_tfidf/per_sample_correlations.csv`

### 当前结果（已有 summary）

- 均值 `ρ = 0.4542`，中位数 `0.4352`，标准差 `0.1457`
- `p < 0.05` 比例：`100%`

### 结论定位

Tier 1 证明：即使用非常朴素的词面特征，结构相似度也与文本相似度存在稳定正相关。

---

## 2. Tier 2：句向量语义实验（独立语义主实验）

### 用到的方法

- `sentence-transformers/all-MiniLM-L6-v2`
- 对所有 slices 统一编码（L2 归一化）
- 语义相似度：`embedding @ embedding.T`（余弦）
- 每个样本内 Spearman 相关

### 做了什么实验

1. 先编码全量 slice 文本（批量 GPU 推理）；
2. 对每个样本恢复对应 embedding 子矩阵；
3. 构建 slice×slice 语义相似度；
4. 与结构相似度做 Spearman 相关；
5. 额外做控制实验（偏相关、top-k、HMM 状态分层、正确/错误对比）。

### 输出文件

- `results/tier2_embedding/per_sample_correlations.csv`
- 相关控制输出在 `results/controls/`

### 当前结果（已有 summary）

- 主结果：均值 `ρ = 0.5704`（显著高于 Tier 1 的 `0.4542`）
- 去位置偏置后的偏 Spearman：均值 `ρ = 0.5533`
- Top-k 检索（k=5）文本相似度均值：  
  - 结构近邻 `0.6601`  
  - 顺序近邻 `0.5450`  
  - 随机 `0.3567`

### 结论定位

Tier 2 是独立语义验证的主证据：结构相似度与语义相似度强相关，而且不只是“相邻位置”造成的伪相关。

---

## 3. Tier 3：Cross-Encoder 细粒度语义重评分实验

### 用到的方法

- `cross-encoder/stsb-distilroberta-base`
- 先按结构相似度分箱抽样 pair（并做 gap 分布加权，减位置偏置）
- Cross-encoder 对句对直接打分（比 bi-encoder 更细粒度）
- 与结构相似度做 Spearman 相关

### 做了什么实验

1. 从全量 slice 对中按结构相似度分层抽样（默认 `10` 个 bin，每 bin 最多 `500` 对）；
2. 用 cross-encoder 对 `(slice_i, slice_j)` 直接评分；
3. 评分归一化到 `[0,1]`；
4. 计算采样对上的 `Spearman(struct_sim, ce_score)`。

### 输出文件

- `results/tier3_crossencoder/pair_scores.csv`

### 当前结果（已有 summary）

- `ρ = 0.7370`，`n_pairs = 5000`，`p ≈ 0`

### 结论定位

Tier 3 使用更强的语义匹配器后，结构-语义一致性进一步提升，支持“结构信号确实刻画语义关系”。

---

## 4. Tier 4：Source-Model 稀疏嵌入补充实验（同源验证）

### 用到的方法

- 使用 DeepSeek-R1 缓存中的神经元激活 `w_sum` + `keys`
- 构造稀疏向量（每个 slice 一行 CSR），L2 归一化后算余弦相似度
- 与结构相似度做 Spearman 相关
- 另做 source-embedding 版本 top-k 检索对照

### 做了什么实验

1. 从 `cache_neuron_output.../rows` 读取 `sample_row_ptr / row_ptr / keys / w_sum`；
2. 为每条样本构建 slice×feature 稀疏矩阵；
3. 得到 source-model 的 slice×slice 余弦相似度；
4. 与结构相似度做相关分析；
5. 做结构近邻 vs 顺序近邻 vs 随机近邻对照。

### 输出文件

- `results/tier4_source_model/per_sample_correlations.csv`
- `results/tier4_source_model/topk_retrieval.json`

### 当前结果（已有 summary）

- 均值 `ρ = 0.8268`（明显更高）
- top-k（k=5）文本相似度均值：  
  - 结构近邻 `0.6774`  
  - 顺序近邻 `0.5087`  
  - 随机 `0.3261`

### 结论定位（重要）

Tier 4 是**同源补充验证**：因为结构信号和语义信号都来自同一个源模型内部表征，因此不能作为“独立外部验证”，但可说明模型内部几何一致性非常强。

---

## 5. 四个 Tier 的关系总结

- **Tier 1（弱基线）**：词面层面已有正相关；
- **Tier 2（独立主验证）**：通用句向量下相关更强，且控制实验支持非位置伪相关；
- **Tier 3（强语义评估器）**：细粒度语义打分下相关最高（独立路径中）；
- **Tier 4（同源补充）**：内部表征同源对齐非常强，但独立性较弱。

如果要写论文叙事，可把 Tier 1→3 作为“独立证据强度递增链路”，Tier 4 作为“同源上界/补充证据”。

