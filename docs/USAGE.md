# 使用说明

## 快速开始

```bash
# 运行测试
python3 test_hmm_graph.py

# 结果将保存在 results/ 目录
```

## 项目结构

```
cot_folding_demo/
├── README.md              # Project overview and findings
├── tests/
│   └── test_hmm_graph.py  # Main test script
└── results/               # Output directory
    ├── distance_matrix_sample0.npy    # 168x168 distance matrix
    ├── hmm_states_sample0.npy         # HMM state sequence
    └── report_sample0.json            # JSON report
```

## 修改测试参数

编辑 `test_hmm_graph.py` 的 `main()` 函数：

```python
# 修改测试样本
sample_id = 0  # 改为 1, 2, 3... 测试其他 samples

# 修改 HMM 参数
result = builder.build_graph(sample_id, p_stay=0.9)  # 改为 0.8, 0.95 等
```

## 读取结果

```python
import numpy as np
import json

# 读取距离矩阵
dist = np.load('results/distance_matrix_sample0.npy')
print(f"Distance matrix shape: {dist.shape}")

# 读取 HMM 状态
states = np.load('results/hmm_states_sample0.npy')
print(f"Exploration slices: {(states == 0).sum()}")

# 读取报告
with open('results/report_sample0.json') as f:
    report = json.load(f)
print(f"Total time: {report['total_time_s']:.3f}s")
```

## 批量处理多个 samples

修改 `main()` 函数添加循环：

```python
for sample_id in range(64):  # 处理一个 problem 的 64 个 samples
    result = builder.build_graph(sample_id)
    # 保存结果...
```

## 性能优化建议

1. **并行处理**: 使用 multiprocessing 并行处理多个 samples
2. **KNN 而非完全图**: 只计算 k=10 近邻，减少 95% 计算
3. **Cython/Numba**: 优化 Jaccard 距离计算

## 依赖项

- Python 3.7+
- NumPy
- NAD Cache (alignment 包)
- EE-main (HMM 分割)

## 常见问题

**Q: 如何测试其他数据集？**
A: 修改 `cache_path` 变量指向其他 cache 目录。

**Q: 内存不足怎么办？**
A: 使用 KNN 图而非完全图，或分批处理。

**Q: 如何可视化图结构？**
A: 使用 networkx + matplotlib 绘制图，参考：
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for i in range(n):
    for j in range(i+1, n):
        if dist_matrix[i, j] < threshold:
            G.add_edge(i, j, weight=dist_matrix[i, j])
nx.draw(G)
plt.show()
```
