# CoT Folding: Structural Analysis of LLM Reasoning Trajectories

**Visualize and quantify the "folding structure" of Chain-of-Thought reasoning, inspired by protein folding.**

> **Live Demo**: [http://demo.cot-folding.top](http://demo.cot-folding.top)

---

## Overview

CoT Folding treats a Chain-of-Thought (CoT) reasoning trajectory as a 1D chain analogous to a protein sequence. By measuring pairwise similarity between token segments using **neuron activation patterns** (via NAD — Neuron Activation Decomposition), we reveal how the reasoning "folds" in activation space, forming compact cores, long-range return connections, and drifting branches.

The framework introduces the **Native Fold Score (NFS)** — a fully unsupervised, parameter-free quality metric for CoT reasoning that requires no correctness labels.

| Protein Folding | CoT Folding |
|:---|:---|
| Amino acid residue | 32-token slice |
| 3D spatial coordinates | Neuron activation vectors (sparse key sets) |
| Residue-residue Euclidean distance | Slice-slice Jaccard distance |
| Secondary structure (alpha-helix / beta-sheet) | HMM state (Exploration / Exploitation) |
| Folding stability | Native Fold Score (NFS) |

**Core insight**: High-quality reasoning = **dense backbone core** + **effective long-range return connections** + **minimal unresolved drift**.

---

## Key Findings

### 1. NFS Effectively Discriminates Correct vs. Incorrect Reasoning

Without any labels, NFS separates correct and incorrect CoT samples with strong discriminative power:

| Metric | AIME 2024 (Math) | GPQA (Science) |
|:---|:---:|:---:|
| Samples | 1,920 (30 problems x 64 runs) | 12,672 (198 problems x 64 runs) |
| Model | DeepSeek-R1-0528-Qwen3-8B | DeepSeek-R1-0528-Qwen3-8B |
| Overall Accuracy | 75.36% | 60.17% |
| **AUROC** | **0.7646** | **0.6857** |
| **AUPRC** | **0.8784** | **0.7569** |
| **Cohen's d** | **0.937** (large) | **0.643** (medium) |
| Selective Accuracy (top 10%) | 92.19% | 86.35% |
| Hit@1 (per-problem ranking) | 0.6667 | 0.6263 |
| Majority Voting Accuracy | 80% (24/30) | 69.7% (138/198) |

NFS distribution by correctness:

| | AIME 2024 | GPQA |
|:---|:---:|:---:|
| NFS (correct) | 12.60 +/- 2.66 | 12.72 +/- 3.82 |
| NFS (incorrect) | 9.98 +/- 3.16 | 10.41 +/- 3.21 |

### 2. Segment-Level Analysis Substantially Improves Discrimination

Aggregating from 32-token slices to HMM-contiguous segments (Phase 4-6) yields major improvements:

| Metric | Slice (baseline) | Segment (union) | Segment (avg) |
|:---|:---:|:---:|:---:|
| **AUROC** | 0.7646 | 0.7911 (+0.027) | **0.8290 (+0.064)** |
| **AUPRC** | 0.8784 | 0.9091 (+0.031) | **0.9295 (+0.051)** |
| **Cohen's d** | 0.869 | 0.947 | **1.111** |
| **Top-10% Accuracy** | 92.19% | 97.40% | **99.48%** |
| **Hit@1** | 0.6667 | 0.7333 | **0.8000** |

The segment-average method achieves near-perfect top-10% selective accuracy (99.5%).

### 3. Explore/Exploit Phase Separation Is Robust

Mann-Whitney U tests confirm that HMM-partitioned Explore and Exploit phases are statistically distinct in activation space:

| | AIME 2024 | GPQA |
|:---|:---:|:---:|
| Significantly separated samples | **100%** (1920/1920) | **98.6%** (12500/12672) |
| Mean separation (Jaccard) | 0.0358 | 0.0277 |
| Mean Cohen's d (within-sample) | -0.533 | -0.423 |

### 4. Structural Coherence ≠ Semantic Correctness

Case studies (e.g., AIME Problem 61) reveal:
- **High NFS + Correct** (NFS=15.83): Dense unified backbone, 63.7% exploit coverage, effective return connections
- **Low NFS + Incorrect** (NFS=6.37): Highly fragmented, repetitive expressions, unresolved drift
- **High NFS + Incorrect** (NFS=11.95): Dense but built around wrong intermediate values — internally consistent but semantically wrong

**Conclusion**: NFS detects structural fragmentation but not semantic errors.

### 5. Label-Free RL Checkpoint Ranking

Tracking NFS components across 11 RL training checkpoints (Qwen3-4B-Base, steps 0-1000):

| Checkpoint | Accuracy | NFS mean | B mean | H mean |
|:---|:---:|:---:|:---:|:---:|
| base | 28.61% | 19.93 | 0.234 | 0.048 |
| step-300 | 31.55% | - | - | - |
| **step-600** | **33.19%** (peak) | - | - | - |
| step-1000 | 32.01% | - | - | - |

Best label-free ranking methods (Spearman rho with accuracy):

| Method | rho |
|:---|:---:|
| -beta_NFS (dynamics) | **0.8273** |
| -H_mean (slice) | 0.8091 |
| -B_mean (slice) | 0.8000 |

### 6. Semantic Validation: Structure Reflects Meaning

A 4-tier semantic validation experiment confirms that structural similarity (from neuron activations) genuinely reflects text semantics:

| Tier | Method | Spearman rho | What it proves |
|:---|:---|:---:|:---|
| **1** | TF-IDF cosine | 0.4542 | Even surface-level word overlap correlates with structural similarity |
| **2** | Sentence embeddings (MiniLM-L6) | 0.5704 | Independent semantic model confirms strong correlation |
| **3** | Cross-encoder (DistilRoBERTa) | 0.7370 | Fine-grained semantic scoring shows even higher agreement |
| **4** | Source-model sparse embeddings | 0.8268 | Same-source upper bound — internal geometric consistency |

Controls rule out positional confounds: partial Spearman after removing sequence distance = 0.5533 (Tier 2).

---

## Methodology

### Pipeline Overview

```
NAD Cache (neuron activation data)
    |
    v
[Phase 1] Graph Construction
    |-- Slice extraction (every 32 tokens)
    |-- Entropy / Confidence computation
    |-- 2-state HMM Viterbi => Exploration(0) / Exploitation(1)
    |-- Pairwise Jaccard distance matrix O(n^2)
    |
    v
[Phase 2] Structural Primitive Extraction
    |-- Core: largest dense exploit block (greedy merge)
    |-- Return Edges: long-range contacts (gap > n/4, sim > threshold)
    |-- Drift Branches: explore blocks disconnected from core
    |-- Final Closure: tail-to-core connection strength
    |
    v
[Phase 3] Native Fold Score (NFS)
    |-- B = s_core * f_core                 (Backbone)
    |-- H = mean(return edge strength)      (Hydrogen)
    |-- D* = 1 - G * (1 - D0)              (Drift, gate-adjusted)
    |-- NFS = 100 * (B * H * (1-D*))^(1/3)
    |
    v
[Phase 4-6] Segment-Level Analysis
    |-- Aggregate slices into HMM-contiguous segments
    |-- Repeat primitives + scoring at segment granularity
```

### NFS Formula

```
NFS = 100 * (B * H * (1 - D*))^(1/3)
```

| Component | Name | Source | Formula | Range |
|:---:|:---|:---|:---|:---:|
| **B** | Backbone | Core | core_internal_sim * core_exploit_fraction | [0, 1] |
| **H** | Hydrogen | Return Edges | mean((sim - tau)/(1 - tau) * gap/(n-1)) | [0, ~0.5] |
| **D0** | Drift | Drift Branches | time-weighted unresolved drift ratio | [0, 1] |
| **G** | Gate | Final Closure | (1 + closure_coefficient) / 2 | [0.5, 1] |
| **D*** | Adjusted Drift | D0 + G | 1 - G * (1 - D0) | [0, 1] |

The geometric mean ensures all three dimensions must be healthy — any single zero component collapses NFS to zero.

### Metrics Glossary

<details>
<summary><b>Contact Metrics</b></summary>

- **Contact Threshold (tau)**: `mean(sim) + std(sim)` of upper triangle — only significantly similar pairs count as contacts
- **Total Contacts**: Count of slice pairs with similarity > tau
- **Long-Range Contacts**: Contacts with sequence gap > n/4
- **Folding Degree**: long_range_contacts / total_contacts — higher = more compact folding
- **Contact Order**: Weighted average sequence gap of contacts, normalized — measures fold "span"

</details>

<details>
<summary><b>MDS Metrics</b></summary>

- **MDS Coordinates**: Classical (Torgerson) MDS projects the n*n distance matrix to 2D/3D
- **MDS Stress (Kruskal Stress-1)**: sqrt(sum((d_orig - d_mds)^2) / sum(d_orig^2)) — lower = better 2D/3D representation
- **Radius of Gyration**: RMS distance of all slices to their centroid — measures structural compactness

</details>

<details>
<summary><b>Per-Slice Metrics</b></summary>

- **Entropy**: Aggregated token-level uncertainty per slice — high = model is exploring/uncertain
- **Confidence**: Mean token-level confidence per slice — high = model is certain about its output
- **HMM State**: Binary classification — Explore (0, high entropy) vs Exploit (1, low entropy)

</details>

<details>
<summary><b>NFS Primitives</b></summary>

- **Core (green)**: Largest dense exploit block, greedily merged — the reasoning backbone
- **Return Edges (orange)**: Long-range high-similarity pairs — fold-back verification connections
- **Drift Branches (red)**: Explore blocks with no strong connection to core — dead-end explorations
- **Final Closure (light green)**: Last exploit block's connection strength to core — conclusion lock-in

</details>

<details>
<summary><b>Information Flow</b></summary>

- **Arterial**: Entropy increases or confidence drops — divergent/exploratory transitions
- **Venous**: Entropy decreases or confidence rises — convergent/confirmatory transitions
- **Capillary**: Small changes in both — steady progression
- **Shunt**: Low entropy change but confidence jumps — shortcut without exploration

</details>

<details>
<summary><b>Clustering Statistics</b></summary>

- **Separation**: cross_mean_distance - within_mean_distance — positive = good HMM state distinction
- **Cohen's d**: Effect size of within vs cross distance distributions
- **p-value**: Mann-Whitney U test significance

</details>

---

## Interactive Demo

The live demo at **[http://demo.cot-folding.top](http://demo.cot-folding.top)** visualizes AIME 2024 data (30 problems, 1920 samples) with:

**Views:**
- **Arc Diagram**: Protein-style folding visualization with MDS coordinates, HMM state coloring, and golden long-range contact bonds
- **Contact Map**: n*n similarity heatmap
- **MDS 2D/3D Plot**: Dimensionality reduction projection
- **Phase View**: Segment-level structural overview
- **Structural Comparison**: Correct vs. incorrect sample statistics
- **Batch Overview**: Aggregate statistics across all problems

**Annotation Tracks:**
- HMM state (Explore/Exploit)
- Entropy and Confidence curves
- Information flow type
- Functional decomposition (Core/Return/Drift/Closure)
- Effectiveness labels

**Interaction:**
- Click any slice to read the original reasoning text
- Keyboard shortcuts: `?` (help), `<-`/`->` (problems), `Up`/`Down` (samples), `1-5` (views)
- Dark mode toggle, 2D/3D lens switch

The demo also includes RL training trajectory analysis tracking structural changes across 11 checkpoints.

---

## Repository Structure

```
cot_folding_demo/
|
|-- nfs_pipeline/                # Core NFS computation pipeline (Python)
|   |-- graph_builder.py         # Phase 1: Slice-level graph construction
|   |-- primitives.py            # Phase 2: Structural primitive extraction
|   |-- fold_score.py            # Phase 3: NFS scoring + external validation
|   |-- segment_graph.py         # Phase 4: Segment-level graph construction
|   |-- segment_primitives.py    # Phase 5: Segment-level primitives
|   |-- segment_score.py         # Phase 6: Segment-level NFS scoring
|   `-- graph_features.py        # Classical graph structure features
|
|-- hmm_simple/                  # 2-state HMM segmentation engine
|   |-- core.py                  # Viterbi decoder (_hmm_viterbi_2state)
|   |-- fusion.py                # HMM fusion methods
|   `-- hmmsimple3.py            # Alternative HMM implementation
|
|-- alignment/                   # NAD cache reader (full version)
|-- alignment_lite/              # Lightweight NAD cache reader
|
|-- analysis/                    # Analysis & visualization modules
|   |-- clustering.py            # Explore/Exploit clustering analysis
|   |-- case_study.py            # Single-sample deep dive
|   |-- rl_dynamics.py           # RL checkpoint dynamics
|   |-- effectiveness_flow_eval.py  # Information flow analysis
|   `-- model_ranking_viz.py     # Checkpoint ranking visualization
|
|-- viz/                         # Visualization utilities
|-- tests/                       # Test cases
|-- project_paths.py             # Centralized path resolver (env-aware)
|-- run_rl_pipeline.py           # RL checkpoint pipeline orchestrator
|
|-- web/                         # Interactive web visualization
|   |-- src/                     # React frontend (28 components, ~5K LOC)
|   |   |-- components/          # FoldingArcDiagram, ContactMap, MDS, etc.
|   |   |-- hooks/               # useFoldingState, useURLState, etc.
|   |   |-- App.jsx              # Main application
|   |   `-- api.js               # Data fetching + LRU cache
|   |-- backend/                 # Python data export scripts
|   |-- experiments/             # Semantic validation experiments (Tier 1-4)
|   |-- package.json             # Node.js dependencies
|   |-- vite.config.js           # Vite build configuration
|   |-- deploy_oss.py            # Alibaba Cloud OSS deployment
|   |-- REPORT.md                # Detailed technical documentation
|   `-- DEPLOY.md                # Deployment guide
|
`-- docs/                        # Additional documentation
    |-- RL_PIPELINE.md           # RL checkpoint pipeline guide
    |-- RL_CHECKPOINT_METRICS.md # Checkpoint ranking analysis
    |-- BATCH_REPORT.md          # Batch processing report
    `-- IC_NFS_RL_ANALYSIS.md    # IC-NFS variant analysis
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- NumPy, SciPy
- [NAD (Neuron Activation Decomposition)](https://github.com/) core library for Jaccard distance computation
- Node.js 18+ (for the web frontend)

### Running the NFS Pipeline

```bash
# Phase 1: Build slice-level graphs
python batch_process.py --cache aime24

# Phase 2: Extract structural primitives
python extract_primitives.py --batch

# Phase 3: Compute NFS + external validation
python native_fold_score.py

# Phase 4-6: Segment-level analysis
python segment_batch_process.py
python extract_segment_primitives.py --batch --method avg
python segment_fold_score.py --method avg
```

### Running the RL Pipeline

```bash
# All 11 checkpoints, all phases
python run_rl_pipeline.py

# Specific checkpoints
python run_rl_pipeline.py --checkpoint base step-600

# Only slice-level (Phase 1-3)
python run_rl_pipeline.py --phase 1 2 3
```

### Running the Web Frontend

```bash
cd web
npm install
npm run dev          # Development server on :5031
npm run build        # Production build
```

### Environment Variables

All paths are configurable via environment variables (see `project_paths.py`):

| Variable | Purpose |
|:---|:---|
| `INTRA_COT_NAD_ROOT` | NAD core library path |
| `INTRA_COT_CACHE_BASE` | Neuron activation cache root |
| `INTRA_COT_RL_CACHE_ROOT` | RL checkpoint cache root |
| `INTRA_COT_MODEL_SEARCH_ROOTS` | Tokenizer model search paths |

---

## Citation

If you use this work, please cite:

```bibtex
@software{cot_folding_2025,
  title={CoT Folding: Structural Analysis of LLM Reasoning Trajectories},
  author={Nian, Junjie},
  url={http://demo.cot-folding.top},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
