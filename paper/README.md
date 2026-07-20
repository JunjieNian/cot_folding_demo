# Paper build

The manuscript follows the ICLR 2026 layout used by the reference paper and is self-contained in this directory.

## Build

```powershell
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked final artifact is also copied to `../output/pdf/cot_folding_paper.pdf`.

## Main evidence sources

- Neuron Agreement Decoding (NAD): https://arxiv.org/abs/2510.26277
- Repository overview and benchmark summaries: `../README.md`
- Exact NFS implementation: `../nfs_pipeline/fold_score.py`
- Structural primitives: `../nfs_pipeline/primitives.py`
- Segment graph and score: `../nfs_pipeline/segment_graph.py`, `../nfs_pipeline/segment_score.py`
- Semantic validation artifacts: `../web/experiments/semantic_validation/results/`
- RL checkpoint analysis: `../docs/RL_CHECKPOINT_METRICS.md`
- Demo screenshot: `figures/demo_interface.png`

Large activation caches and full batch matrices are not stored in this Git repository. They are required to regenerate the full benchmark tables from scratch.
