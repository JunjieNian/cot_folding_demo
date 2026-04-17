#!/usr/bin/env python3
"""
HMM-Simple — 2-state HMM segmented scoring for CoT reasoning maturity.

Only one tuneable hyper-parameter: ``transition_weight`` (default 2.0).
Pearson > 0.83 and Hit@3 = 60 % across all benchmark series.
"""

from .core import compute_hmm_features, compute_hmm_score
from .summary import summarize_hmm
from .hmmsimple3 import compute_hmmsimple3_metrics, summarize_hmmsimple3
from .fusion import compute_fusion_score, compute_fusion_v2_score

__all__ = [
    "compute_hmm_features",
    "compute_hmm_score",
    "summarize_hmm",
    "compute_hmmsimple3_metrics",
    "summarize_hmmsimple3",
    "compute_fusion_score",
    "compute_fusion_v2_score",
]
