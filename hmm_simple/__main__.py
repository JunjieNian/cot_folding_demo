#!/usr/bin/env python3
"""
CLI entry-point for HMM-Simple analysis.

Usage
-----
    python -m hmm_simple analyze \\
        --input data.csv \\
        --output report.json \\
        --p-stay 0.9 \\
        --transition-weight 2.0 \\
        --entropy-col entropy \\
        --confidence-col confidence
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import compute_hmm_features, compute_hmm_score
from .summary import summarize_hmm


def _numpy_encoder(obj):
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m hmm_simple",
        description="HMM-Simple CoT reasoning maturity analysis")
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="Analyze a trajectory CSV")
    analyze.add_argument("--input", required=True,
                         help="Path to input CSV file")
    analyze.add_argument("--output", default=None,
                         help="Path to output JSON report")
    analyze.add_argument("--p-stay", type=float, default=0.9,
                         help="HMM self-transition probability (default: 0.9)")
    analyze.add_argument("--transition-weight", type=float, default=2.0,
                         help="Transition-quality weight (default: 2.0)")
    analyze.add_argument("--entropy-col", default="entropy",
                         help="Name of entropy column (default: entropy)")
    analyze.add_argument("--confidence-col", default="confidence",
                         help="Name of confidence column (default: confidence)")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "analyze":
        parser.print_help()
        sys.exit(1)

    # Load data
    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    for col in [args.entropy_col, args.confidence_col]:
        if col not in df.columns:
            print(f"Error: column '{col}' not found in {path}. "
                  f"Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    entropy = df[args.entropy_col].values.astype(np.float64)
    confidence = df[args.confidence_col].values.astype(np.float64)

    print(f"Loaded {len(entropy)} steps from {path}")

    # Compute summary
    summary = summarize_hmm(entropy, confidence,
                            p_stay=args.p_stay,
                            transition_weight=args.transition_weight)

    # Print summary
    print()
    print(summary["text"])

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        features = compute_hmm_features(entropy, confidence,
                                        p_stay=args.p_stay)
        score_result = compute_hmm_score(entropy, confidence,
                                         p_stay=args.p_stay,
                                         transition_weight=args.transition_weight)

        report = {
            "summary": summary,
            "score": score_result.get("score"),
            "features": features,
        }

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=_numpy_encoder)

        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
