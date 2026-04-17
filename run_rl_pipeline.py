#!/usr/bin/env python3
"""
RL Checkpoint NFS Pipeline — 在 NAD RL Cache 上运行 NFS 打分全流程。

驱动 6 阶段 per-checkpoint + 跨 checkpoint 汇总 (直接调用 nfs_pipeline 包):
  Phase 1: graph_builder.run    → slice-level 距离矩阵 + HMM + batch_summary
  Phase 2: primitives.run       → primitives_analysis.json
  Phase 3: fold_score.run       → nfs_analysis.json
  Phase 4: segment_graph.run    → segment-level 距离矩阵 + segment_summary
  Phase 5: segment_primitives.run (x2) → primitives_segment_{union,avg}
  Phase 6: segment_score.run    (x2)   → nfs_segment_{union,avg}
  Phase 7: aggregate            → cross_checkpoint/nfs_trajectory.json
  Phase 7b: rl_dynamics         → cross_checkpoint/rl_dynamics.json

用法:
  python run_rl_pipeline.py                              # 全部 11 个 checkpoint
  python run_rl_pipeline.py --checkpoint base step-600   # 指定 checkpoint
  python run_rl_pipeline.py --phase 1 2 3                # 仅 slice-level
  python run_rl_pipeline.py --phase 4 5 6                # 仅 segment-level
  python run_rl_pipeline.py --aggregate-only             # 仅跨 checkpoint 汇总
  python run_rl_pipeline.py --no-save-matrices           # 不保存 dist_*.npy
  python run_rl_pipeline.py --dry-run                    # 打印命令不执行
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from project_paths import (
    REPO_ROOT,
    default_rl_batch_dir,
    list_rl_checkpoints,
    resolve_rl_cache_path,
)

from nfs_pipeline import graph_builder, primitives, fold_score
from nfs_pipeline import segment_graph, segment_primitives, segment_score
from analysis import rl_dynamics

# Accuracy reference (from eval reports) for trajectory annotation
RL_ACCURACY = {
    "base":      28.61,
    "step-100":  29.48,
    "step-200":  30.49,
    "step-300":  31.55,
    "step-400":  31.68,
    "step-500":  32.64,
    "step-600":  33.19,
    "step-700":  32.99,
    "step-800":  32.11,
    "step-900":  32.39,
    "step-1000": 32.01,
}

ALL_PHASES = [1, 2, 3, 4, 5, 6]


def rl_step_number(checkpoint: str) -> int:
    """Extract numeric RL step from checkpoint name (base -> 0)."""
    if checkpoint == "base":
        return 0
    return int(checkpoint.split("-")[1])


def _run_phase(label: str, func, *args, dry_run: bool = False, **kwargs) -> bool:
    """Run a pipeline phase with timing and error handling."""
    prefix = f"  [{label}]"
    if dry_run:
        print(f"{prefix} [DRY-RUN] {func.__module__}.{func.__name__}()")
        return True

    print(f"{prefix} Running...")
    t0 = time.perf_counter()
    try:
        func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"{prefix} Done in {elapsed:.1f}s")
        return True
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"{prefix} FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def run_checkpoint(
    checkpoint: str,
    phases: list[int],
    *,
    no_save_matrices: bool = False,
    adaptive: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run selected phases for a single checkpoint."""
    cache = resolve_rl_cache_path(checkpoint)
    out = default_rl_batch_dir(checkpoint)

    print(f"\n{'='*80}")
    print(f"  Checkpoint: {checkpoint}  (RL step {rl_step_number(checkpoint)})")
    print(f"  Cache:  {cache}")
    print(f"  Output: {out}")
    print(f"  Phases: {phases}")
    print(f"{'='*80}")

    ok = True

    # Phase 1: slice-level graph construction
    if 1 in phases:
        ok = _run_phase(
            "Phase 1: slice graph",
            graph_builder.run,
            cache, out,
            save_matrices=not no_save_matrices,
            dry_run=dry_run,
        ) and ok

    # Phase 2: slice primitives extraction
    if 2 in phases:
        ok = _run_phase(
            "Phase 2: slice primitives",
            primitives.run,
            out,
            dry_run=dry_run,
        ) and ok

    # Phase 3: slice NFS scoring
    if 3 in phases:
        ok = _run_phase(
            "Phase 3: slice NFS",
            fold_score.run,
            cache, out,
            dry_run=dry_run,
        ) and ok

    # Phase 4: segment-level graph construction
    if 4 in phases:
        ok = _run_phase(
            "Phase 4: segment graph",
            segment_graph.run,
            cache, out, out,  # slice_batch_dir = same as output
            adaptive=adaptive,
            dry_run=dry_run,
        ) and ok

    # Phase 5: segment primitives (union + avg)
    if 5 in phases:
        for method in ["union", "avg"]:
            ok = _run_phase(
                f"Phase 5: segment primitives ({method})",
                segment_primitives.run,
                out, method,
                dry_run=dry_run,
            ) and ok

    # Phase 6: segment NFS (union + avg)
    if 6 in phases:
        slice_nfs = out / "nfs_analysis.json"
        for method in ["union", "avg"]:
            ok = _run_phase(
                f"Phase 6: segment NFS ({method})",
                segment_score.run,
                cache, out, method, slice_nfs,
                dry_run=dry_run,
            ) and ok

    return ok


def _extract_nfs_metrics(data: dict) -> dict:
    """Extract key metrics from an nfs_analysis / nfs_segment JSON."""
    import numpy as np

    metrics = {}
    dist = data.get("nfs_distribution", {})
    metrics["nfs_mean"] = dist.get("mean")
    metrics["nfs_std"] = dist.get("std")

    disc = data.get("discrimination", {})
    metrics["auroc"] = disc.get("auroc")
    metrics["auprc"] = disc.get("auprc")
    metrics["nfs_correct_mean"] = disc.get("nfs_correct_mean")
    metrics["nfs_incorrect_mean"] = disc.get("nfs_incorrect_mean")

    sel = data.get("selective_accuracy", {})
    metrics["selective_acc_top10"] = sel.get("top_10%")
    metrics["selective_acc_top20"] = sel.get("top_20%")

    rank = data.get("ranking", {})
    metrics["hit_at_1"] = rank.get("hit_at_1")
    metrics["pairwise_accuracy"] = rank.get("pairwise_accuracy")

    vote = data.get("voting", {})
    metrics["majority_accuracy"] = vote.get("majority_accuracy")
    metrics["weighted_accuracy"] = vote.get("weighted_accuracy")
    metrics["top1_accuracy"] = vote.get("top1_accuracy")

    samples = data.get("samples", [])
    if samples:
        b_vals = [s["B"] for s in samples if "B" in s]
        h_vals = [s["H"] for s in samples if "H" in s]
        d_vals = [s["D_star"] for s in samples if "D_star" in s]
        if b_vals:
            metrics["B_mean"] = float(np.mean(b_vals))
        if h_vals:
            metrics["H_mean"] = float(np.mean(h_vals))
        if d_vals:
            metrics["D_star_mean"] = float(np.mean(d_vals))

    return metrics


def aggregate_trajectory(checkpoints: list[str], dry_run: bool = False) -> dict | None:
    """Aggregate NFS metrics across checkpoints into nfs_trajectory.json."""
    print(f"\n{'='*80}")
    print("  Phase 7: Cross-checkpoint aggregation")
    print(f"{'='*80}")

    if dry_run:
        print("  [DRY-RUN] Would aggregate across checkpoints")
        return None

    trajectory = []

    for ckpt in checkpoints:
        out = default_rl_batch_dir(ckpt)
        entry: dict = {
            "name": ckpt,
            "rl_step": rl_step_number(ckpt),
            "accuracy": RL_ACCURACY.get(ckpt),
        }

        slice_nfs_file = out / "nfs_analysis.json"
        if slice_nfs_file.exists():
            with open(slice_nfs_file) as f:
                entry["slice"] = _extract_nfs_metrics(json.load(f))
        else:
            print(f"  WARNING: {slice_nfs_file} not found, skipping slice metrics for {ckpt}")
            entry["slice"] = None

        for method in ["union", "avg"]:
            seg_nfs_file = out / f"nfs_segment_{method}.json"
            if seg_nfs_file.exists():
                with open(seg_nfs_file) as f:
                    entry[f"segment_{method}"] = _extract_nfs_metrics(json.load(f))
            else:
                print(f"  WARNING: {seg_nfs_file} not found, skipping segment_{method} for {ckpt}")
                entry[f"segment_{method}"] = None

        trajectory.append(entry)

    cross_dir = REPO_ROOT / "batch_results_rl" / "cross_checkpoint"
    cross_dir.mkdir(parents=True, exist_ok=True)
    output_file = cross_dir / "nfs_trajectory.json"

    output = {"checkpoints": trajectory}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Trajectory saved to: {output_file}")
    print(f"  Checkpoints: {len(trajectory)}")

    print(f"\n  {'Checkpoint':<12} {'Step':>5} {'Acc%':>6} {'NFS_s':>7} {'AUROC_s':>8} "
          f"{'NFS_u':>7} {'AUROC_u':>8}")
    print(f"  {'-'*60}")
    for e in trajectory:
        acc = e["accuracy"] or 0
        s = e.get("slice") or {}
        u = e.get("segment_union") or {}
        nfs_s = s.get("nfs_mean", 0) or 0
        auroc_s = s.get("auroc", 0) or 0
        nfs_u = u.get("nfs_mean", 0) or 0
        auroc_u = u.get("auroc", 0) or 0
        print(f"  {e['name']:<12} {e['rl_step']:>5} {acc:>6.2f} "
              f"{nfs_s:>7.4f} {auroc_s:>8.4f} {nfs_u:>7.4f} {auroc_u:>8.4f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run NFS pipeline across RL checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", nargs="*", default=None,
        help="Checkpoint(s) to process (default: all). e.g. base step-600",
    )
    parser.add_argument(
        "--phase", nargs="*", type=int, default=None,
        help="Phase(s) to run per checkpoint (default: all 1-6). e.g. 1 2 3",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip per-checkpoint processing, only aggregate existing results",
    )
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Skip cross-checkpoint aggregation",
    )
    parser.add_argument(
        "--no-save-matrices", action="store_true",
        help="Do not save dist_*.npy in Phase 1 (saves disk space)",
    )
    parser.add_argument(
        "--no-dynamics", action="store_true",
        help="Skip Phase 7b RL dynamics analysis",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Enable adaptive segment splitting in Phase 4",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    all_ckpts = list_rl_checkpoints()
    if args.checkpoint:
        for c in args.checkpoint:
            if c not in all_ckpts:
                print(f"ERROR: Unknown checkpoint {c!r}. Available: {all_ckpts}")
                sys.exit(1)
        checkpoints = args.checkpoint
    else:
        checkpoints = all_ckpts

    phases = args.phase if args.phase else ALL_PHASES

    print("=" * 80)
    print("  RL Checkpoint NFS Pipeline")
    print("=" * 80)
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Phases:      {phases}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Save matrices: {not args.no_save_matrices}")

    t_start = time.perf_counter()

    if not args.aggregate_only:
        results = {}
        for ckpt in checkpoints:
            ok = run_checkpoint(
                ckpt, phases,
                no_save_matrices=args.no_save_matrices,
                adaptive=args.adaptive,
                dry_run=args.dry_run,
            )
            results[ckpt] = ok

        print(f"\n{'='*80}")
        print("  Per-checkpoint Summary")
        print(f"{'='*80}")
        for ckpt, ok in results.items():
            status = "OK" if ok else "FAILED"
            print(f"  {ckpt:<12} {status}")

    if not args.no_aggregate:
        aggregate_trajectory(checkpoints, dry_run=args.dry_run)

    if not args.no_aggregate and not args.no_dynamics:
        _run_phase(
            "Phase 7b: RL dynamics",
            rl_dynamics.run_rl_dynamics,
            checkpoints,
            dry_run=args.dry_run,
        )

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
