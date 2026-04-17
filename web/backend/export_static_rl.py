#!/usr/bin/env python3
"""
Static Data Exporter for COT Folding Map — RL Multi-Checkpoint.

Reads from FoldingEngine (per-checkpoint) and exports all data as static JSON
files for the RL dataset, including cross-checkpoint trajectory/dynamics.

Output structure:
  data/rl/
    app.json
    checkpoints.json
    trajectory.json
    problems.meta.json
    base/
      overview.json
      problems.index.json
      compare/p{uuid}.json
      samples/p{uuid}/s{sid}.bundle.json
      samples/p{uuid}/s{sid}.text.json
    step-100/
      ... (same structure)
    step-1000/
      ...

Usage:
  python backend/export_static_rl.py [--output-dir PATH] [--skip-text] [--max-problems N]
  python backend/export_static_rl.py --index-only          # Re-export only problems.index.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make the project root importable
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# folding_engine lives alongside this script or is provided externally.
ORIG_BACKEND = Path(__file__).resolve().parent
if str(ORIG_BACKEND) not in sys.path:
    sys.path.insert(0, str(ORIG_BACKEND))

from project_paths import (
    resolve_rl_cache_path,
    default_rl_batch_dir,
    list_rl_checkpoints,
    REPO_ROOT,
)
from folding_engine import FoldingEngine


def load_problem_texts() -> dict[str, dict]:
    """Load problem prompts/ground_truth from evaluation_report_compact.json.

    Returns {problem_id: {prompt, ground_truth, domain1, difficulty, ...}}.
    Only needs to read from one checkpoint (base) since prompts are the same.
    """
    try:
        cache_path = resolve_rl_cache_path("base")
    except FileNotFoundError:
        print("  [WARN] Cannot resolve RL cache for base — no problem text")
        return {}

    report = cache_path / "evaluation_report_compact.json"
    if not report.exists():
        print(f"  [WARN] evaluation_report_compact.json not found: {report}")
        return {}

    with open(report) as f:
        data = json.load(f)

    texts = {}
    for p in data.get("results", []):
        pid = p["problem_id"]
        # Strip the common "Return your final response within \\boxed{}.\nQuestion：\n" prefix
        prompt = p.get("prompt", "")
        prefix = "Return your final response within \\boxed{}.\nQuestion：\n"
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):]
        # Also try variant with non-breaking space or unicode
        for alt in [
            "Return your final response within \\boxed{}.\nQuestion:\n",
            "Return your final response within \\boxed{}.\n",
        ]:
            if prompt.startswith(alt):
                prompt = prompt[len(alt):]
                break

        texts[pid] = {
            "prompt": prompt.strip(),
            "ground_truth": str(p.get("ground_truth", "")),
            "domain": p.get("domain1", ""),
            "difficulty": p.get("difficulty", ""),
        }

    print(f"  Loaded problem text for {len(texts)} problems")
    return texts


def load_nfs_analysis(checkpoint: str) -> dict[tuple[str, int], bool]:
    """Load nfs_analysis.json for a checkpoint.

    Returns {(problem_id, sample_id): is_correct} lookup table.
    """
    batch_dir = default_rl_batch_dir(checkpoint)
    nfs_path = batch_dir / "nfs_analysis.json"
    if not nfs_path.exists():
        print(f"  [WARN] nfs_analysis.json not found for {checkpoint}: {nfs_path}")
        return {}

    with open(nfs_path) as f:
        data = json.load(f)

    lookup = {}
    for s in data.get("samples", []):
        key = (s["problem_id"], s["sample_id"])
        lookup[key] = bool(s.get("is_correct", False))
    print(f"  Loaded is_correct for {len(lookup)} samples from {checkpoint}")
    return lookup


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def write_json(path: Path, data, indent=None):
    """Write JSON file with numpy support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=indent)
    size_kb = path.stat().st_size / 1024
    print(f"  -> {path.name} ({size_kb:.1f} KB)")


def load_cross_checkpoint_data():
    """Load rl_dynamics.json and nfs_trajectory.json."""
    cross_dir = REPO_ROOT / "batch_results_rl" / "cross_checkpoint"

    dynamics_path = cross_dir / "rl_dynamics.json"
    trajectory_path = cross_dir / "nfs_trajectory.json"

    if not dynamics_path.exists():
        print(f"[ERROR] rl_dynamics.json not found: {dynamics_path}")
        sys.exit(1)
    if not trajectory_path.exists():
        print(f"[ERROR] nfs_trajectory.json not found: {trajectory_path}")
        sys.exit(1)

    with open(dynamics_path) as f:
        dynamics = json.load(f)
    with open(trajectory_path) as f:
        nfs_traj = json.load(f)

    return dynamics, nfs_traj


def select_problems(
    dynamics: dict,
    problem_texts: dict[str, dict],
    max_problems: int = 15,
) -> list[dict]:
    """Select representative problems from the 3 categories.

    Default: 6 from h_rises_first, 6 from d_collapses_first, 3 from other.
    Enriches with problem text from evaluation_report.
    """
    cats = dynamics["problem_tracking"]["categorization"]

    # Distribution: roughly 40/40/20 percent of max_problems
    n_h = max(1, round(max_problems * 0.4))
    n_d = max(1, round(max_problems * 0.4))
    n_o = max(1, max_problems - n_h - n_d)

    h_list = cats["h_rises_first"]
    d_list = cats["d_collapses_first"]
    o_list = cats["other"]

    # Evenly sample from each category
    import random
    random.seed(42)

    selected = []
    for src, n, cat_name in [
        (h_list, n_h, "h_rises_first"),
        (d_list, n_d, "d_collapses_first"),
        (o_list, n_o, "other"),
    ]:
        pick = src[:n] if len(src) <= n else random.sample(src, n)
        for pid in pick:
            short_id = pid[:8] if len(pid) > 12 else pid
            info = problem_texts.get(pid, {})
            prompt = info.get("prompt", "")
            # short_prompt: first ~80 chars for display
            short_prompt = prompt[:80].replace("\n", " ")
            if len(prompt) > 80:
                short_prompt += "..."
            selected.append({
                "id": pid,
                "category": cat_name,
                "short_id": short_id,
                "prompt": prompt,
                "short_prompt": short_prompt,
                "ground_truth": info.get("ground_truth", ""),
                "domain": info.get("domain", ""),
                "difficulty": info.get("difficulty", ""),
            })

    print(f"  Selected {len(selected)} problems: "
          f"{n_h} h_rises_first, {n_d} d_collapses_first, {n_o} other")
    return selected


def export_cross_checkpoint_files(
    output_dir: Path,
    dynamics: dict,
    nfs_traj: dict,
    selected_problems: list[dict],
    checkpoints: list[str],
):
    """Export RL-root-level cross-checkpoint files."""
    print("\n[Exporting cross-checkpoint files]")

    # app.json
    write_json(output_dir / "app.json", {
        "dataset": "rl",
        "title": "COT Folding Map - RL Training (Qwen3-4B)",
        "version": "2026-03-rl-v1",
        "defaultCheckpoint": "base",
        "n_checkpoints": len(checkpoints),
    }, indent=2)

    # checkpoints.json — basic info for each checkpoint
    ckpt_list = []
    for item in dynamics["trajectory"]:
        ckpt_list.append({
            "name": item["name"],
            "rl_step": item["rl_step"],
            "accuracy": item["accuracy"],
            "nfs_mean": item["NFS_mean"],
            "auroc": item["auroc"],
        })
    write_json(output_dir / "checkpoints.json", ckpt_list, indent=2)

    # trajectory.json — full trajectory + nfs details
    write_json(output_dir / "trajectory.json", {
        "dynamics": dynamics["trajectory"],
        "nfs": nfs_traj["checkpoints"],
        "metadata": dynamics["metadata"],
        "length_debiasing": dynamics["length_debiasing"],
    }, indent=2)

    # problems.meta.json — selected problems with categories
    write_json(output_dir / "problems.meta.json", {
        "problems": selected_problems,
    }, indent=2)


def export_checkpoint(
    engine: FoldingEngine,
    checkpoint: str,
    output_dir: Path,
    selected_problem_ids: set[str],
    problem_texts: dict[str, dict],
    skip_text: bool = False,
    text_only: bool = False,
    index_only: bool = False,
):
    """Export data for a single checkpoint."""
    ckpt_dir = output_dir / checkpoint

    if not text_only and not index_only:
        # overview.json
        print(f"\n  [{checkpoint}] Exporting overview.json")
        try:
            data = engine.get_batch_overview()
            write_json(ckpt_dir / "overview.json", data)
        except Exception as e:
            print(f"    [WARN] overview: {e}")

    # Load correctness data from nfs_analysis.json
    correctness = load_nfs_analysis(checkpoint)

    # problems.index.json — only selected problems (always re-export to include text)
    print(f"  [{checkpoint}] Exporting problems.index.json")
    all_problems = engine.get_problems()
    result = {"dataset": "rl", "checkpoint": checkpoint, "problems": []}

    for prob in all_problems:
        pid = prob["problem_id"]
        if pid not in selected_problem_ids:
            continue
        samples = engine.get_samples(pid)

        # Annotate each sample with is_correct
        n_correct = 0
        for s in samples:
            key = (pid, s["sample_id"])
            is_correct = correctness.get(key)
            s["is_correct"] = is_correct if is_correct is not None else None
            if is_correct:
                n_correct += 1

        info = problem_texts.get(pid, {})
        prompt = info.get("prompt", "")
        short_prompt = prompt[:80].replace("\n", " ")
        if len(prompt) > 80:
            short_prompt += "..."
        n_total = len(samples)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        result["problems"].append({
            "problem_id": pid,
            "n_samples": prob["n_samples"],
            "accuracy": round(accuracy, 4),
            "processing_time_s": prob.get("processing_time_s"),
            "short_prompt": short_prompt,
            "ground_truth": info.get("ground_truth", ""),
            "samples": samples,
        })
    write_json(ckpt_dir / "problems.index.json", result)

    if index_only:
        print(f"  [{checkpoint}] --index-only: skipping bundles/compare/text")
        return

    if not text_only:
        # compare/ — structural comparison per problem
        print(f"  [{checkpoint}] Exporting compare/")
        compare_dir = ckpt_dir / "compare"
        for prob in result["problems"]:
            pid = prob["problem_id"]
            try:
                cmp_data = engine.get_structural_comparison(pid)
                write_json(compare_dir / f"p{pid}.json", cmp_data)
            except Exception as e:
                print(f"    [WARN] compare p{pid}: {e}")

        # samples/ — bundles
        total = sum(p["n_samples"] for p in result["problems"])
        done = 0
        print(f"  [{checkpoint}] Exporting {total} sample bundles")
        for prob in result["problems"]:
            pid = prob["problem_id"]
            samples_dir = ckpt_dir / "samples" / f"p{pid}"
            for sample in prob["samples"]:
                sid = sample["sample_id"]
                done += 1
                try:
                    folding = engine.get_folding_data(pid, sid)
                    clustering = None
                    try:
                        clustering = engine.get_clustering(pid, sid)
                    except Exception:
                        pass

                    flow = None
                    try:
                        flow = engine.get_flow_data(pid, sid)
                    except Exception:
                        pass

                    functional = None
                    try:
                        functional = engine.get_functional_data(pid, sid)
                    except Exception:
                        pass

                    bundle = {
                        "problem_id": pid,
                        "sample_id": sid,
                        "folding": folding,
                        "clustering": clustering,
                        "flow": flow,
                        "functional": functional,
                    }
                    write_json(samples_dir / f"s{sid}.bundle.json", bundle)

                    if done % 100 == 0 or done == total:
                        print(f"    [{done}/{total}] bundles exported")

                except Exception as e:
                    print(f"    [ERROR] bundle p{pid}_s{sid}: {e}")

    # text export
    if not skip_text:
        export_checkpoint_texts(engine, checkpoint, ckpt_dir, result,
                                fallback_tokenizer=_shared_tokenizer)
    else:
        print(f"  [{checkpoint}] Skipping text export")


# Module-level shared tokenizer — set from the first engine that has one
_shared_tokenizer = None


def export_checkpoint_texts(
    engine: FoldingEngine,
    checkpoint: str,
    ckpt_dir: Path,
    problems_index: dict,
    fallback_tokenizer=None,
):
    """Export per-sample text files for a checkpoint."""
    global _shared_tokenizer
    if engine._reader is None:
        print(f"  [{checkpoint}] [SKIP] NAD reader not available")
        return
    tokenizer = engine._tokenizer or fallback_tokenizer
    if tokenizer is None:
        print(f"  [{checkpoint}] [SKIP] No tokenizer available")
        return
    # Save tokenizer for reuse by later checkpoints
    if _shared_tokenizer is None and tokenizer is not None:
        _shared_tokenizer = tokenizer

    total = sum(p["n_samples"] for p in problems_index["problems"])
    done = 0
    skipped = 0
    print(f"  [{checkpoint}] Exporting {total} text files")

    for prob in problems_index["problems"]:
        pid = prob["problem_id"]
        samples_dir = ckpt_dir / "samples" / f"p{pid}"

        for sample in prob["samples"]:
            sid = sample["sample_id"]
            n_slices = sample.get("n_slices", 0)
            done += 1

            try:
                reader = engine._reader
                token_start = reader.token_row_ptr[sid]
                token_end = reader.token_row_ptr[sid + 1]
                token_ids = reader.token_ids[token_start:token_end]

                full_text = tokenizer.decode(token_ids.tolist())

                items = []
                tokens_per_slice = 32
                for slice_idx in range(n_slices):
                    t_start = slice_idx * tokens_per_slice
                    t_end = min((slice_idx + 1) * tokens_per_slice, len(token_ids))
                    if t_start >= len(token_ids):
                        break

                    before_tokens = token_ids[:t_start].tolist()
                    before_text = tokenizer.decode(before_tokens) if before_tokens else ""
                    before_plus_current = tokenizer.decode(token_ids[:t_end].tolist())

                    char_start = len(before_text)
                    char_end = len(before_plus_current)

                    items.append({
                        "slice_idx": slice_idx,
                        "token_start": int(t_start),
                        "token_end": int(t_end),
                        "char_start": char_start,
                        "char_end": char_end,
                    })

                text_data = {
                    "problem_id": pid,
                    "sample_id": sid,
                    "unit_label": "slice",
                    "full_text": full_text,
                    "items": items,
                }
                write_json(samples_dir / f"s{sid}.text.json", text_data)

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    [WARN] text p{pid}_s{sid}: {e}")

            if done % 200 == 0 or done == total:
                print(f"    [{done}/{total}] texts processed ({skipped} skipped)")

    if skipped > 0:
        print(f"  [{checkpoint}] {skipped} text files skipped")


def main():
    parser = argparse.ArgumentParser(description="Export RL multi-checkpoint static data")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: public/data/rl)")
    parser.add_argument("--skip-text", action="store_true", help="Skip text export (faster)")
    parser.add_argument("--text-only", action="store_true",
                        help="Only export text + problems.index (skip bundles/compare)")
    parser.add_argument("--index-only", action="store_true",
                        help="Only re-export problems.index.json (skip bundles/compare/text)")
    parser.add_argument("--max-problems", type=int, default=15, help="Max problems to select (default: 15)")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Specific checkpoints to export (default: all)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "public" / "data" / "rl"

    print(f"Output dir: {output_dir}")
    print(f"Max problems: {args.max_problems}")

    # Load problem text from evaluation_report
    print("\n[Loading problem texts...]")
    problem_texts = load_problem_texts()

    # Load cross-checkpoint analysis data
    print("\n[Loading cross-checkpoint data...]")
    dynamics, nfs_traj = load_cross_checkpoint_data()

    # Select representative problems (with text enrichment)
    print("\n[Selecting representative problems...]")
    selected_problems = select_problems(dynamics, problem_texts, max_problems=args.max_problems)
    selected_ids = {p["id"] for p in selected_problems}

    # Determine checkpoints to export
    all_checkpoints = list_rl_checkpoints()
    checkpoints = args.checkpoints if args.checkpoints else all_checkpoints
    print(f"\nCheckpoints to export: {checkpoints}")

    # Export cross-checkpoint files
    output_dir.mkdir(parents=True, exist_ok=True)
    export_cross_checkpoint_files(output_dir, dynamics, nfs_traj, selected_problems, checkpoints)

    # Pre-load tokenizer from base checkpoint (all checkpoints use the same tokenizer)
    global _shared_tokenizer
    if not args.skip_text and not args.index_only and _shared_tokenizer is None:
        print("\n[Pre-loading tokenizer from base checkpoint...]")
        try:
            base_cache = resolve_rl_cache_path("base")
            base_engine = FoldingEngine(
                batch_dir=str(default_rl_batch_dir("base")),
                cache_path=str(base_cache),
                granularity="slice",
            )
            if base_engine._tokenizer is not None:
                _shared_tokenizer = base_engine._tokenizer
                print(f"  Tokenizer loaded: {type(_shared_tokenizer).__name__}")
            del base_engine
        except Exception as e:
            print(f"  [WARN] Failed to pre-load tokenizer: {e}")

    # Export per-checkpoint data
    t_start = time.time()
    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'='*60}")

        batch_dir = default_rl_batch_dir(ckpt)
        if not batch_dir.exists():
            print(f"  [SKIP] Batch dir not found: {batch_dir}")
            continue

        try:
            cache_path = resolve_rl_cache_path(ckpt)
        except FileNotFoundError as e:
            print(f"  [WARN] Cache not found, proceeding without text: {e}")
            cache_path = None

        print(f"  Batch dir:  {batch_dir}")
        print(f"  Cache path: {cache_path}")

        t0 = time.time()
        engine = FoldingEngine(
            batch_dir=str(batch_dir),
            cache_path=str(cache_path) if cache_path else None,
            granularity="slice",
        )
        print(f"  Engine initialized in {time.time() - t0:.1f}s")
        print(f"  Problems: {len(engine.get_problems())}")
        print(f"  NAD reader: {'OK' if engine._reader else 'MISSING'}")

        # Save first successful tokenizer for reuse
        if engine._tokenizer is not None and _shared_tokenizer is None:
            _shared_tokenizer = engine._tokenizer
            print(f"  Tokenizer saved for reuse by other checkpoints")

        export_checkpoint(
            engine, ckpt, output_dir, selected_ids,
            problem_texts=problem_texts,
            skip_text=args.skip_text,
            text_only=args.text_only,
            index_only=args.index_only,
        )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Export complete in {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
