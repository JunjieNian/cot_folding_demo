#!/usr/bin/env python3
"""
Static Data Exporter for COT Folding Map — AIME24 only.

Reads from FoldingEngine and exports all data as static JSON files.

Output structure:
  data/aime24/
    app.json
    overview.json
    problems.index.json
    compare/p{pid}.json
    samples/p{pid}/s{sid}.bundle.json
    samples/p{pid}/s{sid}.text.json

Usage:
  python backend/export_static_aime24.py [--batch-dir PATH] [--cache PATH] [--output-dir PATH]
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

from project_paths import default_batch_dir_for_benchmark, resolve_cache_path
from folding_engine import FoldingEngine


def load_nfs_analysis(batch_dir: Path) -> tuple[dict[tuple, bool], dict]:
    """Load nfs_analysis.json for correctness + ground_truth lookup.

    Returns:
      correctness: {(problem_id, sample_id): is_correct}
      ground_truths: {problem_id: answer_str}  (from correct samples)
    """
    nfs_path = batch_dir / "nfs_analysis.json"
    if not nfs_path.exists():
        print(f"  [WARN] nfs_analysis.json not found: {nfs_path}")
        return {}, {}
    with open(nfs_path) as f:
        data = json.load(f)
    correctness = {}
    ground_truths: dict = {}
    for s in data.get("samples", []):
        key = (s["problem_id"], s["sample_id"])
        is_correct = bool(s.get("is_correct", False))
        correctness[key] = is_correct
        if is_correct and s.get("answer") is not None and s["problem_id"] not in ground_truths:
            ground_truths[s["problem_id"]] = str(s["answer"])
    print(f"  Loaded is_correct for {len(correctness)} samples, "
          f"ground_truth for {len(ground_truths)} problems")
    return correctness, ground_truths


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
    print(f"  -> {path.relative_to(path.parent.parent.parent.parent)} ({size_kb:.1f} KB)")


def export_app_json(output_dir: Path):
    """Export app-level configuration."""
    write_json(output_dir / "app.json", {
        "dataset": "aime24",
        "title": "COT Folding Map - AIME24",
        "version": "2026-03-static-v1",
        "defaultProblemId": 1,
        "defaultSampleId": 0,
    }, indent=2)


def export_overview(engine: FoldingEngine, output_dir: Path):
    """Export batch overview."""
    print("\n[Exporting overview.json]")
    data = engine.get_batch_overview()
    write_json(output_dir / "overview.json", data)


def export_problems_index(engine: FoldingEngine, output_dir: Path,
                          correctness: dict | None = None,
                          ground_truths: dict | None = None):
    """Export combined problems + samples index."""
    print("\n[Exporting problems.index.json]")
    problems = engine.get_problems()
    result = {
        "dataset": "aime24",
        "problems": [],
    }
    for prob in problems:
        pid = prob["problem_id"]
        samples = engine.get_samples(pid)

        # Annotate samples with is_correct
        n_correct = 0
        if correctness:
            for s in samples:
                key = (pid, s["sample_id"])
                is_correct = correctness.get(key)
                s["is_correct"] = is_correct if is_correct is not None else None
                if is_correct:
                    n_correct += 1

        n_total = len(samples)
        accuracy = n_correct / n_total if n_total > 0 and correctness else None

        entry = {
            "problem_id": pid,
            "n_samples": prob["n_samples"],
            "processing_time_s": prob.get("processing_time_s"),
            "samples": samples,
        }
        if accuracy is not None:
            entry["accuracy"] = round(accuracy, 4)
        gt = ground_truths.get(pid) if ground_truths else None
        if gt is not None:
            entry["ground_truth"] = gt
        result["problems"].append(entry)
    write_json(output_dir / "problems.index.json", result)
    return result


def export_compare(engine: FoldingEngine, output_dir: Path, problem_ids: list):
    """Export per-problem structural comparison files."""
    print(f"\n[Exporting compare/ for {len(problem_ids)} problems]")
    compare_dir = output_dir / "compare"
    for pid in problem_ids:
        try:
            data = engine.get_structural_comparison(pid)
            write_json(compare_dir / f"p{pid}.json", data)
        except Exception as e:
            print(f"  [WARN] compare p{pid}: {e}")


def export_sample_bundles(engine: FoldingEngine, output_dir: Path, problems_index: dict):
    """Export per-sample bundle files (folding + clustering + flow + functional)."""
    print(f"\n[Exporting sample bundles]")
    total = sum(p["n_samples"] for p in problems_index["problems"])
    done = 0

    for prob in problems_index["problems"]:
        pid = prob["problem_id"]
        samples_dir = output_dir / "samples" / f"p{pid}"

        for sample in prob["samples"]:
            sid = sample["sample_id"]
            done += 1

            try:
                folding = engine.get_folding_data(pid, sid)
                clustering = engine.get_clustering(pid, sid)

                # Try to get flow and functional data
                try:
                    flow = engine.get_flow_data(pid, sid)
                except Exception:
                    flow = None
                try:
                    functional = engine.get_functional_data(pid, sid)
                except Exception:
                    functional = None

                bundle = {
                    "problem_id": pid,
                    "sample_id": sid,
                    "folding": folding,
                    "clustering": clustering,
                    "flow": flow,
                    "functional": functional,
                }
                write_json(samples_dir / f"s{sid}.bundle.json", bundle)

                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] bundles exported")

            except Exception as e:
                print(f"  [ERROR] bundle p{pid}_s{sid}: {e}")


def export_sample_texts(engine: FoldingEngine, output_dir: Path, problems_index: dict):
    """Export per-sample text files."""
    print(f"\n[Exporting sample texts]")

    if engine._reader is None or engine._tokenizer is None:
        print("  [SKIP] NAD reader or tokenizer not available — text export skipped")
        return

    total = sum(p["n_samples"] for p in problems_index["problems"])
    done = 0
    skipped = 0

    for prob in problems_index["problems"]:
        pid = prob["problem_id"]
        samples_dir = output_dir / "samples" / f"p{pid}"

        for sample in prob["samples"]:
            sid = sample["sample_id"]
            n_slices = sample.get("n_slices", 0)
            done += 1

            try:
                # Get the full token sequence for this sample
                reader = engine._reader
                tokenizer = engine._tokenizer
                token_start = reader.token_row_ptr[sid]
                token_end = reader.token_row_ptr[sid + 1]
                token_ids = reader.token_ids[token_start:token_end]

                full_text = tokenizer.decode(token_ids.tolist())

                # Compute character boundaries for each slice
                items = []
                tokens_per_slice = 32  # default slice size
                for slice_idx in range(n_slices):
                    t_start = slice_idx * tokens_per_slice
                    t_end = min((slice_idx + 1) * tokens_per_slice, len(token_ids))
                    if t_start >= len(token_ids):
                        break

                    # Decode segments to find char boundaries
                    before_tokens = token_ids[:t_start].tolist()
                    current_tokens = token_ids[t_start:t_end].tolist()

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
                    print(f"  [WARN] text p{pid}_s{sid}: {e}")

            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] texts processed ({skipped} skipped)")

    if skipped > 0:
        print(f"  [INFO] {skipped} text files skipped due to errors")


def main():
    parser = argparse.ArgumentParser(description="Export AIME24 static data")
    parser.add_argument("--batch-dir", default=None, help="Override batch_results directory")
    parser.add_argument("--cache", default=None, help="Override cache directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: public/data/aime24)")
    parser.add_argument("--skip-text", action="store_true", help="Skip text export (faster)")
    parser.add_argument("--index-only", action="store_true",
                        help="Only re-export problems.index.json (skip bundles/compare/text)")
    args = parser.parse_args()

    # Resolve paths
    batch_dir = Path(args.batch_dir) if args.batch_dir else default_batch_dir_for_benchmark("aime24")
    cache_path = resolve_cache_path(
        args.cache,
        default_benchmark="aime24",
        default_cache_name="cache_neuron_output_1_act_no_rms_20250902_025610",
        required=False,
    )

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "public" / "data" / "aime24"

    print(f"Batch dir:  {batch_dir}")
    print(f"Cache path: {cache_path}")
    print(f"Output dir: {output_dir}")

    if not batch_dir.exists():
        print(f"\n[ERROR] Batch directory not found: {batch_dir}")
        sys.exit(1)

    # Initialize engine
    print("\n[Initializing FoldingEngine...]")
    t0 = time.time()
    engine = FoldingEngine(
        batch_dir=str(batch_dir),
        cache_path=str(cache_path) if cache_path else None,
        granularity="slice",
    )
    print(f"  Engine initialized in {time.time() - t0:.1f}s")
    print(f"  Problems: {len(engine.get_problems())}")
    print(f"  NAD reader: {'OK' if engine._reader else 'MISSING'}")
    print(f"  Tokenizer:  {'OK' if engine._tokenizer else 'MISSING'}")

    # Load correctness data
    print("\n[Loading correctness data...]")
    correctness, ground_truths = load_nfs_analysis(batch_dir)

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if not args.index_only:
        export_app_json(output_dir)
        export_overview(engine, output_dir)

    problems_index = export_problems_index(engine, output_dir, correctness, ground_truths)

    if args.index_only:
        print("\n[--index-only: skipping bundles/compare/text]")
    else:
        problem_ids = [p["problem_id"] for p in problems_index["problems"]]
        export_compare(engine, output_dir, problem_ids)
        export_sample_bundles(engine, output_dir, problems_index)

        if not args.skip_text:
            export_sample_texts(engine, output_dir, problems_index)
        else:
            print("\n[Skipping text export (--skip-text)]")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Export complete in {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
