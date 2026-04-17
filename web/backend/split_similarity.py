#!/usr/bin/env python3
"""
Split similarity_b64 out of each *.bundle.json into a separate *.sim.b64 file.

For every public/data/aime24/samples/p{pid}/s{sid}.bundle.json:
  1. Extract folding.similarity_b64 → write to s{sid}.sim.b64 (plain text)
  2. Delete folding.similarity_b64 from the bundle
  3. Set folding.has_similarity_file = true
  4. Overwrite the bundle in-place (smaller)

This is idempotent: if has_similarity_file is already set, the bundle is skipped.
"""

import json
import os
import sys
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "public" / "data" / "aime24" / "samples"


def split_one(bundle_path: Path) -> dict:
    """Split a single bundle. Returns stats dict."""
    with open(bundle_path, "r") as f:
        bundle = json.load(f)

    folding = bundle.get("folding", {})

    # Already split — skip
    if folding.get("has_similarity_file"):
        return {"status": "skipped", "path": str(bundle_path)}

    sim_b64 = folding.get("similarity_b64")
    if not sim_b64:
        return {"status": "no_sim", "path": str(bundle_path)}

    # Write similarity to sidecar file: s{sid}.bundle.json → s{sid}.sim.b64
    sim_path = bundle_path.parent / bundle_path.name.replace(".bundle.json", ".sim.b64")
    with open(sim_path, "w") as f:
        f.write(sim_b64)

    # Patch bundle
    del folding["similarity_b64"]
    folding["has_similarity_file"] = True

    orig_size = bundle_path.stat().st_size
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    new_size = bundle_path.stat().st_size

    return {
        "status": "split",
        "path": str(bundle_path),
        "sim_path": str(sim_path),
        "orig_size": orig_size,
        "new_size": new_size,
        "sim_len": len(sim_b64),
        "reduction": f"{(1 - new_size / orig_size) * 100:.1f}%",
    }


def main():
    if not SAMPLES_DIR.is_dir():
        print(f"ERROR: samples dir not found: {SAMPLES_DIR}", file=sys.stderr)
        sys.exit(1)

    bundles = sorted(SAMPLES_DIR.glob("*/s*.bundle.json"))
    print(f"Found {len(bundles)} bundle files in {SAMPLES_DIR}\n")

    total_orig = 0
    total_new = 0
    split_count = 0

    for bp in bundles:
        result = split_one(bp)
        status = result["status"]
        if status == "split":
            split_count += 1
            total_orig += result["orig_size"]
            total_new += result["new_size"]
            print(f"  SPLIT {bp.name}: {result['orig_size']//1024}KB → {result['new_size']//1024}KB ({result['reduction']} smaller)")
        elif status == "skipped":
            print(f"  SKIP  {bp.name} (already split)")
        else:
            print(f"  NOSIM {bp.name}")

    print(f"\nDone: {split_count} bundles split out of {len(bundles)} total")
    if split_count > 0:
        print(f"Total: {total_orig // 1024}KB → {total_new // 1024}KB (saved {(total_orig - total_new) // 1024}KB)")


if __name__ == "__main__":
    main()
