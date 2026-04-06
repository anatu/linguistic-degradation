"""
Re-evaluate all v2 checkpoints with the v3 critical-span log-prob gap metric.

No retraining needed — loads existing checkpoints, computes the new metric,
and merges with v2 results into a new JSONL file.
"""

import os
import json
import time
import re
import numpy as np
import torch

import config
from eval.logprob_gap import evaluate_logprob_gap_v3


def load_v2_results():
    """Load v2 results indexed by (arm, ratio, generation, seed)."""
    path = os.path.join(config.RESULTS_DIR, "all_results.jsonl")
    index = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = (r["arm"], r["ratio"], r["generation"], r["seed"])
                index[key] = r
    return index


def parse_checkpoint_name(filename):
    """Parse arm, ratio, generation, seed from checkpoint filename.

    Patterns:
        fixed_0.25_gen1_seed42.pt -> (fixed, 0.25, 1, 42)
        compound_gen0_seed42.pt   -> (compound, 0.25, 0, 42)
    """
    name = filename.replace(".pt", "")

    m = re.match(r"fixed_([\d.]+)_gen(\d+)_seed(\d+)", name)
    if m:
        return "fixed", float(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.match(r"compound_gen(\d+)_seed(\d+)", name)
    if m:
        return "compound", config.COMPOUNDING_RATIO, int(m.group(1)), int(m.group(2))

    return None


def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Re-evaluating checkpoints with v3 critical-span metric\n")

    v2_results = load_v2_results()
    checkpoints = sorted(os.listdir(config.CHECKPOINT_DIR))
    checkpoints = [f for f in checkpoints if f.endswith(".pt")]

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Found {len(v2_results)} v2 results\n")

    results_path = config.RESULTS_V3_PATH
    # Clear previous v3 results if any
    if os.path.exists(results_path):
        os.remove(results_path)

    t_start = time.time()

    for i, ckpt_file in enumerate(checkpoints):
        parsed = parse_checkpoint_name(ckpt_file)
        if parsed is None:
            print(f"  Skipping unrecognized checkpoint: {ckpt_file}")
            continue

        arm, ratio, gen, seed = parsed
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_file)
        print(f"[{i + 1}/{len(checkpoints)}] {ckpt_file} (arm={arm}, ratio={ratio}, gen={gen}, seed={seed})")

        # Reset eval seeds for consistency with v2
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        gap_results = evaluate_logprob_gap_v3(ckpt_path, device=device)

        # Merge with v2 results
        key = (arm, ratio, gen, seed)
        if key in v2_results:
            merged = dict(v2_results[key])
            # Add new v3 fields
            merged["logprob_novel_critical"] = gap_results["logprob_novel_critical"]
            merged["logprob_gap_critical"] = gap_results["logprob_gap_critical"]
            merged["category_gaps_critical"] = gap_results["category_gaps_critical"]
        else:
            print(f"  WARNING: no v2 result found for {key}, saving gap-only result")
            merged = {
                "arm": arm,
                "ratio": ratio,
                "generation": gen,
                "seed": seed,
                **gap_results,
            }

        with open(results_path, "a") as f:
            f.write(json.dumps(merged, default=convert) + "\n")

        print(f"  gap_v2={gap_results['logprob_gap']:.4f}  gap_v3_critical={gap_results['logprob_gap_critical']:.4f}")
        print(f"  novel_full={gap_results['logprob_novel']:.4f}  novel_critical={gap_results['logprob_novel_critical']:.4f}")
        print()

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s. Results saved to: {results_path}")


if __name__ == "__main__":
    main()
