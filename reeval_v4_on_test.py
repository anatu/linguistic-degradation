"""
Step 1c: Re-evaluate existing V4 checkpoints against base_test.npy.

Confirms that the V4 signal (deeper training breaks pretrained buffer)
survives the val/test split before committing to ~110h multi-seed rerun.

Saves results to results/all_results_v4_reeval_test.jsonl
"""

import os
import json
import time
import numpy as np
import torch

import config
from run_experiment_v4 import run_full_eval_v4, convert

CHECKPOINT_DIR = config.CHECKPOINT_DIR
OUTPUT_PATH = os.path.join(config.RESULTS_DIR, "all_results_v4_reeval_test.jsonl")


def parse_checkpoint_name(fname):
    """Extract metadata from checkpoint filename.

    Format: v4_{config}_fixed_{ratio}_gen{gen}_seed{seed}.pt
    """
    base = fname.replace(".pt", "")
    parts = base.split("_fixed_")
    v4_config = parts[0].replace("v4_", "")  # "primary" or "aggressive_lr"
    rest = parts[1]  # "0.00_gen0_seed42"
    ratio_str, gen_str, seed_str = rest.split("_")
    return {
        "v4_config": v4_config,
        "ratio": float(ratio_str),
        "generation": int(gen_str.replace("gen", "")),
        "seed": int(seed_str.replace("seed", "")),
    }


def main():
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Test data: {config.TEST_DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # Collect all V4 checkpoints
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("v4_") and f.endswith(".pt")
    ])
    print(f"\nFound {len(checkpoints)} V4 checkpoints to re-evaluate.\n")

    # Clear output file
    if os.path.exists(OUTPUT_PATH):
        os.rename(OUTPUT_PATH, OUTPUT_PATH + ".bak")
        print(f"Backed up existing results to {OUTPUT_PATH}.bak")

    t_start = time.time()
    for i, fname in enumerate(checkpoints, 1):
        path = os.path.join(CHECKPOINT_DIR, fname)
        meta = parse_checkpoint_name(fname)
        print(f"\n[{i}/{len(checkpoints)}] {fname}")
        print(f"  config={meta['v4_config']}  ratio={meta['ratio']}  gen={meta['generation']}  seed={meta['seed']}")

        results = run_full_eval_v4(path, device)
        results.update(meta)
        results["arm"] = "fixed"
        results["eval_data"] = "base_test.npy"

        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(results, default=convert) + "\n")

        elapsed = time.time() - t_start
        avg_per = elapsed / i
        remaining = avg_per * (len(checkpoints) - i)
        print(f"  [{elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining]")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {len(checkpoints)} checkpoints re-evaluated in {total/60:.1f} minutes.")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
