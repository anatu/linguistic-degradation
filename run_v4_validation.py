"""
V4 validation probe: run 0% control at aggressive LR (1e-4) for 3 generations.

If novel-text log-probs stay flat, the V4 degradation finding is
contamination-driven. If they decline, we're overtraining.
"""

import os
import json
import time
import numpy as np
import torch

import config
from run_experiment_v4 import (
    V4_CONFIGS,
    ensure_dirs,
    run_v4_arm,
)

VALIDATION_RESULTS_PATH = os.path.join(config.RESULTS_DIR, "all_results_v4_validation.jsonl")


def main():
    t_start = time.time()
    ensure_dirs()
    device = config.DEVICE
    seeds = config.SEEDS[:config.NUM_SEEDS]

    print(f"Device: {device}")
    print(f"V4 Validation Probe: 0% control at LR=1e-4, 5000 steps, 3 generations")
    print(f"Seeds: {seeds}")

    # Clear previous validation results
    if os.path.exists(VALIDATION_RESULTS_PATH):
        os.remove(VALIDATION_RESULTS_PATH)

    run_v4_arm(
        "aggressive_lr", V4_CONFIGS["aggressive_lr"],
        [0.0], config.NUM_GENERATIONS, seeds, device, VALIDATION_RESULTS_PATH,
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"VALIDATION PROBE COMPLETE in {elapsed / 3600:.1f} hours")
    print(f"Results saved to: {VALIDATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
