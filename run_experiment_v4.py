"""
V4 experiment: stress-test pretrained buffer with extended fine-tuning.

Tests whether 10x longer fine-tuning (5000 steps vs 500) on contaminated data
can break through GPT-2's pretrained representations and degrade novel-text
comprehension. Reuses all v2/v3 infrastructure, only changes config values.

Arms:
  - Primary: 5000 steps at LR=5e-5 for ratios [0%, 50%, 100%] x 3 generations
  - Aggressive LR: 5000 steps at LR=1e-4 for ratio=100% only x 3 generations
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch

import config
from data.generate_synthetic import generate_synthetic
from data.mix_data import mix_data
from train import train
from eval.novelty_benchmark import evaluate_novelty
from eval.perplexity import evaluate_perplexity
from eval.diversity import evaluate_diversity
from eval.logprob_gap import evaluate_logprob_gap_v3


# ── V4 configurations ──────────────────────────────────────────────────────────

V4_CONFIGS = {
    "primary": {
        "max_steps": 5000,
        "warmup_steps": 500,
        "eval_interval": 250,
        "learning_rate": 5e-5,
    },
    "aggressive_lr": {
        "max_steps": 5000,
        "warmup_steps": 500,
        "eval_interval": 250,
        "learning_rate": 1e-4,
    },
}

V4_RATIOS = [0.0, 0.50, 1.0]
V4_AGGRESSIVE_LR_RATIOS = [1.0]  # only test aggressive LR on 100%
V4_RESULTS_PATH = os.path.join(config.RESULTS_DIR, "all_results_v4.jsonl")


def ensure_dirs():
    for d in [config.RESULTS_DIR, config.CHECKPOINT_DIR, config.LOG_DIR, config.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def append_result(result, results_path):
    with open(results_path, "a") as f:
        f.write(json.dumps(result, default=convert) + "\n")


def run_full_eval_v4(checkpoint_path, device):
    """Run all evaluation metrics including v3 critical-span."""
    print(f"\n--- Evaluating {os.path.basename(checkpoint_path)} ---")
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    novelty = evaluate_novelty(checkpoint_path, device=device)
    ppl = evaluate_perplexity(checkpoint_path, device=device)
    diversity = evaluate_diversity(checkpoint_path, device=device)
    gap = evaluate_logprob_gap_v3(checkpoint_path, device=device)

    results = {**novelty, **ppl, **diversity, **gap}
    print(f"  novelty_accuracy:    {results['novelty_accuracy']:.4f}")
    print(f"  val_perplexity:      {results['val_perplexity']:.2f}")
    print(f"  logprob_gap (full):  {results['logprob_gap']:.4f}")
    print(f"  logprob_gap (crit):  {results['logprob_gap_critical']:.4f}")
    print(f"  distinct_1/2/3:      {results['distinct_1']:.4f} / {results['distinct_2']:.4f} / {results['distinct_3']:.4f}")
    return results


def patch_config(v4_config):
    """Apply v4 config overrides to config.FINETUNE."""
    for k, v in v4_config.items():
        config.FINETUNE[k] = v


def run_v4_arm(config_name, v4_config, ratios, num_generations, seeds, device, results_path):
    """Run a set of fixed-ratio arms with v4 config overrides."""
    original_finetune = dict(config.FINETUNE)
    patch_config(v4_config)

    val_data_path = os.path.join(config.DATA_DIR, "base_val.npy")

    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"V4 [{config_name}] FIXED-RATIO ARM: contamination={ratio:.0%}, steps={v4_config['max_steps']}, lr={v4_config['learning_rate']}")
        print(f"{'='*60}")

        for gen in range(num_generations):
            arm_tag = f"v4_{config_name}_fixed_{ratio:.2f}"

            if gen > 0 and ratio > 0:
                # Generate synthetic from previous generation's checkpoint
                prev_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen - 1}_seed{seeds[0]}.pt")
                base_tokens = np.load(os.path.join(config.DATA_DIR, "base_train.npy"))
                if ratio < 1.0:
                    n_synthetic = int(len(base_tokens) * ratio / (1.0 - ratio))
                else:
                    n_synthetic = len(base_tokens)
                synth_path = os.path.join(config.DATA_DIR, f"synthetic_{arm_tag}_gen{gen}.npy")
                if not os.path.exists(synth_path):
                    generate_synthetic(prev_ckpt, n_synthetic, synth_path, device=device)

                train_data = os.path.join(config.DATA_DIR, f"mixed_{arm_tag}_gen{gen}.npy")
                if not os.path.exists(train_data):
                    mix_data(synth_path, ratio, train_data, seed=config.SEED + gen)

            for seed in seeds:
                print(f"\n--- Gen {gen}, Ratio {ratio:.0%}, Seed {seed}, Config {config_name} ---")
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen}_seed{seed}.pt")
                log_path = os.path.join(config.LOG_DIR, f"{arm_tag}_gen{gen}_seed{seed}.json")

                if gen == 0 or ratio == 0:
                    train_data_path = os.path.join(config.DATA_DIR, "base_train.npy")
                else:
                    train_data_path = os.path.join(config.DATA_DIR, f"mixed_{arm_tag}_gen{gen}.npy")

                # Skip if checkpoint already exists (resumability)
                if os.path.exists(ckpt_path):
                    print(f"  Checkpoint exists, skipping training: {ckpt_path}")
                else:
                    train(train_data_path, val_data_path, ckpt_path, log_path, device=device, seed=seed)

                eval_results = run_full_eval_v4(ckpt_path, device)
                eval_results["generation"] = gen
                eval_results["ratio"] = ratio
                eval_results["seed"] = seed
                eval_results["arm"] = "fixed"
                eval_results["v4_config"] = config_name
                eval_results["v4_max_steps"] = v4_config["max_steps"]
                eval_results["v4_learning_rate"] = v4_config["learning_rate"]
                append_result(eval_results, results_path)

    # Restore original config
    config.FINETUNE.update(original_finetune)


def main():
    t_start = time.time()
    ensure_dirs()
    device = config.DEVICE
    seeds = config.SEEDS[:config.NUM_SEEDS]

    print(f"Device: {device}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Seeds: {seeds}")
    print(f"V4 ratios: {V4_RATIOS}")
    print(f"V4 configs: {list(V4_CONFIGS.keys())}")

    # Prepare base data if needed
    base_train_path = os.path.join(config.DATA_DIR, "base_train.npy")
    if not os.path.exists(base_train_path):
        from data.prepare_base import download_and_tokenize
        print("\n--- Preparing base corpus ---")
        download_and_tokenize()

    # Clear previous v4 results
    if os.path.exists(V4_RESULTS_PATH):
        os.remove(V4_RESULTS_PATH)

    # Phase 1: Primary config (5000 steps, LR=5e-5) for all ratios
    print("\n" + "#" * 60)
    print("PHASE 1: Primary config (5000 steps, LR=5e-5)")
    print("#" * 60)
    run_v4_arm(
        "primary", V4_CONFIGS["primary"],
        V4_RATIOS, config.NUM_GENERATIONS, seeds, device, V4_RESULTS_PATH,
    )

    # Phase 2: Aggressive LR (5000 steps, LR=1e-4) for 100% only
    print("\n" + "#" * 60)
    print("PHASE 2: Aggressive LR (5000 steps, LR=1e-4) — 100% only")
    print("#" * 60)
    run_v4_arm(
        "aggressive_lr", V4_CONFIGS["aggressive_lr"],
        V4_AGGRESSIVE_LR_RATIOS, config.NUM_GENERATIONS, seeds, device, V4_RESULTS_PATH,
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V4 EXPERIMENT COMPLETE in {elapsed / 3600:.1f} hours")
    print(f"Results saved to: {V4_RESULTS_PATH}")
    print(f"Run `python analyze.py --v4` to generate plots.")


if __name__ == "__main__":
    main()
