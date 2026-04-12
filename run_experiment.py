"""
Orchestrator: runs the full generational fine-tuning experiment (v2).

Arms:
  1. Fixed-ratio arms: for each contamination ratio, run NUM_GENERATIONS generations
     where each gen fine-tunes fresh GPT-2 on data with synthetic from gen-1's model.
  2. Compounding arm: fixed 25% contamination, each gen fine-tunes from the prior
     generation's model (not fresh GPT-2).

Each arm × generation is replicated across NUM_SEEDS seeds.
Results are saved incrementally as JSONL.
"""

import os
import sys
import json
import time
import numpy as np
import torch

import config
from data.prepare_base import download_and_tokenize
from data.generate_synthetic import generate_synthetic
from data.mix_data import mix_data
from train import train
from eval.novelty_benchmark import evaluate_novelty
from eval.perplexity import evaluate_perplexity
from eval.diversity import evaluate_diversity
from eval.logprob_gap import evaluate_logprob_gap


def ensure_dirs():
    for d in [config.RESULTS_DIR, config.CHECKPOINT_DIR, config.LOG_DIR, config.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


def run_full_eval(checkpoint_path, device):
    """Run all evaluation metrics on a checkpoint."""
    print(f"\n--- Evaluating {checkpoint_path} ---")
    # Reset eval seeds for consistency
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    novelty = evaluate_novelty(checkpoint_path, device=device)
    ppl = evaluate_perplexity(checkpoint_path, device=device)
    diversity = evaluate_diversity(checkpoint_path, device=device)
    gap = evaluate_logprob_gap(checkpoint_path, device=device)

    results = {**novelty, **ppl, **diversity, **gap}
    print(f"  novelty_accuracy:  {results['novelty_accuracy']:.4f}")
    print(f"  val_perplexity:    {results['val_perplexity']:.2f}")
    print(f"  logprob_gap:       {results['logprob_gap']:.4f} (std={results['logprob_standard']:.4f} novel={results['logprob_novel']:.4f})")
    print(f"  distinct_1/2/3:    {results['distinct_1']:.4f} / {results['distinct_2']:.4f} / {results['distinct_3']:.4f}")
    return results


def append_result(result, results_path):
    """Append a result to the JSONL file."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "a") as f:
        f.write(json.dumps(result, default=convert) + "\n")


def run_fixed_ratio_arm(ratio, num_generations, seeds, device, results_path):
    """Run a fixed-ratio arm: each generation fine-tunes fresh GPT-2."""
    print(f"\n{'='*60}")
    print(f"FIXED-RATIO ARM: contamination={ratio:.0%}")
    print(f"{'='*60}")

    val_data_path = config.VAL_DATA_PATH

    for gen in range(num_generations):
        # For gen > 0, generate synthetic data from the previous gen's first-seed checkpoint
        # (use seed[0] as the canonical model for synthetic generation)
        arm_tag = f"fixed_{ratio:.2f}"

        if gen > 0 and ratio > 0:
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
            print(f"\n--- Gen {gen}, Ratio {ratio:.0%}, Seed {seed} ---")
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen}_seed{seed}.pt")
            log_path = os.path.join(config.LOG_DIR, f"{arm_tag}_gen{gen}_seed{seed}.json")

            if gen == 0 or ratio == 0:
                train_data_path = os.path.join(config.DATA_DIR, "base_train.npy")
            else:
                train_data_path = os.path.join(config.DATA_DIR, f"mixed_{arm_tag}_gen{gen}.npy")

            # 0% control: vary the seed per generation so gen0/gen1/gen2 aren't
            # a deterministic identity (same data + same seed -> same checkpoint).
            # Offset keeps nominal seed in filenames but gives real batch-order variance.
            effective_seed = seed + 1000 * gen if ratio == 0 else seed
            train(train_data_path, val_data_path, ckpt_path, log_path, device=device, seed=effective_seed)

            eval_results = run_full_eval(ckpt_path, device)
            eval_results["generation"] = gen
            eval_results["ratio"] = ratio
            eval_results["seed"] = seed
            eval_results["effective_seed"] = effective_seed
            eval_results["arm"] = "fixed"
            append_result(eval_results, results_path)

        # Cleanup: remove previous gen's checkpoints for non-canonical seeds
        # Keep seed[0] (needed for next gen's synthetic) and current gen's checkpoints
        if gen > 0:
            for seed in seeds[1:]:
                prev_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen - 1}_seed{seed}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    print(f"  Cleaned up: {prev_ckpt}")


def run_compounding_arm(ratio, num_generations, seeds, device, results_path):
    """Compounding arm: each gen fine-tunes from prior gen's model (not fresh GPT-2)."""
    print(f"\n{'='*60}")
    print(f"COMPOUNDING ARM: ratio={ratio:.0%}")
    print(f"{'='*60}")

    val_data_path = config.VAL_DATA_PATH

    for gen in range(num_generations):
        arm_tag = "compound"

        if gen > 0:
            prev_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen - 1}_seed{seeds[0]}.pt")
            base_tokens = np.load(os.path.join(config.DATA_DIR, "base_train.npy"))
            n_synthetic = int(len(base_tokens) * ratio / (1.0 - ratio))
            synth_path = os.path.join(config.DATA_DIR, f"synthetic_{arm_tag}_gen{gen}.npy")
            if not os.path.exists(synth_path):
                generate_synthetic(prev_ckpt, n_synthetic, synth_path, device=device)

            train_data = os.path.join(config.DATA_DIR, f"mixed_{arm_tag}_gen{gen}.npy")
            if not os.path.exists(train_data):
                mix_data(synth_path, ratio, train_data, seed=config.SEED + gen)

        for seed in seeds:
            print(f"\n--- Compounding Gen {gen}, Seed {seed} ---")
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{arm_tag}_gen{gen}_seed{seed}.pt")
            log_path = os.path.join(config.LOG_DIR, f"{arm_tag}_gen{gen}_seed{seed}.json")

            if gen == 0:
                train_data_path = os.path.join(config.DATA_DIR, "base_train.npy")
            else:
                train_data_path = os.path.join(config.DATA_DIR, f"mixed_{arm_tag}_gen{gen}.npy")

            train(train_data_path, val_data_path, ckpt_path, log_path, device=device, seed=seed)

            eval_results = run_full_eval(ckpt_path, device)
            eval_results["generation"] = gen
            eval_results["ratio"] = ratio
            eval_results["seed"] = seed
            eval_results["arm"] = "compound"
            append_result(eval_results, results_path)


def main():
    t_start = time.time()
    ensure_dirs()
    device = config.DEVICE
    seeds = config.SEEDS[:config.NUM_SEEDS]
    print(f"Device: {device}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Contamination ratios: {config.CONTAMINATION_RATIOS}")
    print(f"Generations per arm: {config.NUM_GENERATIONS}")
    print(f"Seeds: {seeds}")

    # Step 1: Prepare base data if needed
    base_train_path = os.path.join(config.DATA_DIR, "base_train.npy")
    if not os.path.exists(base_train_path):
        print("\n--- Preparing base corpus ---")
        download_and_tokenize()
    else:
        print(f"Base data already exists: {base_train_path}")

    results_path = os.path.join(config.RESULTS_DIR, "all_results.jsonl")

    # Step 2: Run fixed-ratio arms
    for ratio in config.CONTAMINATION_RATIOS:
        run_fixed_ratio_arm(ratio, config.NUM_GENERATIONS, seeds, device, results_path)

    # Step 3: Run compounding arm
    run_compounding_arm(config.COMPOUNDING_RATIO, config.NUM_GENERATIONS, seeds, device, results_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE in {elapsed / 3600:.1f} hours")
    print(f"Results saved to: {results_path}")
    print(f"Run `python analyze.py` to generate plots.")


if __name__ == "__main__":
    main()
