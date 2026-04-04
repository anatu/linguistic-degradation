"""
Mix base (human) and synthetic data at a given contamination ratio.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def mix_data(synthetic_path, contamination_ratio, output_path, seed=None):
    """Create a mixed training set: (1-ratio)*human + ratio*synthetic."""
    if seed is not None:
        np.random.seed(seed)

    base_tokens = np.load(os.path.join(config.DATA_DIR, "base_train.npy"))
    total_tokens = len(base_tokens)

    if contamination_ratio == 0.0:
        np.save(output_path, base_tokens)
        print(f"Pure human data: {total_tokens:,} tokens -> {output_path}")
        return output_path

    synthetic_tokens = np.load(synthetic_path)

    # Calculate how many tokens of each
    n_synthetic = int(total_tokens * contamination_ratio)
    n_human = total_tokens - n_synthetic

    # Sample from each source
    if len(synthetic_tokens) < n_synthetic:
        # Repeat synthetic data if not enough
        repeats = (n_synthetic // len(synthetic_tokens)) + 1
        synthetic_tokens = np.tile(synthetic_tokens, repeats)
    synthetic_sample = synthetic_tokens[:n_synthetic]
    human_sample = base_tokens[:n_human]

    # Interleave by chunking and shuffling chunks (preserves local coherence)
    chunk_size = config.BLOCK_SIZE
    human_chunks = [human_sample[i : i + chunk_size] for i in range(0, len(human_sample) - chunk_size, chunk_size)]
    synth_chunks = [synthetic_sample[i : i + chunk_size] for i in range(0, len(synthetic_sample) - chunk_size, chunk_size)]
    all_chunks = human_chunks + synth_chunks
    np.random.shuffle(all_chunks)

    mixed = np.concatenate(all_chunks).astype(np.uint16)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.save(output_path, mixed)
    print(f"Mixed data: {n_human:,} human + {n_synthetic:,} synthetic ({contamination_ratio:.0%}) = {len(mixed):,} tokens -> {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mix_data(args.synthetic, args.ratio, args.output)
