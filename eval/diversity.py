"""
Generation diversity metrics (Distinct-N).
Measures output distribution breadth via n-gram diversity.
"""

import os
import sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.gpt2_finetune import load_checkpoint, generate_tokens


def distinct_n(token_lists, n):
    """Compute Distinct-N: ratio of unique n-grams to total n-grams."""
    total_ngrams = Counter()
    for tokens in token_lists:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            total_ngrams[ngram] += 1
    total = sum(total_ngrams.values())
    unique = len(total_ngrams)
    return unique / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_diversity(model_or_path, device=None):
    """Generate samples and compute Distinct-1,2,3."""
    device = device or config.DEVICE

    if isinstance(model_or_path, str):
        model = load_checkpoint(model_or_path, device=device)
    else:
        model = model_or_path
    model.eval()

    base_data = np.load(os.path.join(config.DATA_DIR, "base_train.npy"))
    gen_cfg = config.GENERATE
    prompt_len = config.DIVERSITY_PROMPT_LEN
    gen_len = config.DIVERSITY_GEN_LEN
    num_samples = config.DIVERSITY_NUM_SAMPLES

    all_generated = []
    for i in range(num_samples):
        start = np.random.randint(0, len(base_data) - prompt_len)
        prompt = torch.tensor(
            base_data[start : start + prompt_len].astype(np.int64)
        ).unsqueeze(0).to(device)
        output = generate_tokens(
            model, prompt,
            max_new_tokens=gen_len,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            device=device,
        )
        generated = output[0, prompt_len:].cpu().tolist()
        all_generated.append(generated)

    d1 = distinct_n(all_generated, 1)
    d2 = distinct_n(all_generated, 2)
    d3 = distinct_n(all_generated, 3)

    return {
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_3": d3,
        "num_samples": num_samples,
        "gen_length": gen_len,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    result = evaluate_diversity(args.checkpoint)
    print(f"Distinct-1: {result['distinct_1']:.4f}, Distinct-2: {result['distinct_2']:.4f}, Distinct-3: {result['distinct_3']:.4f}")
