"""
Evaluate a model on the linguistic novelty benchmark.
Measures comprehension via multiple-choice cloze: which option gets highest log-prob?
"""

import os
import sys
import json
import numpy as np
import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.gpt2_finetune import load_checkpoint


def load_benchmark(path=None):
    path = path or config.BENCHMARK_PATH
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


@torch.no_grad()
def score_option(model, tokenizer, context_text, option_text, device):
    """Compute the mean log-probability of option tokens given the context."""
    context_ids = tokenizer.encode(context_text)
    option_ids = tokenizer.encode(" " + option_text)

    full_ids = context_ids + option_ids
    max_len = model.config.n_positions
    if len(full_ids) > max_len:
        full_ids = full_ids[-max_len:]
        option_start = len(full_ids) - len(option_ids)
    else:
        option_start = len(context_ids)

    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(device)
    outputs = model(input_ids)
    log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)

    option_log_probs = []
    for i, token_id in enumerate(option_ids):
        pos = option_start + i - 1  # logits at pos predict token at pos+1
        if 0 <= pos < log_probs.size(0):
            option_log_probs.append(log_probs[pos, token_id].item())

    return np.mean(option_log_probs) if option_log_probs else float("-inf")


def evaluate_novelty(model_or_path, device=None, benchmark_path=None):
    """Evaluate novelty benchmark. Accepts a model or checkpoint path."""
    device = device or config.DEVICE
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)

    if isinstance(model_or_path, str):
        model = load_checkpoint(model_or_path, device=device)
    else:
        model = model_or_path
    model.eval()

    examples = load_benchmark(benchmark_path)
    correct = 0
    total = 0
    log_probs_correct = []
    results_by_category = {}

    for ex in examples:
        context = ex["context"].replace("___", "")
        scores = {}
        for option in ex["options"]:
            scores[option] = score_option(model, tokenizer, context, option, device)

        predicted = max(scores, key=scores.get)
        is_correct = predicted == ex["correct"]
        correct += int(is_correct)
        total += 1
        log_probs_correct.append(scores[ex["correct"]])

        cat = ex.get("category", "unknown")
        if cat not in results_by_category:
            results_by_category[cat] = {"correct": 0, "total": 0}
        results_by_category[cat]["correct"] += int(is_correct)
        results_by_category[cat]["total"] += 1

    accuracy = correct / total if total > 0 else 0.0
    mean_log_prob = np.mean(log_probs_correct) if log_probs_correct else float("-inf")

    category_accuracy = {
        cat: d["correct"] / d["total"] for cat, d in results_by_category.items()
    }

    return {
        "novelty_accuracy": accuracy,
        "novelty_log_prob": mean_log_prob,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    results = evaluate_novelty(args.checkpoint)
    print(json.dumps(results, indent=2))
