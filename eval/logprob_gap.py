"""
Log-prob gap metric: measures how much worse the model is at predicting tokens
in novel linguistic constructions compared to standard text.

A widening gap under contamination = evidence for distribution stagnation.
A constant gap = contamination causes uniform degradation, not specific competence loss.
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


@torch.no_grad()
def mean_logprob_on_tokens(model, input_ids, device):
    """Compute mean per-token log-probability for a sequence.

    For each token at position t (t >= 1), compute log P(token_t | token_0..t-1).
    Returns the mean over all predicted positions.
    """
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    max_len = model.config.n_positions
    if input_ids.size(1) > max_len:
        input_ids = input_ids[:, :max_len]

    outputs = model(input_ids)
    logits = outputs.logits  # (1, T, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # For each position t, logits[0, t] predicts token at t+1
    # So we compare logits[0, 0:T-1] against input_ids[0, 1:T]
    targets = input_ids[0, 1:]  # (T-1,)
    preds = log_probs[0, :-1]  # (T-1, vocab)

    token_log_probs = preds[torch.arange(len(targets)), targets]  # (T-1,)
    return token_log_probs.mean().item(), len(targets)


def compute_standard_logprob(model, device, val_data_path=None, num_passages=None, passage_length=None):
    """Compute mean log-prob on standard (validation) text passages."""
    val_data_path = val_data_path or os.path.join(config.DATA_DIR, "base_val.npy")
    num_passages = num_passages or config.LOGPROB_NUM_STANDARD_PASSAGES
    passage_length = passage_length or config.LOGPROB_PASSAGE_LENGTH

    data = np.load(val_data_path)
    total_logprob = 0.0
    total_tokens = 0

    for i in range(num_passages):
        start = np.random.randint(0, len(data) - passage_length - 1)
        passage = torch.tensor(data[start:start + passage_length].astype(np.int64))
        lp, n = mean_logprob_on_tokens(model, passage, device)
        total_logprob += lp * n
        total_tokens += n

    return total_logprob / total_tokens if total_tokens > 0 else float("-inf")


def _char_to_token_offsets(tokenizer, sentence, char_start, char_end):
    """Map character offsets to token offsets via prefix encoding.

    Returns (token_start, token_end) where the span covers
    tokens[token_start:token_end] in the full sentence encoding.
    """
    prefix_ids = tokenizer.encode(sentence[:char_start])
    full_prefix_ids = tokenizer.encode(sentence[:char_end])
    return len(prefix_ids), len(full_prefix_ids)


@torch.no_grad()
def mean_logprob_on_span(model, input_ids, span_start_tok, span_end_tok, device):
    """Compute mean per-token log-probability over a token span.

    Runs the model on the full input_ids (for left-context), but only
    scores tokens in positions [span_start_tok, span_end_tok).
    """
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    max_len = model.config.n_positions
    if input_ids.size(1) > max_len:
        input_ids = input_ids[:, :max_len]

    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # logits[0, t] predicts token at t+1
    # To score token at position p, we read log_probs[0, p-1, token_id]
    # Clamp span to valid prediction range: tokens at positions 1..T-1 can be scored
    score_start = max(span_start_tok, 1)
    score_end = min(span_end_tok, input_ids.size(1))

    if score_start >= score_end:
        return float("-inf"), 0

    targets = input_ids[0, score_start:score_end]
    preds = log_probs[0, score_start - 1:score_end - 1]
    token_log_probs = preds[torch.arange(len(targets)), targets]

    return token_log_probs.mean().item(), len(targets)


def compute_novel_logprob_critical(model, device, tokenizer, benchmark_path=None):
    """Compute mean log-prob on critical spans of novel construction sentences."""
    benchmark_path = benchmark_path or config.BENCHMARK_PATH

    examples = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    total_logprob = 0.0
    total_tokens = 0
    category_stats = {}
    skipped = 0

    for ex in examples:
        sentence = ex["context"].replace("___", ex["correct"])
        critical_span = ex.get("critical_span")
        if not critical_span:
            skipped += 1
            continue

        char_start = sentence.find(critical_span)
        if char_start == -1:
            # Try case-insensitive match for spans that start with capital
            lower_start = sentence.lower().find(critical_span.lower())
            if lower_start != -1:
                char_start = lower_start
            else:
                print(f"  WARNING: critical_span not found in sentence: {critical_span!r}")
                skipped += 1
                continue

        char_end = char_start + len(critical_span)
        span_start_tok, span_end_tok = _char_to_token_offsets(tokenizer, sentence, char_start, char_end)

        input_ids = torch.tensor(tokenizer.encode(sentence), dtype=torch.long)
        if len(input_ids) < 2:
            skipped += 1
            continue

        lp, n = mean_logprob_on_span(model, input_ids, span_start_tok, span_end_tok, device)
        if n == 0:
            skipped += 1
            continue

        total_logprob += lp * n
        total_tokens += n

        cat = ex.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"logprob_sum": 0.0, "token_count": 0}
        category_stats[cat]["logprob_sum"] += lp * n
        category_stats[cat]["token_count"] += n

    if skipped > 0:
        print(f"  Critical-span eval: skipped {skipped}/{len(examples)} items")

    mean_novel = total_logprob / total_tokens if total_tokens > 0 else float("-inf")

    category_logprobs = {}
    for cat, stats in category_stats.items():
        if stats["token_count"] > 0:
            category_logprobs[cat] = stats["logprob_sum"] / stats["token_count"]

    return mean_novel, category_logprobs


def compute_novel_logprob(model, device, tokenizer, benchmark_path=None):
    """Compute mean log-prob on novel construction sentences (correct completions filled in)."""
    benchmark_path = benchmark_path or config.BENCHMARK_PATH

    examples = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    total_logprob = 0.0
    total_tokens = 0
    category_stats = {}  # cat -> {logprob_sum, token_count}

    for ex in examples:
        # Fill in the correct answer to get the complete sentence
        sentence = ex["context"].replace("___", ex["correct"])
        input_ids = torch.tensor(tokenizer.encode(sentence), dtype=torch.long)

        if len(input_ids) < 2:
            continue

        lp, n = mean_logprob_on_tokens(model, input_ids, device)
        total_logprob += lp * n
        total_tokens += n

        cat = ex.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"logprob_sum": 0.0, "token_count": 0}
        category_stats[cat]["logprob_sum"] += lp * n
        category_stats[cat]["token_count"] += n

    mean_novel = total_logprob / total_tokens if total_tokens > 0 else float("-inf")

    category_logprobs = {}
    for cat, stats in category_stats.items():
        if stats["token_count"] > 0:
            category_logprobs[cat] = stats["logprob_sum"] / stats["token_count"]

    return mean_novel, category_logprobs


def evaluate_logprob_gap(model_or_path, device=None):
    """Compute the full log-prob gap metric.

    Returns dict with:
        logprob_standard: mean log-prob on standard val text
        logprob_novel: mean log-prob on novel constructions
        logprob_gap: standard - novel (positive = novel is harder)
        category_gaps: per-category gap values
    """
    device = device or config.DEVICE
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)

    if isinstance(model_or_path, str):
        model = load_checkpoint(model_or_path, device=device)
    else:
        model = model_or_path
    model.eval()

    lp_standard = compute_standard_logprob(model, device)
    lp_novel, category_logprobs = compute_novel_logprob(model, device, tokenizer)

    gap = lp_standard - lp_novel  # positive means novel is harder

    category_gaps = {
        cat: lp_standard - cat_lp
        for cat, cat_lp in category_logprobs.items()
    }

    return {
        "logprob_standard": lp_standard,
        "logprob_novel": lp_novel,
        "logprob_gap": gap,
        "category_gaps": category_gaps,
    }


def evaluate_logprob_gap_v3(model_or_path, device=None):
    """Compute both full-sentence and critical-span log-prob gap metrics.

    Returns dict with all v2 keys plus:
        logprob_novel_critical: mean log-prob on critical spans only
        logprob_gap_critical: standard - novel_critical
        category_gaps_critical: per-category critical-span gaps
    """
    device = device or config.DEVICE
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)

    if isinstance(model_or_path, str):
        model = load_checkpoint(model_or_path, device=device)
    else:
        model = model_or_path
    model.eval()

    lp_standard = compute_standard_logprob(model, device)
    lp_novel, category_logprobs = compute_novel_logprob(model, device, tokenizer)
    lp_novel_critical, category_logprobs_critical = compute_novel_logprob_critical(model, device, tokenizer)

    gap = lp_standard - lp_novel
    gap_critical = lp_standard - lp_novel_critical

    category_gaps = {
        cat: lp_standard - cat_lp
        for cat, cat_lp in category_logprobs.items()
    }
    category_gaps_critical = {
        cat: lp_standard - cat_lp
        for cat, cat_lp in category_logprobs_critical.items()
    }

    return {
        "logprob_standard": lp_standard,
        "logprob_novel": lp_novel,
        "logprob_gap": gap,
        "category_gaps": category_gaps,
        "logprob_novel_critical": lp_novel_critical,
        "logprob_gap_critical": gap_critical,
        "category_gaps_critical": category_gaps_critical,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v3", action="store_true", help="Include critical-span metrics")
    args = parser.parse_args()
    if args.v3:
        results = evaluate_logprob_gap_v3(args.checkpoint)
    else:
        results = evaluate_logprob_gap(args.checkpoint)
    print(json.dumps(results, indent=2))
