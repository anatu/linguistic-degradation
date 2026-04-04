"""
Generate synthetic text from a fine-tuned model checkpoint.
Uses HuggingFace's generate() with KV cache for speed.
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.gpt2_finetune import load_checkpoint


def generate_synthetic(checkpoint_path, num_tokens, output_path, device=None, batch_size=4):
    """Generate synthetic tokens from a checkpoint using HF generate with KV cache."""
    device = device or config.DEVICE
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    gen_cfg = config.GENERATE
    tokens_per_seq = gen_cfg["max_new_tokens"]
    tokens_per_batch = tokens_per_seq * batch_size
    num_batches = (num_tokens // tokens_per_batch) + 1
    prompt_len = 10

    base_data = np.load(os.path.join(config.DATA_DIR, "base_train.npy"))

    all_tokens = []
    total_generated = 0

    print(f"Generating {num_tokens:,} tokens in {num_batches} batches of {batch_size} (KV cached)...")
    for _ in tqdm(range(num_batches), desc="Generating"):
        starts = np.random.randint(0, len(base_data) - prompt_len, size=(batch_size,))
        prompts = torch.stack([
            torch.tensor(base_data[s:s + prompt_len].astype(np.int64))
            for s in starts
        ]).to(device)

        # Use HF generate with KV cache — much faster than manual loop
        with torch.no_grad():
            output = model.generate(
                prompts,
                max_new_tokens=tokens_per_seq,
                do_sample=True,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                pad_token_id=model.config.eos_token_id,
            )

        generated = output[:, prompt_len:].cpu().numpy()
        all_tokens.append(generated.reshape(-1))
        total_generated += generated.size

        if total_generated >= num_tokens:
            break

    all_tokens = np.concatenate(all_tokens)[:num_tokens].astype(np.uint16)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_tokens)
    print(f"Saved {len(all_tokens):,} synthetic tokens to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_synthetic(args.checkpoint, args.num_tokens, args.output)
