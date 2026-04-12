"""
Download and preprocess the base corpus (WikiText-103).
Tokenizes with tiktoken GPT-2 encoding, saves as numpy arrays.
"""

import os
import sys
import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_and_tokenize():
    from datasets import load_dataset

    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    enc = tiktoken.get_encoding(config.TOKENIZER_NAME)

    def tokenize_split(split_name):
        texts = ds[split_name]["text"]
        all_tokens = []
        for text in texts:
            text = text.strip()
            if text:
                all_tokens.extend(enc.encode(text, allowed_special=set()))
        return np.array(all_tokens, dtype=np.uint16)

    # Tokenize train split
    print("Tokenizing train split...")
    train_tokens = tokenize_split("train")
    if len(train_tokens) > config.MAX_BASE_TOKENS:
        print(f"Capping train tokens from {len(train_tokens):,} to {config.MAX_BASE_TOKENS:,}")
        train_tokens = train_tokens[: config.MAX_BASE_TOKENS]
    print(f"Train tokens: {len(train_tokens):,}")

    # Tokenize validation split (use the dataset's own val split) and split it
    # in half: first half -> val (checkpoint selection), second half -> test (final metrics).
    # This prevents leakage between the data used for best-checkpoint selection
    # and the data used to report perplexity / log-prob metrics.
    print("Tokenizing validation split...")
    val_all = tokenize_split("validation")
    split_idx = len(val_all) // 2
    val_tokens = val_all[:split_idx]
    test_tokens = val_all[split_idx:]
    print(f"Validation tokens: {len(val_tokens):,}  Test tokens: {len(test_tokens):,}")

    # Save
    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_path = config.TRAIN_DATA_PATH
    val_path = config.VAL_DATA_PATH
    test_path = config.TEST_DATA_PATH
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    np.save(test_path, test_tokens)
    print(f"Saved: {train_path} ({train_tokens.nbytes / 1e6:.1f} MB)")
    print(f"Saved: {val_path} ({val_tokens.nbytes / 1e6:.1f} MB)")
    print(f"Saved: {test_path} ({test_tokens.nbytes / 1e6:.1f} MB)")

    return train_path, val_path, test_path


if __name__ == "__main__":
    download_and_tokenize()
