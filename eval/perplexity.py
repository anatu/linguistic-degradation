"""
Standard validation perplexity evaluation.
"""

import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.gpt2_finetune import load_checkpoint


@torch.no_grad()
def evaluate_perplexity(model_or_path, val_data_path=None, device=None, num_batches=100):
    """Compute perplexity on validation data."""
    device = device or config.DEVICE
    val_data_path = val_data_path or config.TEST_DATA_PATH

    if isinstance(model_or_path, str):
        model = load_checkpoint(model_or_path, device=device)
    else:
        model = model_or_path
    model.eval()

    data = np.load(val_data_path)
    block_size = config.BLOCK_SIZE
    batch_size = config.FINETUNE["batch_size"]

    losses = []
    for _ in range(num_batches):
        ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)
        outputs = model(x, labels=y)
        losses.append(outputs.loss.item())

    mean_loss = np.mean(losses)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"val_loss": mean_loss, "val_perplexity": ppl}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    result = evaluate_perplexity(args.checkpoint)
    print(f"val_loss: {result['val_loss']:.4f}, val_perplexity: {result['val_perplexity']:.2f}")
