"""
Training loop for the distribution stagnation experiment (v2).
Supports fine-tuning pretrained GPT-2 with gradient accumulation.
"""

import os
import json
import time
import math
import numpy as np
import torch

import config
from model.gpt2_finetune import load_pretrained, save_checkpoint


class TokenDataset:
    """Simple dataset that serves random contiguous chunks from a token array."""

    def __init__(self, data_path, block_size):
        self.data = np.load(data_path)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size, device):
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, size=(batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, lr, lr_min):
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return lr_min
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr - lr_min)


@torch.no_grad()
def estimate_loss(model, dataset, eval_steps, batch_size, device):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(batch_size, device)
        outputs = model(x, labels=y)
        losses.append(outputs.loss.item())
    model.train()
    return np.mean(losses)


def train(
    train_data_path,
    val_data_path,
    checkpoint_path,
    log_path,
    device=None,
    seed=None,
    model_name=None,
):
    """Fine-tune pretrained GPT-2 on the given data. Returns the trained model."""
    device = device or config.DEVICE
    seed = seed if seed is not None else config.SEED
    model_name = model_name or config.MODEL_NAME
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    tc = config.FINETUNE
    block_size = config.BLOCK_SIZE

    train_dataset = TokenDataset(train_data_path, block_size)
    val_dataset = TokenDataset(val_data_path, block_size)

    model, _ = load_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.95),
    )

    grad_accum = tc["gradient_accumulation_steps"]

    log = {
        "train_data": train_data_path,
        "val_data": val_data_path,
        "model_name": model_name,
        "config_finetune": tc,
        "device": device,
        "seed": seed,
        "steps": [],
    }

    best_val_loss = float("inf")
    t0 = time.time()

    print(f"Fine-tuning {model_name} on {device} for {tc['max_steps']} steps (grad_accum={grad_accum})...")
    optimizer.zero_grad(set_to_none=True)

    for step in range(tc["max_steps"]):
        lr = get_lr(step, tc["warmup_steps"], tc["max_steps"], tc["learning_rate"], tc["lr_min"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        accum_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = train_dataset.get_batch(tc["batch_size"], device)
            outputs = model(x, labels=y)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += outputs.loss.item()
        accum_loss /= grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % tc["eval_interval"] == 0 or step == tc["max_steps"] - 1:
            val_loss = estimate_loss(model, val_dataset, tc["eval_steps"], tc["batch_size"], device)
            elapsed = time.time() - t0

            log["steps"].append({
                "step": step,
                "train_loss": accum_loss,
                "val_loss": val_loss,
                "lr": lr,
                "elapsed": elapsed,
            })

            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"step {step:5d} | train_loss {accum_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f} | lr {lr:.2e} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, checkpoint_path,
                    model_name=model_name,
                    step=step,
                    val_loss=val_loss,
                )

    # Save log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. Best val_loss: {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.2f})")
    return model, log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default=os.path.join(config.DATA_DIR, "base_train.npy"))
    parser.add_argument("--val_data", default=os.path.join(config.DATA_DIR, "base_val.npy"))
    parser.add_argument("--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "gen0.pt"))
    parser.add_argument("--log", default=os.path.join(config.LOG_DIR, "gen0.json"))
    args = parser.parse_args()
    train(args.train_data, args.val_data, args.checkpoint, args.log)
