"""
HuggingFace GPT-2 wrapper for the distribution stagnation experiment.
Provides a consistent interface for training, evaluation, and generation.
"""

import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_pretrained(model_name="gpt2"):
    """Load a pretrained GPT-2 model and tokenizer."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name}: {n_params / 1e6:.1f}M parameters")
    return model, tokenizer


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load a fine-tuned model from our checkpoint format."""
    # Always load to CPU first to avoid MPS alignment issues
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name", "gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def save_checkpoint(model, checkpoint_path, model_name="gpt2", step=0, val_loss=None, extra=None):
    """Save a checkpoint in our format."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    data = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "step": step,
        "val_loss": val_loss,
    }
    if extra:
        data.update(extra)
    torch.save(data, checkpoint_path)


@torch.no_grad()
def generate_tokens(model, prompt_ids, max_new_tokens, temperature=1.0, top_p=0.95, device="cpu"):
    """Generate tokens using nucleus sampling. Returns full sequence including prompt."""
    model.eval()
    idx = prompt_ids.to(device)
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)

    for _ in range(max_new_tokens):
        # Truncate to model's max position embeddings if needed
        max_len = model.config.n_positions
        idx_cond = idx if idx.size(1) <= max_len else idx[:, -max_len:]

        outputs = model(idx_cond)
        logits = outputs.logits[:, -1, :] / temperature

        # Nucleus (top-p) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs_sorted = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
        mask = cumulative_probs - probs_sorted >= top_p
        sorted_logits[mask] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        idx_in_sorted = torch.multinomial(probs, num_samples=1)
        idx_next = sorted_indices.gather(1, idx_in_sorted)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
