"""
Configuration for the Distribution Stagnation PoC experiment (v2).
Fine-tuning GPT-2 with contaminated data.
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benchmark", "novelty_examples.jsonl")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
SEEDS = [42, 137, 256]  # for multi-seed replication

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"          # HuggingFace model name ("gpt2" = 124M, "gpt2-medium" = 355M)
TOKENIZER_NAME = "gpt2"      # tiktoken encoding name (used by data prep)
BLOCK_SIZE = 256              # reduced from 1024 to fit MPS memory

# ── Fine-Tuning ──────────────────────────────────────────────────────────────
FINETUNE = dict(
    batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    max_steps=500,
    warmup_steps=50,
    eval_interval=50,
    eval_steps=20,
    grad_clip=1.0,
    lr_min=1e-6,
    gradient_accumulation_steps=8,  # effective batch = 32
)

# ── Data ─────────────────────────────────────────────────────────────────────
BASE_CORPUS = "wikitext-103"
VAL_FRACTION = 0.05
MAX_BASE_TOKENS = 2_000_000    # reduced for tractable synthetic generation

# ── Experiment sweep ─────────────────────────────────────────────────────────
CONTAMINATION_RATIOS = [0.0, 0.10, 0.25, 0.50, 1.0]
NUM_GENERATIONS = 3
NUM_SEEDS = 1                  # single seed first for fast directional signal
COMPOUNDING_RATIO = 0.25

# ── Synthetic generation ─────────────────────────────────────────────────────
GENERATE = dict(
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=256,
)

# ── Diversity evaluation ─────────────────────────────────────────────────────
DIVERSITY_NUM_SAMPLES = 50
DIVERSITY_PROMPT_LEN = 10
DIVERSITY_GEN_LEN = 100

# ── Log-prob gap evaluation ──────────────────────────────────────────────────
LOGPROB_NUM_STANDARD_PASSAGES = 200  # random val passages for standard log-prob
LOGPROB_PASSAGE_LENGTH = 128         # tokens per passage

# ── V3/V4 results ────────────────────────────────────────────────────────────
RESULTS_V3_PATH = os.path.join(RESULTS_DIR, "all_results_v3.jsonl")
RESULTS_V4_PATH = os.path.join(RESULTS_DIR, "all_results_v4.jsonl")
