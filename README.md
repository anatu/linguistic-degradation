# Distribution Stagnation PoC

Proof-of-concept experiment testing whether training language models on synthetic-contaminated data degrades their ability to comprehend novel linguistic constructions.

## Status: V4 Experiment Complete

- **V1 (from-scratch training):** Abandoned — 30M model had no baseline competence.
- **V2 (fine-tuning GPT-2 124M, 500 steps):** Complete. At moderate contamination (10-50%), the log-prob gap widens through distribution narrowing, not degraded novel-text processing.
- **V3 (critical-span metric):** Complete. Scoring only construction-defining tokens amplifies the gap but confirms the same qualitative finding as V2.
- **V4 (extended fine-tuning, 5000 steps):** Complete. Deeper training breaks through the pretrained buffer. At 100% contamination with LR=1e-4, novel-text comprehension degrades monotonically (-7.79 → -9.01 → -9.95 nats). At 50% with LR=5e-5, a small but real signal emerges (-7.90 → -7.97).
- **Phase 1 publication-readiness fixes (2026-04-12):** val/test split (separate `base_test.npy` for final metrics vs. `base_val.npy` for checkpoint selection), seeded log-prob passage sampling, and per-generation seed offsetting at 0% contamination so the control arm has genuine variance. See `plans/V4_IMPROVEMENTS.md`.
- **Phase 2 V2 multi-seed replication (2026-04-13):** Complete. 54 cells at n=3 (seeds 42, 137, 256), 14.5h wall-clock on MPS. Contamination effects are 5–50× their standard deviation; all V2 qualitative findings survive. V4 multi-seed rerun (~110h) not yet launched.

**Key findings:**
1. At shallow fine-tuning (500 steps), synthetic contamination narrows what models *generate* (D-1: 0.42 → 0.04) without affecting what they *comprehend* — a generation-comprehension asymmetry.
2. At deeper fine-tuning (5000 steps), contamination does degrade comprehension of novel constructions. The pretrained representations act as a buffer, not a permanent shield.

## Project Structure

```
├── plans/                # Research plans and improvement roadmaps
│   ├── PLAN.md           # V1 research plan (from-scratch training, abandoned)
│   ├── PLAN_v2.md        # V2 research plan (fine-tuning GPT-2)
│   └── V4_IMPROVEMENTS.md # Publication-readiness improvement plan
├── config.py             # Experiment configuration
├── train.py              # Fine-tuning loop (GPT-2 + gradient accumulation)
├── run_experiment.py     # V2 orchestrator
├── reeval_v3.py          # V3 re-evaluation script (critical-span metric)
├── run_experiment_v4.py  # V4 orchestrator (extended fine-tuning)
├── analyze.py            # Plot generation and analysis (supports --v3, --v4)
├── model/
│   ├── gpt.py            # Custom small GPT (V1, kept for reference)
│   └── gpt2_finetune.py  # HuggingFace GPT-2 wrapper (V2+)
├── data/
│   ├── prepare_base.py   # Download and tokenize WikiText-103 (produces base_train/val/test)
│   ├── generate_synthetic.py  # Generate synthetic text from checkpoints
│   └── mix_data.py       # Mix human + synthetic data at ratios
├── eval/
│   ├── novelty_benchmark.py   # Multiple-choice cloze evaluation
│   ├── perplexity.py          # Validation perplexity
│   ├── diversity.py           # Distinct-N metrics
│   └── logprob_gap.py         # Log-prob gap metric (full-sentence + critical-span)
├── benchmark/
│   └── novelty_examples.jsonl # 51 novel linguistic constructions (with critical_span annotations)
├── kb/                   # Project knowledge base
│   ├── v1_experiment_findings.md
│   ├── v2_experiment_findings.md
│   ├── v3_experiment_findings.md
│   ├── v4_experiment_findings.md
│   ├── benchmark_critique.md
│   └── lit_review.md
├── report/
│   ├── main.tex          # LaTeX results report
│   ├── references.bib    # Bibliography
│   └── main.pdf          # Compiled report
└── results/
    ├── all_results.jsonl      # V2 raw results
    ├── all_results_v3.jsonl   # V3 results (critical-span re-evaluation)
    ├── all_results_v4.jsonl   # V4 results (extended fine-tuning)
    ├── checkpoints/           # Model checkpoints
    ├── logs/                  # Training logs
    └── plots/                 # Generated figures
```

## Running Experiments

```bash
# Use conda base Python (has torch + transformers)
PYTHON=/opt/anaconda3/bin/python

# V2: original experiment (~12 hours on MPS)
$PYTHON run_experiment.py

# V3: re-evaluate checkpoints with critical-span metric (~1 min)
$PYTHON reeval_v3.py

# V4: extended fine-tuning stress test (~30 hours on MPS)
$PYTHON run_experiment_v4.py

# Generate plots
$PYTHON analyze.py          # V2 plots
$PYTHON analyze.py --v3     # V3 plots (includes critical-span comparison)
$PYTHON analyze.py --v4     # V4 plots
```

## Dependencies

```
torch>=2.0
transformers>=4.35
tiktoken
numpy
datasets
matplotlib
seaborn
scipy
tqdm
```
