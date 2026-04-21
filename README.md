# Distribution Narrowing Under Synthetic Data Contamination

Multi-seed experiment testing whether training language models on synthetic-contaminated data degrades their ability to comprehend novel linguistic constructions.

## Status: Paper Draft Complete

- **V2 (fine-tuning GPT-2 124M, 500 steps):** Complete, n=3 seeds. Moderate contamination (10-50%) narrows distribution without degrading novel-text comprehension.
- **V3 (critical-span metric):** Complete. Scoring only construction-defining tokens amplifies the contamination signal.
- **V4 (extended fine-tuning, 5000 steps):** Complete, n=3 seeds (60.2h on MPS). Deeper training breaks the pretrained buffer — gap widens to 2.30±0.09 (p<0.001) at 100% contamination.
- **Paper:** 16-page draft at `report/main.tex` covering all results with error bars, 10 figures, per-category appendix, expanded related work.

**Key findings:**
1. At shallow fine-tuning (500 steps), synthetic contamination narrows what models *generate* (D-1: 0.43 → 0.04) without affecting what they *comprehend* — a generation-comprehension asymmetry.
2. At deeper fine-tuning (5000 steps), contamination breaks through pretrained representations and degrades comprehension of novel constructions (novel-text log-prob: -7.16 → -8.09 nats at 100% contamination gen 1).
3. Generation diversity (Distinct-N) is the earliest warning signal, collapsing even at 10% contamination.
4. Log-prob gap metrics become unreliable above PPL ~1400 due to degenerate collapse.

## Project Structure

```
├── plans/                # Research plans and improvement roadmaps
│   └── V4_IMPROVEMENTS.md # Publication-readiness improvement plan (9/10 items done)
├── config.py             # Experiment configuration
├── train.py              # Fine-tuning loop (GPT-2 + gradient accumulation)
├── run_experiment.py     # V2 orchestrator (shallow fine-tuning sweep)
├── run_experiment_v4.py  # V4 orchestrator (deep fine-tuning stress test)
├── reeval_v3.py          # V3 re-evaluation script (critical-span metric)
├── reeval_v4_on_test.py  # Step 1c: re-evaluate V4 checkpoints on base_test.npy
├── analyze.py            # Plots, tables, stats, training curves, per-category analysis
├── requirements.txt      # Pinned dependencies
├── model/
│   └── gpt2_finetune.py  # HuggingFace GPT-2 wrapper
├── data/
│   ├── prepare_base.py   # Download and tokenize WikiText-103 (90/5/5 train/val/test)
│   ├── generate_synthetic.py  # Generate synthetic text from checkpoints
│   └── mix_data.py       # Mix human + synthetic data at ratios
├── eval/
│   ├── novelty_benchmark.py   # Multiple-choice cloze evaluation
│   ├── perplexity.py          # Validation perplexity (on base_test.npy)
│   ├── diversity.py           # Distinct-N metrics
│   └── logprob_gap.py         # Log-prob gap metric (full-sentence + critical-span)
├── benchmark/
│   └── novelty_examples.jsonl # 51 novel linguistic constructions (with critical_span)
├── report/
│   ├── main.tex          # LaTeX paper (16 pages, Natu & Claude Opus 4.6)
│   ├── references.bib    # Bibliography (10 references)
│   └── main.pdf          # Compiled PDF
└── results/
    ├── all_results.jsonl           # V2 multi-seed results (54 entries, n=3)
    ├── all_results_v3.jsonl        # V3 results (critical-span re-evaluation)
    ├── all_results_v4.jsonl        # V4 multi-seed results (36 entries, n=3)
    ├── all_results_v4_reeval_test.jsonl  # Step 1c re-eval on base_test.npy
    ├── checkpoints/                # Model checkpoints (~475MB each)
    ├── logs/                       # Training logs (step-by-step JSON)
    └── plots/                      # Generated figures (13 PNGs)
```

## Running Experiments

```bash
# Use conda base Python (has torch + transformers)
PYTHON=/opt/anaconda3/bin/python

# V2: shallow fine-tuning sweep (~14.5 hours on MPS, n=3)
$PYTHON run_experiment.py

# V4: deep fine-tuning stress test (~60 hours on MPS, n=3)
$PYTHON run_experiment_v4.py

# Generate all plots and analysis
$PYTHON analyze.py --v4

# Compile paper
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Dependencies

See `requirements.txt` for pinned versions. Key packages:

```
Python 3.13+
torch==2.10.0
transformers==5.3.0
tiktoken==0.12.0
numpy>=2.3.0
```

Tested on Apple Silicon (MPS backend) with block_size=256, batch_size=4.
