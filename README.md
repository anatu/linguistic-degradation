# Distribution Stagnation PoC

Proof-of-concept experiment testing whether training language models on synthetic-contaminated data degrades their ability to comprehend novel linguistic constructions.

## Status: V4 Experiment Ready to Run

- **V1 (from-scratch training):** Abandoned — 30M model had no baseline competence.
- **V2 (fine-tuning GPT-2 124M, 500 steps):** Complete. At moderate contamination (10-50%), the log-prob gap widens through distribution narrowing, not degraded novel-text processing. Only 100% contamination degrades novel-text comprehension, coinciding with catastrophic model collapse.
- **V3 (critical-span metric):** Complete. Refined the log-prob gap to score only construction-defining tokens. Amplifies the signal but confirms the same qualitative finding as V2.
- **V4 (extended fine-tuning, 5000 steps):** Infrastructure built, ready to run. Stress-tests whether 10x deeper fine-tuning can break through GPT-2's pretrained representations.

**Key finding so far:** Synthetic contamination dramatically narrows what models *generate* (D-1 drops from 0.42 to 0.04) without meaningfully affecting what they *comprehend*. Log-prob scoring is a single forward pass that can still recognize novel patterns; generation is autoregressive sampling that systematically avoids low-probability paths in a peakier distribution.

## Project Structure

```
├── PLAN.md               # V1 research plan (from-scratch training, abandoned)
├── PLAN_v2.md            # V2 research plan (fine-tuning GPT-2)
├── FINDINGS.md           # V2 results and interpretation
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
│   ├── prepare_base.py   # Download and tokenize WikiText-103
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
│   ├── benchmark_critique.md
│   └── lit_review.md
├── report/
│   ├── main.tex          # LaTeX results report
│   ├── references.bib    # Bibliography
│   └── main.pdf          # Compiled report
└── results/
    ├── all_results.jsonl      # V2 raw results
    ├── all_results_v3.jsonl   # V3 results (critical-span re-evaluation)
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
