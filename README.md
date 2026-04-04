# Distribution Stagnation PoC

Proof-of-concept experiment testing whether training language models on synthetic-contaminated data degrades their ability to comprehend novel linguistic constructions.

## Status: PoC Complete (v2)

The v2 experiment (fine-tuning GPT-2 124M) is complete with directional results across 5 contamination ratios and 3 generations. See `FINDINGS.md` for full analysis.

**Key finding:** At moderate contamination (10-50%), the log-prob gap between standard and novel text widens — but through improved in-distribution confidence, not degraded novel-text processing. Only at 100% contamination does novel-text processing genuinely degrade, coinciding with catastrophic model collapse.

## Project Structure

```
├── PLAN.md               # v1 research plan (from-scratch training, abandoned)
├── PLAN_v2.md            # v2 research plan (fine-tuning GPT-2)
├── FINDINGS.md           # Results and interpretation
├── config.py             # Experiment configuration
├── train.py              # Fine-tuning loop (GPT-2 + gradient accumulation)
├── run_experiment.py     # Orchestrator for full generational sweep
├── analyze.py            # Plot generation and statistical analysis
├── model/
│   ├── gpt.py            # Custom small GPT (v1, kept for reference)
│   └── gpt2_finetune.py  # HuggingFace GPT-2 wrapper (v2)
├── data/
│   ├── prepare_base.py   # Download and tokenize WikiText-103
│   ├── generate_synthetic.py  # Generate synthetic text from checkpoints
│   └── mix_data.py       # Mix human + synthetic data at ratios
├── eval/
│   ├── novelty_benchmark.py   # Multiple-choice cloze evaluation
│   ├── perplexity.py          # Validation perplexity
│   ├── diversity.py           # Distinct-N metrics
│   └── logprob_gap.py         # Log-prob gap metric (novel vs standard)
├── benchmark/
│   └── novelty_examples.jsonl # 51 novel linguistic construction examples
├── report/
│   ├── main.tex          # LaTeX results report
│   ├── references.bib    # Bibliography
│   └── main.pdf          # Compiled report
└── results/
    ├── all_results.jsonl  # Raw results (all conditions)
    └── plots/             # Generated figures
```

## Running the Experiment

```bash
# Prepare base data (downloads WikiText-103)
python data/prepare_base.py

# Run full experiment (~12 hours on MPS)
python run_experiment.py

# Generate plots and analysis
python analyze.py
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
