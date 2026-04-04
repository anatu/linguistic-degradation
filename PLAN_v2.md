# PLAN_v2.md — Distribution Stagnation PoC (Transfer Learning Redesign)

## Why v2?

The v1 experiment (PLAN.md) trained a 30M-parameter GPT from scratch on WikiText-103 and evaluated it on a linguistic novelty benchmark. It failed to produce usable signal because:

1. **The baseline model could not do the task.** All models — clean and contaminated — scored an identical 18/51 (0.3529) on the novelty benchmark. The model was answering from token frequency priors, not from any kind of pragmatic inference. With no baseline competence, there was nothing to degrade.
2. **The novelty benchmark requires real language understanding.** Constructions like "that's a choice" (understated negative evaluation) or "sir, this is a Wendy's" (deflating seriousness) require pragmatic reasoning that a 30M model trained for 2000 steps simply cannot do.
3. **Val perplexity showed weak, potentially insignificant dose-response.** The 0.25 contamination arm degraded perplexity by ~12% (218 → 244), but with n=1 per condition and no replication, this isn't statistically testable. And perplexity was always "the boring metric" — it replicates Shumailov et al., not the novel hypothesis.
4. **Diversity metrics showed no signal.** Distinct-1/2/3 were flat across all conditions.

**The fundamental problem:** we cannot test whether contamination destroys linguistic competence if the model never had linguistic competence to begin with.

---

## v2 Approach: Fine-Tuning GPT-2

Instead of training from scratch, we fine-tune OpenAI's pretrained GPT-2 (124M parameters). This gives us:

- **A model that already understands language.** GPT-2 was trained on ~40GB of internet text (WebText). It can do basic pragmatic inference, handle idioms, and process complex syntax. This means the baseline should score well above chance on the novelty benchmark.
- **A realistic experimental scenario.** In the real world, no one trains language models from scratch on contaminated data. What actually happens is people fine-tune pretrained models on new data that increasingly contains synthetic text. Our experiment now models this directly.
- **Faster iteration.** Fine-tuning 124M parameters for a few hundred steps is fast on MPS. We can run the full sweep in hours, not days.
- **A stronger test of the hypothesis.** If fine-tuning on contaminated data degrades a model that *can* do pragmatic inference, that's direct evidence for distribution stagnation. If it doesn't, the hypothesis is weakened.

### What Changes from v1

| Aspect | v1 (from-scratch) | v2 (fine-tune) |
|--------|-------------------|----------------|
| Base model | Custom 30M GPT (random init) | Pretrained GPT-2 124M (HuggingFace) |
| Training | Train from scratch, 2000 steps | Fine-tune, ~500 steps, lower LR |
| Tokenizer | tiktoken gpt2 | HuggingFace GPT2Tokenizer (same vocab) |
| Model code | Custom `model/gpt.py` | `transformers.GPT2LMHeadModel` |
| Data format | uint16 numpy arrays | Same (reuse existing base data pipeline) |
| Eval: novelty | Multiple-choice cloze (same) | Same benchmark, same scoring logic |
| Eval: perplexity | Same | Same |
| Eval: diversity | Same | Same |
| Eval: NEW | — | **Log-prob gap metric** (see Section 3.3) |
| Contamination ratios | [0.0, 0.10, 0.25, 0.50, 0.75, 0.90] | [0.0, 0.10, 0.25, 0.50, 0.75, 1.0] |
| Generations | 5 | 5 |
| Replication | n=1 per condition | n=3 seeds per condition |

### What Stays the Same

- The research question and core hypothesis
- The linguistic novelty benchmark (51 examples)
- The base corpus (WikiText-103, already tokenized)
- The data mixing strategy (chunk-wise interleaving)
- The synthetic generation approach (nucleus sampling from prior model)
- The generational loop structure (fixed-ratio arms + compounding arm)
- The evaluation framework (novelty accuracy, perplexity, diversity)
- The analysis and plotting code (with updates for new metrics)

---

## Phase 0: Precondition Validation (CRITICAL — DO THIS FIRST)

Before running the full experiment, we must verify that the pretrained GPT-2 baseline can actually do the novelty benchmark. If it can't, we're back to the v1 problem and need a different benchmark.

### 0.1 Zero-Shot Baseline

1. Load pretrained GPT-2 (124M) from HuggingFace — no fine-tuning.
2. Run the novelty benchmark exactly as-is.
3. **Success criterion: accuracy > 0.50** (significantly above the 0.25 chance level, demonstrating that the model is using context to select answers, not just token priors).
4. Record per-category accuracy to identify which construction types the model handles.

### 0.2 Fine-Tuned Baseline

1. Fine-tune GPT-2 on clean WikiText-103 for 500 steps.
2. Run the novelty benchmark again.
3. **Expected outcome:** Accuracy should stay roughly the same or improve slightly. Fine-tuning on encyclopedia text shouldn't hurt pragmatic inference — it might help via general language modeling improvement, or slightly hurt if it overwrites some of GPT-2's web-text knowledge.
4. This establishes the "generation 0" baseline for the experiment.

### 0.3 Decision Gate

- If zero-shot accuracy > 0.50: proceed with the experiment as designed.
- If zero-shot accuracy is 0.35–0.50: the model has marginal competence. Consider:
  - Using GPT-2 medium (355M) instead
  - Revising the benchmark to include easier examples
  - Adding a new metric (log-prob gap, see 3.3) that's more sensitive
- If zero-shot accuracy < 0.35: the benchmark is too hard even for GPT-2. Must redesign the evaluation before proceeding.

---

## Phase 1: Infrastructure Changes

### 1.1 Model Layer: Replace Custom GPT with HuggingFace GPT-2

**File: `model/gpt2_finetune.py`** (new file)

This module wraps HuggingFace's GPT-2 for use in our pipeline:

```python
# Responsibilities:
# - Load pretrained GPT-2 from HuggingFace (with caching)
# - Provide a consistent interface matching our existing train/eval code:
#     model(input_ids, labels=None) → (logits, loss)
#     model.generate(input_ids, max_new_tokens, temperature, top_p) → token_ids
# - Save/load checkpoints in our format (state_dict + metadata)
# - Support both GPT-2 small (124M) and medium (355M) as a config option
```

**Key design decisions:**
- Use `GPT2LMHeadModel.from_pretrained("gpt2")` for the base model
- Use `GPT2Tokenizer.from_pretrained("gpt2")` — this is the same 50257 BPE vocab as tiktoken's gpt2 encoding, so our existing tokenized data is compatible
- Wrap in a thin adapter class that matches the interface expected by `train.py` and `eval/*.py`
- The `generate()` method should use HuggingFace's `model.generate()` with `do_sample=True, top_p=0.95, temperature=1.0`

**File: `model/gpt.py`** (keep for reference, no longer used in main pipeline)

### 1.2 Config Changes

**File: `config.py`** — update with new settings:

```python
# ── Model ──
MODEL_NAME = "gpt2"           # HuggingFace model name ("gpt2" = 124M, "gpt2-medium" = 355M)
MODEL_BLOCK_SIZE = 1024        # GPT-2's native context window (up from 256)

# ── Fine-Tuning ──
FINETUNE = dict(
    batch_size=8,              # smaller batch for 124M model on MPS
    learning_rate=5e-5,        # much lower LR for fine-tuning (vs 3e-4 for from-scratch)
    weight_decay=0.01,         # lighter regularization
    max_steps=500,             # fewer steps needed for fine-tuning
    warmup_steps=50,
    eval_interval=50,
    eval_steps=20,
    grad_clip=1.0,
    lr_min=1e-6,
    gradient_accumulation_steps=4,  # effective batch = 32
)

# ── Experiment ──
CONTAMINATION_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]  # added 1.0, the maximally pessimistic case
NUM_GENERATIONS = 5
NUM_SEEDS = 3                  # replications per condition for statistical testing
SEEDS = [42, 137, 256]        # fixed seed list

# ── Synthetic generation ──
GENERATE = dict(
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=512,        # longer generations (GPT-2 has 1024 context)
)
```

**Why these hyperparameter choices:**
- **LR 5e-5:** Standard fine-tuning LR for GPT-2 (used in the original GPT-2 paper and most downstream tasks). Too high and we catastrophically forget pretrained knowledge; too low and the contaminated data has no effect (which would be a null result for boring reasons).
- **500 steps:** Enough to adapt to the corpus's style/domain without fully overwriting pretrained representations. We want the model to "absorb" the training data distribution — including its contamination — while retaining its general language capabilities.
- **Batch size 8 with 4x gradient accumulation:** Effective batch of 32 matches v1, but fits in MPS memory for 124M params.
- **1.0 contamination ratio:** This is the Shumailov full-replacement protocol. If even 100% synthetic contamination doesn't degrade novelty accuracy on a model that CAN do the task, the hypothesis is in serious trouble. This is the "put up or shut up" arm.
- **3 seeds:** Minimum needed for basic statistical testing (can compute mean and standard deviation, run t-tests between conditions).

### 1.3 Training Loop Adaptation

**File: `train.py`** — modify to support fine-tuning mode:

Changes needed:
1. Accept a `pretrained_model_name` argument (default: `"gpt2"`)
2. Load pretrained weights instead of random init
3. Use the fine-tuning hyperparameters from `config.FINETUNE`
4. Adjust the data loader for GPT-2's 1024-token context window
5. Add gradient accumulation support
6. Checkpoint saving: save full model state + the pretrained model name for reproducibility

**The training loop structure stays the same** — it's still: load data → forward → backward → step → eval → log. The only changes are the model source and hyperparameters.

### 1.4 Tokenizer Compatibility

**Critical check:** Our existing `base_train.npy` and `base_val.npy` were tokenized with tiktoken's `gpt2` encoding. HuggingFace's `GPT2Tokenizer` uses the same BPE vocabulary (50257 tokens, same merges). However, there may be minor differences in special token handling.

**Action:** Write a validation script that:
1. Takes 100 random text samples from WikiText-103
2. Tokenizes with both tiktoken and HuggingFace GPT2Tokenizer
3. Asserts the token IDs are identical
4. If they differ: re-tokenize the base data with HuggingFace's tokenizer

This is a one-time check. If the encodings match (they should — both are the original GPT-2 BPE), we reuse the existing numpy data files. If not, we re-run `data/prepare_base.py` with the HuggingFace tokenizer.

### 1.5 Synthetic Generation Updates

**File: `data/generate_synthetic.py`** — update to use HuggingFace model:

Changes:
1. Load model from our fine-tuned checkpoint (not the custom GPT)
2. Use `model.generate()` from HuggingFace with `do_sample=True`
3. Increase `max_new_tokens` to 512 (GPT-2 handles longer context)
4. Keep the prompt-from-base-data strategy (sample random 10-token prompts)

The output format stays the same: uint16 numpy array of token IDs.

### 1.6 Data Pipeline

**Files: `data/prepare_base.py`, `data/mix_data.py`** — minimal changes:

- `prepare_base.py`: May need to re-tokenize with HuggingFace tokenizer (see 1.4). Update `block_size` references from 256 to 1024.
- `mix_data.py`: Update chunk size to use the new block_size (1024). Logic otherwise identical.

---

## Phase 2: Evaluation Updates

### 2.1 Novelty Benchmark (Adapt Existing)

**File: `eval/novelty_benchmark.py`** — update model loading:

Changes:
1. Load HuggingFace GPT-2 model from checkpoint instead of custom GPT
2. Use HuggingFace tokenizer instead of tiktoken
3. The scoring logic (compute log-prob of each option given context, pick highest) stays identical

The benchmark file (`benchmark/novelty_examples.jsonl`) is unchanged.

### 2.2 Perplexity (Adapt Existing)

**File: `eval/perplexity.py`** — update model loading:

Same pattern: swap custom GPT for HuggingFace model. The perplexity computation (mean cross-entropy loss on val batches) is identical.

### 2.3 Diversity (Adapt Existing)

**File: `eval/diversity.py`** — update model loading and generation:

Use HuggingFace's `model.generate()` for sampling. Distinct-N computation unchanged.

### 2.4 NEW: Log-Prob Gap Metric

**File: `eval/logprob_gap.py`** (new file)

This is a new, more sensitive metric that doesn't rely on multiple-choice accuracy (which is coarse and can be at ceiling or floor). Instead, it measures:

**"How much worse is the model at predicting tokens in novel constructions compared to standard text?"**

#### Design:

1. **Prepare two token sets:**
   - **Standard set:** 200 random passages from the WikiText validation set (text the model has seen during fine-tuning). Compute mean per-token log-probability.
   - **Novel set:** The 51 novelty benchmark examples, but instead of multiple-choice, compute the mean per-token log-probability of the *correct completion* in its full sentential context.

2. **Compute the gap:**
   ```
   logprob_gap = mean_logprob(standard_set) - mean_logprob(novel_set)
   ```
   This is always positive (novel constructions are harder). The question is whether contamination makes the gap *wider* — i.e., the model gets relatively worse at novel constructions compared to standard text.

3. **Why this is more sensitive than accuracy:**
   - Accuracy is binary (right/wrong) and can be identical across models if they all pick the same option.
   - Log-prob is continuous and changes with every weight update. Even if two models pick the same answer, one might assign 0.3 probability and the other 0.1.
   - The *gap* metric controls for overall model quality degradation. If contamination hurts everything equally, the gap stays constant. If it disproportionately hurts novel constructions, the gap widens. This directly tests the hypothesis.

4. **Additional refinement — per-category gaps:**
   Compute the gap separately for each novelty category (slang, code-switching, neologisms, etc.) to identify which types of linguistic novelty are most affected.

#### Output format:
```json
{
    "logprob_standard": -3.42,
    "logprob_novel": -5.87,
    "logprob_gap": 2.45,
    "category_gaps": {
        "novel_slang": 2.31,
        "code_switching": 3.12,
        "creative_neologism": 2.58,
        "rhetorical_subversion": 2.21,
        "pragmatic_inference": 2.44
    }
}
```

### 2.5 NEW: Benchmark Expansion (Optional, Post-Validation)

If the Phase 0 validation shows the model has marginal competence (0.35–0.50 accuracy), we should expand the benchmark:

1. **Add easier examples** that test basic novel-construction comprehension (e.g., common 2020s slang that GPT-2 likely hasn't seen but can infer from context).
2. **Add graded difficulty levels** so we can measure *where* competence breaks down, not just whether it does.
3. **Target: 100+ examples** for more statistical power.

This is optional and depends on Phase 0 results.

---

## Phase 3: Experiment Design

### 3.1 Arms

The experiment has the same structure as v1 but with replication and the 100% contamination arm:

#### Fixed-Ratio Arms (6 arms)

For each ratio in **[0.0, 0.10, 0.25, 0.50, 0.75, 1.0]**:
- **Generation 0:** Fine-tune pretrained GPT-2 on clean WikiText-103.
- **Generation 1:** Generate synthetic text from Gen-0's model. Mix with base corpus at the specified ratio. Fine-tune a fresh copy of pretrained GPT-2 on the mixed data.
- **Generation 2:** Generate synthetic from Gen-1's model. Mix. Fine-tune fresh GPT-2.
- **...through Generation 4.**

Each generation starts from the same pretrained GPT-2 weights. This isolates the data contamination effect from model-weight compounding.

**Note on the 0.0 arm:** This is the control. Each generation fine-tunes on pure WikiText-103 with a different seed. Any variation across generations in this arm is seed noise, establishing the noise floor for comparison.

**Note on the 1.0 arm:** This is 100% synthetic data (no human data mixed in). This is Shumailov's full-replacement protocol, adapted to fine-tuning. It's the "worst case" — if novelty accuracy doesn't degrade even here, the hypothesis is falsified at this scale.

#### Compounding Arm (1 arm)

- Fixed 25% contamination ratio.
- **Generation 0:** Fine-tune GPT-2 on clean WikiText.
- **Generation 1:** Fine-tune *Gen-0's fine-tuned model* (not fresh GPT-2) on 25% synthetic from Gen-0.
- **Generation 2:** Fine-tune *Gen-1's model* on 25% synthetic from Gen-1.
- **...through Generation 4.**

This compounds both data contamination and weight drift. It models the realistic scenario: a model that's been fine-tuned on contaminated data is then further fine-tuned on data generated by itself (or similar contaminated models).

#### Replication

Each arm × generation is run with **3 seeds** (42, 137, 256). This gives us:
- 6 ratios × 5 generations × 3 seeds = 90 runs (fixed arms)
- 1 arm × 5 generations × 3 seeds = 15 runs (compounding)
- **Total: 105 fine-tuning runs**

At ~5 minutes per run on MPS (500 steps, batch 8, gradient accumulation 4), this is approximately **8-9 hours** total. Runnable overnight.

### 3.2 What Each Run Produces

Per run, we save:
1. **Checkpoint:** Fine-tuned model weights + metadata (arm, ratio, generation, seed)
2. **Training log:** Per-step train_loss, val_loss, lr, elapsed time
3. **Evaluation results:** Saved immediately after training (not batched at the end like v1)
   - `novelty_accuracy`, `novelty_log_prob`, per-category accuracy
   - `val_loss`, `val_perplexity`
   - `distinct_1`, `distinct_2`, `distinct_3`
   - `logprob_gap`, `logprob_standard`, `logprob_novel`, per-category gaps

All results are appended to a single JSONL file (`results/all_results.jsonl`) as they complete, so no data is lost if the experiment is interrupted.

### 3.3 Checkpoint Management

With 105 runs producing ~500MB each, storing all checkpoints requires ~52GB. This is too much.

**Strategy:**
- Only keep checkpoints that are needed for the next generation's synthetic data generation.
- For each arm, we need Gen N's checkpoint to generate synthetic data for Gen N+1. Once Gen N+1 is trained, Gen N's checkpoint can be deleted.
- Exception: keep all Gen-0 and Gen-4 checkpoints (first and last generation) for post-hoc analysis.
- Save evaluation results immediately — they're small (JSON) and don't require the checkpoint.

**Disk budget:** At any given time, we need at most 3 checkpoints per arm (current gen × 3 seeds) = ~1.5GB per arm. With 7 arms running sequentially, peak disk usage is ~1.5GB for checkpoints + results.

---

## Phase 4: Analysis

### 4.1 Primary Analysis: Novelty Accuracy vs. Contamination

**Plot: Heatmap with error bars**
- X-axis: contamination ratio [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
- Y-axis: generation number [0, 1, 2, 3, 4]
- Color: mean novelty_accuracy across 3 seeds
- Annotation: mean +/- std in each cell

**Statistical test:** For each (ratio, generation) pair, compare novelty_accuracy to the corresponding generation in the 0.0 control arm using a two-sample t-test (n=3 per group). Apply Bonferroni correction for multiple comparisons.

**The hypothesis predicts:** A gradient from top-left (high accuracy, low contamination, early generation) to bottom-right (low accuracy, high contamination, late generation). The 0.0 row should be flat.

### 4.2 Key Comparison: Perplexity vs. Novelty Divergence

**Plot: Dual-axis line plot**
- X-axis: generation number
- Left Y-axis (blue): novelty_accuracy
- Right Y-axis (red): val_perplexity
- One line per contamination ratio

**What we're looking for:** If novelty_accuracy drops faster than perplexity degrades (or drops while perplexity stays flat), that's the strongest evidence for the thesis — the model is maintaining general language quality while losing competence on novel constructions specifically.

### 4.3 Log-Prob Gap Analysis

**Plot: Line plot**
- X-axis: generation number
- Y-axis: logprob_gap (novel minus standard)
- One line per contamination ratio, with error bands (3 seeds)

**The hypothesis predicts:** The gap widens with contamination and generations. The 0.0 control should be flat.

**Per-category breakdown:** Heat map where X = ratio, Y = novelty category, color = gap_delta (change from gen 0 to gen 4). This identifies which types of linguistic novelty are most vulnerable to contamination.

### 4.4 Compounding vs. Fixed Comparison

**Plot: Line plot comparing compounding arm (25% ratio) to fixed 25% arm**
- Both plotted with error bands
- If compounding degrades faster, this confirms the "recursion trap" hypothesis

### 4.5 100% Contamination Deep Dive

**Plot: All metrics for the 1.0 arm across generations**
- Novelty accuracy
- Val perplexity
- Log-prob gap
- Distinct-N

This arm is the stress test. If 5 generations of 100% synthetic fine-tuning don't degrade novelty accuracy, the hypothesis is unlikely to hold at any contamination level.

### 4.6 Summary Statistics

A single table saved to `results/summary.csv`:

| Arm | Ratio | Gen | Seed | novelty_acc | val_ppl | logprob_gap | d1 | d2 | d3 |
|-----|-------|-----|------|-------------|---------|-------------|----|----|-----|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Plus a condensed version with means and stds across seeds.

---

## Phase 5: Implementation Plan (Ordered)

### Step 1: Tokenizer Compatibility Check
- Write and run a script to verify tiktoken gpt2 == HuggingFace GPT2Tokenizer
- If they match: reuse `base_train.npy` and `base_val.npy`
- If not: re-tokenize with HF tokenizer
- **Time: 10 minutes**

### Step 2: Model Wrapper
- Create `model/gpt2_finetune.py`
- Implement `load_pretrained()`, `forward()`, `generate()`, `save_checkpoint()`, `load_checkpoint()`
- Test: load GPT-2, generate 100 tokens, verify output is coherent English
- **Time: 30 minutes**

### Step 3: Precondition Validation (Phase 0)
- Run zero-shot GPT-2 on novelty benchmark
- If accuracy > 0.50: proceed
- If not: stop and reassess before investing more time
- **Time: 5 minutes (evaluation only)**

### Step 4: Update Config
- Add fine-tuning hyperparameters
- Add new experiment parameters (seeds, 1.0 ratio, etc.)
- **Time: 10 minutes**

### Step 5: Update Training Loop
- Modify `train.py` to support fine-tuning mode (pretrained model loading, gradient accumulation, new hyperparams)
- Test: fine-tune GPT-2 on WikiText for 50 steps, verify loss decreases
- **Time: 30 minutes**

### Step 6: Update Synthetic Generation
- Modify `data/generate_synthetic.py` to use HF model
- Test: generate 1000 tokens from fine-tuned checkpoint
- **Time: 15 minutes**

### Step 7: Update Data Pipeline
- Modify `data/mix_data.py` for new block_size (1024)
- Verify mixing produces correct ratios
- **Time: 10 minutes**

### Step 8: Update Evaluation Suite
- Modify `eval/novelty_benchmark.py` for HF model
- Modify `eval/perplexity.py` for HF model
- Modify `eval/diversity.py` for HF model
- Create `eval/logprob_gap.py` (new metric)
- Test all four on the fine-tuned baseline from Step 5
- **Time: 45 minutes**

### Step 9: Update Orchestrator
- Modify `run_experiment.py` for:
  - Fine-tuning instead of from-scratch training
  - Multiple seeds per condition
  - Incremental result saving (JSONL)
  - Checkpoint cleanup (only keep what's needed for next gen)
  - The 1.0 contamination arm
- **Time: 30 minutes**

### Step 10: Smoke Test
- Run one full arm (e.g., ratio=0.25, 2 generations, 1 seed) end-to-end
- Verify: training → synthetic generation → mixing → training → evaluation → results saved
- Check that novelty accuracy actually varies (not stuck at one value)
- **Time: ~20 minutes**

### Step 11: Full Experiment Run
- Run all arms with 3 seeds
- **Time: ~8-9 hours (overnight)**

### Step 12: Analysis
- Update `analyze.py` for new metrics and multi-seed analysis
- Generate all plots described in Phase 4
- Write summary of findings
- **Time: 1 hour**

**Total implementation time (excluding overnight run): ~3.5 hours**

---

## Decision Points and Fallbacks

### If GPT-2 Small (124M) scores < 0.50 on novelty benchmark:
- **Fallback A:** Try GPT-2 Medium (355M). Slower to fine-tune but more capable. Everything else stays the same — just change `MODEL_NAME` in config.
- **Fallback B:** Revise the benchmark to include constructions that GPT-2 can handle. This is less desirable because it weakens the "genuinely novel" claim.
- **Fallback C:** Abandon multiple-choice accuracy entirely. Rely on the log-prob gap metric as the primary measure. This is viable but less intuitive to present.

### If novelty accuracy doesn't degrade even at 100% contamination:
- The hypothesis is not supported at this scale. This is a valid and publishable negative result. Report it honestly.
- Consider whether the hypothesis might hold at larger scales (requires more compute) or with different types of contamination (e.g., contamination from a different model family, not self-contamination).

### If novelty accuracy degrades but so does perplexity, proportionally:
- This suggests contamination causes *uniform* degradation, not *specific* loss of novelty comprehension. The log-prob gap metric would show no widening.
- This is a weaker result: contamination makes the model worse at everything, not specifically at novel language. Still interesting but doesn't support the "distribution stagnation" framing specifically.

### If the log-prob gap widens while accuracy stays flat:
- Accuracy is too coarse to detect the effect, but the continuous metric catches it. This supports the hypothesis but means we need to present the evidence carefully (log-prob gap is less intuitive than accuracy for a general audience).

---

## Reproducibility

### Seeds and Determinism
- All random seeds are fixed and recorded per run
- PyTorch deterministic mode enabled where possible
- Each result is tagged with: arm, ratio, generation, seed, model_name, all hyperparameters

### Incremental Saving
- Results saved to JSONL after each run completes (not batched at the end)
- Training logs saved per run as JSON
- Checkpoints saved with full metadata

### Dependencies
```
torch>=2.0
transformers>=4.35
tiktoken
numpy
datasets (huggingface)
matplotlib
seaborn
scipy (for statistical tests in analysis)
tqdm
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `config.py` | MODIFY | New hyperparams, model name, seeds, 1.0 ratio |
| `model/gpt2_finetune.py` | CREATE | HuggingFace GPT-2 wrapper |
| `model/gpt.py` | KEEP | No changes, kept for reference |
| `train.py` | MODIFY | Fine-tuning mode, gradient accumulation, pretrained loading |
| `data/prepare_base.py` | MODIFY (maybe) | Re-tokenize if HF tokenizer differs from tiktoken |
| `data/generate_synthetic.py` | MODIFY | Use HF model for generation |
| `data/mix_data.py` | MODIFY | Update block_size to 1024 |
| `eval/novelty_benchmark.py` | MODIFY | Use HF model + tokenizer |
| `eval/perplexity.py` | MODIFY | Use HF model |
| `eval/diversity.py` | MODIFY | Use HF model |
| `eval/logprob_gap.py` | CREATE | New log-prob gap metric |
| `run_experiment.py` | MODIFY | Fine-tuning, multi-seed, incremental saving, checkpoint cleanup |
| `analyze.py` | MODIFY | New metrics, error bars, statistical tests |
| `scripts/validate_tokenizer.py` | CREATE | One-time tokenizer compatibility check |
| `scripts/validate_baseline.py` | CREATE | Phase 0 precondition check |
