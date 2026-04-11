# V4 Improvements: Publication-Readiness Plan

Prioritized improvements from the ML Best Practices Audit, ordered by impact on publication-readiness at a top venue (NeurIPS/ICML/ICLR).

---

## 1. Multi-Seed Replication (CRITICAL)

**Problem:** All results are n=1 per condition. No error bars, no valid significance tests. The 0% control produces the exact same model 3 times (identical metrics to 6+ decimal places), so "experimental stability" is tautological, not empirical.

**Fix:**
- Set `NUM_SEEDS = 3` in `config.py` (seeds 42, 137, 256 are already defined)
- Re-run V2 full sweep (5 ratios × 3 generations × 3 seeds = 45 runs × ~25 min each ≈ 19 hours)
- Re-run V4 stress test (2 configs × relevant ratios × 3 generations × 3 seeds ≈ 110 hours)
- Update all tables in `main.tex` to report mean ± std
- Replace t-tests in `analyze.py` with valid multi-seed comparisons
- With 3 seeds, the 0% control will have genuine variance (different random batches during training → slightly different checkpoints)

**Files to modify:**
- `config.py:57` — change `NUM_SEEDS = 1` → `NUM_SEEDS = 3`
- `run_experiment.py` — already loops over seeds; should work without code changes
- `run_experiment_v4.py` — same
- `analyze.py` — update tables/plots to show error bands
- `report/main.tex` — all tables need mean ± std columns

**Estimated cost:** ~130 hours wall-clock on MPS (V2 + V4). Consider running V2 multi-seed first as it's faster and sufficient for a first submission.

---

## 2. Fix Val/Test Data Leakage

**Problem:** The validation set (`base_val.npy`) serves double duty:
1. **Checkpoint selection:** `train.py:143` saves best checkpoint by val_loss on `base_val.npy`
2. **Final evaluation:** `eval/perplexity.py` and `eval/logprob_gap.py:51` compute metrics on the same `base_val.npy`

This means the model is optimized to minimize loss on the same data used for final perplexity and standard log-prob reporting. Absolute numbers are optimistic. Relative ordering across conditions is likely preserved but this is a methodological flaw reviewers will flag.

**Fix:**
- Split current `base_val.npy` into two: `base_val.npy` (for checkpoint selection) and `base_test.npy` (for final metrics)
- Update `data/prepare_base.py` to produce a 3-way split: 90% train / 5% val / 5% test
- Update `eval/perplexity.py` and `eval/logprob_gap.py` to default to `base_test.npy`
- Re-run evaluation on all existing checkpoints (no retraining needed — just re-evaluate)
- Document the split in the paper's Methods section

**Files to modify:**
- `data/prepare_base.py` — add test split
- `eval/perplexity.py` — change default data path
- `eval/logprob_gap.py:51` — change default val_data_path
- `config.py` — add `TEST_DATA_PATH`

**Risk:** Absolute perplexity numbers will change slightly. Relative patterns should be preserved.

---

## 3. Seed the Standard Log-Prob Evaluation

**Problem:** `eval/logprob_gap.py:59` uses `np.random.randint` to select 200 random validation passages without resetting the seed. The standard log-prob component depends on call order and prior random state. Re-evaluating the same checkpoint can give different values.

**Fix:**
- Add a deterministic seed reset at the start of `compute_standard_logprob()`:
  ```python
  rng = np.random.RandomState(seed=config.SEED)
  # use rng.randint instead of np.random.randint
  ```
- Or pass through an explicit `rng` parameter for reproducibility
- Verify that existing results are not affected (re-evaluate one checkpoint and compare)

**Files to modify:**
- `eval/logprob_gap.py:59` — use seeded RNG

---

## 4. Add Environment Specification

**Problem:** No `requirements.txt`, `environment.yml`, or `pyproject.toml`. Dependencies listed in README by name only, no versions. Torch/transformers version differences can affect numerical results.

**Fix:**
- Run `pip freeze` in the experiment environment and create `requirements.txt` with pinned versions
- At minimum pin: `torch`, `transformers`, `tiktoken`, `numpy`, `datasets`, `matplotlib`, `seaborn`, `scipy`, `tqdm`
- Add Python version to README (e.g., Python 3.10+)
- Document MPS-specific constraints (block_size=256, batch_size=4)

**Files to create:**
- `requirements.txt` — pinned versions
- Or `environment.yml` if using conda

---

## 5. Integrate V3/V4 into the Paper

**Problem:** `report/main.tex` only covers V2 results. V4's finding that deeper training breaks the pretrained buffer is the most interesting result and should be the paper's climax, not an undocumented follow-up.

**Fix:**
- Add a Section 4.7 or restructure Results as:
  - 4.1–4.6: V2 results (current content)
  - 4.7: Critical-span validation (V3) — confirms mechanism, adds methodological confidence
  - 4.8: Stress test — deeper training breaks the pretrained buffer (V4)
- Add V4 tables: primary vs aggressive LR at 100% contamination over 3 generations
- Add V4 validation probe (0% control at 1e-4) as evidence that degradation is contamination-driven
- Update the abstract and conclusion to reflect the full V2→V4 narrative arc
- Add V4-specific figures from `results/plots/` (v4_lr_comparison.png, v4_decomposed_by_config.png)

**Files to modify:**
- `report/main.tex` — major additions to Results and Discussion
- `report/references.bib` — if V4 cites additional work

---

## 6. Add Training Loss Curves to the Paper

**Problem:** Training logs exist in `results/logs/*.json` with step-by-step train_loss and val_loss, but no learning curve plots are generated or included in the paper.

**Fix:**
- Add a `plot_training_curves()` function to `analyze.py` that:
  - Loads training logs for key conditions (0%, 50%, 100% at V2 and V4)
  - Plots train_loss and val_loss vs step
  - Highlights where best checkpoint was saved
  - Shows divergence at high contamination
- Include as a figure in the paper (e.g., supplementary or appendix)
- This provides evidence of training convergence (or lack thereof at 100%)

**Files to modify:**
- `analyze.py` — add `plot_training_curves()` function

---

## 7. Report Compute Budget

**Problem:** No wall-clock time, hardware spec, or compute cost reported anywhere. NeurIPS checklist requires this.

**Fix:**
- Add to paper Methods section:
  - Hardware: Apple Silicon (specify chip: M1/M2/M3 Pro/Max, RAM)
  - Runtime per condition: ~25 min per V2 run (500 steps), ~4 hours per V4 run (5000 steps)
  - Total compute: V2 ~12 hours, V3 ~1 minute (re-eval only), V4 ~37 hours
  - Total including failed experiments and V1
- Add to README as well

**Files to modify:**
- `report/main.tex` — Methods section
- `README.md` — add hardware/runtime section

---

## 8. Handle 0% Control Identity

**Problem:** At 0% contamination, every generation is fine-tuned on identical data from identical pretrained weights with the same seed, producing the exact same model. The paper says "constant gap of 0.489 across all generations, confirming experimental stability" — but this is a deterministic identity, not an empirical finding.

**Fix (choose one):**
- **(A) Different seeds per generation (recommended):** For the 0% control, use different seeds for each generation even though data is the same. This gives genuine variance from random batch ordering, dropout, etc.
- **(B) Acknowledge explicitly:** State in the paper that the 0% control is deterministically identical across generations, and frame it as a sanity check rather than evidence of stability.

**Impact:** With option (A) + multi-seed replication (#1), the 0% control produces 9 slightly different models (3 generations × 3 seeds), giving a real variance baseline.

---

## 9. Per-Item Benchmark Analysis

**Problem:** Only per-category breakdowns reported. With 51 items across 5 categories, there may be informative within-category variance. Which specific constructions degrade first?

**Fix:**
- Add per-item log-prob tracking to `eval/logprob_gap.py` (return per-item scores, not just means)
- Rank items by sensitivity to contamination: `gap(100%, gen 1) - gap(0%, gen 0)` per item
- Report top-5 and bottom-5 most affected constructions
- Check if slang items (e.g., "no cap") are more fragile than structural items (e.g., code-switching)
- Add a supplementary table or appendix figure

**Files to modify:**
- `eval/logprob_gap.py` — return per-item scores alongside aggregates
- `analyze.py` — add per-item analysis and ranking
- `report/main.tex` — add supplementary table

---

## 10. Expand Related Work

**Problem:** Related work covers Shumailov et al. and two follow-ups, but misses several important papers.

**Papers to add:**
- Alemohammad et al. (2023) "Self-Consuming Generative Models Go MAD" — closest work on recursive training loops
- Briesch et al. (2023) — synthetic data quality degradation
- Hataya et al. (2023) — "Will Large-scale Generative Models Corrupt Future Datasets?"
- Martínez et al. (2023) — distribution shift under synthetic augmentation
- Quinonero-Candela et al. (2009) — foundational dataset shift reference

**Files to modify:**
- `report/main.tex` — expand Section 2
- `report/references.bib` — add entries

---

## Execution Order

Recommended implementation order based on dependencies and impact:

```
Phase 1 (Methodological fixes — do before any new runs):
  [2] Fix val/test split
  [3] Seed standard log-prob eval
  [8] Handle 0% control identity

Phase 2 (Replication — the long compute phase):
  [1] Multi-seed replication (V2 first, then V4)

Phase 3 (Analysis & writing — while compute runs or after):
  [6] Training loss curves
  [9] Per-item benchmark analysis
  [5] Integrate V3/V4 into paper
  [10] Expand related work
  [7] Report compute budget
  [4] Environment specification
```

Phase 1 should be done before Phase 2, as you don't want to run 130 hours of compute with known methodological flaws. Phase 3 can largely overlap with Phase 2.
