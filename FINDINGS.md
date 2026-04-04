# Findings: Distribution Stagnation PoC

## Summary

We tested whether training language models on progressively synthetic-contaminated data destroys their ability to comprehend genuinely novel linguistic constructions. Using GPT-2 (124M) fine-tuned on WikiText-103 with varying contamination ratios across multiple generations, we found **partial support for the hypothesis** — but the mechanism is different from what was originally predicted.

**Key finding:** Synthetic contamination does not make novel language *harder* for the model to process (except at 100% contamination). Instead, it makes standard language *easier* while leaving novel language unchanged. The model's representational capacity for novel constructions doesn't degrade — it simply fails to benefit from the improved distributional fit that contaminated training provides for in-distribution text. This is distribution *narrowing*, not distribution *stagnation* in the strong sense.

---

## Experimental Setup

- **Base model:** GPT-2 124M (pretrained, HuggingFace)
- **Training:** Fine-tuned for 500 steps on WikiText-103 (2M tokens), LR=5e-5, effective batch=32
- **Contamination ratios:** 0% (control), 10%, 25%, 50%, 100%
- **Generations:** 3 per arm (gen 0 = clean baseline, gen 1-2 = contaminated)
- **Compounding arm:** 25% contamination with model-weight inheritance across generations
- **Seeds:** 1 per condition (directional signal only, not statistically powered)
- **Evaluation metrics:**
  - Novelty accuracy (multiple-choice cloze on 51 novel linguistic constructions)
  - Validation perplexity (on held-out WikiText)
  - Log-prob gap (difference in mean log-prob between standard and novel text)
  - Distinct-N (generation diversity)

---

## Results

### 1. Log-Prob Gap: The Primary Metric

| Arm | Gen 0 | Gen 1 | Gen 2 |
|-----|-------|-------|-------|
| 0% (control) | 0.489 | 0.489 | 0.489 |
| 10% | 0.489 | 0.641 | 0.597 |
| 25% | 0.489 | 0.682 | 0.616 |
| 50% | 0.489 | 0.777 | 0.647 |
| 100% | 0.489 | **1.390** | **1.129** |
| Compound (25%) | 0.489 | 0.682 | 0.615 |

**Observations:**
- The gap widens with contamination in a clear dose-response pattern. Higher contamination → wider gap.
- The 0% control is perfectly flat (0.489 across all generations), confirming the experimental setup is stable.
- The 100% arm shows a dramatic gap widening (0.489 → 1.390), nearly 3x the baseline.
- All contaminated arms show partial recovery at gen 2 (gap narrows slightly from gen 1). This is unexpected and may indicate the model is reaching a contaminated equilibrium.

### 2. Decomposing the Gap: Standard vs. Novel Log-Probs

| Arm | Metric | Gen 0 | Gen 1 | Gen 2 |
|-----|--------|-------|-------|-------|
| 0% | standard LP | -5.729 | -5.729 | -5.729 |
| 0% | novel LP | -6.218 | -6.218 | -6.218 |
| 10% | standard LP | -5.729 | -5.624 | -5.638 |
| 10% | novel LP | -6.218 | -6.264 | -6.235 |
| 25% | standard LP | -5.729 | -5.565 | -5.597 |
| 25% | novel LP | -6.218 | -6.247 | -6.213 |
| 50% | standard LP | -5.729 | -5.460 | -5.522 |
| 50% | novel LP | -6.218 | -6.237 | -6.169 |
| 100% | standard LP | -5.729 | **-5.041** | **-5.187** |
| 100% | novel LP | -6.218 | **-6.432** | **-6.317** |

**Critical observation: The gap widening is driven by two different mechanisms depending on contamination level.**

- **At 10-50% contamination:** The gap widens primarily because standard text gets easier (standard LP improves from -5.73 toward -5.46) while novel text stays approximately constant (~-6.22). The model is becoming more confident on in-distribution text without losing novel-construction capability.

- **At 100% contamination:** Both mechanisms are at work. Standard text gets much easier (-5.73 → -5.04) AND novel text gets genuinely harder (-6.22 → -6.43). This is the only condition where we see actual degradation of novel-text processing — the strong version of the hypothesis.

### 3. Validation Perplexity

| Arm | Gen 0 | Gen 1 | Gen 2 |
|-----|-------|-------|-------|
| 0% | 230 | 230 | 230 |
| 10% | 230 | 243 | 244 |
| 25% | 230 | 254 | 254 |
| 50% | 230 | 283 | 279 |
| 100% | 230 | **897** | **1686** |

- Perplexity degrades monotonically with contamination, as expected (replicates Shumailov et al.).
- The 100% arm shows catastrophic perplexity collapse (230 → 1686 by gen 2) — classic model collapse.
- At 10-50%, perplexity degrades modestly (5-22% increase), consistent with the model absorbing some distribution shift while retaining most of its language modeling capability.

### 4. Generation Diversity (Distinct-N)

| Arm | D-1 (Gen 0) | D-1 (Gen 1) | D-1 (Gen 2) |
|-----|-------------|-------------|-------------|
| 0% | 0.421 | 0.420 | 0.423 |
| 10% | 0.425 | 0.261 | 0.288 |
| 25% | 0.424 | 0.239 | 0.160 |
| 50% | 0.422 | 0.187 | 0.152 |
| 100% | 0.420 | 0.108 | **0.040** |

**Diversity collapses dramatically under contamination.** This is the clearest and most robust signal in the experiment:
- Even 10% contamination cuts unigram diversity nearly in half (0.42 → 0.26).
- 100% contamination by gen 2 reduces D-1 to 0.04 — the model is generating near-repetitive text.
- D-2 and D-3 follow the same pattern with even steeper drops.
- The 0% control is perfectly stable.

This directly confirms the "distribution narrowing" aspect of model collapse: the model's output vocabulary shrinks dramatically when trained on synthetic data, even at modest contamination levels.

### 5. Novelty Accuracy

Novelty accuracy stayed essentially flat across all conditions (0.29-0.37), confirming our earlier finding that this metric is non-discriminative for GPT-2 on this benchmark. The model answers from token-frequency priors regardless of contamination. The log-prob gap is the appropriate metric for this experimental scale.

Interestingly, the 100% arm showed a *slight increase* in accuracy at gen 2 (0.37 vs 0.31 baseline). This is likely noise, but could also reflect the model becoming more "generic" in its predictions — defaulting to common completions that happen to overlap with the benchmark's correct answers.

### 6. Compounding Arm

The compounding arm (25% contamination with weight inheritance) produced nearly identical results to the fixed 25% arm:

| | Fixed 25% | Compound 25% |
|--|-----------|-------------|
| Gen 1 gap | 0.682 | 0.682 |
| Gen 2 gap | 0.616 | 0.615 |
| Gen 2 D-1 | 0.160 | 0.179 |

Over 3 generations, compounding shows no additional degradation beyond what the fixed-ratio arm produces. This suggests the contamination effect saturates quickly — the model reaches a contaminated equilibrium by gen 1 that doesn't worsen with further compounding. More generations would be needed to determine if divergence emerges later.

---

## Interpretation: What Does This Mean for the Hypothesis?

### The Original Hypothesis

> When models are trained on data increasingly dominated by previous-generation LLM outputs, they lose the ability to perform zero-shot comprehension of held-out linguistic innovations. This is not a scaling problem or a data-quantity problem — it is a structural competence ceiling imposed by recursive interpolation within a narrowing distribution.

### What We Found

**The hypothesis is partially supported, but the mechanism is weaker than predicted.**

1. **Supported: Distribution narrowing is real and dose-dependent.** Generation diversity collapses dramatically under contamination, confirming that synthetic data narrows the model's representational distribution. This is the core mechanism the hypothesis relies on.

2. **Partially supported: The log-prob gap widens with contamination.** Models trained on contaminated data do become *relatively* worse at processing novel linguistic constructions compared to standard text. However:

3. **Not supported (at moderate contamination): Novel-construction processing doesn't actively degrade.** At 10-50% contamination, novel-text log-probs stay approximately constant. The gap widens because the model gets better at standard text, not because it gets worse at novel text. The model doesn't *lose* competence — it just doesn't *generalize* its improved distributional fit to out-of-distribution constructions.

4. **Supported at extreme contamination only:** At 100% synthetic data (full Shumailov replacement), novel-text log-probs do genuinely degrade (-6.22 → -6.43). This is the only condition that supports the strong "competence loss" version of the hypothesis. But this also comes with catastrophic perplexity collapse (230 → 1686), so it's not a case of hidden degradation — everything falls apart.

### The Revised Picture

The data suggests a more nuanced story than "contamination destroys linguistic competence":

**At realistic contamination levels (10-50%):** The model develops a narrower but more confident distribution over in-distribution text. Novel constructions aren't hurt — they're just left behind. The model becomes a specialist rather than a generalist. This is concerning for a different reason than the hypothesis predicted: it's not that the model *can't* process novel language, it's that the *gap* between what it handles well and what it handles poorly widens. In practice, this means contaminated models would be increasingly biased toward conventional language patterns.

**At extreme contamination (100%):** Full model collapse occurs, degrading everything including novel-construction processing. This confirms Shumailov et al. and shows that the collapse does affect comprehension (not just generation), but it's not a subtle or hidden effect — it's visible in all metrics simultaneously.

---

## Limitations

1. **No statistical replication.** All results are n=1 per condition. The dose-response patterns are suggestive but not statistically testable. The partial recovery at gen 2 could be noise.

2. **Only 3 generations.** The compounding effect may require more generations to manifest. Shumailov et al. used up to 10 generations.

3. **Novelty benchmark limitations.** The multiple-choice accuracy metric was non-discriminative at this scale. The log-prob gap metric is more sensitive but less intuitive.

4. **Small model, short fine-tuning.** GPT-2 124M fine-tuned for 500 steps may not absorb contamination deeply enough. Longer training or larger models might show stronger effects.

5. **Single seed for synthetic generation.** All arms used the same seed-42 model for synthetic data generation, creating correlation between conditions.

---

## Recommendations for Follow-Up

1. **Add seed replication (3 seeds)** to make the results statistically testable. Priority: the 25%, 50%, and 100% arms.

2. **Extend to 5-7 generations** to see if the gen 2 partial recovery continues or if degradation resumes.

3. **Try longer fine-tuning (2000+ steps)** to let the model absorb contamination more deeply.

4. **Test the threshold hypothesis:** Is there a contamination ratio between 50% and 100% where novel-text log-probs start to genuinely degrade? A sweep at [60%, 70%, 80%, 90%] would identify this threshold.

5. **Per-category analysis:** The benchmark includes 5 categories (slang, code-switching, neologisms, rhetorical subversion, pragmatic inference). Analyze whether contamination affects some categories more than others.
