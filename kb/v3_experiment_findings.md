# V3 Experiment Findings

## Motivation
The v2 log-prob gap metric averaged per-token log-prob over entire novel sentences, potentially diluting the signal from the few tokens carrying novel meaning. Most tokens in sentences like "He showed up two hours late with coffee and said 'no ___' like nothing happened" are common English — the novelty lives in "no cap," not the surrounding words.

V3 tests whether scoring only the critical construction tokens reveals degradation that full-sentence averaging masked.

## Setup
- **No retraining.** Re-evaluated all 18 existing v2 checkpoints with a new metric.
- Each of the 51 benchmark items was annotated with a `critical_span` field — the minimal contiguous token sequence carrying the novel meaning (e.g., "no cap", "touch grass", "Sir, this is a Wendy's", "her villain era").
- The new metric computes log-prob only over the critical span tokens, while still running the full sentence through the model (so the span tokens get full left-context).
- Character-to-token offset mapping via prefix encoding to handle BPE boundary issues.

## Results

### Critical-Span Log-Prob Gap

| Contamination | Gen 0 | Gen 1 | Gen 2 |
|---|---|---|---|
| 0% (control) | 1.162 | 1.162 | 1.162 |
| 10% | 1.162 | 1.299 | 1.259 |
| 25% | 1.162 | 1.332 | 1.279 |
| 50% | 1.162 | 1.421 | 1.303 |
| 100% | 1.162 | 2.033 | 1.704 |
| Compound 25% | 1.162 | 1.331 | 1.278 |

Compared to v2 full-sentence gap (baseline 0.489), the critical-span gap is uniformly ~0.67 nats wider (baseline 1.162). This confirms that construction-defining tokens are genuinely harder for the model than surrounding filler.

### Decomposed Critical-Span Log-Probs

| Contamination | Novel Critical Gen 0 | Novel Critical Gen 1 | Novel Critical Gen 2 |
|---|---|---|---|
| 0% (control) | -6.906 | -6.906 | -6.906 |
| 10% | -6.906 | -6.952 | -6.927 |
| 25% | -6.906 | -6.925 | -6.902 |
| 50% | -6.906 | -6.913 | -6.861 |
| 100% | -6.906 | **-7.103** | -6.908 |

- At 10-50% contamination: critical-span novel log-probs are essentially **flat** (~-6.86 to -6.95), same as the v2 full-sentence finding.
- At 100% gen 1: critical-span novel log-prob drops to **-7.103** (vs -6.906 baseline), a 0.197 nats degradation — larger than the v2 full-sentence drop of 0.214 nats (-6.218 to -6.432). The critical tokens are hit comparably to full sentences.
- At 100% gen 2: partial recovery to -6.908, nearly back to baseline.

### Comparison: V2 Full-Sentence vs V3 Critical-Span

| Metric | Baseline | 100% Gen 1 | Delta |
|---|---|---|---|
| V2 full-sentence gap | 0.489 | 1.390 | +0.901 |
| V3 critical-span gap | 1.162 | 2.033 | +0.871 |
| V2 novel LP | -6.218 | -6.432 | -0.214 |
| V3 novel LP (critical) | -6.906 | -7.103 | -0.197 |

The delta (change from baseline) is nearly identical between v2 and v3. The critical-span metric amplifies the absolute gap but doesn't change the relative signal.

## Key Takeaway

**The critical-span metric does not change the qualitative story.** The "too easy" hypothesis is not confirmed in the way we expected:

1. The critical tokens are indeed harder than filler (gap is ~0.67 nats wider at baseline), confirming the dilution concern was real in absolute terms.
2. But the **pattern across contamination conditions is the same** — at moderate contamination, novel construction tokens are not getting harder. The gap widening is still driven by standard text getting easier.
3. The dilution from full-sentence averaging was not hiding a different conclusion about degradation at moderate contamination.

## Interpretation

The lack of novel-text degradation at moderate contamination is likely a real finding, not a measurement artifact. The contamination (model-generated Wikipedia-style prose mixed back into training data) doesn't actively suppress novel constructions — it just fails to include them. The model trains on a blander version of English, which increases confidence on standard patterns without necessarily degrading representations of unusual constructions. Active degradation only occurs at 100% contamination where the model sees zero real data, coinciding with catastrophic model collapse.

## Why Comprehension Is Constant While Generation Degrades

The v2 diversity metrics show massive generation collapse (D-1: 0.42 → 0.04 at 100%/gen2), yet novel-text log-probs barely move. This seems contradictory but reflects an asymmetry between comprehension and generation in autoregressive models:

**Comprehension (log-prob scoring) is a single forward pass.** You present "no cap" and the model assigns a probability. It just needs the right weights to *recognize* the pattern.

**Generation is autoregressive sampling.** At each step, the model picks one token from its entire vocabulary. If contamination has made the distribution even slightly peakier — more confident on high-frequency tokens — then sampling systematically avoids the low-probability paths that lead to novel constructions.

**The critical insight:** A model can still assign reasonable probability to "no cap" while never sampling it. A token with 0.1% probability looks fine in a log-prob evaluation, but it essentially never wins the sampling lottery against the 50 tokens that each have 1-2% probability.

Contamination appears to be **redistributing probability mass within the distribution** — concentrating it on high-frequency patterns — without zeroing out the tail. The tail is still there, just a smaller share of a peakier distribution. That's enough to kill generation diversity but not enough to meaningfully change the log-prob of any individual tail token.

**Implication:** The v2/v3 finding isn't "novel comprehension is robust to contamination." It's that **log-prob is the wrong metric for what's actually happening.** The real degradation is in the model's *willingness to traverse* novel-construction paths during generation, not its *ability to score* them when presented. This reframes the experiment toward generation-based evaluation.

## What This Suggests for Next Steps

1. **Generation-based metric:** Instead of "can the model still score novel text?", ask "can the model still produce novel constructions when prompted?" The diversity collapse suggests this is where the real degradation lives.
2. **Adversarial contamination:** Synthetic data that actively paraphrases novel constructions into standard English, so the model learns to *prefer* the standard form.
3. **Longer/deeper fine-tuning** (5000+ steps, higher LR) to overwrite pretrained representations rather than just nudging them.
4. **Train from scratch** with a model large enough for baseline competence, so all knowledge comes from data we control.
5. **Post-2019 benchmark constructions** genuinely absent from GPT-2's pretraining, removing the pretrained safety net.
