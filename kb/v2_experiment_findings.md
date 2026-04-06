# V2 Experiment Findings

## Setup
- GPT-2 (124M) fine-tuned on WikiText-103 (~2M tokens)
- Contamination ratios: 0% (control), 10%, 25%, 50%, 100%
- 3 generations per arm (gen 0 = clean baseline)
- Compounding arm: 25% with weight inheritance
- 500 steps/generation, LR=5e-5, batch=4, 8-step grad accumulation
- Single seed per condition (directional signal only)

## Core Result
**Hypothesis partially supported, but mechanism differs from prediction.**

At moderate contamination (10-50%), the log-prob gap widens through **distribution narrowing** — standard text becomes easier while novel text stays ~constant. The model becomes a specialist, not a worse understander.

Only at 100% contamination does genuine novel-text degradation occur, coinciding with catastrophic model collapse.

## Key Metrics

### Log-Probability Gap (standard LP - novel LP)
| Contamination | Gen 0 | Gen 1 | Gen 2 |
|---|---|---|---|
| 0% (control) | 0.489 | 0.489 | 0.489 |
| 10% | 0.489 | 0.641 | 0.597 |
| 25% | 0.489 | 0.682 | 0.616 |
| 50% | 0.489 | 0.777 | 0.647 |
| 100% | 0.489 | 1.390 | 1.129 |
| Compound 25% | 0.489 | 0.682 | 0.615 |

### Decomposed Gap
- **10-50%:** Standard LP improves (-5.73 -> ~-5.5), novel LP constant (~-6.22). Gap widens from standard side.
- **100%:** Standard LP improves (-5.73 -> -5.04) AND novel LP degrades (-6.22 -> -6.43). Both mechanisms active.

### Validation Perplexity
- Control stable at 230. Moderate contamination: 5-22% increase. 100%: 230 -> 1686 (catastrophic collapse).

### Generation Diversity (Distinct-1)
Strongest signal. Even 10% contamination nearly halves unigram diversity (0.42 -> 0.26). At 100%/gen2: 0.04 (near-repetitive).

### Novelty Accuracy
Flat across conditions (~0.29-0.37). Non-discriminative for GPT-2 at this scale.

### Compounding Arm
Nearly identical to fixed 25% — effects saturate quickly, no additional degradation from weight inheritance.

## Limitations
1. No statistical replication (n=1 per condition)
2. Only 3 generations
3. Small model, brief fine-tuning (500 steps)
4. Single seed for synthetic generation
