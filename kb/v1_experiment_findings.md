# V1 Experiment Findings

## Setup
- Custom 30M-parameter GPT trained from scratch on WikiText-103
- 2000 training steps
- Contamination ratios: 0%, 10%, 25%, 50%, 75%, 90%
- Evaluated on the same 51-example linguistic novelty benchmark (multiple-choice cloze)
- Metrics: novelty accuracy, validation perplexity, Distinct-N diversity

## Result: No Usable Signal

The v1 experiment failed to produce meaningful results. It was abandoned in favor of the v2 fine-tuning approach.

### Why It Failed

1. **Baseline model could not do the task.** All models — clean and contaminated — scored an identical 18/51 (0.3529) on the novelty benchmark. The model answered from token-frequency priors, not pragmatic inference. With no baseline competence, there was nothing to degrade.

2. **The benchmark requires real language understanding.** Constructions like "that's a choice" (understated negative evaluation) or "sir, this is a Wendy's" (deflating seriousness) require pragmatic reasoning that a 30M model trained for 2000 steps cannot do.

3. **Validation perplexity showed weak, potentially insignificant dose-response.** The 0.25 contamination arm degraded perplexity by ~12% (218 -> 244), but with n=1 and no replication, this isn't statistically testable. Perplexity was always "the boring metric" — it replicates Shumailov et al., not the novel hypothesis.

4. **Diversity metrics showed no signal.** Distinct-1/2/3 were flat across all conditions.

## Key Lesson

**You cannot test whether contamination destroys linguistic competence if the model never had linguistic competence to begin with.** This motivated the v2 redesign: fine-tuning pretrained GPT-2 (124M) instead of training from scratch, so the baseline model starts with real language understanding that can potentially be degraded.

## What Changed for V2
- Switched from training-from-scratch (30M custom GPT) to fine-tuning pretrained GPT-2 (124M)
- Added the log-prob gap metric (continuous, more sensitive than binary accuracy)
- Added 100% contamination arm (Shumailov full-replacement protocol)
- Reduced training steps (500 vs 2000) with lower LR appropriate for fine-tuning
