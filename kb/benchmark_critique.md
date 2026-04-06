# Novel-Text Benchmark Critique

## Current Design
- 51 cloze sentences across 5 categories: novel slang (20), creative neologisms (14), rhetorical subversion (8), pragmatic inference (6), code-switching (3)
- Log-prob gap metric averages per-token log-prob over the entire sentence, then compares novel vs standard (WikiText-103 validation passages)

## "Too Easy" Hypothesis
The novel-text log-probs stayed flat (~-6.22) at 10-50% contamination. One explanation: the benchmark sentences are too easy at the token-prediction level, masking genuine degradation.

### Evidence Supporting This
1. **Token dilution:** Novel sentences are short (~15 tokens), mostly common English. The few truly novel tokens are diluted by easy surrounding context when averaging log-prob over the full sentence.
2. **Narrow baseline gap:** Only 0.49 nats difference between standard (-5.73) and novel (-6.22) at gen 0 — the model doesn't find novel text much harder to begin with.
3. **Accuracy/log-prob disconnect:** MCQ accuracy is near chance (~0.30 on 4-option), meaning the model can't discriminate correct novel usage from distractors, yet it predicts individual tokens reasonably well.

### Critical Token Analysis
Many blank-position tokens are common English words (crying, texted, awake, guys, email, field, slammed, come, caring) — a model can assign high log-prob from frequency priors alone. The novelty often lives in the **construction pattern** (the surrounding frame), not the blank token.

Strongest candidates for critical-token analysis (genuinely unusual tokens): beige, NPC, Irish-exited, Wendy's, FOMO, leche, llamo, ick, bop, doom.

## Proposed Improvements
1. **Critical-token log-prob:** Compute log-prob only on the answer span rather than the full sentence. However, this only helps for items where the blank itself is novel.
2. **Full construction span:** Score the entire novel construction (e.g., "that's a whole mood", "no cap", "touch grass") rather than just the blank, to capture distributional novelty.
3. **Harder novel examples:** Longer constructions, more unusual token sequences, constructions requiring genuinely rare n-grams.
4. **Adversarial standard control:** Standard-distribution sentences matched in length/structure to novel ones, to rule out sentence-length artifacts.
5. **Novel passages:** Paragraphs written entirely in novel registers (no easy surrounding context to dilute signal).
