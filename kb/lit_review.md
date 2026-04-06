# Literature Review: Model Collapse & Synthetic Data Contamination

## 1. Shumailov et al. — The Foundational Model Collapse Paper

**Citation:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson. "The Curse of Recursion: Training on Generated Data Makes Models Forget." arXiv:2305.17493 (2023); published as "AI models collapse when trained on recursively generated data" in *Nature*, vol. 631, pp. 755–759 (July 2024).

**Setup:** Fine-tuned OPT-125M on wikitext2 using a full replacement protocol — each generation's training set is entirely replaced with synthetic output from the prior model. Also tested on VAEs and GMMs. 5 random seeds per experiment.

**Findings:**
- Perplexity degraded by 20–28 points over successive generations (from ~34 baseline).
- **Tail-trimming mechanism:** Low-probability tokens (P(token) <= 1/M) have expected sample count < 1 and disappear first. Distribution variance collapses to zero.
- Later generations produce hallucination artifacts the original model would never generate, while losing coverage of rare but valid outputs.
- With 10% original data preserved, degradation was "only minor" — early hint that mixing helps.

**Relevance:** Establishes that collapse preferentially destroys rare/tail phenomena. Novel linguistic constructions are exactly the low-probability events that disappear first.

---

## 2. Gerstgrasser et al. — Accumulation Avoids Collapse

**Citation:** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, et al. "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data." arXiv:2404.01413 (2024). Published at COLM.

**Setup:** GPT-2 (9M params) and Llama2 (12M–125M) pretrained on TinyStories. Compared two protocols: (a) replace old data with synthetic each generation, (b) accumulate all prior data. Also tested GeoDiff on molecules and VAEs on CelebA.

**Findings:**
- **Replacement:** Validation loss increases across iterations for all model sizes — confirms collapse.
- **Accumulation:** Validation loss remains stable; collapse is avoided. Held across all architectures.
- **Theoretical bound:** Under accumulation, test error bounded by σ²·d/(T−d−1)·π²/6, independent of iteration count. Under replacement, error grows linearly as σ²·d/(T−d−1)·n.
- Even under accumulation, there is a constant-factor degradation (π²/6 ≈ 1.644× base error) — diversity does not fully recover.

**Relevance:** Protocol matters enormously. Real web data accumulates, but if training pipelines preferentially select recent (increasingly synthetic) data, the effective protocol resembles replacement.

---

## 3. Dohmatob, Seddik et al. — Statistical Threshold for Collapse

**Citation:** Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Debbah. "How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse." arXiv:2404.05090 (2024).

**Setup:** Theoretical analysis of next-token-prediction with Softmax classifier. Two scenarios: fully synthetic (each gen trains only on prior gen's output) and partially synthetic (N real + n synthetic samples). Validated on GPT-2-style models trained on tiny Shakespeare.

**Findings:**
- **Fully synthetic:** Total collapse occurs inevitably with exponential probability.
- **Partially synthetic threshold:** To maintain ε-closeness to the original distribution, synthetic data must satisfy n ≤ O(log(N·ε)) — synthetic data can only grow **logarithmically** relative to real data. This is an exponentially restrictive constraint.
- Even small proportions of synthetic data can trigger distributional shift.

**Relevance:** The logarithmic threshold means even modest contamination ratios (10–25%) may exceed the safe mixing bound for large corpora, affecting rare constructions first.

---

## 4. Alemohammad et al. — Self-Consuming Models Go MAD

**Citation:** Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, et al. "Self-Consuming Generative Models Go MAD." arXiv:2307.01850 (2023). Published at ICLR 2024.

**Setup:** StyleGAN2 on FFHQ (faces) and DDPM on MNIST. Three loop types: fully synthetic, synthetic augmentation (fixed real data), and fresh data (new real data each generation). Measured FID, precision (quality), and recall (diversity).

**Findings:**
- **Fully synthetic, unbiased:** Both precision and recall decrease; FID increases. "Model Autophagy Disorder" (MAD).
- **Fully synthetic with quality filtering:** Precision maintained but diversity collapses *faster*. The quality-diversity tradeoff accelerates collapse along the diversity axis. Eventually converges to near-identical outputs.
- **Fresh data loops:** Can maintain stability but require sufficient fresh real data per generation.
- MADness observable within 5–10 generations.

**Relevance:** The precision/recall decomposition is key. Models may produce superficially correct text (maintaining precision) while losing diversity (recall) — directly mapping to the concern that standard constructions remain fine while novel ones are lost.

---

## 5. Guo et al. — The Curious Decline of Linguistic Diversity

**Citation:** Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, Chloe Clavel. "The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text." arXiv:2311.09807 (2023).

**Setup:** OPT-350M fine-tuned over 6 recursive iterations on three tasks: news summarization (XL-SUM), scientific abstracts (ACL Anthology), and story generation (WritingPrompts). Measured lexical (TTR, Distinct-n, Self-BLEU), semantic (Sentence-BERT), and syntactic (Weisfeiler-Lehman graph kernel on dependency trees) diversity.

**Findings:**

| Metric | Task | Human | Iter 6 | Decline |
|---|---|---|---|---|
| TTR | News | 7.36 | 3.66 | −50% |
| TTR | Stories | 2.23 | 0.61 | −73% |
| Distinct-2 | Scientific | 35.4% | 13.3% | −62% |
| WL Syntactic | News | 3.17 | 0.82 | **−74%** |

- **Syntactic diversity** was the most severely affected dimension (up to 74% decline).
- **Semantic diversity** was the most stable.
- **High-entropy tasks** (story generation) showed the steepest decline.

**Relevance:** Most directly relevant paper. Demonstrates that recursive synthetic training causes massive syntactic diversity loss — precisely the structural diversity needed to handle novel linguistic constructions. The finding that high-entropy/creative tasks degrade fastest suggests comprehension of novel constructions would be especially vulnerable.

---

## 6. Dohmatob et al. — A Tale of Tails

**Citation:** Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, Julia Kempe. "A Tale of Tails: Model Collapse as a Change of Scaling Laws." arXiv:2402.07043 (2024). Published at ICML 2024.

**Setup:** Theoretical analysis of how neural scaling laws evolve under synthetic contamination. Empirical validation on transformers (arithmetic tasks) and Llama2 (text generation).

**Findings:**
- Four phenomena: (1) loss of scaling (performance plateaus), (2) shifted scaling with generation count, (3) **un-learning of skills** — models lose capabilities they previously had, (4) grokking when mixing human and synthetic data.
- Synthetic contamination fundamentally changes the scaling law exponent. More (contaminated) data no longer helps.

**Relevance:** The "un-learning of skills" finding directly supports the hypothesis that training on synthetic data can cause selective capability loss, not just overall perplexity degradation.

---

## 7. Dohmatob et al. — Strong Model Collapse

**Citation:** Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe. "Strong Model Collapse." arXiv:2410.04840 (2024). Published at ICLR 2025.

**Setup:** Theoretical analysis within scaling law frameworks, with experiments on language models and feed-forward networks.

**Findings:**
- Even **1% synthetic data** can trigger measurable model collapse.
- In certain regimes, **larger models worsen collapse** rather than mitigating it.
- "Strong" collapse: performance degradation persists regardless of scale.

**Relevance:** The 1% threshold is alarming for real-world conditions. If current estimates of AI-generated web content are even roughly correct, virtually all new training corpora are contaminated well above this threshold.

---

## 8. Feng, Dohmatob et al. — Verification as Mitigation

**Citation:** Yunzhen Feng, Elvis Dohmatob, Pu Yang, Francois Charton, Julia Kempe. "Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification." arXiv:2406.07515 (2024). Published at ICLR 2025.

**Setup:** Theoretical analysis with GMMs + linear classifiers/verifiers. Empirical tasks: matrix eigenvalue computation (transformers) and news summarization (LLMs).

**Findings:**
- Both tasks exhibit collapse on unverified synthetic data.
- Even imperfect verifiers can prevent collapse if they exceed a quality threshold.
- Key insight: "it is easier to verify than to generate" — verification is computationally cheaper and can serve as a scalable filter.

**Relevance:** Suggests a mitigation path: a verifier that detects whether training text preserves linguistic diversity could filter synthetic data. However, building such a verifier for novel constructions is itself hard.

---

## 9. Briesch, Sobania & Rothlauf — LLMs Suffer From Their Own Output

**Citation:** Martin Briesch, Dominik Sobania, Franz Rothlauf. "Large Language Models Suffer From Their Own Output: An Analysis of the Self-Consuming Training Loop." arXiv:2311.16822 (2023, revised 2024).

**Setup:** Used logic expressions for objectively verifiable evaluation of LLM output. Studied varying proportions of synthetic data across training iterations. Compared full synthetic (100% replacement) with partial (mixed) cycles.

**Findings:**
- Generated outputs remained **correct** but experienced significant **diversity decline**.
- Adding fresh human data slowed but could not fully prevent quality degradation.
- Full synthetic cycle reduced diversity to a single point (complete collapse).
- First systematic separation of "correctness" from "diversity" using verifiable tasks.

**Relevance:** The correctness-vs-diversity separation is central to our project. Models trained on synthetic data can still handle standard constructions correctly while losing the capacity to process novel ones.

---

## 10. Schaeffer et al. — Position: Model Collapse Does Not Mean What You Think

**Citation:** Rylan Schaeffer, Joshua Kazdan, Alvan Caleb Arulandu, Sanmi Koyejo. "Position: Model Collapse Does Not Mean What You Think." arXiv:2503.03150 (March 2025).

A position paper arguing the literature uses 8 distinct and sometimes conflicting definitions of "collapse," grouped into 3 families: (1) test loss on real data, (2) distributional deformation, (3) scaling behavior. Argues that prominent collapse predictions rely on assumptions (full replacement) that poorly match real-world conditions, and that the most realistic scenarios show collapse is avoidable. Recommends focusing on "specific harms more likely under society's current trajectory."

**Relevance:** Important counterpoint. The authors (who include Gerstgrasser et al. co-authors) argue the field should move beyond generic "collapse" claims toward studying specific downstream harms — which is exactly what our project does by measuring comprehension of novel constructions rather than just perplexity.

---

## Synthesis: Gap This Project Fills

Across the literature, several findings converge:

1. **Tail trimming is the primary mechanism** (Shumailov, Dohmatob). Novel constructions are low-probability tail events — first casualties of collapse.
2. **Syntactic diversity is hit hardest** (Guo et al.). 74% decline over 6 generations directly implicates structural novelty.
3. **Correctness masks diversity loss** (Briesch et al., Alemohammad et al.). Standard benchmarks may not detect the degradation.
4. **The threshold is extremely low** (Dohmatob "Strong Model Collapse"). Even 1% synthetic contamination is measurable; safe mixing is logarithmic, not linear.
5. **Scale does not save you** (Dohmatob "A Tale of Tails", "Strong Model Collapse"). Larger models can worsen collapse in some regimes.

**The gap:** No existing paper directly measures whether model collapse degrades **comprehension** of novel linguistic constructions (as opposed to generation diversity or perplexity). All prior work measures perplexity, generation diversity, or task accuracy on standard benchmarks. Our "linguistic degradation" framing — testing whether contaminated models can understand constructions absent from their training distribution — is novel.
