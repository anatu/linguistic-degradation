# ArXiv Preprint Preparation Plan

Checklist for preparing the current paper draft (`report/main.tex`) for arXiv submission. The paper reports multi-seed experiments on linguistic novelty degradation under synthetic data contamination, with two training regimes (shallow/deep). All experiments and analysis are complete; this plan covers writing and formatting tasks only.

---

## TODOs

### 1. Expand References (~40 total)
**Status:** Not started
**Current:** 10 references. Need ~30 more.

Areas to cover:
- Dataset shift foundations (Quinonero-Candela et al., 2009; Moreno-Torres et al., 2012)
- Data curation and quality (Longpre et al., 2023; Penedo et al., 2023)
- Synthetic data quality and augmentation (wider than the 4 papers already cited)
- LLM evaluation methods (perplexity, diversity metrics, log-prob scoring)
- Model collapse follow-ups and extensions beyond Shumailov
- Fine-tuning dynamics and catastrophic forgetting
- Linguistic novelty and out-of-distribution language processing

Weave citations into existing prose — don't just pad the bibliography.

### 2. Add Broader Impact / Ethics Section
**Status:** Not started

Short section (~0.5 pages) covering:
- Implications for training data pipelines as synthetic text proliferates online
- Risk that contaminated models disadvantage novel/minority linguistic forms
- Limitations of the proof-of-concept scope (small model, English only)

### 3. Polish Abstract
**Status:** Not started (do last)

Rewrite after all other changes are finalized. Current abstract is solid but should reflect the final paper state including expanded related work and any prose changes.

### 4. ArXiv Formatting
**Status:** Not started

Options:
- (A) Keep current generic `article` class — perfectly acceptable for arXiv
- (B) Switch to NeurIPS style — common for ML arXiv preprints, signals venue intent

Either way, add a footnote on the title page indicating preprint status.

### 5. Final Proofread
**Status:** Not started (do last)

- Check all cross-references resolve
- Verify all figures/tables are referenced in text
- Consistent notation throughout
- No orphan acronyms
- Compile cleanly with zero warnings
