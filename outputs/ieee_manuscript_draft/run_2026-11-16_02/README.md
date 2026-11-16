# RecheckLab manuscript draft

This directory contains a complete first IEEE conference-style manuscript draft and its supporting files.

- `main.tex` ? manuscript source using the `IEEEtran` conference class.
- `references.bib` ? bibliography; the 2026 SEVRA, Calibration Floor, and risk-aware self-correction works are arXiv v1 preprints as of 16 November 2026, not represented as peer-reviewed publications.
- `figures/accuracy_vs_tokens.png` ? accuracy/token comparison for the four evaluated cohorts.
- `figures/source_development_accuracy_vs_tokens.png` ? separate descriptive plot for the model/threshold selection partition.
- `figures/help_hurt_by_completion.png` ? recorded HELP/HURT distribution by completion state.
- `tables/` ? copied core analysis CSVs plus `CALL_VS_TOKEN_REDUCTION.csv`, a question-weighted recomputation from the frozen response records.
- `reproduce_additional_manuscript_analysis.py` ? reproduces the call/token table and the two accuracy-cost figures without training or inference.
- `MANUSCRIPT_GAPS.md` ? five high-impact remaining issues and concrete remedies.

The numeric claims draw on `outputs/ieee_manuscript_preparation/run_2026-11-16_01/STATISTICAL_RESULTS.md`, its supporting CSVs, and `ADDITIONAL_BENCHMARKS.md`. The call-conditioned token means were independently recalculated from frozen response records using canonical question hashes and equal question weights. The source MATH holdout, CSQA target test, and previously inspected historical partitions are not fresh confirmation. MATH500 and GSM8K remain exploratory recorded-response diagnostics.

Build from this directory with a LaTeX installation that includes IEEEtran and BibTeX, for example:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Author names and affiliations were not supplied, so the manuscript uses ?Anonymous Authors.? No LLM inference, retraining, application edit, or frozen data/model/threshold modification was performed for this manuscript update.
