# Manuscript-preparation run 2026-11-16_01

This run contains a read-only analysis of the frozen RecheckLab v0.2 classifier and the existing SEVRA recorded-response outcomes. It did not train a gate, select a threshold, change any label, call Ollama, or edit the application.

## Reproduce

The original run is immutable: the script refuses to overwrite its nonempty run directory. To reproduce from the `recheck_transfer` project root, create a new versioned empty directory and use the existing environment:

```powershell
$run = 'outputs/ieee_manuscript_preparation/run_reproduction_YYYYMMDD_01'
New-Item -ItemType Directory -Path $run
.\.venv\Scripts\python.exe -B outputs\ieee_manuscript_preparation\run_2026-11-16_01\analyze_frozen_v02.py --output-dir $run
```

The script reuses archived probabilities/decisions from `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/analysis_predictions.csv` for source development, source test and CSQA target test and fails unless row order, group, label, probability and decision match the saved artifact. For MATH500/GSM8K it applies the unchanged saved artifact only (no fitting or threshold adaptation). Pair outcomes and tokens are read from `data/raw/` and `outputs/v0.2/data/prepared/`.

## Files

- `PAPER_CONTRIBUTIONS.md` — claim boundary and evidence required.
- `STATISTICAL_RESULTS.md` — paper-ready summary and statistical interpretation.
- `ADDITIONAL_BENCHMARKS.md` — MATH500/GSM8K compatibility and diagnostics.
- `RESEARCH_QUESTIONS.md` — frozen questions and evidence status.
- `REVIEWER_OBJECTIONS.md` — likely objections and honest responses.
- `MANUSCRIPT_OUTLINE.md` — IEEE-style structure.
- `MAIN_POLICY_COMPARISON.csv`, `CLASSIFIER_METRICS_CONFUSION.csv`, `CONFUSION_MATRICES.csv`, `PAIRED_COMPARISONS.csv`, `GATE_HEURISTIC_AGREEMENT.csv`, `HELP_HURT_COMPLETION_BREAKDOWN.csv`, `DATASET_PROTOCOL_COMPARISON.csv`, `CLAIMS_EVIDENCE.csv` — reproducible tables.
- `accuracy_vs_tokens.png`, `help_hurt_by_completion.png` — figures. Recorded token cost is not measured latency.
- `INPUT_SHA256.json`, `RUN_MANIFEST.json`, `PRESERVATION_CHECK.json` — provenance and integrity.

Bootstrap intervals resample canonical questions (10,000 replicates). Repeated outcomes for the same question stay in one cluster and average within question. Paired sign-flip tests are Monte Carlo, unadjusted and descriptive. No confirmatory claim is made from previously exposed tests or the explored external replay splits.
