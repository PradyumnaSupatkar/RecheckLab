# Disagreement analysis run

This run independently audits the frozen RecheckLab v0.2 gate against the completion heuristic on existing recorded-response data. It performs no model fitting, threshold selection, LLM inference, or application/manuscript editing.

## Files

- `DISAGREEMENT_AUDIT.md` — counts, paired policy effects, confidence intervals, costs, and scope.
- `FAILURE_MODES.md` — completion, answer-change, HELP/NO_CHANGE/HURT, and token-cost breakdowns.
- `CASE_STUDIES.md` — five redacted illustrative transcript cases; no benchmark keys or answer strings.
- `PAPER_SUBSECTION.md` — concise manuscript-ready results prose and table.
- `analyze_disagreements.py` — reproduces all numeric outputs from the frozen artifact and records.
- `tables/DISAGREEMENTS.csv` — every discordant rollout with full canonical question hash, score, threshold, actions, outcomes, and signed action-token delta; no response text or answer keys.
- `tables/DISAGREEMENT_QUESTION_EFFECTS.csv` — 16 question-level discordance clusters.
- `tables/COHORT_POLICY_EFFECTS.csv` — policy comparison, question-cluster uncertainty, cost, and disagreement summary for all four cohorts.
- `tables/FAILURE_MODE_CROSSTAB.csv` — mechanical event stratification.
- `figures/gate_vs_completion_effects.png` — paired ML-minus-completion accuracy and total-token differences with question-cluster bootstrap intervals.
- `RUN_MANIFEST.json` — artifact digest, threshold, seed, and verified cohort disagreement counts.

Run from the repository root with `python outputs/ieee_disagreement_analysis/run_2026-11-16_01/analyze_disagreements.py`. The script checks the v0.2 artifact and dataset hashes against the saved input snapshot before analysis and verifies the source-test and target-test gate decisions against their saved predictions.

The reports intentionally contain no gold answer fields or answer strings. Short response excerpts in the case-study report have answer-bearing content redacted. MATH500/GSM8K protocol identity is only partially verified, and all results are retrospective on previously inspected response records.
