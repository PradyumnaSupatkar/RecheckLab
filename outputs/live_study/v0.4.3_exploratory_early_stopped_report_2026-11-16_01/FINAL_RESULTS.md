# RecheckLab v0.4.3 — Exploratory early-stopped evaluation

## Confirmed results

The frozen selection contained 60 MATH Level 4–5 questions. The run stopped by explicit user instruction after **43 complete initial/revision pairs** (86 naturally completed LLM calls). These 43 pairs alone form the primary four-policy analysis; incomplete attempts are excluded from that paired comparison and listed separately.

| Policy | Correct / 43 | Accuracy | Rechecks | Mean total tokens/question | Replayed latency/question |
|---|---:|---:|---:|---:|---:|
| never_recheck | 41/43 | 0.953 | 0/43 | 1239.3 | 67.44s |
| always_recheck | 42/43 | 0.977 | 43/43 | 3062.2 | 109.96s |
| completion_heuristic | 41/43 | 0.953 | 0/43 | 1239.3 | 67.44s |
| frozen_v0.2_ml_gate_ood | 41/43 | 0.953 | 0/43 | 1239.3 | 67.44s |

Transitions in the complete-pair set: HELP 1, HURT 0, NO_CHANGE 42. Frozen ML versus completion decisions disagreed on 0/43 questions. Initial/revision extraction succeeded on 43/43 and 43/43 complete calls, respectively. The deterministic exact grader matched the saved correctness values on 86/86 answer records.

## Attempt accounting

Distinct selected questions attempted: 47/60. Incomplete initial calls: 4; incomplete revision calls: 0; terminally interrupted calls: 3; timeout-related incomplete calls: 2; unattempted questions: 13. Question 1 attempt 1 remains the inherited incomplete v0.4.2 attempt; no attempt 2 was made.

The evaluation was stopped early due to excessive runtime. This is **not** a completed 60-question study. Selection and runtime censoring may bias the observed subset, and the confidence intervals are descriptive. No significance, equivalence, or generalization claim is made.

## Supporting files

See `POLICY_COMPARISON.csv`, `QUESTION_LEVEL.csv`, `INCOMPLETE_ATTEMPTS.md`, `STATISTICAL_ANALYSIS.md`, `HARDWARE_AND_RUNTIME.md`, and `EXPERIMENT_LIMITATIONS.md`. `make_final_report.py` regenerates the analysis from the existing local journals and checkpoints without inference.
