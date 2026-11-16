# Statistical analysis

Primary analysis n = 43 complete question pairs. All intervals are descriptive; the study stopped before the planned 60 questions.

## Accuracy and 95% Wilson intervals

| Policy | Correct / n | Accuracy | 95% Wilson CI | Rechecks | Recheck rate | Mean tokens/question | Mean replay latency/question (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| never_recheck | 41/43 | 0.953 | [0.845, 0.987] | 0/43 | 0.000 | 1239.3 | 67.44 |
| always_recheck | 42/43 | 0.977 | [0.879, 0.996] | 43/43 | 1.000 | 3062.2 | 109.96 |
| completion_heuristic | 41/43 | 0.953 | [0.845, 0.987] | 0/43 | 0.000 | 1239.3 | 67.44 |
| frozen_v0.2_ml_gate_ood | 41/43 | 0.953 | [0.845, 0.987] | 0/43 | 0.000 | 1239.3 | 67.44 |

## Paired accuracy differences

Differences are policy A minus policy B, using 20,000 paired question-level bootstrap resamples (seed 20261121). A confidence interval spanning zero is inconclusive; this small, early-stopped dataset does not establish equivalence or significance.

| Policy A | Policy B | Difference | Bootstrap 95% interval |
|---|---|---:|---:|
| never_recheck | always_recheck | -0.023 | [-0.070, 0.000] |
| never_recheck | completion_heuristic | 0.000 | [0.000, 0.000] |
| never_recheck | frozen_v0.2_ml_gate_ood | 0.000 | [0.000, 0.000] |
| always_recheck | completion_heuristic | 0.023 | [0.000, 0.070] |
| always_recheck | frozen_v0.2_ml_gate_ood | 0.023 | [0.000, 0.070] |
| completion_heuristic | frozen_v0.2_ml_gate_ood | 0.000 | [0.000, 0.000] |

## HELP/HURT/NO_CHANGE

Transitions among complete pairs: HELP 1, HURT 0, NO_CHANGE 42.
Frozen ML gate and completion heuristic disagreed on 0/43 questions.

| Policy | HELP captured | HELP missed | HURT selected | HURT avoided |
|---|---:|---:|---:|---:|
| never_recheck | 0 | 1 | 0 | 0 |
| always_recheck | 1 | 0 | 0 | 0 |
| completion_heuristic | 0 | 1 | 0 | 0 |
| frozen_v0.2_ml_gate_ood | 0 | 1 | 0 | 0 |

## Extraction and exact grading

Initial answers parsed: 43/43 (100.0%). Revision answers parsed: 43/43 (100.0%). Unparseable or absent answers are scored incorrect under the frozen evaluation rule. Independent exact `Fraction` checks disagreed with saved initial/revision correctness on 0 answer records. Gold keys were read only for reporting, after the gate decisions had been journaled.
