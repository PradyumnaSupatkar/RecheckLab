# Metric sensitivity — no retrospective grade replacement

## A: Frozen original evaluation

All original booleans, labels and keys remain unchanged. This is **always-minus-never** gain, not gate gain and not a causal estimate of semantic reasoning repair.

| Partition | Initial % | Always % | Difference pp [95% paired question-bootstrap CI] |
|---|---|---|---|
| source_train | 79.5663 | 91.8265 | +12.2602 [10.2585, 14.2619] |
| source_development | 80.7500 | 91.0000 | +10.2500 [7.0000, 13.7500] |
| source_test | 74.7500 | 90.2500 | +15.5000 [11.7500, 19.2500] |
| target_adaptation_pool | 77.0492 | 71.5164 | -5.5328 [-7.8893, -3.1762] |
| target_test | 76.1255 | 72.8513 | -3.2742 [-4.7749, -1.7053] |
| math500_replay | 59.0000 | 75.5000 | +16.5000 [13.0000, 20.1025] |
| gsm8k_replay | 93.4041 | 93.4041 | +0.0000 [-0.7202, 0.7582] |

Intervals use 2,000 paired question-cluster resamples, seed 20261116. These describe empirical question variation conditional on this solver/protocol and recorded rollouts; they do not capture systematic grader errors, uncertainty about mechanisms or model-to-model generalization. Previously inspected tests are descriptive. No multiple-comparison significance claim is made.

Independently recomputed completion accuracies: source training 92.2435%, development 92.0000%, source test 90.0000%, target adaptation 78.2787%, target test 76.6712%, MATH500 replay 75.7000%, GSM8K replay 94.4655%. Complete costs, recheck rates and HELP/HURT contributions are in `SUBGROUP_POLICIES.csv`.

Archived frozen-gate results from `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/policy_metrics.csv`: development 92.00% / 4,081.3425 tokens, source test 90.00% / 4,412.825 tokens, target test 76.6712% / 2,334.8465 tokens. The matching completion costs independently recomputed here are 4,121.0125 / 4,402.4200 / 2,334.8465. No fresh ML inference, retraining, or alternative-grade gate evaluation was performed. Nothing here demonstrates learned superiority.

## B: Conservative recovery — bounded diagnostic, not an alternative grader

**Not executed as population regrading.** The seven reviewed training records support narrow representation-recovery suspicions, but authoritative MATH keys and a pinned extraction/equivalence contract are unavailable. A general parser inferred from those positives would risk asymmetric false acceptance. Gold-looking intermediate expressions, option letters in explanations and discarded candidates are ineligible. No proposed alternative correctness values are assigned; `alternative_grade_applied=False` for every case.

Why each proposed review is warranted is logged in `CORRECTION_CASES.csv` and the seven-case table: Unicode exponent representation (253,1355), retained answer wrapper (1203), repeated final marker (1226,1494), retained LaTeX text brace (1373), missing multiline extraction (572). A validated future change must record original answer/grade, exact asserted span, trusted key, permitted transformation, revised grade, adjudicators and rationale, for both initial and revised outputs. Distinguish exact numeric equality from expression equivalence; never drop ordered-pair delimiters as a general rule.

Illustrative arithmetic only: if independent adjudication eventually changed the six nonempty initial grades from false to true while all other grades remained fixed, the training always-minus-never gain would fall by 6/1199 = **0.5004 pp**. Including 572 would subtract **0.5838 pp**. This is a conditional accounting example, not a corrected accuracy result, not a full symmetric audit, and not an estimated population artifact rate. It does not justify correcting unseen cases.

The separate key-consistency check uses the existing live numeric normalizer only on already-extracted GSM8K numeric strings, and exact A–E strings for CSQA. It finds no supported-syntax mismatch; it is **not** conservative transcript recovery and does not validate unsupported outputs.

## C: Completion-controlled analysis

True “valid completed answer” requires adjudication. Available proxies are:

- `stop_nonempty`: stopped initial call and nonempty publisher-extracted answer.
- `strict_FINAL_proxy`: additionally a nonempty FINAL/FINAL ANSWER assertion on the last nonempty transcript line, using the frozen regex. This establishes marker syntax only, not correctness, absence of contradiction, or mathematical validity. The marker rule omits some valid multiline/symbolic formats and must not be advertised as a universal evaluator.

Both use initial-state evidence only. Their denominators differ from the overall cohort. Eligibility is not selected by correctness. Within eligible question groups the recorded outcomes remain equally weighted; all repeated question clusters are retained together for dependence accounting. No better rollout is chosen. These tables are population stratifications, not regraded full-benchmark accuracies.

| Partition | Subset | Questions | Never % | Always % | Gain pp | Never / always tokens |
|---|---|---|---|---|---|---|
| source_train | stop_nonempty | 918 | 98.5839 | 97.8214 | -0.7625 | 1929.7 / 3761.9 |
| source_train | strict_FINAL_proxy | 913 | 98.6857 | 97.8094 | -0.8762 | 1925.3 / 3753.0 |
| source_development | stop_nonempty | 305 | 99.6721 | 98.3607 | -1.3115 | 1873.3 / 3704.8 |
| source_development | strict_FINAL_proxy | 301 | 99.6678 | 98.3389 | -1.3289 | 1860.9 / 3684.9 |
| source_test | stop_nonempty | 293 | 97.9522 | 98.2935 | +0.3413 | 1881.3 / 3717.5 |
| source_test | strict_FINAL_proxy | 288 | 97.9167 | 98.2639 | +0.3472 | 1875.2 / 3704.9 |
| target_adaptation_pool | stop_nonempty | 464 | 81.0345 | 73.9224 | -7.1121 | 2008.2 / 4642.8 |
| target_adaptation_pool | strict_FINAL_proxy | 464 | 81.0345 | 73.9224 | -7.1121 | 2008.2 / 4642.8 |
| target_test | stop_nonempty | 708 | 78.8136 | 74.8588 | -3.9548 | 2100.3 / 4569.7 |
| target_test | strict_FINAL_proxy | 708 | 78.8136 | 74.8588 | -3.9548 | 2100.3 / 4569.7 |
| math500_replay | stop_nonempty | 273 | 98.1685 | 97.8022 | -0.3663 | 2050.6 / 3995.8 |
| math500_replay | strict_FINAL_proxy | 270 | 98.1481 | 97.7778 | -0.3704 | 2033.4 / 3973.7 |
| gsm8k_replay | stop_nonempty | 1281 | 95.7065 | 94.6136 | -1.0929 | 1022.8 / 2670.9 |
| gsm8k_replay | strict_FINAL_proxy | 1281 | 95.7065 | 94.6136 | -1.0929 | 1022.8 / 2670.9 |

Within these subsets completion skips everyone, so its accuracy and cost equal never recheck. This is definitional, not an algorithmic improvement. Training stop/nonempty gain is −0.7625 pp versus +12.2602 pp overall; development is −1.3115 versus +10.25; source test is +0.3413 versus +15.50. The difference is strong evidence that whole-cohort gains depend on initial-state mixture. It does not prove that the length stratum gains are merely formatting, or that controlling initial generation would preserve the same answers.

The full-cohort decomposition is exact: gain = sum over initial-state strata of stratum weight × (HELP−HURT)/stratum question mass. Reporting only completed outputs changes the estimand; do not replace the original whole-cohort number with it. Do not call the strict proxy a causal intervention or a rule improving benchmark scores.

## D: Ambiguity-aware interpretation

Every mechanism remains unresolved unless independently supported. The number of genuine semantic HELP events can range from zero to all published HELP events under current information; reviewed suspects do not create a defensible population-wide causal lower bound. If one restricts content contributions to the *published changing events*, their net contribution can lie in the conservative bounds below. These are identification bounds, **not confidence intervals** or corrected total accuracies; unobserved discrepancies among NO_CHANGE records could broaden an actual regrading analysis further.

| Partition | Potential content contribution lower pp | Upper pp | Length-associated HELP % [bootstrap CI] |
|---|---|---|---|
| source_train | -1.2510 | 13.5113 | 95.06 [91.51, 98.15] |
| source_development | -1.5000 | 11.7500 | 100.00 [100.00, 100.00] |
| source_test | -0.7500 | 16.2500 | 95.38 [89.86, 100.00] |
| target_adaptation_pool | -7.7869 | 2.2541 | 54.55 [23.53, 81.26] |
| target_test | -4.7067 | 1.4325 | 38.10 [11.11, 66.67] |
| math500_replay | -2.2000 | 18.7000 | 95.72 [91.26, 99.01] |
| gsm8k_replay | -1.2509 | 1.2509 | 84.85 [66.67, 100.00] |

The development ratio bootstrap is degenerate at 100% because all observed HELP have length flags. It is not evidence that the population proportion is exactly 100%; resampling cannot invent unseen categories. These intervals are conditional sampling summaries and do not resolve finalizer confounding or label interpretation.

## Interpretation

There is verified sensitivity to completion-based cohort definition and strong bounded evidence of representation-sensitive grades. There is **no verified alternative grading result** and no measured fraction of genuinely repaired reasoning. A symmetric, blinded manual audit with trusted keys is the minimum next validity step. Changing extraction to maximize reported accuracy would answer the wrong question.
