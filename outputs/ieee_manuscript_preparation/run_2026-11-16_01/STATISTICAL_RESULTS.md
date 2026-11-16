# Statistical results from frozen v0.2 replay

## Scope and analysis

Read-only calculations use the saved `outputs/v0.2/models/source_gate.joblib` and recorded paired active-verification outcomes. No model was fitted, threshold changed, outcome relabeled, or LLM called. The frozen classifier is class-balanced `HistGradientBoostingClassifier` (`max_iter=100`, `max_leaf_nodes=7`, `min_samples_leaf=20`, `learning_rate=0.05`, `l2_regularization=5`, `random_state=42`) on 19 initial-only features, with class balancing through question-weighted sample weights. The frozen threshold is 0.25968145276947807 and harm weight is 1.0; the gate rechecks when `P(HELP) - P(HURT) > threshold`. Completion means `base_done_reason == "length"`; other explicit states skip. The gate is `rechecklab.predictor.decide_recheck` using the artifact probabilities and frozen threshold.

Policies are evaluated on the same rows. Accuracy and costs average outcome rows within canonical question hash, then average questions. This handles repeated rollouts without treating them as independent questions. Cluster bootstrap resamples whole questions (10,000 draws, percentile intervals). Paired sign-flip tests are Monte Carlo (20,000 draws), two-sided, unadjusted for multiple policy comparisons; the minimum reportable Monte Carlo p is 0.00005 and is not an exact p-value. Intervals are conditional on the finite recorded response library; they do not capture the full LLM generation distribution. Sparse improvements can yield a positive percentile bootstrap interval while the sign-flip test remains inconclusive; CSQA completion versus never has this pattern (+0.55 pp; bootstrap 95% CI +0.14 to +1.09; unadjusted p=0.121), so do not claim significance.

The full policy table, including accuracy intervals, recheck rate, HELP/HURT events, initial/recheck/total tokens and observed Pareto status, is `MAIN_POLICY_COMPARISON.csv`. Oracle rows are hindsight-only. `PAIRED_COMPARISONS.csv` contains accuracy and token differences. `GATE_HEURISTIC_AGREEMENT.csv` records policy disagreements. `CONFUSION_MATRICES.csv` and `CLASSIFIER_METRICS_CONFUSION.csv` give both raw rollout counts and question-equivalent confusion mass, alongside question-weighted classification metrics.

## Core policy comparison

| Cohort | Policy | Final accuracy (95% question bootstrap CI) | Recheck | Total recorded tokens/question | Tokens saved vs always | HELP question-equivalents selected | HURT equivalents selected |
|---|---|---:|---:|---:|---:|---:|---:|
| MATH source test | Never | 74.75% (70.50–79.00) | 0.00% | 3,182.0 | 44.64% | 0 | 0 |
|  | Always | 90.25% (87.25–93.00) | 100.00% | 5,747.4 | 0.00% | 65 | 3 |
|  | Completion | 90.00% (87.00–92.75) | 26.75% | 4,402.4 | 23.40% | 62 | 1 |
|  | Frozen ML | 90.00% (87.00–92.75) | 27.25% | 4,412.8 | 23.22% | 62 | 1 |
| CSQA target test | Never | 76.13% (72.99–79.13) | 0.00% | 2,243.6 | 52.47% | 0 | 0 |
|  | Always | 72.85% (69.78–75.85) | 100.00% | 4,720.0 | 0.00% | 10.5 | 34.5 |
|  | Completion | 76.67% (73.53–79.67) | 3.41% | 2,334.8 | 50.53% | 4 | 0 |
|  | Frozen ML | 76.67% (73.53–79.67) | 3.41% | 2,334.8 | 50.53% | 4 | 0 |
| MATH500 | Never | 59.00% (54.60–63.40) | 0.00% | 4,312.5 | 46.92% | 0 | 0 |
|  | Always | 75.50% (71.90–79.00) | 100.00% | 8,124.9 | 0.00% | 93.5 | 11 |
|  | Completion | 75.70% (72.00–79.20) | 45.40% | 7,062.9 | 13.07% | 89.5 | 6 |
|  | Frozen ML | 75.30% (71.60–78.80) | 43.60% | 6,933.6 | 14.66% | 87.5 | 6 |
| GSM8K | Never | 93.40% (92.04–94.69) | 0.00% | 1,180.4 | 59.74% | 0 | 0 |
|  | Always | 93.40% (92.08–94.62) | 100.00% | 2,932.0 | 0.00% | 16.5 | 16.5 |
|  | Completion | 94.47% (93.21–95.64) | 2.88% | 1,331.4 | 54.59% | 14 | 0 |
|  | Frozen ML | 94.35% (93.10–95.56) | 2.50% | 1,313.1 | 55.22% | 12.5 | 0 |

Source development (previously inspected; descriptive only): never 80.75%, always 91.00%, completion 92.00% at 4,121.0 tokens/question, and frozen ML 92.00% at 4,081.3 tokens/question. The source-development model/threshold selection history makes this an optimistic descriptive comparison, not confirmation.

The completion rule captures 62/65 HELP events on source test (95.38%; 62 helpful among 107 recheck-equivalent calls, 57.94% helpful precision), 4/10.5 on CSQA target test (38.10%; 4/25 recheck-equivalents, 16.00% helpful precision), 89.5/93.5 on MATH500 (95.72%; 39.43% helpful precision), and 14/16.5 on GSM8K (84.85%; 36.84% helpful precision). It avoids 2/3 source-test harmful revisions, all 34.5 target-test harmful equivalents, 5/11 MATH500 harmful equivalents, and all 16.5 GSM8K harmful equivalents in the recorded outcomes. These counts are fractional question-equivalents where two rollouts are averaged; they are not fractional individual questions. `MAIN_POLICY_COMPARISON.csv` contains corresponding missed-HELP, avoided-HURT and precision/capture fields for every policy.

The same accuracy point can conceal different costs: on MATH source test the gate differs from completion on two rollout rows, both NO_CHANGE, and costs 10.41 more tokens/question. On CSQA all 1,466 rollout decisions are identical. The gate and completion differ on 18 MATH500 rollout rows (9 question clusters) and 10 GSM8K rows (5 clusters); neither difference improves observed accuracy. For MATH500 ML is 0.40 pp below completion (paired cluster bootstrap CI -0.90 to 0.00 pp; sign-flip p=0.252). For GSM8K ML is 0.11 pp below completion (CI -0.30 to 0.00; p=0.496). These are exploratory, unadjusted comparisons.

## Frozen three-class classifier metrics

Accuracy here means argmax prediction of HELP / NO_CHANGE / HURT, not final answer accuracy. The confusion table order is true rows and predicted columns in that order; `CONFUSION_MATRICES.csv` records raw rollout-row counts. Classification metrics weight each question as one, split across its rollout rows. NO_CHANGE dominance makes accuracy look high while the model can miss harmful events; class-specific recall and macro-F1 are essential.

| Cohort | Questions (rows) | Q-equal support HELP/NC/HURT | Three-class accuracy | Macro-F1 (95% cluster CI) | Weighted-F1 | Log loss (95% cluster CI) | HELP P/R/F1 | HURT P/R/F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MATH source development* | 400 (401) | 47 / 347 / 6 | 85.00% | 0.559 (0.494–0.636) | 0.868 | 0.392 (0.321–0.467) | .518/.915/.662 | .077/.167/.105 |
| MATH source test | 400 (400) | 65 / 332 / 3 | 85.25% | 0.542 (0.509–0.572) | 0.869 | 0.411 (0.334–0.492) | .588/.923/.719 | 0/0/0 |
| CSQA target test | 733 (1,466) | 10.5 / 688 / 34.5 | 91.54% | 0.394 (0.334–0.455) | 0.900 | 0.322 (0.260–0.388) | .160/.381/.225 | 0/0/0 |
| MATH500 | 500 (1,000) | 93.5 / 395.5 / 11 | 70.20% | 0.465 (0.432–0.500) | 0.732 | 0.728 (0.643–0.816) | .417/.914/.573 | .026/.045/.033 |
| GSM8K | 1,319 (2,638) | 16.5 / 1,286 / 16.5 | 94.39% | 0.496 (0.432–0.552) | 0.953 | 0.213 (0.184–0.243) | .370/.606/.460 | .041/.091/.056 |

*Source development was used for model/threshold selection; show it as development evidence, not held-out generalization.

Example raw confusion matrices (HELP, NO_CHANGE, HURT; true rows, predicted columns):

**MATH source test**

```text
[[60,  5,  0],
 [41,281, 10],
 [ 1,  2,  0]]
```

**CSQA target test**

```text
[[ 8,  13,  0],
 [42,1334,  0],
 [ 0,  69,  0]]
```

Question-equivalent CSQA confusion mass (each question has total weight one):

```text
[[ 4.0,   6.5, 0.0],
 [21.0, 667.0, 0.0],
 [ 0.0,  34.5, 0.0]]
```

The target model never predicts HURT. HURT precision is therefore undefined as a mathematical ratio (the saved display convention uses zero); recall is zero. These records do not justify a harm-prevention or safety claim.

## Paired policy effects and Pareto view

Completion minus never is +15.25 pp on MATH source test (95% CI +11.75 to +18.75), +16.70 pp on MATH500 (+13.40 to +20.10), and +1.06 pp on GSM8K (+0.57 to +1.59). On CSQA it is +0.55 pp (+0.14 to +1.09) with the sparse-event caveat above. The ML-minus-completion effect is exactly 0 on source test and CSQA because their selected recheck outcomes happen to preserve equal accuracy; on MATH500 and GSM8K it is slightly negative as reported above.

Among implementable fixed operating points (oracle excluded), observed Pareto points are: source test—Never, Completion, Always; CSQA—Never and Completion/ML (coincident); MATH500—Never, Completion and Frozen ML; GSM8K—Never, Completion and Frozen ML. Source-development Never and Frozen ML are the two points. The source-test Always result is the highest accuracy but costs the most; the oracle appears as `hindsight-only`, not as an achievable policy. Pareto status is descriptive over these four policies and does not establish optimality over untested policies.

## HELP/HURT and completion

On source test, 62 of 65 HELP rollout rows have `length` status; completion selects them and selects 1 HURT row (with 2 additional HURT rows among `stop`). On CSQA, completion selection is only 25 of 733 questions (50 of 1,466 rollout rows): it captures 8 HELP rows (4 question-equivalents) and no HURT rows in this saved test, while the full set has 69 HURT rows (34.5 question-equivalents). MATH500 has two rollouts per question; length-state outcomes have HELP 179 / 454 rows and HURT 12 / 454, while stop-state has 8 HELP and 10 HURT / 546. GSM8K length has 28 HELP / 76 rows and no HURT; stop has 5 HELP / 2,562 and 33 HURT. These subgroup rates are descriptive and do not justify a causal claim that length termination itself causes correctability.

The clustered HURT rate intervals are in `MAIN_POLICY_COMPARISON.csv`; selection reduces harm events relative to always-recheck in these fixed logs, but event sparsity and one frozen protocol limit inference. Individual class and confusion support counts are in `CLASSIFIER_METRICS_CONFUSION.csv` and `CONFUSION_MATRICES.csv`.

Completion is the largest explicit gating signal, but only its observed association is established. Source test selects 107/400 length-status questions and captures 62/65 HELP rows; the frozen gate has 109 selected rows and captures the same 62 HELP rows. In CSQA the completion policy and gate decisions are identical. MATH500 and GSM8K show a few rollout disagreements, not accuracy gains. The Phase 2 training-only feature diagnostic found just six completed/nonempty beneficial switches and a nested-OOF residual-score AUC of 0.449; that signal is too sparse to claim the remaining 18 surface features reliably predict residual value. Completion status patterns also vary substantially across tasks, so a protocol shift can change their usefulness. See `outputs/ieee_phase2_feasibility/run_2026-11-16_01/feature_associations.csv` and `OVERRIDE_OPPORTUNITY.md`; these prior analyses were exploratory, not causal.

## Computation and limitations

Token savings count `base_actual_tokens + selected * action_actual_tokens`; neutral rechecks still incur tokens. Source-development gate mean total cost (4,081.3) is 26.09% below always (5,521.7), consistent with the earlier reported approximately 4,081 versus 5,522. The exact replay table gives 23.22% savings on MATH source test and 50.53% on CSQA target test. These are replay estimates only. `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/timing_evidence.json` has no per-call timing columns and explicitly records no trustworthy measured latency savings. Offline model scoring overhead is not a live timing measurement and is not reported as an efficiency gain.

All historical test sets have been inspected repeatedly. MATH500/GSM8K are additional published evaluation splits but are not pristine here: this analysis makes them newly scored with the frozen artifact, not newly collected or independently generated evidence. No significance claim is made for ML versus completion. The many cohort/policy comparisons are unadjusted and intended to describe rather than select a policy.
