# Override opportunity and bounded CPU experiment

16 November 2026. Only `outputs/v0.2/data/prepared/source_train.parquet` and, after candidate freezing, `source_development.parquet` were opened for outcome analysis. No source-test, target-test, or adaptation outcomes entered this run. Historical test numbers in the preceding audit were read as background, not selection data. Preservation hashes read original files as bytes without parsing test labels.

## Definitions and validity limitations

Completion rechecks iff `base_done_reason == 'length'`. “Completed” means `stop`; `unknown` is separate. A nonempty publisher-extracted `base_answer` is an **answer-presence proxy**, not proof of syntactic or mathematical validity. MATH answers include expressions, sets, and ordered pairs; applying the live A–E/numeric parser would silently change the protocol.

We therefore report both completed/nonempty and a narrower completed/simple-numeric-syntax subset (signed integer, decimal, or fraction spelling). The latter is a syntactic diagnostic, not a complete MATH validator. Other nonempty strings are **unassessed**, not automatically invalid. No outcomes or keys were changed. “Initially wrong” groups require outcome labels and are diagnostic strata only, never gate inputs.

HELP = wrong→correct, HURT = correct→wrong, NO_CHANGE = unchanged correctness, not unchanged wording. Each question has total weight one; the development duplicate has weight 1/2 per row. Group counts below overlap and must not be added together.

## Opportunity counts

| Partition / group | Questions | HELP | HURT | NO_CHANGE | Initial errors | Beneficial switches vs completion | Accuracy headroom within group, pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training: all | 1,199 | 162 | 15 | 1,022 | 245 | 10 | 0.8340 |
| Training: completed, nonempty | 918 | 6 | 13 | 899 | 13 | 6 | 0.6536 |
| Training: completed, simple numeric syntax | 599 | 0 | 6 | 593 | 2 | 0 | 0 |
| Training: incomplete/length | 278 | 154 | 2 | 122 | 229 | 2 | 0.7194 |
| Training: missing answer | 1 | 1 | 0 | 0 | 1 | 1 | 100.0, single case |
| Training: unknown completion | 2 | 1 | 0 | 1 | 2 | 1 | 50.0, two cases |
| Training: nonempty initially wrong, outcome-only diagnostic | 244 | 161 | 0 | 83 | 244 | 7 | 2.8689 |
| Development: all | 400 | 47 | 6 | 347 | 77 | 2 | 0.5000 |
| Development: completed, nonempty | 305 | 0 | 4 | 301 | 1 | 0 | 0 |
| Development: completed, simple numeric syntax | 213 | 0 | 2 | 211 | 0 | 0 | 0 |
| Development: incomplete/length | 94 | 47 | 2 | 45 | 75 | 2 | 2.1277 |
| Development: missing answer | 2 | 1 | 0 | 1 | 2 | 0 | 0 |
| Development: nonempty initially wrong, outcome-only diagnostic | 75 | 46 | 0 | 29 | 75 | 0 | 0 |

Full strata, row counts, wrong-answer numeric subsets, and costs are in `OPPORTUNITY.csv`. Development has 401 rows/400 questions. For completed/nonempty, 306 rows become 305 question-equivalents. No original MATH difficulty level is inferred.

The **10 training opportunities** comprise eight beneficial added rechecks outside the heuristic and two beneficial skips avoiding HURT inside it. The eight additions comprise six completed/nonempty, one missing answer, and one unknown completion. Conversely, blindly reversing completion risks **167 accuracy-harming switches**: 13 added harmful rechecks and 154 skipped fixes. The two development opportunities are both skips of harmful revisions; **there is no recorded additional helpful correction outside completion on development**.

### Hindsight costs and bounds

Two oracles are distinguished:

- **Accuracy-improving-switch oracle:** reverse completion only on an accuracy-beneficial transition; retain its neutral calls. This isolates additional correction/harm-avoidance opportunity.
- **Minimum-token accuracy oracle:** select only observed HELP; also remove all neutral calls. Both are unattainable hindsight policies. Neutral-call savings do not establish learned semantic-error prediction.

| Partition | Completion accuracy / tokens | Accuracy-switch oracle accuracy / tokens | HELP-only oracle accuracy / tokens |
|---|---|---|---|
| Training | 92.2435% / 4,096.207 | 93.0776% / 4,099.617 | 93.0776% / 3,494.369 |
| Development | 92.0000% / 4,121.013 | 92.5000% / 4,079.885 | 92.5000% / 3,396.988 |

Training's eight added fixes cost **16,592 total tokens**, and skipping its two harmful calls saves **12,503**, for a net **+3.410 tokens/question** under the accuracy-switch oracle. The six completed/nonempty fixes alone cost **11,795 tokens**, or **12.849 per member of that subgroup**. Development's two hindsight harmful-call skips save **16,451 total tokens**, or **41.128/question** overall. These costs are observed outcome diagnostics, never inputs to deployment decisions.

### Crucial training-transcript inspection

All six completed/nonempty HELP cases already display the mathematical result accepted after revision. The saved extracted strings differ mainly in format:

| Training example | Initial extracted representation | Revised representation | Interpretation |
|---|---|---|---|
| 253 | `3x³-5x²+8x-35` | `3x^3-5x^2+8x-35` | Same polynomial, superscript versus caret notation |
| 1203 | `<answer>(3-1)</answer>` | `(3-1)` | Initial transcript explicitly gives `(3, -1)` and verifies both equations; wrapper/normalization difference |
| 1226 | `Finalanswer:(0.50.6)` | `(0.50.6)` | Initial transcript gives `(0.5, 0.6)`; extra answer-prefix artifact |
| 1355 | `36x³+8x²+27x+6` | `36x^3+8x^2+27x+6` | Same polynomial, superscript versus caret notation |
| 1373 | `}frac(29,4)` | `frac(29,4)` | Initial transcript derives 29/4; stray-brace/normalization difference |
| 1494 | `Finalanswer:-3x^2+9x-11` | `-3x^2+9x-11` | Same polynomial, answer-prefix artifact |

The additional missing-answer example 572 already displays `1/4` in its initial transcript, while `base_answer` is empty. Exact records are preserved in `training_completed_HELP_evidence.json` (seven completed HELP rows, including that missing answer).

This is evidence of **extraction/normalization-sensitive apparent recovery**, not six verified semantic corrections. It does not reproduce the publisher's grader or justify changing the archived labels. No alternate scoring was used for fitting, policy selection, or reported accuracy. No format-specific rule was added after this inspection. An independently versioned grading sensitivity study would be necessary to quantify corrected scoring across a full cohort.

## Predictability using existing initial-only features

All 19 production features were retained. No features were selected after inspecting their associations. Within completed/nonempty training answers, raw-direction univariate AUC for beneficial switches is **0.7175 for response log-length**, **0.4940 for uncertainty markers**, **0.4852 for backtracking**, and **0.4910 for question log-word count**. There are only **six positive events**. These are same-sample exploratory associations, with multiple correlated features, not validated accuracy gains or causal explanations.

The residual model's outer-OOF score AUC is **0.4490** for those six completed/nonempty positives. In the incomplete group it is **0.5960**, based on only **two** beneficial skips. Constant completion indicators cannot explain within-stratum variation. These results provide no persuasive generalizable residual signal; they do not prove that all possible models/features would fail. See `feature_associations.csv` for every inspected feature, including direction rather than selecting whichever orientation looks strongest.

## One predeclared candidate, no search expansion

Implemented in `../feasibility.py`; fixed choices are in `analysis_plan.json` written before model fitting/development loading.

- Ridge regression, **alpha=100**, standardized original 19 features; scaler and fit use question-equal weights within each training fold. One model family, no alpha search, no class balancing.
- Fit target is signed accuracy advantage of switching completion: `(1−2h) * d`.
- Deployment score subtracts `(1−2h) * 0.01 * c_hat/1000`, where `c_hat` is the fit-partition mean revision cost for the completion state. Thus deployment sees a training-derived cost estimate, never actual future tokens. One probability point per 1,000 extra tokens is an explicit illustrative trade-off, not a discovered optimal business cost.
- A strict score margin chooses overrides. Grid: **0, .01, .025, .05, .1, .2, 1,000,000**; the last is the effective no-override option in this run. Among margins with no empirical accuracy loss and no increased HURT on inner validation, maximize realized net utility, then minimize tokens, then prefer larger margin. These empirical constraints are **not** a confidence guarantee.
- **Five outer / three inner stratified group folds**, seed 46 with documented inner offsets. Each outer margin sees inner OOF data only. Outer held-out predictions evaluate it. There are 15 inner fits, five outer fits, and one final fit: **21 small CPU fits**.
- Final margin uses training outer-OOF scores as a tuning surface, not an unbiased reported validation score. Candidate coefficients, scaler, cost estimates, and threshold were frozen to JSON before development loading. The development comparison is descriptive because that partition was inspected previously.
- The frozen ML gate is never retrained. Its source-training results are explicitly **in-sample**; its historical development results are selection-set results, not a fair fresh independent comparator.
- No new production artifact format, deployed model, gate schema, or Streamlit change.

## Actual policy results

### All source-training questions

| Policy / evaluation status | Accuracy % | Delta vs completion, pp | Avg. tokens | Recheck % | Overrides % | Additional HELP / HURT | Beneficial overrides missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Completion, fixed | 92.2435 | 0 | 4,096.207 | 23.1860 | 0 | 0 / 0 | 10 |
| Frozen ML, in-sample | 93.0776 | +0.8340 | 4,054.771 | 23.3528 | 1.6681 | +8 / −2 | 0 |
| Residual, nested OOF | 92.2435 | 0 | 4,109.928 | 23.9366 | 0.7506 | 0 / 0 | 10 |
| Residual ranking, matched rate diagnostic | 92.1601 | −0.0834 | 4,086.508 | 23.1860 | 0.3336 | −1 / 0 | 10 |
| HELP-only oracle, unattainable | 93.0776 | +0.8340 | 3,494.369 | 13.5113 | 11.0092 | +8 / −2 | 0 |

The residual OOF policy adds **nine neutral rechecks**, capturing none of the 10 beneficial switches. It costs **13.721 additional tokens/question**: **0.335% more** at equal accuracy. The matched-rate diagnostic selects exactly the completion quota using initial-only predicted recheck-benefit ranks and stable group-ID tie-breaking. It loses one helpful correction. It is a batch diagnostic, not a second selected candidate or a test-tuned operating point. Matched recheck rate is not necessarily matched token cost.

### All source-development questions

| Policy / evaluation status | Accuracy % | Delta vs completion, pp | Avg. tokens | Recheck % | Overrides % | Additional HELP / HURT | Beneficial overrides missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Completion, descriptive | 92.0000 | 0 | 4,121.013 | 23.50 | 0 | 0 / 0 | 2 |
| Frozen ML, retrospective selection set | 92.0000 | 0 | 4,081.343 | 23.25 | 0.75 | 0 / 0 | 2 |
| Frozen residual, descriptive confirmation | 92.0000 | 0 | 4,121.013 | 23.50 | 0 | 0 / 0 | 2 |
| Residual ranking, matched rate diagnostic | 91.7500 | −0.25 | 4,120.813 | 23.50 | 0.50 | −1 / 0 | 2 |
| HELP-only oracle, unattainable | 92.5000 | +0.50 | 3,396.988 | 11.75 | 11.75 | 0 / −2 | 0 |

Outer margins were **.1, .1, no-override, no-override, 0**. Final training-only selection chose **no overrides**. This fallback is an acceptable negative outcome; it does not demonstrate a learned advantage. Frozen ML's 0.963% development token reduction versus completion is historical, not a new result from this candidate.

### Completed-answer performance

On the 918 completed/nonempty training questions, completion and residual OOF both achieve **98.5839%**. Residual rechecks **0.8715%** (eight questions), increasing average tokens from **1,929.656 to 1,947.577**, with no useful overrides. On 305 completed/nonempty development questions, both achieve **99.6721%**, zero rechecks, and **1,873.272 tokens**. Development simple-numeric-syntax cases (213 questions) have **100% initial accuracy**, so no semantic improvement is available there. This small syntax subset is not representative of all MATH answers.

### Paired uncertainty and interpretation

`POLICY_COMPARISON.csv` contains 30 measured rows: two partitions × five policies × three scopes. It includes question-equivalent HELP/HURT, missed beneficial overrides, tokens, rates, and paired intervals. Two thousand bootstrap draws resample questions, keeping repeated rollouts together.

- Residual OOF accuracy difference: **[0, 0] pp**; token difference: **[5.177, 25.175]** tokens/question.
- Frozen residual development accuracy and token differences: **[0, 0]** because decisions are identical to completion.
- Matched-rate development accuracy difference: **[−0.75, 0] pp**; token difference **[−13.028, 12.428]**.
- Frozen ML development token difference: **−39.670**, interval **[−104.268, 4.605]**.

These are conditional empirical paired intervals. Degenerate zero intervals reflect no observed differences, not proof of population equivalence or safety. OOF fits overlap; the bootstrap does not propagate fitting/selection uncertainty. Development is previously exposed. No population superiority or statistical-significance claim is made. There is no reason to spend additional compute on inference from these results.

## Reproduction and safeguards

From the project root, using a **new** run name:

```powershell
.\.venv\Scripts\python.exe -B outputs/ieee_phase2_feasibility/feasibility.py another_fresh_run
.\.venv\Scripts\python.exe -B outputs/ieee_phase2_feasibility/verify_feasibility.py another_fresh_run
```

The experiment refuses an existing run directory; verification evidence uses exclusive creation. No package installs or networking occur in either script. Source data are read-only. `frozen_candidate.json` contains the experimental linear coefficients, not a production artifact. `training_oof_predictions.csv`, `development_predictions.csv`, and `folds.json` make numerical comparisons auditable.

Core references: `rechecklab/data.py:34,74,233,248` labels/group validation; `features.py:88,117` initial allowlist; `metrics.py:9,22` weighting/policy accounting; `predictor.py:121,303` frozen predictions/decisions. Experiment references: `../feasibility.py:52` strata, `:65` opportunity, `:90` fold fitting, `:98` inference-time score, `:106` margin selection, `:121` paired intervals, `:154` matched-rate diagnostic, `:161` execution and freeze-before-development order.
