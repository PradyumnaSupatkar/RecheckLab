# Frozen ML gate versus completion: disagreement audit

## Scope and method

This is a read-only audit of the active v0.2 histogram-gradient-boosting gate on the four recorded-response cohorts. It does not retrain, tune the threshold, generate LLM responses, or edit correctness labels. The artifact was loaded with SHA-256 `9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439`; the frozen harm weight is 1.0 and threshold is 0.25968145276947807. The score is `p(HELP) - harm_weight * p(HURT)` and the gate rechecks only when this score is strictly greater than the frozen threshold. Completion rechecks when `base_done_reason == length`.

Question identity uses `rechecklab.data.question_hash`: Unicode NFKC normalization, case folding, whitespace collapse, then SHA-256. All outcome, accuracy, call, and token metrics average repeated rollouts within each question before weighting questions equally. Question-cluster percentile intervals use 10,000 bootstrap resamples. They condition on the finite recorded outcomes and are not evidence of equivalence or fresh generalization.

The gate was recomputed from the frozen artifact for all cohorts. On MATH source test and CSQA target test, recomputed gate decisions matched the saved v0.2 prediction file. The 30 discordant records are enumerated in [`tables/DISAGREEMENTS.csv`](tables/DISAGREEMENTS.csv); every row includes its full canonical question hash, rollout, initial status, score and threshold, actions, correctness transition, and signed token difference. It contains no prompt text or gold answer strings. [`tables/DISAGREEMENT_QUESTION_EFFECTS.csv`](tables/DISAGREEMENT_QUESTION_EFFECTS.csv) gives the corresponding question-level rows.

## Recomputed disagreement counts

| Cohort | Discordant rollout rows | Distinct question groups | Direction | Transition labels on disagreements |
|---|---:|---:|---|---|
| MATH source test | 2 | 2 | ML rechecks; completion skips | 2 NO_CHANGE |
| CommonsenseQA target test | 0 | 0 | Identical decisions | None |
| MATH500 | 18 | 9 | ML skips; completion rechecks | 4 HELP, 14 NO_CHANGE |
| GSM8K | 10 | 5 | ML skips; completion rechecks | 3 HELP, 7 NO_CHANGE |
| **Total** | **30** | **16** |  | **7 HELP, 23 NO_CHANGE, 0 HURT** |

The four paired-response cohorts therefore have 30 rollout disagreements but only 16 independent question clusters. Every MATH500 and GSM8K discordance occurs on a `length` initial status; each question contributes two recorded intervention rollouts. The two MATH source-test disagreements are separate, single-rollout `stop` questions. CSQA has 1,466 identical rollout decisions.

## Paired policy effects

Accuracy and token intervals are question-cluster bootstrap 95% percentile intervals. “Revision tokens” are selected `action_actual_tokens` averaged per question; total replay tokens include the same initial-generation cost under both policies.

| Cohort | ML accuracy | Completion accuracy | ML − completion accuracy (95% CI, pp) | ML / completion recheck rate | ML / completion revision tokens per question | ML / completion total tokens per question | ML − completion total tokens per question (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MATH source test | 90.000% | 90.000% | 0.000 [0.000, 0.000] | 27.25% / 26.75% | 1,230.808 / 1,220.403 | 4,412.825 / 4,402.420 | +10.405 [0.000, 26.920] |
| CommonsenseQA target test | 76.6712% | 76.6712% | 0.000 [0.000, 0.000] | 3.4106% / 3.4106% | 91.242 / 91.242 | 2,334.847 / 2,334.847 | 0.000 [0.000, 0.000] |
| MATH500 | 75.300% | 75.700% | −0.400 [−0.900, 0.000] | 43.60% / 45.40% | 2,621.098 / 2,750.326 | 6,933.638 / 7,062.866 | −129.228 [−222.945, −49.962] |
| GSM8K | 94.3518% | 94.4655% | −0.1137 [−0.3033, 0.0000] | 2.5019% / 2.8810% | 132.706 / 150.968 | 1,313.093 / 1,331.355 | −18.262 [−36.967, −4.439] |

Across the 16 discordant question groups, the learned policy has no positive accuracy difference: five groups lose accuracy because the gate skips a recorded HELP transition (three MATH500 and two GSM8K groups), while 11 groups have no accuracy difference. No discordant row is HURT. The lower token use on MATH500 and GSM8K is therefore an accuracy-cost trade-off, not a demonstrated ML improvement. MATH source test adds cost with no accuracy change; CSQA is identical. The small fixed-log differences do not establish non-inferiority, equivalence, or future policy value.

## Row-level cost attribution

- The two MATH source-test gate-only rechecks cost 1,718 and 2,444 actual action tokens (4,162 across the rows). Both initial answers are parsed, stop-completed, and correct; both recorded revisions are also correct and leave the parsed answer unchanged.
- MATH500 gate skips 18 completion-selected records. Their recorded action token sum is 129,228 across rollout rows; question-equal average total cost is 129.228 tokens/question lower under ML. The four HELP-labeled rows account for 17,813 of those rollout tokens; the 14 NO_CHANGE rows account for 111,415.
- GSM8K gate skips 10 completion-selected records with 48,175 total recorded action tokens across rollout rows; question-equal average total cost is 18.262 tokens/question lower under ML. HELP-labeled rows account for 10,448 rollout tokens and NO_CHANGE rows for 37,727.

The rollout sums above are descriptive sums across saved action rows. Policy means and bootstrap intervals use question-equal weighting, averaging each question's repeated rollouts first.

## Interpretation limits

Outcome labels are inherited from the SEVRA release. A changed parsed answer does not establish a semantic correction, and a NO_CHANGE label can mean either correct-to-correct or wrong-to-wrong. The public release omits gold answer strings. The output record's parsed base answer is present in every disagreement, but 28 discordances have `length` status and `base_finalizer_used=True`; this says the stored initial answer passed through a finalizer, not that the truncated reasoning text itself contained a complete answer. Raw texts were consulted only for the redacted illustrative cases in [`CASE_STUDIES.md`](CASE_STUDIES.md). They do not support a definitive formatting-only versus reasoning-correction classification.

MATH500 and GSM8K protocol identity is only partially verified because the public response records do not publish the full prompts or exact checkpoint digest. All four cohorts are reused responses, not a new inference experiment. Historical and external replay results remain exploratory as described in the manuscript.
