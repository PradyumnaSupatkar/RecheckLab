# Implementation audit and pilot investigation

Date: 4 October 2026. The original files were read before changing implementation/results. `BASELINE_v0.1_SHA256.json` records 24 unchanged original raw/prepared/model/report files. New outputs are separate in `outputs/v0.2`.

| Starting observation | Finding and correction |
|---|---|
| Five overloaded UI tabs | Confirmed. Replaced with exactly Demo and Results; collection/setup instructions moved to README. |
| Every target argmax is NO_CHANGE | Confirmed in the shipped matrix: HELP 21, NO_CHANGE 1376, HURT 69 rows all predicted NO_CHANGE; macro-F1 0.322777. This is weak transfer, not a class-column bug. |
| Limited-label adaptation does not improve | Confirmed: every v0.1 mean is below frozen accuracy 76.6712%. Retained baseline; v0.2 also has no higher mean. |
| Manual metadata disagrees with evaluation | Confirmed hardcoded finalizer/completion signals. Recorded replay now uses the selected row's actual values; saved evaluation features are checked against UI inputs. |
| Missing numeric features become zero | Confirmed numeric fallback and negative-count clipping. Required signals now reject missing/nonfinite/invalid values. Published empty completion status is explicitly unknown, with a trained indicator; this is documented separately from numeric missingness. |
| Predictor metrics count rows, policy metrics average questions | Confirmed. Fitting, predictor metrics, threshold objectives, policy metrics and uncertainty now consistently weight independent questions. Raw counts remain labeled diagnostics. |
| API substitutes defaults for a partial custom dataset pair | Confirmed `source is None or target is None`. Partial pairs now fail before public loading or output writes. |
| Protocol/artifact metadata unchecked | Strengthened solver/action/checkpoint/sidecar compatibility, schema/class order/probability validation and matching result hashes. Public full prompts/checkpoint remain unavailable. |

Additional fixes: preserve numeric-looking answer/ID text when loading CSV/JSONL; validate all initial signals across repeated attempts; recompute canonical hashes rather than trusting provided group IDs; reject non-recheck actions; bind the adapted threshold and adaptation plot/tables to the result bundle; clear displayed predictions when input or model changes.

The pipeline continues to derive all four correctness transitions correctly. Gold keys, initial/final correctness, outcome labels, revised responses/answers, future tokens and post-action annotations do not enter features. UI prediction projects onto initial inputs and uses the same predictor/decision function as evaluation. No truth-based override or invented answer is added.

## Model investigation

Fixed majority, logistic/boosting and class-balanced variants were compared using source development only. The selection rule was declared as largest question-weighted development macro-F1, tie smaller log loss, then fixed candidate order. v0.1 selected by development log loss; this change explicitly investigates rare-class performance rather than silently changing the original baseline. Every candidate's held-out results are retained in `outputs/v0.2/reports/model_comparison.csv`, with per-class scores/matrices/log loss in experiment metadata.

The selected balanced boosting model reaches target macro-F1 about 0.394 (majority/unbalanced boosting about 0.323). Balanced logistic regression is worse, about 0.110. Balancing does not universally solve transfer. The selected model recovers some HELP classifications but HURT recall remains zero. Unbalanced logistic has better target log loss than the selected model, but target results did not select the winner.

Source maths has many more helpful transitions and very few harmful ones compared with CommonsenseQA. Outcome prevalence and initial feature means are saved for source training versus the reserved adaptation pool, alongside test diagnostics. These shifts are plausible explanations for transfer weakness, not established causal explanations. No feature ablation or independent multi-seed/model replication was performed.

The learned frozen target policy equals the completion rule in accuracy, harmful flips, recheck rate and replay tokens in this run. A better macro-F1 therefore does not establish a better policy. Five threshold-adaptation subset sequences share the same fixed test and overlap each other; their SD is not an independent-experiment confidence interval. Budget200/repeat0 is displayed by a predeclared rule, not selected because it looks favorable. No tested budget has higher mean accuracy than the frozen policy.

## Remaining limits

One solver family/direction/split; inherited correctness annotations without gold keys; extremely rare source HURT labels; incomplete released generation settings; uncalibrated probabilities; only threshold adaptation; replay token estimates excluding classifier overhead. The experiment evaluates one specified bounded recheck and makes no compute-optimality, unrestricted-reasoning, novelty, or guaranteed-improvement claim. The optional Ollama collector remains a separate documented CLI; its automated tests mock inference and do not constitute newly generated research data.

Completed verification is recorded in `GUIDE_PROGRESS.md` and `outputs/v0.2/reports/TEST_RESULTS.txt`.
