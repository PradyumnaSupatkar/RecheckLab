# RecheckLab experiment 1: model and balancing comparison

Selected **hgb_full_baseline**, threshold **-0.0456048059794106**; fit on source training only using the unchanged 19 features.

It won by pooled nested-CV utility **152.000** question-equivalents; nested-CV accuracy **92.2435%**, average extra tokens **1546.274**.

Selection used 5 stratified group outer folds and at least 3 inner folds per outer fit. For each configuration, inner OOF predictions tuned its threshold before scoring the untouched outer fold. All configurations used identical splits. Utility then recheck tokens determined ranking; remaining ties used declared candidate order.

Class balancing uses fit-subset question-weighted class mass raised to power 0, 0.5 or 1, normalized to preserve total sample weight. Logistic scaling uses question weights only. Logistic C values are 0.1, 0.5, 2; max_iter=1500. HGB uses 100 iterations, 7 maximum leaves, minimum leaf size 20, learning rate 0.05, L2=5. All estimator seeds are 42.

The selected model's pooled outer OOF scores set its final threshold using training labels only. The selection JSON and compatible model artifact were saved before development was loaded. Development did not change any choice.

| Source-development policy | Accuracy % | Helpful fixes | Harmful flips | Recheck % | Average total tokens | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected CV gate | 91.7500 | 47.000 | 3.000 | 78.5000 | 5070.836 | 0.558724 | 0.391613 |
| Existing frozen gate | 92.0000 | 47.000 | 2.000 | 23.2500 | 4081.343 | 0.558724 | 0.391613 |
| Completion rule | 92.0000 | 47.000 | 2.000 | 23.5000 | 4121.012 | N/A | N/A |
| Always recheck | 91.0000 | 47.000 | 6.000 | 100.0000 | 5521.676 | N/A | N/A |
| Keep initial answer | 80.7500 | 0.000 | 0.000 | 0.0000 | 3005.635 | N/A | N/A |

Selected minus existing frozen development accuracy: **-0.2500 percentage points**.
Helpful/harmful counts are question-equivalents; repeated rollout rows share weight. Action-only baselines have no three-class probabilities, so their classifier metrics are N/A.

## Interpretation and safeguards

- Each outer threshold is selected without its outer validation labels. Selecting the best configuration among these outer CV scores still introduces winner-selection optimism.
- The final threshold is optimized on the selected configuration's source-training OOF predictions. Its optimized training utility is a tuning statistic, not an unbiased estimate.
- Source development was previously inspected, and the existing frozen gate was selected on it historically. This one-pass development comparison is descriptive, not fresh confirmation.
- No source-test or target-test outcomes are loaded for selection or evaluation. Any future evaluation on those previously inspected tests must be labeled exploratory.
- One partition and one CV seed; HURT events are rare; no improvement in target accuracy is established.

No new dataset, LLM response, download, application change or deployment. All writes are inside outputs/accuracy_experiments/. integrity.json verifies original project files byte-for-byte (excluding existing environment/cache directories). Integrity hashing reads bytes without parsing test outcomes.

Runtime: 14.599 seconds. See candidate_comparison.csv for all 11 configurations, cv_fold_metrics.csv and cv_predictions.csv for nested decisions, and source_development_comparison.csv for the single development evaluation.
