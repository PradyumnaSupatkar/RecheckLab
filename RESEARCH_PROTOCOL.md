# Protocol v0.2: one specified bounded recheck

## Research question and units

Predict HELP/HURT/NO_CHANGE for one frozen initial response followed by one specified recheck. A separate gate chooses whether to spend the revision call. This pilot studies maths-to-CommonsenseQA transfer and threshold-only adaptation. It does not evaluate arbitrary amounts of uninterrupted internal reasoning.

HELP = incorrect initial to correct revised answer. HURT = correct initial to incorrect revised answer. NO_CHANGE = either incorrect-to-incorrect or correct-to-correct, regardless of wording changes. Skip means keep the initial answer; no ground-truth-driven override or invented correction is permitted.

A canonical question is the independent unit. Unicode NFKC, case folding and whitespace normalization define its SHA256 group. All initial attempts and revision rollouts of an identical question share a partition. Rollout rows share total weight one per question in fitting, model selection, threshold objectives and both types of evaluation metrics. Raw row counts are supplementary diagnostics.

## Data and compatibility

Use only the pinned SEVRA `active_verify` records: source MATH train, target CommonsenseQA. Labels are derived from inherited correctness annotations; answer keys are absent. Model `qwen3:4b`, action description and published protocol identity must agree across tasks. The card declares a 4096-token base generation cap and up to 512 non-reasoning finalizer tokens. Public logs lack checkpoint digest and complete prompts/settings, so full replication compatibility cannot be established.

Custom source and target must both be supplied. Collector sidecars bind model digest, instructions, budget, temperature, thinking and seeds; different protocols/checkpoints are rejected. Only `recheck`/`active_verify` actions are accepted. Identical questions cannot occur in source and target. All initial metadata and actual integer token counts are validated; token estimates are never substituted.

Source split: 60% train / 20% development / 20% test, seed42. Target: 40% reserved adaptation pool / 60% fixed test, seed43. Repartitioning a formerly published evaluation release defines a new experiment and cannot reproduce the original paper's scores. Model settings and decisions are fixed before target test labels are evaluated.

## Pre-recheck inputs and missingness

The shared feature builder reads only question, initial response, parsed initial answer, actual initial token count, completion status and initial finalizer flag. Gold answers, correctness, outcomes, revised text/answers/tokens and post-action annotations are excluded. Model/protocol metadata are compatibility checks, not predictive features.

The 18 v0.1 text/initial-signal features remain, plus `initial_completion_unknown`. Two published source rows have an empty completion reason; only the public loader maps these to the explicit `unknown` training category. Custom metadata must explicitly name `stop`, `length` or `unknown`. An empty parsed answer is explicit absence, while missing parsed-answer metadata is rejected. Numeric features and finalizer flags have no imputation. Uncertainty words are linguistic features, not measured confidence.

## Fixed candidates and source selection

Fit on source training only: majority class (training class priors for probabilities), logistic regression, histogram gradient boosting, and class-balanced versions of each learned model. Logistic C=0.5/max_iter1500; boosting100 iterations/7 leaves/L2=5/learning_rate0.05; seed42. Class balancing multiplies question weights by inverse class mass; it is not probability calibration.

Select the largest question-equal source-development macro-F1. Ties at 12 decimal places use smaller question-equal multiclass log loss, then fixed order: majority, logistic, balanced logistic, boosting, balanced boosting. This replaces the v0.1 development-logloss rule to investigate rare-class performance. No candidate or setting is chosen from target results. All candidates' held-out metrics are reported diagnostically without reselection.

Probability columns have explicit order `[HELP, NO_CHANGE, HURT]`. Validate class identity/order, schema version/fingerprint, model metadata, shape, finiteness, [0,1] range and row sums. Missing fitted classes have probability zero in the fixed order. These probabilities are uncalibrated model estimates.

## Decision threshold and adaptation

Score = P(HELP) - harm_weight*P(HURT), default harm_weight1. Recheck iff score > threshold; this exact shared function is used by evaluation and UI. Scan unique source-development scores plus always/never boundaries. Maximize question-equal HELP minus weighted HURT. Tie-break by fewer realized revision tokens and then larger threshold. Token cost is only a tie-breaker in this run: no priced cost objective or compute-optimal decision claim is made.

Threshold-only adaptation keeps classifier weights/probabilities frozen. Sample only the target adaptation pool using five nested sequences, seeds142-146, at budgets25/50/100/200/400 independent questions. Optimize each cutoff on that sample alone. Test labels never determine training, hyperparameters, calibration, thresholds or budgets. The displayed adapted policy is budget200, repeat0, declared before test evaluation; small custom runs use the smallest feasible requested budget. All subsets and unfavorable results are retained.

## Evaluation and uncertainty

Policies: keep initial, always recheck, completion rule (`base_done_reason == length`), frozen source gate, and predeclared adapted gate. Accuracy, helpful/harmful events, recheck rate and tokens first average rows within questions, then questions. Harmful flips are shown over all questions, with initial-correct conditional rates in detailed tables. Realized tokens include the initial call plus the selected revision prompt-plus-generation count. They are replay estimates, excluding gate CPU cost/latency/energy/cache pricing.

Prediction diagnostics: per-class precision/recall/F1, macro-F1, question-weighted confusion matrix, raw row matrix/counts and multiclass log loss (probabilities clipped at1e-15 solely for finite scoring). Every class participates in macro-F1, even with zero support. Majority baseline is explicit. Six hundred question bootstrap draws give macro-F1/log-loss intervals and paired accuracy-gain intervals versus keeping initial, conditional on the fixed fitted model and partition. Adaptation error bars are SD across overlapping subsets, not confidence intervals or independent full experiments.

## Reproducibility and interpretation

Keep original raw/prepared data, model and reports untouched; verify their recorded SHA256 baseline manifest. Rerun to `outputs/v0.2` or a new separate folder. Save the selected model, frozen/adapted thresholds, complete schema/class order/protocol, partition groups, environment versions, candidate scores, features and predictions. The UI checks hashes and recomputes features/probabilities/decisions; its future-answer reveal cannot affect prediction.

Balanced training improving classification does not prove policy superiority. Report rare HURT source labels, zero recalls, domain shift, completion-rule ties and unfavorable adaptation means. No established probability calibration, causal claim about reasoning duration, sufficient label count, universal gain, or novelty is assumed. Public provenance and underlying benchmark terms remain attached.
