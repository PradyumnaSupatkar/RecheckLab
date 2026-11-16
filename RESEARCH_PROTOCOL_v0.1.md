# Protocol v0.1

## Research question

Does a source-trained LLM self-correction predictor transfer to a different task, and does adjusting its decision threshold using a limited number of target examples improve that transfer?

## Unit and outcomes

An independent question is the main unit. A row contains its fixed initial answer and one recorded revision rollout. Outcomes are HELP (incorrect to correct), HURT (correct to incorrect), or NO CHANGE (either correct to correct or incorrect to incorrect). Changed wording or changed incorrect answers do not imply changed correctness.

Canonicalized question hashes, not row IDs, determine partitions. Different initial attempts of a repeated question also stay in the same partition. All revision rollouts of the same initial attempt stay together.

## Data and partitions

Source: active verification records from the SEVRA MATH training split. Target: active verification records from CommonsenseQA. Gold keys are absent; correctness comes from the published annotations. Public records are inherited; the project does not claim to have generated them.

Source train/development/test fractions: 0.6/0.2/0.2, seed 42. Target adaptation/test fractions: 0.4/0.6, seed 43. Their question hashes must be disjoint. The reserved target test cannot be used for choosing a model, adaptation method, threshold, or label budget.

The original target release was an evaluation split. Reserving part of it for adaptation defines a new protocol; it prevents direct claims of reproducing the paper's published target scores.

## Features

The allowlist covers question length, numeric content, initial response length, word diversity, uncertainty/backtracking/checking expressions, initial actual token count, completion flags, and initial answer presence. Correctness, gold answers, revised text, revised answers, and future token counts are excluded. Public difficulty/verification annotations are also excluded to avoid relying on features whose estimation protocol would need replication.

Uncertainty words are linguistic features, not calibrated confidence. Predicted outcome probabilities are model estimates, not guaranteed risks.

## Models and decision

Compare logistic regression and histogram gradient boosting using only source development cross-entropy. Probability columns have an explicit fixed class order. Fit the classifier on source training data only.

Revision score = P(HELP) minus harm_weight × P(HURT). Default harm_weight = 1.

Choose the source threshold on source development utility, breaking ties in favor of fewer realized rechecking tokens. At runtime, recheck if score exceeds that frozen threshold.

For each target sample budget, estimate a new threshold using only sampled target adaptation questions and their recorded outcomes. No source-classifier weights or probability estimates are updated. Five nested sample sequences are drawn with seeds 142–146. Tested budgets are 25, 50, 100, 200, and 400 questions. This baseline is deliberately called threshold adaptation, not full calibration.

## Baselines and metrics

Keep the first answer; always recheck; recheck only an initially budget-exhausted response; frozen source gate; target-threshold-adapted gate.

Accuracy and event rates average rollouts within each question, then average questions. Report harmful flips as both a fraction of all questions and conditional on initial correctness. Report recovery conditional on initial incorrectness. Token accounting includes realized initial and selected revision prompt-plus-generation tokens. It excludes classifier CPU overhead and is a replay estimate.

Macro-F1 and confusion matrices describe the prediction problem. Because NO CHANGE is frequent, classifier accuracy alone is not a useful headline metric. Paired bootstrap intervals over questions describe accuracy differences against keeping the initial answer, conditional on this partition and the recorded runs. Adaptation SD bars describe subset variability and are not confidence intervals.

## Interpretation rules

Do not choose the best target-test result and present it as a preselected model. Compare the learned gate against both unconditional policies and the simple completion baseline. Retain unfavorable adaptation results and rare-class counts. If a sample contains too few harmful examples, state that limitation rather than inferring a reliable harm probability.

This pilot does not establish novelty, causality about reasoning length, or a universal benefit from more labeled data. Collection errors, truncated reasoning, invalid final formats, and stochastic revisions require separate audits.

## Next extensions

Collect own GSM8K/BoolQ response pairs under a fixed documented model and revision instruction. Audit extraction. Repeat source/target directions, source split seeds, and solver checkpoints. Compare threshold-only adaptation with probability calibration and target-data classifier fitting. Add random budget-matched gates and a longer first solve. Evaluate end-to-end latency and gate overhead when actual inference is available.
