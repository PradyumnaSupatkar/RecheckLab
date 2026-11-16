# IEEE-style manuscript outline

## Abstract

State the question: whether a lightweight learned gate improves selective LLM self-verification over completion-based decisions, and how policy accuracy and recorded inference cost vary by task. Identify the secondary SEVRA replay corpus, frozen Qwen3-4B active-verify responses and 19-feature gate. Report the key negative finding (tie on CSQA, tie/slight cost disadvantage on source test, no added accuracy on MATH500/GSM8K), plus selective-computation results with calibrated wording. Say token counts are replay estimates, not measured latency. Include the exposed-test and protocol limitations. Do not call the source holdout MATH Level 5.

## 1. Introduction

- Motivation: verification can help or harm; rechecking every answer has computation cost.
- Distinguish policy-level benefit of selecting calls from incremental benefit of a learned selector.
- Research questions RQ1–RQ4 in `RESEARCH_QUESTIONS.md`.
- Contributions limited to paired replay comparison, cross-task policy/cost description, and negative ML-vs-completion evidence.

## 2. Related work and distinction from SEVRA

- Summarize selective verification and adaptive inference with verified primary references from `outputs/ieee_phase4_discovery/run_2026-11-16_01/REFERENCES.md`.
- Closest direct antecedent: Sajib Acharjee Dip, Dawei Zhou, and Liqing Zhang, *Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning*, arXiv v1 submitted 18 June 2026 ([primary record](https://arxiv.org/abs/2606.19808)); it is also the source of the recorded recovery-outcome release.
- Describe SEVRA as the provenance of the paired response records and its learned recoverability-aware controller.
- Explain that this manuscript analyzes a distinct lightweight, initial-only HELP/NO_CHANGE/HURT gate and its completion heuristic using the published records; it does not reproduce all SEVRA claims or claim a new controller.
- State that novelty is an empirical question and that primary-data independence is limited.

## 3. Research questions

List RQ1 policy improvement over never; RQ2 ML incremental value over completion; RQ3 cross-task heterogeneity; RQ4 three-class event prediction. Predeclare the primary comparisons in a future protocol before any fresh data collection. The present report is retrospective and cannot be called preregistered.

## 4. Methodology

- Source frozen v0.2 artifact: balanced histogram gradient boosting, 19 initial-response features, HELP/NO_CHANGE/HURT order, threshold 0.25968145276947807, harm weight 1.0.
- Policies: never, always, `base_done_reason == length`, frozen artifact; hindsight oracle shown only as an unattainable bound.
- Outcomes inherited from SEVRA paired labels; define HELP/HURT/NO_CHANGE through base/final correctness as implemented in `rechecklab/data.py:34-86`.
- Question hash grouping; question-equal weights and rollout averaging; token cost formula; metric definitions.
- State classification argmax metrics differ from downstream policy accuracy.

## 5. Experimental setup

Use a dataset/protocol table from `DATASET_PROTOCOL_COMPARISON.csv`:

- MATH source development: 400 questions, 401 active-verify rows; selection data, descriptive.
- MATH source test: 400 questions, 400 rows; historical/exposed.
- CSQA target test: 733 questions, 1,466 rows; historical/exposed, transfer domain.
- MATH500: 500 questions, 1,000 rows; pinned SEVRA external replay diagnostic.
- GSM8K: 1,319 questions, 2,638 rows; pinned SEVRA external replay diagnostic.

Report zero canonical query overlap among these cohorts, and separately verify train-source disjointness. Describe model-family/action and action-description hash checks, plus absent checkpoint/prompt provenance. No model retraining or new LLM inference.

## 6. Results

1. Main policy table: accuracy and cluster CI, recheck, total tokens, token savings, HELP/HURT, Pareto status.
2. ML classification table and confusion matrix; emphasize HURT detection failures and class imbalance.
3. Paired effect sizes with question-cluster intervals and unadjusted paired randomization tests; distinguish sparse CSQA evidence.
4. Cost frontier figure and completion-state HELP/HURT plot.
5. Completion vs gate decision agreement and dataset/protocol table.
6. External cohort diagnostics, separately labeled.

Use figures `accuracy_vs_tokens.png`, `help_hurt_by_completion.png`; CSV source tables are the machine-readable result tables.

## 7. Discussion and negative findings

- Selective action can have large replay accuracy value on some tasks, little effect on others, and negative value for always-recheck on CSQA.
- Completion captures most observed useful MATH source events; policy behavior varies by task.
- Frozen ML adds no demonstrated accuracy benefit to completion; any token changes are small/modest and not live latency.
- MATH500/GSM8K do not rescue a learned-gating advantage.
- The measured associations do not establish that completion status causes recoverability.

## 8. Threats to validity

- Secondary SEVRA records; one solver family/action; possible prompt/checkpoint mismatch not auditable.
- MATH source split is not verified official Level 5.
- Upstream labels/gold omitted; no independent regrading.
- Repeated rollouts clustered, but only two in CSQA/MATH500/GSM8K; bootstrap conditional on saved pool.
- Historical test exposures and retrospective analysis; no fresh independent confirmation.
- Protocol, solver, parser and prompt shifts; threshold and source-dev selection optimism.
- Token accounting does not measure latency, GPU load, energy, or memory.
- Sparse HURT and multiple unadjusted comparisons.

## 9. Conclusion

Conclude that on these recorded outcomes, selective rechecking can change the accuracy-cost trade-off, but the learned 19-feature gate does not show a meaningful held-out advantage over completion gating. The result is a bounded empirical negative finding; broader deployment or superiority claims require prospective, independent records and measured system costs.
