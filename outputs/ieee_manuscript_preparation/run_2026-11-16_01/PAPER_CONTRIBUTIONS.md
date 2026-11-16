# Proposed paper contributions and claim boundaries

**Working title:** *RecheckLab: A Cross-Domain Empirical Evaluation of Learned Gating for Efficient LLM Self-Verification*

## Three defensible contributions, conditional on final manuscript verification

1. **A paired replay comparison of selective policies.** We compare never-recheck, always-recheck, a completion rule and one frozen 19-feature gate on the same recorded initial attempts and active-verification outcomes. Each canonical question has total weight one; its repeated intervention rollouts are averaged within question. This isolates the policy decision from differences in generated responses. It is evidence about recorded-response replay, not live execution.
2. **A cross-task account of the accuracy/computation trade-off.** The frozen artifact can be applied without fitting to the pinned SEVRA MATH500 and GSM8K active-verification logs as well as the historical MATH-training holdout and CommonsenseQA target data. Results differ by task: completion gives 75.70% on MATH500 with 13.07% fewer recorded total tokens than always-recheck; on GSM8K it gives 94.47% with 54.59% token savings. These additional cohorts are exploratory external replay diagnostics, not untouched confirmation: the same release omits exact generation prompts and checkpoint digest.
3. **A candid negative result about the incremental value of learning.** The frozen ML gate ties completion on CSQA (identical decisions) and MATH source test (same 90.00% accuracy, slightly more tokens); it is also 0.40 pp below completion on MATH500 and 0.11 pp below on GSM8K. In source development it ties accuracy while using 39.67 fewer tokens/question. These results support the statement that selective computation can trade accuracy and replay token cost, but do not establish that learning improves the policy over a cheap completion signal.

The first two are useful empirical contributions only if the paper clearly identifies the records as secondary SEVRA replay, gives the protocol and provenance limitations, and avoids presenting the outcomes as fresh model-run evidence. Contribution 3 is the strongest supported negative result. The paper should not claim an improved classifier, robust harm prevention, measured latency reduction, or official MATH Level 5 evaluation.

## Verified limitations to put in the abstract/discussion

- MATH source development/test are partitions of the SEVRA MATH `train` split; the source test is not verified official MATH Level 5.
- The active v0.2 model is class-balanced histogram gradient boosting (`max_iter=100`, `max_leaf_nodes=7`, `min_samples_leaf=20`, `learning_rate=0.05`, `l2_regularization=5`, seed 42) over 19 initial-only features; the saved decision threshold is 0.25968145276947807 for `P(HELP) - P(HURT)`. It was selected with source-development results, so development performance is not an independent final estimate.
- Historical target and source test results were repeatedly inspected. They are exploratory historical results, not fresh confirmatory evaluation.
- `data/raw/DATASET_CARD.md` identifies Qwen3-4B, active verification, a 4,096-token base limit and up to 512-token finalization. It omits checkpoint digest and full prompts; the raw benchmark files have no per-cohort metadata sidecars.
- Recheck token counts are observed replay fields, not on-device measurements. `timing_evidence.json` records no trustworthy per-call timing; no latency or energy savings are established.
- HURT is rare in MATH source test (3 of 400) and CSQA target test (69 rollout rows but 34.5 question-equivalents among 733 questions); the frozen gate does not reliably detect HURT. Do not describe it as a safety guarantee.
- The “oracle” chooses HELP rollouts using future observed correctness. It is an unattainable hindsight diagnostic and must never be described as an implementable policy.

## Evidence required before submission

The current analysis is paper preparation, not a fresh independent validation. Before a defensible empirical submission, freeze one primary claim and protocol; verify the original SEVRA extraction/grading and release provenance; run a truly independent, preregistered evaluation with question-level grouping; report repeated-seed or rollout uncertainty; and measure actual wall-clock cost on the intended runtime if efficiency is a claim. To support learned-gate superiority, a new untouched cohort must show a predeclared, practically meaningful gain over completion at a matched cost (or a cost reduction at a predeclared noninferiority margin). If that does not happen, retain the negative conclusion and frame learning as not justified under the studied setting.

## Source evidence

- Frozen classifier and threshold: `outputs/v0.2/models/source_gate.joblib`; metadata in `outputs/v0.2/reports/experiment_metadata.json`.
- Grouping and labels: `rechecklab/data.py:34-86`; `rechecklab/metrics.py:9-55`.
- Feature allowlist: `rechecklab/features.py:20-31,88-115`.
- Frozen prediction compatibility: `rechecklab/predictor.py:121-139,200-228,303-340`.
- Dataset provenance: `data/raw/DATASET_CARD.md`, `data/raw/release_manifest.json`.
- Existing inference timing limitation: `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/timing_evidence.json`.
- New numerical outputs: `MAIN_POLICY_COMPARISON.csv`, `PAIRED_COMPARISONS.csv`, `CLASSIFIER_METRICS_CONFUSION.csv`, `DATASET_PROTOCOL_COMPARISON.csv` in this run directory.
