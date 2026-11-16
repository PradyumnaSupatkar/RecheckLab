# Phase 2 verdict: NO-GO for investing in residual gating as the main contribution

**Date:** 16 November 2026. **Scope:** one lightweight offline feasibility candidate, existing source training/development, no new LLM responses. This is a research investment decision, not a proof that every conceivable residual model fails.

## Executive decision

**Do not spend weeks or new Ollama inference on the proposed residual gate in its current form.** The combination of weak novelty, very few beneficial deviations, format-sensitive apparent corrections, and no measured cross-validated improvement makes it a poor primary conference direction.

The experiment was run, not skipped. One ridge model family used 21 small CPU fits, nested grouped validation, question weights, fixed compute penalty, and training-derived cost estimates. It did not improve upon completion. The final training-only decision was to retain completion everywhere. No candidate was deployed or saved in the production artifact format.

The word **conservative** here denotes empirical margin/fallback behavior. It does not denote a statistical safety guarantee. This sample cannot reliably calibrate such a guarantee.

## Seven research questions, answered directly

| Question | Finding | Evidence category |
|---|---|---|
| 1. Enough correctable error after completion? | Only **10/1,199** beneficial training switches, at most **+0.834 pp** accuracy. Six are completed/nonempty apparent fixes whose initial transcripts already show the accepted mathematical result. Development has **zero extra fixes** outside completion and only **two avoidable harms**. | Hindsight opportunity; development descriptive, not policy selection |
| 2. Can the current features predict them? | Response length has a same-sample association, but six positives cannot support confidence. Nested-OOF residual score AUC is **0.449** in the completed/nonempty group; no useful switches were captured. | Exploratory association and held-out-fold diagnostic; not a universal impossibility result |
| 3. Does residual gating beat completion on held-out training folds? | **No.** Both achieve **92.2435%**, while residual uses **4,109.928 vs 4,096.207 tokens/question**. Nine extra calls are all neutral. | Nested grouped OOF estimate |
| 4. Does improvement survive development? | There is no improvement to confirm. Training-only margin selection chooses no overrides; candidate equals completion at **92.00% / 4,121.013 tokens**. The rate-matched ranking diagnostic falls to **91.75%**. | Frozen candidate, previously inspected development descriptive confirmation |
| 5. Large enough effect for affordable measurement? | Probably not for this event rate. Even the training hindsight ceiling is below one point, and observed semantic opportunity is weaker. A 50–300-question pilot will contain very few beneficial deviations. | Planning inference under stated assumptions |
| 6. Distinct from previous research? | Not yet. Baseline fallback, risk-aware revision, and cost-sensitive routing are established; the exact combination was not verified as identical prior art, but a different regression target/margin is insufficient novelty. | Verified literature comparison plus reviewer judgment |
| 7. Scientifically worthwhile for IEEE? | This bounded diagnostic is worthwhile for avoiding an unproductive direction. By itself it is not a substantial paper. A broader independently designed measurement-validity study might be worthwhile if it answers a specific unresolved question. | Research judgment, no acceptance prediction |

## Compact measured comparison

| Partition / policy | Accuracy % | Difference vs completion, pp | Avg. tokens | Recheck % | Beneficial overrides captured |
|---|---:|---:|---:|---:|---:|
| Training / Completion | 92.2435 | 0 | 4,096.207 | 23.186 | 0 |
| Training / Residual nested OOF | 92.2435 | 0 | 4,109.928 | 23.937 | 0 |
| Training / Matched-rate ranking diagnostic | 92.1601 | −0.0834 | 4,086.508 | 23.186 | 0 |
| Development / Completion | 92.0000 | 0 | 4,121.013 | 23.500 | 0 |
| Development / Frozen historical ML | 92.0000 | 0 | 4,081.343 | 23.250 | 0 |
| Development / Frozen residual | 92.0000 | 0 | 4,121.013 | 23.500 | 0 |
| Development / Matched-rate ranking diagnostic | 91.7500 | −0.2500 | 4,120.813 | 23.500 | 0 |

The existing frozen ML gate obtains all 10 training opportunities **in-sample**; that is not evidence of generalization and is explicitly labeled in the full CSV. Its development token advantage is retrospective because the original model/threshold were selected on development. The new residual candidate provides no gain over either fair heuristic comparison.

Question-cluster bootstrap for residual OOF gives accuracy difference **[0, 0] pp** and added tokens **[5.177, 25.175]**. Identical observed correctness causes the zero interval; it is not a population-equivalence guarantee. The bootstrap conditions on saved predictions and does not refit the nested procedure. No significance or independent-test confirmation claim is made.

## Why more inference is unlikely to be a good immediate investment

Using the training beneficial-switch rate **10/1,199 = 0.834%** only as a planning illustration, 300 independent questions would yield roughly **2.5** beneficial opportunities on average. Obtaining even 30 would require about **3,600 questions / 7,200 initial-plus-recheck calls** at that same rate. Thirty events is not automatically enough for learning or safety calibration, and the rate will change with solver/protocol/domain.

The earlier local pilots suggest a rough **60–120 seconds/call** planning range, making 7,200 calls about **120–240 hours**, potentially worse for long math reasoning. These are estimates, not measured runtime for this proposed experiment. No calls are authorized or started here. The bottleneck is event support and interpretation, not insufficient neural model size.

Crucially, the six apparent completed/nonempty training corrections are not good examples of newly fixed reasoning. Their initial responses already contain the accepted answers, with differences in tags, answer prefixes, or mathematical notation. Historical labels are preserved; the finding limits what can be learned or claimed from them. We should not collect a large dataset to fit a residual semantic-error detector without first establishing that its target events are semantic.

## One recommended alternative contribution

**Annotation-stable measurement of LLM recovery value under constrained generation.**

Research question: *How much apparent benefit of selective verification survives fixed, mathematically defensible answer extraction and completion controls, and which policy comparisons remain stable across those choices?*

This is more promising than another gate variation because the present training inspection exposes a concrete measurement issue. It is not claimed as globally novel; compare it carefully with prior papers' parser/protocol diagnostics before committing. The contribution would need a systematic, reproducible decomposition—not a few cherry-picked malformed examples or a higher rescored accuracy.

Minimal next action, **only after approval**:

1. Predeclare a random development-only audit cohort, blinded to HELP/HURT when selecting it. Retain all original annotations. Obtain official trusted keys for the chosen math questions with documented provenance before any alternate grading; do not invent keys or treat a revised answer as gold.
2. Define a small, task-appropriate family of extraction/normalization rules and independent adjudication procedures before examining its results. Separate semantic correction, answer-extraction recovery, truncation recovery, harmful semantic changes, and unresolved cases. Treat representation-equivalent output consistently.
3. Compare policy accuracy, harm, tokens, and oracle bounds across the frozen scoring variants; report intervals and disagreements, not a selected favorable scoring rule. Have a second human review ambiguous math cases if feasible. This first stage needs **zero LLM calls**.
4. Proceed to a new independent live sample only if that audit identifies a substantive unresolved mechanism and a sufficiently frequent event type. Reuse benchmark questions; freeze protocol and evaluation IDs. Do not propose a large call budget until its event rate and scoring reliability are established.

Stop if a focused literature review shows this measurement question already adequately answered, trusted grading cannot be established, or almost all variation is confined to a few known formatting artifacts. A negative finding can still inform the project, but should not be inflated into an original algorithm paper.

## Files, reproduction, and preservation

- `NOVELTY_ASSESSMENT.md`: verified primary sources and the residual-policy equivalence argument.
- `OVERRIDE_OPPORTUNITY.md`: detailed groups, costs, transcript findings, method, results, and limits.
- `POLICY_COMPARISON.csv`: **30 actual measured rows**, with partition, scope, and evaluation-status labels; no placeholder candidate results.
- `OPPORTUNITY.csv`, `TRAINING_OPPORTUNITY.csv`, `feature_associations.csv`: empirical breakdowns.
- `analysis_plan.json`, `folds.json`, `frozen_candidate.json`, saved predictions: reproducibility evidence.
- `../feasibility.py`: the bounded experiment, fresh-run refusal, and original-file checks.
- `../verify_feasibility.py`: independent arithmetic and saved-coefficient verification without fitting.

Independent verification reproduced **270 numerical metric comparisons**, checked group separation, reconstructed development scores from saved linear coefficients, and rehashed **320 original files unchanged**. No test outcome files were loaded by these calculations. The fresh output root is `outputs/ieee_phase2_feasibility/`; no existing report, dataset, model, source file, or frozen protocol was modified.

Relevant code: production completion/decision contracts in `rechecklab/predictor.py:303` and `rechecklab/experiment.py:61–67`; initial feature extraction `features.py:88`; question-equal metrics `metrics.py:9,22`; experimental fit/score/selection `../feasibility.py:90,98,106`; experiment freeze-before-development order in `main()` at `:161` onward.

## Final researcher assessment

**If our goal is acceptance into a legitimate IEEE conference, should we invest in conservative residual gating, or pursue a different contribution?**

**Pursue a different, measurement-focused contribution unless new independent evidence changes the opportunity assessment.** Do not invest weeks in this residual gate on the strength of its name or fallback mechanism. The current idea is closely related to established work, the available beneficial deviations are sparse and often format-sensitive, and the one bounded candidate did not help. The appropriate next investment is a small, rigorous study that establishes what “correction” actually measures here—not more classifier tuning or a larger inference job.
