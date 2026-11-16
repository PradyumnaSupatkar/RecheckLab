# RecheckLab — IEEE Conference Readiness Audit

**Audit date: 16 November 2026. Analysis only; no recommendations implemented.**

## 1. Executive verdict

**Ready for a candid professor discussion; not ready as a strong original-method conference submission.** The existing results support selective rechecking as an economical alternative to always rechecking on these recorded attempts. They do **not** establish that the learned gate is better than a completion heuristic, a general semantic-error detector, a reliable harm detector, or a latency-saving deployed system.

The largest obstacle is **novelty, followed by independent experimental evidence**. The public data come from SEVRA, whose paper already studies learned recoverability gates, frozen-solver verification, harmful revisions, serving metadata, and budget allocation. Merely adding histogram boosting, transferring to CommonsenseQA, and building a dashboard is not a compelling new scientific contribution. Attribution is already present in `DATA_SOURCES.md:5–15`; the manuscript must make the derivative nature equally explicit. A later, directly related September 2026 preprint further reduces room for a generic “risk-aware selective revision” claim (Section 5).

The defensible direction is a **controlled study of the marginal value of learning beyond completion/format signals**, with a small, independently evaluated extension for harm-aware routing if the data support it. A well-designed negative result can be useful, but “our classifier ties a heuristic” alone is not enough for a substantial research paper.

**Important correction:** the audited 400-question source test is a question-group holdout from the SEVRA MATH training file. It is not established as an official **MATH Level 5** subset, and it is not MATH500. Do not present its 90% as a Level 5 or official MATH test score.

Evidence labels used below:

- **Verified now:** recomputed from the saved artifact/records or inspected in current code.
- **Recorded evidence:** present in archived experiments, without rerunning their training/inference.
- **Limitation:** a boundary on what those observations establish.
- **Proposal/hypothesis:** requires future implementation or evidence; no improvement is implied.

## 2. Scope, architecture, and engineering assessment

### Audit scope and verification

Recursively inventoried the project excluding `.venv`, `.git`, `__pycache__`, and `.pytest_cache`: **319 existing files, 56 Python files, 18 Parquet files**. All 56 Python files passed in-memory AST parsing. The tests contain **181 `test_` methods**. Reviewed the core data, feature, prediction, experiment, live collection/training/validation, UI, tests, provenance, and archived analysis paths.

Recomputed v0.2 policy/classifier results from the actual joblib artifact and prepared partitions. The independent recorded-policy verifier passed its saved probability, decision, score, feature, policy, and 25 adaptation-run checks. Executed only the three read-only tests in `tests/test_recorded_policy_results.py`: **3/3 passed**. The entire test suite was **not** run: many tests create temporary artifacts or fit synthetic fixtures, which is unnecessary for this analysis-only audit. Syntax and three tests do not establish full functional coverage. No browser/UI session, Ollama call, training, threshold search, package installation, or data/model download occurred.

The inventory/hash baseline excludes this report. Its combined SHA256 is `c3f2fb7f006f8767a003e49a0fae9e4860d31a7b9bd1356cc2ba67795716d742`; construction hashes sorted relative paths followed by each file's SHA256 bytes. The final check reproduced this exact hash and count: **all 319 protected files unchanged, with no additional non-cache project file except this report**.

### Architecture

| Layer | Actual implementation | Assessment |
|---|---|---|
| Public paired data | `rechecklab/data.py:34 transition_label`, `:74 validate_pairs`, `:259 public_frames` | Reads published initial/revised answers and correctness; only `active_verify` enters v0.2. Strong structural validation; inherited grading remains a provenance limitation. |
| Grouping/splits | `data.py:39 question_hash`, `:233 group_partition`, `:248 assert_disjoint` | Canonical question groups, whole-group partitioning, recomputed disjointness. Exact normalization is not semantic deduplication or a pretraining-contamination check. |
| Initial-only features | `features.py:23 INITIAL_REQUIRED`, `:88 extract_one`, `:117 extract_features` | Six required initial fields produce 19 features; no gold, revised answer, correctness, or future token features. Mostly surface counts and completion signals. |
| Classifier/policy | `predictor.py:121 probabilities`, `:236 fit_predictor`, `:297 revision_score`, `:303 decide_recheck`, `:308 choose_threshold` | Three-class probabilities followed by a distinct utility-score threshold. Probabilities are not calibrated risk guarantees. |
| Experiment orchestration | `experiment.py:44–55`, `:75–86`, `:144–172` | Source fitting/selection precedes test scoring; target adaptation has a separate pool and predeclared displayed budget. Writers need stronger overwrite protection. |
| Evaluation | `metrics.py:9 question_weights`, `:22 policy_metrics`, `:60 predictor_metrics`, `:101 paired_accuracy_interval` | Averages repeated outcomes within each question. Existing bootstrap is conditional on a fixed model; it does not cover all selection uncertainty. |
| Independent replay checks | `scripts/verify_recorded_policies.py:27 independent_metrics`, `:183 verify_recorded_policies` | Valuable consistency checks against saved predictions and reports. |
| Local study | `study_data.py`, `study_protocol.py`, `study_collection.py`, `study_training.py`, `study_validation.py` | Separate keys, pinned protocols, append-only call evidence, backend checks, and guarded fitting. Incomplete collection is not a completed experiment. |
| Streamlit | `app.py:24 RUN_ROOT`, `:81 validate_replay`, `:136 validate_file_hashes`, `:211 render_recorded`, `:325 render_live`, `:458 main` | Active root is `outputs/v0.2`. Recorded and experimental live use are distinguished. UI correctness is useful engineering, not research novelty. |

Active model: **class-balanced histogram gradient boosting**, 19 features, `max_iter=100`, `max_leaf_nodes=7`, `min_samples_leaf=20` (default), `learning_rate=0.05`, `l2_regularization=5`, seed 42. Class balancing uses inverse question-weighted class mass. Source-development macro-F1 selects the classifier, with log loss as tie-break; source-development HELP−HURT selects the threshold with token use as tie-break. Frozen threshold **0.25968145276947807**, harm weight **1.0**; recheck iff `P(HELP) − P(HURT) > threshold`.

### Engineering findings requiring attention before publication

1. **Verified overwrite hazard, not observed data corruption.** `experiment.py:31–33` prevents writing to the project root but not to an existing run directory; `:144–172` writes fixed filenames, and `predictor.py:179–186 save_artifact` uses `joblib.dump` without exclusive creation. A mistaken rerun can replace a frozen output. Future runs should require a fresh directory and manifest; do not change existing archives.
2. **Verified stale status pointer.** `reports/live_study/CURRENT_STATUS.json` describes v0.3.1, while `outputs/live_study/v0.3.2/collection_status.json` exists and records a partial later run. The latter has `ended_at=null`; this is not evidence that a collector is currently running. A future run registry should separate implementation status, collection status, and paper-included runs.
3. **Multiple execution protocols are easy to conflate.** Public replay, `collector.py`/`live.py` experimental UI calls, v0.3.x studies, and generation pilots differ. `README.md` explicitly documents the UI's 768-token, requested nonthinking protocol; `study_protocol.py:60–92` defines a different 1,024-token thinking protocol. A matching model tag does not imply matching checkpoint or response distribution.
4. **Frozen numeric parser limitation is real but guarded.** `study_protocol.py:188 normalize_number` uses Decimal arithmetic; `study_validation.py:24 audit_numeric_precision` detects precision loss, including nonterminating fractions. This is unsuitable as a general symbolic-MATH grader. It does not invalidate public v0.2, which uses publisher grades. Any future parser correction needs a new version and a predeclared sensitivity analysis.
5. **Replay reproducibility is stronger than generation reproducibility.** Pinned requirements, artifact/schema fingerprints, manifests, and raw logs are strengths. Exact historical solver prompts/checkpoint digest are not established by the pinned local release (`data.py:278–283`). Newer upstream documentation may add details; it cannot retroactively prove the exact generation configuration of every pinned row.
6. **No confirmed removable dead module.** Old v0.1 artifacts and overlapping v0.3.x runs are preserved history, not automatically dead code. Similar public/live/study helpers and duplicated metric calculations create maintenance burden, but independent metric implementations also help auditing. Do not consolidate frozen paths casually. No exhaustive dynamic call-coverage or semantic clone analysis was performed.
7. **Tests are substantial but narrower than scientific evidence.** See `tests/test_prediction_contract.py:110` for future-field exclusion, `:131` for strict threshold semantics, `tests/test_study_collection_audit.py` for collection safeguards, and `tests/test_study_validation.py` for numeric precision. Mocked inference tests cannot demonstrate backend reliability, live accuracy, or reproducibility across Ollama versions.

## 3. Independently verified experimental results

### Partitions and labels

Source split: 60/20/20 of canonical question groups, seed 42. Target adaptation/test: 40/60, seed 43 (`experiment.py:44–48`). All five partitions passed recomputed group-disjointness validation.

| Partition | Questions | Rows | HELP rows | NO_CHANGE rows | HURT rows |
|---|---:|---:|---:|---:|---:|
| Source training | 1,199 | 1,199 | 162 | 1,022 | 15 |
| Source development | 400 | 401 | 47 | 348 | 6 |
| Source test | 400 | 400 | 65 | 332 | 3 |
| Target adaptation | 488 | 976 | 22 | 878 | 76 |
| Target test | 733 | 1,466 | 21 | 1,376 | 69 |

HELP is wrong→correct; HURT correct→wrong; NO_CHANGE includes both correct→correct and wrong→wrong. NO_CHANGE does **not** mean text or answer remained identical. Each question has total weight one; target outcomes have weight 1/2 each. The duplicated source-development question contributes one unit, so its weighted NO_CHANGE support is 347, not 348.

### Policy results

All figures below were recomputed now. HELP/HURT are question-equivalent counts, not independently sampled rollout counts. Token totals are recorded prompt-plus-generation usage. Savings are against always rechecking **within the same partition**.

| Partition / policy | Final accuracy % | Recheck % | Avg. total tokens | Tokens saved % | HELP / HURT selected |
|---|---:|---:|---:|---:|---:|
| MATH source test / Never | 74.7500 | 0 | 3,182.018 | 44.636 | 0 / 0 |
| MATH source test / Always | 90.2500 | 100 | 5,747.435 | 0 | 65 / 3 |
| MATH source test / Completion | 90.0000 | 26.750 | 4,402.420 | 23.402 | 62 / 1 |
| MATH source test / ML | 90.0000 | 27.250 | 4,412.825 | 23.221 | 62 / 1 |
| MATH source test / Hindsight oracle | 91.0000 | 16.250 | 3,707.835 | 35.487 | 65 / 0 |
| CSQA target test / Never | 76.1255 | 0 | 2,243.604 | 52.466 | 0 / 0 |
| CSQA target test / Always | 72.8513 | 100 | 4,719.982 | 0 | 10.5 / 34.5 |
| CSQA target test / Completion | 76.6712 | 3.411 | 2,334.847 | 50.533 | 4 / 0 |
| CSQA target test / ML | 76.6712 | 3.411 | 2,334.847 | 50.533 | 4 / 0 |
| CSQA target test / Hindsight oracle | 77.5580 | 1.432 | 2,291.702 | 51.447 | 10.5 / 0 |

Oracle selects only observed HELP outcomes, skipping neutral/harmful outcomes. Its accuracy is `mean_question(mean_rollout(max(base_correct, final_correct)))`. It neither selects the better rollout nor represents an attainable initial-only policy. Identical initial inputs can have different stochastic outcomes; an oracle can distinguish them, the deployed gate cannot. The ceiling applies only to these recorded one-recheck pairs, not to new solvers, prompts, or aggregation algorithms. **80% CSQA is impossible by selecting between these fixed paired answers under the current scoring.**

**ML versus completion:** source-test disagreements are exactly two questions, IDs **1430 and 1457**, both NO_CHANGE. ML adds those calls; correctness is identical pointwise. Consequently no subgroup of this fixed source test can demonstrate an ML accuracy advantage. Target decision vectors are exactly identical. The observed deployable accuracy–token frontier is never/completion/always on source test, and never plus coincident completion/ML on target. The oracle is excluded from deployable frontiers.

Source development must remain separate:

| Policy | Accuracy % | Recheck % | Avg. total tokens | Savings vs always % |
|---|---:|---:|---:|---:|
| Never | 80.75 | 0 | 3,005.635 | 45.567 |
| Always | 91.00 | 100 | 5,521.676 | 0 |
| Completion | 92.00 | 23.50 | 4,121.013 | 25.367 |
| ML | 92.00 | 23.25 | 4,081.343 | 26.085 |

The development gate saves 39.67 tokens/question versus completion, with equal accuracy. This is **selection-set performance**, not independent evidence that ML is more efficient generally.

### Classifier quality versus policy quality

| Partition | Three-class accuracy | Macro-F1 | Log loss | HELP precision / recall | HURT precision / recall |
|---|---:|---:|---:|---:|---:|
| Source development | 85.00% | 0.558724 | 0.391613 | 51.81% / 91.49% | 7.69% / 16.67% |
| Source test | 85.25% | 0.541671 | 0.411111 | 58.82% / 92.31% | 0% / 0% |
| Target test | 91.5416% | 0.393647 | 0.322128 | 16.00% / 38.10% | undefined / 0% |

Target predicts no HURT; stored zero precision is a convention. Its NO_CHANGE prevalence is 93.8608%, so always predicting NO_CHANGE beats 91.5416% classification accuracy. Argmax classification and the binary score-threshold policy are different tasks. Source policy captures 62/65 HELP (95.38%), with 62/109 helpful precision (56.88%) and 1/109 harmful selections (0.92%). Target captures 4/10.5 HELP (38.10%); only 16% of selected outcomes help and 84% are neutral. Zero observed selected HURT is not a safety guarantee and is not successful three-class harm detection.

No trustworthy historical solver wall-time columns exist in the three evaluated prepared partitions. Archived offline gate benchmarking reports roughly 4.13 ms/source and 4.00 ms/target per resident-model single-record call, including feature extraction (`outputs/ml_efficiency_evaluation/evaluation_2026-11-03/gate_overhead.json`, benchmark implementation `evaluate_ml_efficiency.py:135`). These timings were not remeasured here. Zero median process CPU time reflects timer granularity, not zero work. **Token savings and calls avoided are computation proxies, not measured LLM latency or energy savings.**

### Earlier optimization: recorded negative evidence

`outputs/accuracy_experiments/experiment_01_2026-11-03/REPORT.md` records 11 nested-CV configurations. The winner remained balanced HGB; its training-OOF-selected threshold produced source-development **91.75%**, versus frozen **92.00%**, with **5,070.836** rather than **4,081.343** tokens. No promotion is justified.

`outputs/final_accuracy_audit/review_2026-11-03/REPORT.md` records: the added visible-character/token interaction gave **91.75%** and **4,840.241** development tokens; a training-selected length-or-missing rule tied **92.00%** with **4,125.155** tokens. A threshold tuned on 244 adaptation groups scored **78.6885%** on the other 244, versus **79.0984%** frozen/completion. These are archived experiments, not rerun training in this audit. Their selection safeguards are useful; reused development sets are not fresh confirmation.

## 4. Critical scientific weaknesses

### 4.1 The learned advantage is absent

The strongest comparator is completion, not always rechecking. Relative to always, the system is useful; relative to completion, the learned component is unnecessary on the target and slightly more expensive on the source test. A paper centered on “ML improves efficiency” would over-attribute benefits to ML. Report the policy tie prominently and distinguish the selective-computation contribution from the learning contribution.

A different setting could reveal a learned advantage, but choosing that setting because its test outcomes favor ML is invalid. Predeclare protocol changes and evaluation cohorts. If a completion-sufficient regime persists, report its boundary conditions rather than continue searching for a favorable subset.

### 4.2 Rare harms and distribution shift

Only **15 source-training HURT** events support the harm model; source test has three. Target adaptation has 76/976 HURT rows versus source training 15/1,199, while HELP prevalence falls sharply. Balanced weighting cannot manufacture missing diverse examples, and it changes score calibration. A harm-aware score is not evidence of learned harm discrimination.

Avoid a separate high-capacity harm classifier. Consider a regularized three-class or signed-utility model with a small, independent calibration set, but stop if effective harm support is inadequate. Conservative fallback to a fixed rule is preferable to unsupported safety claims. Artificially inducing truncation or selecting failures to obtain HURT examples changes the population and must not be hidden.

### 4.3 Completion recovery can masquerade as reasoning correction

Completion/finalizer features are legitimate pre-recheck signals. The issue is attribution: repairing an unfinished answer differs from correcting a completed reasoning error. Analyze both, plus valid-wrong→correct and correct→invalid transitions. Feature ablation alone does not prove causality; initial-budget interventions are needed to study the mechanism.

The local source test lacks `level`, `type`, `subject`, or `category` fields. Its difficulty values are 0.45/0.55/0.65/0.75/0.85 with counts 37/82/75/107/99. Original MATH has levels 1–5 and seven subjects, but those values cannot be inferred from these proxies. The upstream description calls difficulty a static heuristic estimate. [Original MATH paper, 2021](https://arxiv.org/html/2103.03874v2), [SEVRA metadata description](https://arxiv.org/html/2606.19808v1).

### 4.4 Statistical exposure and uncertainty

The original splits prevent exact cross-partition question leakage; the feature allowlist excludes future outcomes. Nonetheless, repeated inspection and follow-up work mean these tests are no longer fresh confirmation for future designs. Source-development classifier and threshold selection on the same sample also creates optimism. Nested CV helps, but picking a winner among CV configurations remains selection.

Five adaptation subset repetitions share the same test questions and overlap; their SD is not a confidence interval over independent datasets or solver runs (`AUDIT.md:28`). Two target rechecks share one initial response. They estimate conditional revision variability, not variation across new initial generations. A question bootstrap must retain all its rollouts and paired policies. Training-set randomness and initial-generation randomness need separate replication.

No arithmetic defect was found in current replay checks. Public grading remains inherited: local numeric parsing is not used to create v0.2 labels. The prior grading audit found no standard-letter/key mismatch and documented malformed transcripts as scoring sensitivity. This audit does not silently re-extract answers or turn those cases into extra successes. Exact-text grouping cannot exclude paraphrases or solver pretraining contamination.

### 4.5 Limited system and causal claims

Replay is appropriate for comparing policies when initial and recheck outcomes were recorded for every question under a fixed action protocol. It avoids collecting only gate-selected revisions. It does not identify counterfactual outcomes for a different prompt, budget, checkpoint, context, or verifier. It also does not establish actual serving latency under caching, load, batching, or failures. The offline collection paid for every call; projected deployment savings are not money or energy already saved during collection.

A credible system comparison must include a longer initial solve under a comparable total budget. A weak initial budget can make all recovery policies look artificially valuable. Compare achieved accuracy/cost curves, not merely equal maximum-token caps.

## 5. Related work and novelty assessment

Targeted primary-source review through **16 November 2026**, not an exhaustive systematic review and not coverage of future November publications. Publication status below is limited to what was verified. Search snippets and similarly named papers were not treated as reliable citations without checking primary pages.

| Work and verified date/status | Relation to RecheckLab and resulting requirement |
|---|---|
| [Self-Refine: Iterative Refinement with Self-Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html), Madaan et al., NeurIPS 2023 | Iterative feedback/refinement predates RecheckLab. Its difference is the decision to pay for revision, not the idea of revising an answer. |
| [Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/forum?id=IkmD3fKBPQ), Huang et al., ICLR 2024; preprint October 2023 | Harm from intrinsic self-correction is established. RecheckLab must explain when its policy helps, without generalizing a particular prompt/model result to all self-correction. |
| [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176), Chen et al., 9 May 2023 preprint; [author-hosted TMLR paper, December 2024](https://lchen001.github.io/papers/2024_FrugalGPT_TMLR.pdf) | Learned cost-aware cascades are established. RecheckLab routes actions on one solver rather than among models; this distinction alone is insufficient novelty. |
| [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), Ong et al., June 2024 preprint | Relevant learned-routing efficiency baseline concept. Reimplementing a large router is unnecessary, but cost-matched simple routing is essential. No venue claim is needed here. |
| [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314), Snell et al., 6 August 2024 preprint | Prompt-dependent allocation and difficulty-sensitive compute are established. “Spend more computation on difficult questions” is not a novel hypothesis by itself. |
| [PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier](https://arxiv.org/abs/2506.10406), Jiang et al., 12 June 2025 preprint | Selective revision also appears in trained generative-verifier systems. RecheckLab's CPU gate is cheaper, but cheapness needs measured cost and competitive baselines. RL replication is outside the resource budget. |
| [Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning](https://arxiv.org/abs/2606.19808), Dip, Zhou and Zhang, 18 June 2026 preprint | **Closest antecedent and data source.** Already includes recoverability gates and budget-aware verification. Its longer-initial comparison and completion-state analysis overlap proposed RecheckLab explanations. A manuscript must state exactly what is replicated and what is new. |
| [When Should LLMs Trust Their Own Revisions? A Risk-Aware Study of Intrinsic Self-Correction](https://arxiv.org/abs/2609.35832), Zhang, 23 September 2026 preprint | Directly studies HELP/HURT trade-offs and selective revision from initial-response signals across multiple models. “Risk-aware rechecking” and “sometimes simple policies win” cannot be claimed as first discoveries. |
| [Verification-Gated Redraw and Repair for Local Small Language Models: A Paired Evaluation](https://arxiv.org/abs/2611.00707), Rykhtik, 31 October 2026 preprint | A resource-bounded paired local recovery study with executable checks. Relevant methodological comparator, but code-task checks differ from commonsense evidence. Small non-significant studies must delimit their claims. |

**Reviewer judgment:** the current contribution is incremental and scientifically weak as a new algorithm, but its reproducible implementation is a good platform. A potentially meaningful extension needs a sharply specified empirical estimand, strong cheap baselines, independent replication, and results that change a deployment decision. None of the three hypotheses below is claimed to be globally novel; a further focused literature check is required before claiming priority.

## 6. Five highest-impact improvement opportunities

| Priority / proposal | Research impact and novelty potential | Complexity and compute | Main risk |
|---|---|---|---|
| 1. Define the marginal value over completion and parsing heuristics | Very high validity impact; moderate potential for a careful boundary-of-usefulness study | Low CPU cost on replay; independent live sample later | A broad negative result may duplicate prior work unless mechanisms and protocol coverage add insight |
| 2. Separate truncation recovery from semantic correction; include a longer-initial baseline | High attribution value; moderate novelty in a tightly controlled replication | Moderate engineering; roughly three calls/question for short initial + revision + longer initial | Longer initial dominates; this is a legitimate result, not a failed study |
| 3. Freeze claims/splits and use paired question-level uncertainty | Essential credibility, low algorithmic novelty | Low CPU; bootstrap existing predictions, no LLM cost | Wide intervals reveal insufficient power; never conceal them |
| 4. One regularized utility/risk-aware gate with calibration checks | Moderate potential, only if it beats rules at matched cost or harm | Low CPU fitting; reuses paired data; calibration needs disjoint data | Fifteen training harms are too few for robust tail-risk estimates; over-conservative policies skip all |
| 5. Independent local replication with timing and resource logging | High external-validity value; modest novelty alone | Sequential 4B quantized inference, controlled RAM, staged collection | Backend drift, truncation, low event counts, thermal/caching variability |

Method triage:

- **Cost-sensitive/expected utility:** scientifically appropriate. Target `E[correct_after − correct_before | initial_state] − lambda * E[extra_tokens | initial_state]`. Choose lambda from an explicit deployment objective, not to make a plot attractive. Threshold on predicted cost, never actual unseen revision cost at deployment. The current code already supports a cost term during offline threshold selection; adding it is not itself novel.
- **Calibration:** compare unweighted regularized multinomial logistic with the frozen balanced HGB; use cross-fitted calibration and Brier/log-loss/reliability diagnostics. Do not fit isotonic curves for a handful of harms. Calibrated probabilities can still fail under domain shift.
- **Small features:** parse validity, option-reference consistency, visible-length/generated-token ratio, and lexical uncertainty are feasible. Test incremental value beyond completion on training folds; the ratio variant already failed to improve policy utility. A new feature is not evidence until ablated and confirmed.
- **Binary HELP detector versus three-class versus signed utility:** a small defensible comparison. HELP-only classification ignores harmful costs unless explicitly weighted. Ordinal HELP/NC/HURT treatment is not automatically justified.
- **Learning-to-rank/pairwise learning:** low priority. With two actions, direct net utility is simpler; ranking may ignore calibrated absolute harm/cost. Do not double-count every pair of correlated rollouts as independent training data.
- **Uncertainty:** entropy of the gate is available but not calibrated solver uncertainty. Multiple LLM samples or semantic entropy require extra calls; charge them to the policy. Token log probabilities are not present in the historical schema and cannot be invented offline.
- **Risk constraints:** promising only with independent validation and an explicit risk definition (harm/all questions versus harm/selected). A confidence upper bound must account for threshold search, group correlation, and shift. Ordinary empirical quantiles are not a safety certificate.
- **Dynamic/domain thresholds:** use known domain and runtime metadata, not hidden correctness. Compare against equally adapted heuristics and account for the labeled adaptation cost. Avoid a proliferating threshold per small subgroup.

## 7. Three focused research hypotheses

### H1 — Completion signals explain most apparent learned-gating value

**Hypothesis:** after controlling initial completion reliability and realized compute, a learned gate has little additional utility on these compact-model workloads; any remaining advantage concentrates in valid, normally completed attempts.

**Method:** preregister completion-only, parse-only, completion-or-invalid, and a 19-feature gate. Compare completion-only features, text-only features, and full features with identical grouped training folds. Evaluate the full cohort and predefined completed-valid/truncated-invalid strata; do not train on test strata. Add a short-budget versus completion-reliable-budget intervention on the same benchmark questions, with revision attached to the shared short initial response.

**Potential originality:** a controlled estimate of marginal learning benefit across budget regimes, rather than another generic selective-verification algorithm. Existing SEVRA completion analyses make mere subgroup plots insufficient.

**Implementation/cost:** small CPU ablation module plus a versioned collector protocol; no LLM fitting. Existing replay starts at zero new calls. Live budget study requires approximately 3N calls (short initial, revision, longer initial).

**Evidence required:** paired accuracy/cost differences and intervals, fixed rate/cost comparisons, parse-failure accounting, and at least two task types. A predeclared equivalence/noninferiority margin needs adequate power; failure to reject a difference is not equivalence. Failure cases: almost no completed-but-correctable errors, protocol differences, or longer initial solving dominating every gate. Publish those boundaries honestly.

### H2 — Learning when to depart from a strong heuristic is more stable than learning every decision

**Hypothesis:** a conservative residual policy improves net utility only where its expected gain over completion is supported, otherwise falling back to completion.

**Method:** let the default decision be completion. Fit a small regularized signed-benefit model; predict the incremental utility of switching that decision, including predicted extra cost. Use cross-fitting and a separately calibrated override margin; include always-fallback as a baseline. Restrict the search to one model family and a few preregistered margins.

**Potential originality:** an auditable override/fallback design and measured adaptation burden for this setting; safe policy improvement itself is established, so do not claim invention. Domain-specific risk claims require domain-specific evidence.

**Implementation/cost:** CPU-only on paired logs, then the same independent paired live sample as H1; no extra rechecks beyond collecting the action outcomes. An initial 3–7 days of implementation/validation is plausible, excluding collection.

**Evidence required:** improved lower-confidence net utility versus completion or less compute at a justified noninferiority margin, override coverage, failure analysis, and harm confidence bounds. Failure cases: too few override events, no distinguishable signal, or conservative bounds forcing the default everywhere. Do not relax the margin after inspecting test labels.

### H3 — Repeated-outcome uncertainty predicts when adaptation is worth paying for

**Hypothesis:** disagreement among revision outcomes limits the value of initial-only routing, and shrinkage toward the heuristic is more robust than aggressive threshold adaptation when label budgets are small.

**Method:** train with per-question empirical outcome proportions and compare with current row-wise question weighting. Proper weighted cross-entropy on identical inputs is already equivalent to using these soft proportions; therefore that conversion alone is **not** a new algorithm. The candidate extension is explicitly estimating uncertainty in the action value, shrinking low-support estimates, and measuring held-out policy stability as adaptation labels increase.

**Potential originality:** an empirical characterization of conditional revision noise and adaptation economics for lightweight gates, not the existence of stochasticity or soft labels.

**Implementation/cost:** CPU analysis first. The target has only two revisions per initial, which is weak evidence about each question's true probability. A preregistered random subset could receive a third revision later, plus a smaller independent-initial replication block; these address different uncertainty sources.

**Evidence required:** question-level held-out utility and coverage, sensitivity to which rollout supplies training outcomes, and benefits after charging annotation/inference costs. Failure cases: soft labels produce the same solution, uncertainty bounds are uninformative, or observed disagreement is protocol noise. Avoid calling this causal treatment-effect discovery: all recorded outcomes depend on a fixed generator and protocol.

**Priority:** pursue H1 first, H2 only if training evidence shows useful residual opportunities, and H3 as a bounded robustness analysis rather than a second major project.

## 8. Datasets and baselines

| Dataset | Value | Recommendation and caveat |
|---|---|---|
| Existing MATH source split | No-cost development of the mechanistic analysis | Call it a SEVRA MATH-training holdout. Current difficulty proxies do not prove official levels. |
| Existing MATH500 replay | Separate mathematical cohort, already on disk | Useful frozen-policy external diagnostic; check overlaps and prior exposure, and never tune on it. It is not “MATH Level 5.” |
| Existing GSM8K replay and official snapshot | Arithmetic versus commonsense shift; simpler exact numeric grading | Prefer before downloading another benchmark. Check whether prior analyses exposed it; a fresh split of unused official training questions is an internal holdout, not the official test. |
| CommonsenseQA | Reveals harm and transfer failure; local official keys available | Good complement to math. Historical public replay covers official validation; use unused official training questions for a new internally held-out study, clearly labeled. Exclude all prior manifests, not just completed calls. |
| ARC-Challenge | Science MCQ gives a third reasoning domain without symbolic grading | Best optional additional benchmark. Freeze original IDs/splits and option-label handling before inference. [Clark et al., March 2018](https://arxiv.org/abs/1803.05457). |
| MMLU-Pro | Broad knowledge/reasoning coverage | Defer. Ten options require a new parser beyond A–E; categories multiply comparisons and longer prompts add cost. [Wang et al., June 2024; NeurIPS 2024 Datasets and Benchmarks](https://arxiv.org/abs/2406.01574). |
| Genuine MATH Level 5 | Strong stress test | Optional, after official metadata verification and with a matched easier-level control. Harder can mean more persistent errors, not more recoverable errors; Level 5-only sampling can make the study less informative and more expensive. |

Minimal baseline set: never, always, completion, parse-invalid, completion-or-invalid, one shallow tree or regularized logistic policy, frozen ML, matched-rate random selection, and an unattainable HELP-only oracle. Fix rule definitions using training only. Show a longer-initial baseline for a system-level efficiency claim. A small self-consistency/redraw comparison is optional and must include all sampling costs; it is not the same intervention as verification. Do not compare policies at their most favorable test-selected thresholds.

## 9. Detailed experimental design and live evidence

### What the existing local records actually establish

Recounted main journals: v0.3.0 **19 pairs (7 HELP, 0 HURT)**; v0.3.1 **52 (20 HELP, 0 HURT)**; v0.3.2 **11 (4 HELP, 0 HURT)**. They reuse questions and differ in frozen backends; do not pool them. `reports/live_study/CURRENT_STATUS.json` records backend-version blocking and absent HURT training support. `study_training.py:32` requires at least five examples/class to run; that is an execution guard, not a statistically adequate sample size.

`study_protocol.py:60–92` freezes Qwen3:4b, 1,024 generated tokens/phase, context 4,096, thinking true, greedy sampling, no finalizer, and seeds 42/43. Recheck construction uses the full saved response (`:223`, `:235`; `collector.py:89` combines thinking and content). Generation pilots subsequently used visible-only history. These protocols are scientifically different.

Recomputed the five-question preflight from `outputs/generation_experiments/preflight_5_2026-11-03/calls.jsonl`: 20 calls, **8 length stops, 7 invalid finals**. Initial valid/correct: 3/5 and 2/5; A: 4/5 and 2/5; B: 3/5 and 2/5; C: 3/5 and 3/5. These were operational tests, not model selection evidence. The stored failure audit attributes six invalid cases to no visible content and one to truncated visible analysis before FINAL; no completed valid FINAL was rejected.

Recomputed `completion_3_2026-11-03/records.jsonl`: concise 2,048-token condition B gives **1/3 valid, 1/3 correct, two length stops**; identical concise prompt at 3,072 tokens gives **3/3 valid, 2/3 correct, zero length stops**. Mean latencies are **86.37 s** and **95.44 s**, respectively. The three cases deliberately include past failures. A is an archived reference on another Ollama version, so A→B is not a clean prompt-effect comparison. The archived RAM monitor reports minimum **4.101 GiB** in this later experiment, versus **0.591 GiB** in the earlier preflight; background load and monitoring differ. This supports guarded completion testing, not a guarantee that 3,072 tokens are always enough.

Hardware planning uses the stated Ryzen 7 4800H and the saved hardware snapshot confirming RTX 3050 Laptop **4,096 MiB VRAM** and about **16 GB physical RAM**. No current GPU/model was loaded or probed in this audit. Qwen3-4B-Instruct-2507 is an officially documented nonthinking variant and is a reasonable optional future compact comparator; its hardware fit and accuracy here are unmeasured. Do not assume the installed Thinking-2507 checkpoint can be converted by `think=false`. [Official Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

### Minimum credible staged protocol

1. **Freeze the question before the model:** primary contrast ML versus completion at a fixed token budget or a stated accuracy noninferiority margin. Predeclare one harm endpoint and correction-family handling. Do not make “80%” the success criterion for gate research.
2. **Keep old data developmental.** Lock new question IDs, parser, solver digest, backend, prompts, budgets, seeds, and exclusions before generation. Separate gate-training, calibration, and final evaluation groups. With limited new data, evaluate the old gate strictly as frozen transfer; do not pretend it is trained for the new protocol.
3. **Operational phase:** 10–20 development-only questions, excluded from final evaluation, to establish completion/context/resource reliability. Choose one initial/recheck protocol and stop modifying it. Target reliability is a criterion to test, not a conclusion from 3/3.
4. **Feasibility phase:** 50 development questions with one initial plus one revision each. This is about **100 calls**, not a publication-sized rare-harm evaluation. It can estimate parsing failure, runtime, and whether enough correctable errors exist to justify expansion.
5. **Independent phase:** start with a fixed **300-question evaluation** after freezing decisions, preferably balanced across two domains. With 100 separate development questions, this totals **400 questions / 800 calls** for one initial and one revision each. This is a resource-bounded pilot for large effects, not enough to resolve a 0.5-point difference or establish near-zero harm. Set the sample size using development variance and the smallest meaningful effect before unblinding evaluation.
6. **Paired evaluation:** all policies share exactly one initial response and its recorded revision. Collect revisions for every question offline, even if the gate would skip, to avoid missing-outcome bias. Test keys are used only after generation and frozen policy decisions. Record incomplete calls separately; do not silently replace failures or omit hard questions.
7. **Statistics:** report effect sizes and 95% question-cluster bootstrap intervals for accuracy, harm, tokens, and paired policy differences. Resample entire questions with all trajectories. Use exact McNemar only for one paired binary outcome per independent question; for repeated/fractional outcomes use cluster-aware methods. Correct a small predeclared family of contrasts (e.g., Holm), not every exploratory plot as if confirmatory. A fixed model's bootstrap does not incorporate model-selection uncertainty.
8. **Replication:** cheap gate-training seeds/grouped folds on development data; a separately selected block of new initial seeds for generator robustness if budget permits. Do not call recheck-only seeds independent initial trials. Report each seed, not only the best.
9. **Cost/timing:** separate prompt, generated, and total tokens; record wall time, load/prefill/decode where available, cache policy, failed calls, gate overhead, peak sampled RAM/VRAM, and cold versus warm starts. Summing recorded phase times gives a replay latency estimate, not a directly timed end-to-end deployment trial. A small counterbalanced live schedule is needed for a measured serving-speed claim.
10. **Resource controls:** one loaded model and one sequential request; context-fit checks reserve the requested output space; no hidden retries/finalizers. Use explicit available-RAM floors and in-flight cancellation, as the completion pilot does. Fix a daily time/call cap and stop on memory/backend drift. Do not resume a frozen run with different settings.

At roughly **60–120 seconds/call** as a planning range informed by the local pilots, 100 calls cost approximately **1.7–3.3 hours**, and 800 calls **13–27 hours**. Math, longer outputs, loading, and failures can make this substantially slower. This is an estimate, not a throughput guarantee. Spread sessions across days; no overnight unbounded collection is needed.

For scale: at an observed accuracy near 80%, an ordinary 300-question independent accuracy interval has approximate half-width **4.5 percentage points**. Paired differences can be more precise depending on discordance, but sample size must use that paired variance. Zero harmful events among 25 independent selected questions gives a rough one-sided 95% upper bound around **11.3%**, not proof of safety; even zero among 100 is about **3%**. Repeated rollouts do not turn 25 questions into 50 independent safety trials.

## 10. Compute-efficient roadmap

These are options for approval, not tasks begun by this audit. Effort estimates assume a student familiar with the repository and exclude publication writing/review delays.

| Plan | Work and evidence | Effort | New inference | Expected impact / risk |
|---|---|---|---|---|
| **A — Essential** | Correct dataset/claim naming; explicit related-work comparison; retain heuristic tie; paired CIs and predefined completion/validity strata; run registry/overwrite guard; reproducible paper tables and limitations | 3–5 focused days, CPU-only | **0** for a transparent replay paper; optional 10–20-question operational check adds 20–40 calls | High credibility, low novelty. Suitable professor/replication discussion; not automatically a competitive full paper. |
| **B — Strong, recommended** | Plan A + H1; completion-reliable protocol; one small utility baseline; two domains; 100 development + 300 locked evaluation; long-initial control on those questions; training-only ablations | 2–4 weeks part-time plus collection | **800 paired calls**, plus **400 longer-initial calls = 1,200**; operational checks additional. Approximately 20–40 hours at the planning rate, potentially more for math | High empirical value, moderate novelty. May show heuristics or long initial dominate. That is acceptable if study is rigorous and adds independent evidence. |
| **C — Ambitious** | Plan B + H2 conservative override and bounded H3; one additional compact model/protocol; independent initial-seed subset; risk/adaptation sensitivity | 4–8 weeks part-time | Example cap: **2,400 calls** for two models' 3-call × 400-question comparisons, plus **300 calls** for a second initial+revision block on 75 questions/model; **2,700 total**, operational checks extra | Higher potential, high inconclusive-result risk. About 45–90 hours at the planning rate; no GPU training. Proceed only if development identifies residual signal. |

All gate methods can run on CPU. Local inference may partially offload the 4B model; do not assume complete VRAM residency. Optional university GPU time can accelerate **the same pinned protocol**, but hardware/backend changes must be recorded and not silently mixed into a timing comparison. Free cloud availability, quotas, privacy/terms, and exact backend compatibility require verification before use; cloud access is not necessary for Plan A or B.

Avoid: RL/QLoRA gate training, large embedding models, exhaustive hyperparameter sweeps, many prompt variants, a new agent framework, UI redesign, and benchmark accumulation without a hypothesis. These add cost faster than they add evidence.

## 11. IEEE-style reviewer assessment

Scores are judgments of the **current project as a research submission**, not acceptance probabilities or judgments of the student's ability.

| Dimension | Score / 10 | Reason and concrete route upward |
|---|---:|---|
| Originality | **2** | Core idea and public data directly overlap prior selective-verification work. Improve through a specific controlled question and clearly differentiated evidence, not a renamed classifier. |
| Technical contribution | **3** | Sound classical gate and contracts, but no demonstrated gain over the simple heuristic. A justified residual/risk method must beat strong baselines or expose useful limits. |
| Methodological rigor | **6** | Good grouping, keys, versioning, and initial-only contracts. Improve selection accounting, contamination disclosure, paired uncertainty, protocol matching, and overwrite prevention. |
| Experimental evidence | **4** | Reproducible replay on two domains, but one solver/source split, heavily inspected tests, rare harms, and tiny local pilots. Independent protocol-controlled replication is the main need. |
| Reproducibility | **7** | Strong saved logs, hashes, requirements, and tests. Historical generation provenance is incomplete; status pointers and multiple protocol paths complicate reproduction. |
| Practical relevance | **7** | Avoiding harmful/unnecessary revision on small hardware is real. Demonstrate marginal value beyond completion and measure serving latency before claiming deployment benefit. |
| Overall submission readiness | **3** | A good research prototype and honest replication base; insufficient new evidence for a strong original-method paper. |

Likely reviewer objections: “This largely reproduces the source paper”; “Why use ML when completion does the same thing?”; “Is recovery merely a token-budget artifact?”; “Where are the matched-cost longer-initial baseline and independent test?”; “How can 15 training harms support safety?”; “Why is MATH-training replay described as Level 5?”; “Where are real latency measurements?”

### Venue fit, without acceptance promises

- **IEEE ICDM research track:** established, demanding data-mining venue. Current method/evidence is below a convincing main-track bar. A stronger generalizable learning/evaluation result and rigorous novelty comparison would be needed. The [official 2026 research CFP](https://icdm2026.neu.edu.cn/calls/list.htm) verifies scope; future deadlines and tracks need fresh checking.
- **IEEE International Conference on Big Data:** established competitive main track; a small routing demo is not enough, and data-systems relevance must be real. A focused relevant workshop may be a better home for a rigorous replication/negative study. The [official 2026 CFP](https://bigdataieee.org/BigData2026/calls/papers/) lists main/workshop structures. Do not assume a workshop has the same review or archival status as the main track.
- **IEEE Conference on Artificial Intelligence (CAI):** broader topical fit for empirical AI evaluation or an application-focused contribution, subject to the actual track. The [IEEE Signal Processing Society listing](https://signalprocessingsociety.org/event-names/cai) confirms the series and points to the official forthcoming edition. Broader scope does not establish easier acceptance; verify its CFP, review criteria, dates, fees, and proceedings arrangements.
- **Relevant IEEE-associated workshops or student research forums:** more realistic initial targets after Plan A/B, particularly for reproducibility and resource-constrained evaluation. No specific currently open workshop was verified, so none is promised. A demo track may recognize engineering but is not equivalent to a full research-paper contribution.

Use official IEEE society/conference records to verify sponsorship and publication status. Do not rely on a familiar-looking domain or “IEEE-format” marketing. No acceptance probabilities or guaranteed indexing claims are warranted.

## 12. Final recommended research direction

**Suggested framing:** “When does learning add value to selective LLM revision? A completion-controlled, resource-bounded evaluation.”

The current strongest contribution is an auditable evaluation system that exposes a meaningful negative result: on these historical records, a learned gate largely reproduces a cheap serving-state heuristic. Its implementation is more mature than its scientific claim. Strengthen the paper by making the baseline tie the starting point, controlling the generation protocol, and measuring where learning has—or lacks—marginal value.

Do not promise that improved features, harder questions, calibration, a new benchmark, or a stronger model will produce a learned advantage. If no advantage survives fair comparisons, the appropriate conclusion is that ML is not justified in the tested regime. A carefully delimited explanation, independently replicated, is more defensible than inflated accuracy or a complex model without benefit.

## If this were my research project, here is exactly what I would do.

1. **Essential, now:** show the professor the verified source/target policy table and explicitly say the learned gate does not beat completion. Correct “MATH Level 5” and separate classifier accuracy from downstream answer accuracy.
2. **Essential, before a paper:** write a one-page contribution comparison against SEVRA and the September 2026 risk-aware study. If the proposed claim is already answered there, narrow or change it before running anything.
3. **Essential, cheap:** freeze one primary question: marginal utility of learning beyond completion under a reliable initial-generation protocol. Add paired uncertainty and completion/parse-state analyses to the existing replay, with all historical tests labeled exposed. Preserve old results.
4. **Essential for live claims:** after approval, complete a small operational check; select one protocol on development only. Three successful completions are insufficient reliability evidence. Use the existing 3,072-token result as a starting hypothesis, not a universal setting.
5. **Recommended:** run Plan B in stages with a stop/go decision after 50 development questions. If correctable completed errors are too scarce, do not train a more complicated gate. Study the completion/compute boundary or stop with a bounded negative result.
6. **Recommended:** charge all computation, include a longer-initial baseline, reserve evaluation IDs before collection, and report uncertainty and failures. Obtain a fresh independent evaluation before claiming an improvement.
7. **Optional:** implement the conservative override method only if grouped training validation shows residual value. One extra domain or compact nonthinking solver can test generality; both are optional, not prerequisites for an honest first paper.
8. **Distractions:** optimizing the dashboard, chasing 80% with old pairs, relabeling malformed outputs, mining test subgroups, larger classifier sweeps, fine-tuning an LLM, and collecting thousands of unplanned calls.

**Decision:** present the project now as a rigorous prototype with a clear negative baseline result and an actionable research plan. Do not yet present it as a novel superior learned-gating method or submit it unchanged as a strong IEEE research paper. No recommendation in this report has been implemented.
