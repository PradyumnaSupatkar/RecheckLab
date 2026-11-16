# RecheckLab Phase 4 — research landscape and feasibility

**16 November 2026. Discovery only.** No new model, controller, benchmark, live generation or statistical experiment was implemented. Scores and research hypotheses are judgments, not measured performance. References R01–R22 and venue sources V1–V4 resolve in [REFERENCES.md](REFERENCES.md).

## Executive assessment

The strongest **investigation** is D1: **cost-aware, paired evaluation of fixed stochastic inference policies with explicit harm uncertainty**. This changes the research object from deciding whether to recheck to deciding what evidence is needed to compare policies reliably. Its first test can use CPU-only replay. **Novelty is plausible but unconfirmed; none of the eight candidates currently earns an unconditional paper-development GO.** Allocate at most four working days to the leading candidate's derivation, baseline comparison and kill test after approval. Do not spend six weeks merely applying established confidence sequences to a renamed outcome.

The alternatives ranked next are D2, interruption-safe answer delivery, and D3, protocol-transfer sentinel adequacy. They are fallback investigations with stronger prior-art or inference-cost risks, not equally endorsed projects. Five other directions are screened and rejected as principal contributions rather than dressed up to fill the list.

## Repository evidence inspected

No requested report family is missing. The active project root is `recheck_transfer/`; the outer directory is a packaging folder. The following were read or checked, without changing them:

| Evidence | Location and implication |
|---|---|
| Original readiness audit | `IEEE_RESEARCH_AUDIT.md`: architecture, historical result verification, provenance and overwrite risks. |
| Active v0.2 | `outputs/v0.2/models/source_gate.joblib`; `data/prepared/*.parquet`; `reports/split_manifest.json`. Existing model/report evidence only, no Phase 4 fit or new score. |
| Residual feasibility | `outputs/ieee_phase2_feasibility/run_2026-11-16_01/FEASIBILITY_VERDICT.md`, `POLICY_COMPARISON.csv`: no held-out residual advantage. |
| Measurement audit | `outputs/ieee_phase3_measurement/run_2026-11-16_01/{IEEE_PHASE3_VERDICT,MEASUREMENT_AUDIT,METRIC_SENSITIVITY,NOVELTY_REVIEW}.md` and measured CSVs: completion association is not causal attribution. |
| Efficiency | `outputs/ml_efficiency_evaluation/evaluation_2026-11-03/{TECHNICAL_REPORT,PROFESSOR_SUMMARY}.md`, `policy_metrics.csv`: token accounting is not live latency. |
| Local operations | `outputs/generation_experiments/{preflight_5_2026-11-03,completion_3_2026-11-03}/REPORT.md`, summary/protocol files; `outputs/live_study/v0.3.{0,1,2}/questions.jsonl`. |
| Reusable source | `rechecklab/data.py:34,39,74,233,259`; `features.py:23,88,117`; `metrics.py:9,22,101`; `predictor.py:121,200,303`; `study_data.py:128 benchmark_candidates`; `study_protocol.py:173,188,215,235`; collection journal/read functions in `study_collection.py`. Grouping, initial-state schemas, outcome records, keys and frozen protocol checks are useful foundations. |

Historical findings remain: source-test ML/completion 90.00%; target-test ML/completion 76.6712%, with identical target decisions. Source disagreements are two neutral calls. Training-held-out residual accuracy ties completion at 92.2435%; development ties at 92.00%. These are archived results, **not rerun as new Phase 4 experiments**. The source is a SEVRA MATH-training holdout, not verified Level 5. No candidate below is selected by re-optimizing exposed test outcomes.

The previous completion study's B/C comparison has six new calls, about 86–95 seconds mean latency, and minimum sampled available RAM 4.1015 GiB. The earlier preflight had 20 calls, seven invalid outputs and a reported low-memory episode. A and B also differed in Ollama version/streaming, so they do not isolate a prompt effect. The laptop is feasible for bounded sequential work, not demonstrated safe for an unattended large study. A 3,072-token budget helped three selected cases; it is not a general completion guarantee.

## What the literature rules out

| Tempting claim | Closest primary sources | Research consequence |
|---|---|---|
| “Learn when verification helps” | R01, R03 | Already the direct problem; another classical classifier is weak novelty. |
| “Most gains are format repair” | R02 | Directly anticipated, with more extensive controls than our replay. |
| “Use GPU resources to choose compute” | R04, R10 | Live cost and deadline-aware controllers already exist. Mere hardware substitution is insufficient. |
| “Evaluate with fewer samples” | R05–R08 | Active testing, sequential bounds and repetition allocation are established. A specialized combination needs an observable advantage over strong adapted baselines. |
| “Detect unsafe transfer and abstain” | R14–R16 | Generic monitored abstention is already developed; no label-free correctness guarantee under arbitrary shift. |
| “Deterministic checking or independent verification improves answers” | R20–R22 | Already studied; focus on a genuinely unanswered mechanism or reject. |
| “Fixed seeds are insufficient; control benchmarking state” | R11–R13 | Established reproducibility problem; another timing checklist alone is not a strong contribution. |

**Gap status:** verified prior-art overlaps are numerous. R07 explicitly identifies heterogeneous costs/correlated outcomes as limitations, establishing a limitation of that paper—not a confirmed global research gap. D1's joint policy-contrast/harm/cost setting is a plausible specialization requiring a closest-method reduction check. D2's transaction-level answer preservation and D3's narrow sentinel-transfer question are also plausible but unverified. No absence-of-keyword-search result is treated as novelty.

The search covered primary work through the stated cutoff; it was not exhaustive, and some IEEE/ACM full texts were inaccessible. Preprint claims were inspected, not independently reproduced. The recommended methods reuse known statistical/system ideas and must credit them. Any eventual paper needs both a distinct result and a correctly scoped claim.

## Data independence: what is actually available

A small read-only inventory checked every `outputs/live_study/**/questions.jsonl`, the generation preflight question manifest, all `data/raw/*.parquet` queries, and existing benchmark snapshots. It used `data.question_hash`, source question IDs and the benchmark's canonical MCQ option order. It did not select questions or inspect candidate answers to maximize performance.

| Local official snapshot | Rows | Absent from checked manifests/replay by exact canonical matching |
|---|---:|---:|
| GSM8K training | 7,473 | 6,673 |
| GSM8K test | 1,319 | 0 |
| CommonsenseQA training | 9,741 | 9,536 |
| CommonsenseQA validation | 1,221 | 0 |

Snapshots are under `outputs/live_study/v0.3.0/raw/`, GSM8K revision `740312add88f781978c0658806c59bc2815b9866` and CSQA revision `94630fe30dad47192a8546eb75f094926d47e155`. Keys are published `####` numeric answers or `answerKey`; keep them outside prompts and acquisition features. No comparable authoritative local MATH key join was established in Phase 3. Do not make MATH semantic correctness the sole endpoint of a new paper before resolving that provenance.

These counts establish **candidate availability**, not certified untouched status. Exclude every reserved manifest ID even if no call was completed, check all call journals/standalone question files and near-duplicate stems before selection, and obtain the researcher's confirmation that no untracked experiment used them. Exact hashing does not detect paraphrases or pretraining contamination. Official training items can support an independently sampled study holdout, but must not be described as an official test-set result.

Future split rule: draw development and evaluation IDs once from the remaining pool, stratified only by dataset (and predeclared input-length bands if needed), before any response collection. Keep every initial/revision/repeat/protocol variant of a question in one partition. Lock the question manifest and keys separately. Historical source train/dev and exposed tests can support exploration, implementations and clearly labeled replay diagnostics; none can certify fresh generalization. A protocol change creates new responses, not a newly untouched question.

## Common methodological requirements

1. Define whether the estimand is a finite recorded library, a future random rollout on a fixed benchmark, or a population of new questions. Reusing two logged rollouts in simulation does not identify a high-precision per-question response probability.
2. Distinguish classifier accuracy, final correctness, harm, valid delivery, inference calls, tokens and measured wall time. Initial generation, pilot fitting/profiling, retries and failed attempts count toward cost. No energy claim without appropriate instrumentation; GPU-only telemetry is not laptop wall energy.
3. Match initial attempts across policies where the policy problem permits it; independent fresh-generation comparisons need independent seeds and question blocking. Do not use the better recorded revision. A common seed does not guarantee a valid common-random-number coupling across different prompts.
4. Fixed-policy evaluation must not quietly become policy optimization. Any method that selects its best policy on evaluation data needs selection-aware inference or another holdout. Use group-level uncertainty and predeclared multiplicity handling; optional stopping requires valid sequential inference.
5. Include the strongest simple competitor. D1 must beat uniform sampling **restricted to policy disagreement**, not only wasteful full-pool evaluation. D2 must beat timeout-plus-retain, not only an unsafe overwrite implementation. D3 must beat a random sentinel sample, not only no monitoring.
6. Count missing/invalid/deferred outcomes explicitly. For answer-delivery endpoints they are failures; for semantic accuracy they are a separate, reported category. Never exclude failed requests from the denominator to improve reliability.

## Compute and publication boundary

Use one existing model sequentially; CPU statistics are the priority. Timing plans below use a **planning envelope of 60–180 seconds/call**, informed by tiny archived operational samples, not a validated forecast. Thus 300 calls imply about 5–15 active inference hours and 900 about 15–45 hours, excluding cooldown, annotation and recovery. A new request shape may be slower. Stop rather than compensating for unsafe memory by changing settings mid-study. A second machine is optional for external validity, not a prerequisite for the first kill test.

The 2–6 week target is reasonable for one bounded study **after** its kill test and novelty check succeed; it is not reasonable for developing a new statistical theory, runtime engine and broad multi-model benchmark simultaneously.

## IEEE venue fit, honestly scoped

| Venue family / track | Possible fit | Evidence needed and publication distinction |
|---|---|---|
| IEEE ICDM research | D1 if a genuinely nontrivial evaluation method emerges | Established demanding data-mining venue; proof/assumptions, strong baselines and external evidence needed. V1 states IEEE proceedings and regular/short designation after review. A narrow laptop case study is a poor fit. |
| IEEE Big Data research | D1 evaluation methodology, D3 if broader evidence exists | V2 includes benchmarking/evaluation frameworks. Need general evaluation value beyond a tiny app; not automatically easier because broad. Workshop calls are separate and may differ in review and archival status. |
| ACM/IEEE SEC research | D2 if there is a real systems mechanism | V3 emphasizes implemented systems and performance evaluation. One machine narrows claims; multi-session workload controls and strongest fallback baseline are essential. Position papers are separately judged, not equivalent evidence to a full research paper. |
| IEEE BigComp regular/short research | A bounded, rigorous D1 or D2 study | V4 supports smart-computing scope and archival papers. This may be a more scope-compatible target for a student project; no claim of easy acceptance or known acceptance probability. |
| Workshops / demos / student forums | Early mechanism or negative-result feedback | A peer-reviewed workshop may be useful if the exact CFP fits. SEC demos have archival extended abstracts, but a dashboard demo does not satisfy the desired full scientific contribution. Forum eligibility and archival status must be checked individually. |

The inspected 2026 calls are closed. Use them for scope, not an immediate submission deadline; recheck future calls. Quality and contribution must determine track choice. Do not choose a vaguely IEEE-associated event simply for indexing or a participation certificate.

## Preservation

Only the six requested Phase 4 report files are created in this fresh directory. Baseline inventory: **357 existing non-cache files**, combined SHA-256 `e8e89eee31343bad23ce72df84e38fafb2b3e21c64155b231e742619d524945b`. Construction: sort relative paths; hash each path's UTF-8 bytes followed by that file's SHA-256 bytes, excluding `.git`, `.venv`, caches, `node_modules`, and this Phase 4 directory. Final verification reproduced this exact count and digest: all 357 original files remain unchanged. No source, application, model, protocol or earlier report is changed.
