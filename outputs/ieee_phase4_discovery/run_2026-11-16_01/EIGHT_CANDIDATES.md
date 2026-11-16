# Eight distinct research questions — screened, not eight endorsements

All proposed gains are hypotheses. No candidate was trained or evaluated in Phase 4. Reference IDs resolve in [REFERENCES.md](REFERENCES.md); evidence and independence rules are in [RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md). The eight questions concern evaluation acquisition, delivery reliability, transfer monitoring, oracle interpretation, output contracts, executable evidence, context anchoring and benchmarking design respectively. They are not eight classifiers.

## D1 — Cost-Aware Certification of Paired LLM Policy Differences

**Question:** Can we compare two fixed inference policies' accuracy and harmful-revision rates with fewer paid rechecks, while retaining valid uncertainty under adaptive acquisition and stopping?

**Hypothesis:** After eliminating known-zero policy-agreement contributions, heterogeneity in contrast variance and recheck cost sometimes permits further savings beyond uniform disagreement sampling; sometimes it does not. A meaningful contribution is a validated benefit boundary plus a simple valid acquisition design, not higher solver accuracy.

**Prior-art test:** R05 active evaluation, R06 sequential LLM evaluation, R07 rollout allocation and R08 off-policy confidence sequences cover almost every ingredient. Generic adaptive sampling is already solved. The narrow possible gap is **joint paired accuracy/harm contrasts with shared initial attempts, clustered repeats and heterogeneous generation costs**. R07 explicitly leaves cost/correlation extensions open, but that does not establish a global gap. Originality is **plausible/unverified**. An IEEE reviewer will reject mere application of Neyman allocation to a new dataset unless the problem creates a distinct result or useful empirical failure boundary.

**Method:** Known disagreement mask; a few initial-only strata; uniform random sampling within each stratum; predictable allocation between strata; simultaneous confidence bounds for accuracy and harm differences; an explicit inconclusive outcome. Reuse pair schema, group IDs, cost accounting and fixed policy decisions. New work: acquisition simulator, statistical validity argument, cost model and coverage checks. No new gate.

**Data/baselines:** Existing multi-rollout CSQA/GSM8K/MATH500 for exposed replay diagnostics; unmanifested GSM8K/CSQA training questions for independent collection. Compare full enumeration, uniform full-pool, uniform disagreement, fixed stratified allocation, and an adapted sequential/active estimator. Ablate disagreement pruning, cost awareness, paired versus unpaired estimation and cluster handling. Primary endpoint: paid tokens to a fixed joint precision target at valid coverage; secondary: calls, interval width, false certification and measured acquisition overhead.

**Feasibility:** CPU-only kill test, zero fresh calls; 3–6 weeks if successful. Possible independent library: 300 questions × (one initial + two rechecks) = 900 calls, 15–45 active hours under the provisional envelope. If this is too costly, publish no generality claim from replay alone. Main risks: tiny differences, wide honest intervals, cost-prediction error, two-rollout limitations and a standard estimator matching the proposal. **Inconclusive risk: high. Rank 1; conditional investigation only.**

## D2 — Interruption-Safe Answer Delivery for Local LLM Verification

**Question:** Under a deadline or resource abort, can a revision transaction improve delivered-correct-answer rate without losing a previously available answer, beyond a simple retain-on-timeout implementation?

**Hypothesis:** Separating committed answer state from speculative revision state, and bounding the time to cancel/drain a revision, can matter when resource interruptions occur mid-call. Meaningful contribution requires a mechanism and benefit beyond ordinary fallback; “save the old answer” alone is standard engineering.

**Prior-art test:** R04 resource-aware control already charges deferrals; R10 handles interruption and deadlines; R01 protects initial answers through selective actions. The possible gap is application-level answer commit semantics plus measured cancellation latency on a memory-constrained discrete-GPU runtime. **Plausible/unverified**, and “runs on RTX 3050” is not novelty. Exact equivalent systems may exist outside the inspected set.

**Method:** State machine: initial answer committed → revision pending → valid candidate committed atomically, or rollback to original on timeout/backend failure. An admission rule uses measured deadline slack and conservative remaining-time estimates, not gold or a learned HELP model. New components: deadline wrapper, server-side cancellation verification, transaction journal and safe resource telemetry. Reuse streaming collector, parser and RAM guard patterns.

**Data/baselines:** Random existing GSM8K/CSQA training items, not only previously failed questions. Never recheck; full verification; timeout-and-retain; resource-blind deadline admission; adapted resource-aware admission. All baselines use safe output retention and the same model. Ablate atomic commit, resource signals and cost prediction. Primary endpoint: fraction of **all** requested questions delivered correctly by deadline. Secondary: valid-delivery rate, harmful commits, cancellation lag, p95 latency and minimum RAM. Abort/failure is counted, not censored away.

**Feasibility:** Old logs support design but not a controlled benefit. Mock/fault-injection kill test needs zero model calls; a later operational test needs about 40–80; bounded study about 300–600 sequential calls, 5–30 active hours, 3–6 weeks. No intentional OOM; use safe client timeouts and bounded background load only after separate approval. Risk: HTTP cancellation does not cancel GPU work, or retain-on-timeout already solves everything. One-machine evidence limits external validity. **Inconclusive risk: high. Rank 2, backup only.**

## D3 — What Can Small Sentinel Sets Certify About Controller Transfer?

**Question:** Can a small predeclared paired sentinel set detect harmful changes in a frozen controller's utility after an inference-protocol change, and when must it refuse to certify transfer?

**Hypothesis:** Completion/format summaries alone may miss changes in the conditional recovery/harm distribution; a small randomized labelled sentinel may be more informative than a larger unlabelled drift check. The contribution would be a measured detectability boundary, not a new abstention classifier or an assertion of label-free safety.

**Prior-art test:** R14/R15 already monitor harmful risk shifts; R16 combines shift detection and abstention; R17/R18 establish protocol sensitivity. **Known substantial overlap.** A narrow controller-relative sentinel adequacy study remains plausible but unverified. General drift monitoring, conformal abstention or threshold adaptation is rejected as a new claim.

**Method:** Freeze controller and grader. Compare predeclared protocol pairs, changing only one factor at a time, e.g. output cap or wording, within a verified context limit. Monitor paired utility loss relative to completion on a randomized sentinel sample using an existing sequential method. Include harmless changes and nonstationary baselines. New work: protocol matrix, blinded sentinel allocation and failure-boundary analysis; reuse manifests, hashes, frozen predictions and journals. No retraining or threshold tuning.

**Data/baselines:** Old records cannot identify a controlled shift effect. Use remaining CSQA/GSM8K training questions for development and independent evaluation. Baselines: no monitor, fixed random sentinel, generic feature-distribution test, supervised risk monitor, prediction-powered monitor. Ablate paired sampling, completion stratification and true-label access. Primary: number of paid outcomes required to detect a predeclared harmful utility drop at controlled false-alarm rate; secondary: missed shifts, benign alarms, delays and abstention rate.

**Feasibility:** CPU diagnostic starts with old protocol metadata but can only rule out opportunities, not validate transfer. A 30-question × two-protocol × initial/recheck pilot needs 120 calls; a 150-question comparison needs 600, roughly 10–30 hours. Three to six weeks. Multiple shifts and rare harms quickly exceed this budget; do not claim a 1-pp safety guarantee from 30 sentinels. Risk: monitor reduces to R15 with no new insight, no harmful shift exists, or all monitors are underpowered. **Inconclusive risk: very high. Rank 3, weak fallback blueprint rather than a GO.**

## D4 — Which Oracle Frontiers Are Attainable From Initial State?

**Question:** How much of a reported hindsight accuracy–cost frontier requires knowing a future rollout, rather than just knowing whether the initial answer is wrong?

**Hypothesis:** Outcome-conditioned oracle **cost** can be substantially optimistic, even when its accuracy has a simpler interpretation. This matters because a controller can be compared unfairly with a free hindsight action selector.

**Prior-art test:** R01/R03 explicitly label oracles as hindsight; R07 uses an oracle only as a diagnostic. This is mostly an interpretation/theory correction, not an algorithmic breakthrough. **Important analytical kill:** for fixed initial correctness b∈{0,1} and recheck correctness Y, `E[max(b,Y)] = max(b,E[Y])`. Thus there is no stochastic Jensen accuracy gap for this binary keep/recheck setup. An oracle knowing initial correctness could attain that accuracy by rechecking every initially wrong answer, although it would also pay for unsuccessful rechecks. Claims of an additional accuracy gap from rollout randomness alone would be false here.

**Method/data:** Derive accuracy/cost bounds under explicit information sets; contrast per-outcome oracle with gold-initial-state oracle and deployable policies. Existing paired data suffice for empirical accounting; no keys used for deployable rules. Reuse `policy_metrics`, question groups and tokens. New: bound derivations and a table, CPU only. Baselines: HELP-only hindsight, recheck-all-initially-wrong hindsight, completion, never/always. Ablate repeated-rollout averaging versus illegal best-of-two. Primary: oracle cost gap; secondary: sensitivity to neutral outcome cost.

**Feasibility:** Zero calls, 2–4 days for a note; extra repeated data would not make the elementary identity novel. Risk is not lack of power but lack of contribution. No standalone IEEE paper justified by this observation. **Rank 4; reject as main direction, retain as a useful evaluation clarification.**

## D5 — Non-Destructive Finalization Contracts

**Question:** When a finalizer is instructed only to expose an existing answer, does it preserve the asserted answer content?

**Hypothesis:** Grammar compliance need not imply content preservation; a finalization call can create a new answer. A meaningful result would quantify a contract violation under independent annotation and provide a defensible preservation mechanism, not simply recover more parseable answers.

**Prior-art test:** R02's forced re-extraction directly overlaps; R19 establishes structured decoding; R01 already has finalizers. An answer-preserving contract is narrower than Phase 3 but **high prior-art overlap**, not a proven new research gap.

**Method:** Identify an unambiguous asserted answer without reference to gold; deterministic renderer when possible, abstain otherwise; compare with an instruction-only finalizer and a constrained finalizer. Reuse saved raw/visible channels, parsers, hashes and official keys. New: human assertion annotation and transformation log. No finalizer may search gold for a convenient span. Baselines: no finalization, deterministic extraction, unconstrained finalization, grammar-constrained finalization. Ablate contradiction/multiple-assertion handling and visible versus thinking access.

**Data/metrics:** Existing selected MATH examples are inadequate; CSQA/GSM8K keys help grade outcomes but do not prove intent. Primary: rate of changing the asserted answer; secondary: valid-output recovery, new harm, tokens and ambiguity. Two independent annotators and symmetric before/after assessment required.

**Feasibility:** Annotation kill test on 40 random transcripts, zero calls. Later 100–200 finalizer calls, about 2–10 hours depending on actual short-output runtime, 2–4 weeks including annotation. Risk: preservation is trivial when extraction succeeds and unidentifiable when it fails; direct prior art covers the finding. **Rank 7; reject as principal direction unless a new contract-specific gap is established.**

## D6 — Executable Counterexamples Instead of Generic Self-Critique

**Question:** Does a narrowly sound counterexample, such as a failed substitution in explicitly supplied equations, localize useful revisions more efficiently than generic self-critique?

**Hypothesis:** Verifiable negative evidence can reduce unnecessary correction compared with fluent but unsupported critique. Contribution would require a formally delimited witness class, explicit abstention and a new localization/cost finding.

**Prior-art test:** R20 program-aided reasoning, R21 key-condition checking and R01's candidate-specific checks substantially anticipate this. “Add SymPy/Python” is not novel. **Known overlap**, with a possible narrow witness-soundness question not established as open.

**Method:** Deterministically parse only a predeclared explicit-equation grammar; verify a candidate substitution or arithmetic equality; return a counterexample or abstain. A contradiction can reject that calculation, but absence of contradiction cannot certify the whole word-problem solution. Do not execute unrestricted model code. Reuse responses/grading; new safe parser, witness checker and targeted prompt.

**Data/baselines/ablations:** Existing MATH raw responses lack trusted keys; broad GSM8K natural-language translation remains the difficult part. Use a predefined syntactic eligibility rule before labels, report coverage on the complete benchmark, and manually validate translation. Baselines: no check, generic verification, PAL-style execution, key-condition verification. Ablate witness versus generic error flag and parser coverage. Primary: correct fixes per paid recheck within eligible cohort; secondary: false witnesses, harm and full-cohort coverage.

**Feasibility:** CPU witness audit first; 200–400 later calls, 4–20 hours, 4–6 weeks. Cannot depend on downloading a formal-math dataset now. Risk: too few supported questions, unsound extraction, comparison conflated with solving. **Rank 6; reject main direction on novelty/coverage grounds.**

## D7 — Causal Effect of Showing the Initial Answer to a Verifier

**Question:** Does exposing the initial answer and rationale to a verifier help error localization or mainly anchor its decision?

**Hypothesis:** The effect differs between initially right and wrong answers and can be separated by randomizing the context while holding question, solver and compute protocol fixed. The contribution would be a new, reproducible boundary condition, not another improved prompt.

**Prior-art test:** R22 deliberately uses independent verification; R21 and R03 already study revision evidence/prompt effects. **Substantially covered.** An additional compact-model replication is insufficient unless it reveals an unexplained mechanism.

**Method:** Randomize question-only verification, initial-answer-only context and full initial rationale. Use the same initial response for all conditions and fixed budgets; independent revision seeds, with a paired question analysis. New causal protocol; reuse request logs/keys. No manipulated gold answers or hand-picked wrong examples.

**Data/baselines/metrics:** Existing A/B/C preflight is too small and confounded to answer this. Fresh random CSQA/GSM8K questions required. Baselines: independent re-solve, ordinary verification, factored independent checks; no-answer baseline. Ablate rationale exposure and equal-cap versus actual-token accounting. Primary: context treatment effect on harmful revisions; secondary: helpful fixes, final accuracy, valid outputs and cost. Predeclare initially-correct/wrong strata using offline keys only, never deployable features.

**Feasibility:** 20-question kill pilot × (initial + 3 revisions) = 80 calls; meaningful study 100–200 questions = 400–800 calls, 7–40 hours, 3–5 weeks. Rare HURT events make modest effects difficult to estimate. **Rank 8; reject main direction, use only as a justified ablation in another study.**

## D8 — Cache and Execution-Order Confounding in Policy Benchmarks

**Question:** Can warm cache, model-loading and run-order effects reverse measured latency rankings between inference policies, and can a blocked crossover evaluation remove that confounding cheaply?

**Hypothesis:** Token-matched policies can have different measured costs because of prompt reuse and state carryover. A meaningful contribution requires a reproducible ranking reversal and a better experimental design beyond ordinary warmup/randomization.

**Prior-art test:** R12 directly addresses isolation/repeated benchmarking, R13 prompt reuse, R11 runtime reproducibility. **Known strong overlap**; “measure actual latency” is essential practice, not novelty.

**Method:** Randomized policy order within question/session blocks, separate cold-load and warm-service estimands, cache-hit proxies and prompt-evaluation durations. No backend replacement or cache engine implementation. Reuse streaming request logs, timings and seeds; new scheduler and instrumentation validation. Baselines: fixed-order token proxy, warmup-only, randomized crossover and sequential isolation. Ablate order blocking, loading separation and cache state.

**Data/metrics:** Historic replay has no trustworthy latency; tiny live logs motivate only. Random unmanifested GSM8K/CSQA items and repeated sessions required. Primary: stability and uncertainty of policy latency differences; secondary: correctness, p95 latency, prefill/decode split and cache-hit evidence. Do not infer energy from latency. If Ollama cannot expose or control the needed state, report that before collecting.

**Feasibility:** About 48–96 calls for a mechanistic kill test, 300–600 for broader session replication, 5–30 hours, 2–4 weeks. Same-machine repeated sessions are not independent hardware replication. **Rank 5; reject standalone novelty, require its controls in D2 and any future efficiency claim.**

## Ranking interpretation

`CANDIDATE_RANKING.csv` contains all eight criteria, costs and risks. Weights: originality 25%, problem importance 15%, technical contribution 20%, feasibility 10%, hardware fit 5%, reuse 5%, defensibility 10%, IEEE fit 10%. Convenience/reuse cannot compensate for absent originality. A high numerical score does not override a decisive prior-art failure; ranks 4–8 are rejected main-paper directions. Scores are subjective triage with roughly ±1-point uncertainty per criterion, not acceptance probabilities. The top three blueprints specify ways to disprove their key assumptions before committing to implementation.
