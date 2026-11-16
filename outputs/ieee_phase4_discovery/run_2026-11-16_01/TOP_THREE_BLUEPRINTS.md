# Top three blueprints — proposals requiring approval and kill tests

No experiment below was executed. Each numerical target is a **proposed decision criterion**, not an expected result. References resolve in [REFERENCES.md](REFERENCES.md). A failed novelty reduction check overrides a promising benchmark result.

## 1. Leading candidate: Cost-Aware Sequential Evaluation of Paired LLM Inference Policies

### 1–5. Question, hypothesis, conditional contribution and competitors

**Working title:** *When Is a Recheck Policy Worth Evaluating? Cost-Aware Certification of Paired Accuracy and Revision Harm.*

**Research question:** Given fixed policies sharing the same initial attempt and revision action, can one estimate their accuracy and harm differences to a declared precision at lower generation cost than the best simple evaluation design?

**Hypothesis:** Known agreement structure and heterogeneous contrast variance/cost create exploitable evaluation savings, but rare-harm certification and weak cost predictions impose identifiable failure boundaries.

**Conditional contribution:** A minimal acquisition design with correctly justified joint uncertainty, plus a reproducible characterization of when it improves evaluation efficiency and when it cannot. Do not claim a better gate, more accurate LLM or a newly invented confidence sequence. The proposed specialization must prove more useful than applying standard methods unchanged.

**Closest competitors:** R05 active testing, R06 sequential LLM evaluation, R07 stochastic rollout allocation, R08 off-policy confidence sequences. The scope difference is paired policy contrasts and joint harm endpoints with paid generation, not discovering adaptive evaluation. These sources are mandatory comparisons, not just citations in an introduction.

### 6–7. Architecture and mathematical target

Freeze a bank of policies **before** evaluation: never, always, completion, existing ML, and optionally a fixed hash-random 50% policy as a design stress control. Report every predeclared pair; the identical ML/completion case is an exact-zero sanity check, not evidence of statistical efficiency. Do not manufacture a favorable policy pair after inspecting its outcomes.

For question i with one fixed initial state, let b_i be initial correctness, Y_ir correctness of recheck rollout r, and a_i,b'_i∈{0,1} the two fixed policy decisions. Define

`D_ir = (a_i - b'_i)(Y_ir - b_i)`

`H_ir = (a_i - b'_i) 1{b_i=1 and Y_ir=0}`.

Targets are question-equal means Δ_A = N⁻¹Σ_i E_r[D_ir] and Δ_H = N⁻¹Σ_i E_r[H_ir]. Agreement implies both contrasts are exactly zero without generating a recheck. This identity is elementary and **not the novelty claim**. It requires the same revision action; it does not apply to two different recheck prompts or solvers without additional potential-outcome data.

For finite replay, replace E_r with the mean of the **recorded** rollouts. Reveal the whole question cluster together and charge all revealed rollout tokens. Never simulate arbitrarily many independent outcomes from two records and claim they estimate a true rollout distribution. In future collection, fix the number of repeats per revealed question; revisiting questions adaptively is a separate extension with extra validity obligations.

Create a few predeclared strata on decision direction, completion and initial-length bands. Initial correctness may also be used by the **evaluator**, because the initial response and trusted key have already been paid for/available; all baselines must receive this information equally. It never enters the deployed policy or model prompts. Select the next stratum predictably from previous observations; sample an unrevealed question uniformly within it. Keep nonzero exploration until a stratum is exhausted.

A practical allocation heuristic is `n_h proportional to w_h * sigma_h / sqrt(c_h)`, with shrunk contrast-variance estimates and training/past-only cost estimates. This is classical cost-adjusted Neyman allocation, not a new theorem. A joint version prioritizes whichever endpoint currently dominates the required interval width. Actual future token usage cannot select the next question.

Start with established finite-population or bounded-observation confidence sequences per stratum, combined through known weights. A conservative audit implementation can allocate α across endpoints/strata and use an α_t∝1/t² spending schedule over within-stratum sample counts; an experienced statistics reviewer must check the without-replacement and adaptive-stratum argument. More efficient betting bounds are a baseline/implementation improvement, not novelty by themselves. Do not claim coverage until the construction and assumptions have been checked.

Total evaluator cost is `C_initial(all N) + C_pilot + C_acquired_rechecks + C_CPU`. A result about recheck-call savings must also report total-cost savings; common initial generation may dominate.

### 8–11. Evaluation, baselines, endpoints and statistics

**Development:** use historical source training for design debugging and source development for a frozen replay comparison. Public MATH500/GSM8K/CSQA records are exposed external stress libraries, not independent confirmation. Retain all grading limitations. Freeze the policy bank, strata, costs, endpoints and tolerances before any final diagnostic library is unmasked to the acquisition procedure.

**Five baselines:** (1) full enumeration as exact finite-library reference; (2) uniform full-pool paired sampling; (3) uniform disagreement-only sampling; (4) fixed cost-adjusted stratification with a held-out pilot; (5) an adapted R06/R08 sequential estimator with the same input/cost information. R07 is an additional allocation comparator where repeated-rollout budgets support its assumptions; do not fake an 8–64-repeat evaluation from two recorded repeats.

**Primary endpoint:** paid recheck tokens required to achieve simultaneous accuracy/harm interval precision, at demonstrated nominal coverage. Proposed replay targets: accuracy half-width 2 pp, harm half-width 1 pp, familywise confidence 95%. Also report 1/0.5 and 5/2 pp sensitivity, without choosing whichever favors the method. A cost cap yields “inconclusive,” never a forced winner. Secondary endpoints: total tokens, acquired calls, interval width versus cost, CPU overhead and false sign/equivalence declarations.

For noninferiority/equivalence statements, predeclare distinct practical margins and require the appropriate entire interval to satisfy them. Intervals containing zero do not establish equivalence. If multiple policy pairs are certified, allocate error across them or state that only one preregistered pair is confirmatory.

Run repeated CPU random-acquisition replays against each known finite-library mean. Report coverage with binomial Monte Carlo uncertainty, paired cost differences across sampling seeds and cluster-level sensitivity. This validates the **evaluation procedure on a finite library**, not the LLM's unknown population accuracy. No expensive call is made during this replay.

### 12–15. Compute, milestones, failure scenarios and kill test

**First four days, after approval:** day 1 reduction/validity note against R05–R08; day 2 implement strong simple baselines in an isolated directory; day 3 bounded replay; day 4 reviewer-style decision. No model fitting is necessary. If the design is already a direct special case with no nontrivial empirical question, stop on day 1.

**Kill experiment:** at most 500 random-acquisition replays per predeclared comparison during triage, CPU only. Include trivial agreement, rare-harm and broad-disagreement cases. Do not tune acquisition on exposed test outcomes. Reject if there is a coverage failure, if no policy contrast needs acquisition, or if the proposal fails to reduce median paid recheck tokens by a proposed **20% practical threshold** against the strongest simple valid comparator across at least two predeclared dataset libraries at equal precision. Also reject if a library shows over 10% cost degradation without a predeclared, observable warning condition. These thresholds are effort-allocation preferences, not discoveries or acceptance criteria.

Monte Carlo 95% coverage has about 1 percentage-point standard error at 500 replays; apparent 94–96% differences cannot establish validity. A simulation check cannot replace the mathematical argument. A successful triage would warrant a larger CPU-only coverage check and independent collection, not an immediate claim.

**Possible independent confirmation:** after an operational gate, 60 development questions (30/domain) and 240 reserved evaluation questions (120/domain), each one initial plus two revisions: 900 total calls. All policies consume the same fixed revision protocol; exhaustive collection supplies a finite-library reference for a withheld adaptive evaluator. Simulated acquisition savings on this library remain simulated; actual acquisition time needs a separately approved prospectively sampled run. Report current actual total collection cost honestly.

Plan 15–45 active inference hours plus CPU/annotation over weeks 2–5, writing in week 6. This is optional after the kill test, not authorized now. 240 evaluation questions may support an estimator-efficiency study but will not establish tiny semantic accuracy differences or stringent rare-harm safety. If intervals remain wide, report the limitation or stop; do not silently enlarge the experiment.

**Failure scenarios:** strong baseline wins; adaptive sampling chases noise; few disagreements; pilot costs erase savings; harm endpoint forces near-enumeration; missing grader trust; cost/runtime mismatch; direct prior-art equivalence. All are legitimate reasons to terminate this direction.

## 2. Backup: Deadline-Safe Answer Transactions for Local LLM Revision

### 1–5. Definition and claim boundary

**Working title:** *Preserving Answer Availability During Interrupted Local LLM Verification.*

**Question/hypothesis:** Can transaction-level retention and measured cancellation/admission control improve correct delivery by a deadline beyond safe timeout-and-retain? The conditional claim is a demonstrated failure mechanism and remedy, not “resource-aware inference is new.” Closest work: R04, R10 and R01. A platform change is insufficient novelty, and counting deferrals is already established.

### 6–7. Architecture and formulation

Keep an immutable committed initial answer. Start a speculative revision only if permitted by a predeclared deadline-slack rule. Commit a fully parsed revision atomically; on cancellation/failure retain initial state. The controller uses runtime telemetry and past latency, not correctness labels. Invalid initial responses remain invalid; retaining them is not successful recovery.

For deadline d, define `Q_d = N^-1 sum_i 1{a valid correct answer is delivered by d}`. Secondary harm is a committed correct→wrong transition; unavailable, deferred and invalid outputs remain in the denominator. Model latency L and cancellation/drain time K separately: stopping the client does not prove the server released the GPU. Resource guards are operational constraints, not mathematical OOM guarantees.

New architecture: journaled commit state, monotonic timestamps, cancellation acknowledgment/drain observation, admission controller and telemetry. Reuse existing collector/parser/protocol structures, without modifying frozen implementations. No runtime kernel work or second concurrent model is planned.

### 8–11. Procedure, baselines and evidence

Develop on 20 random unreserved questions. Freeze timeout/admission settings from operational timings, not correctness gains. Compare five policies: never; full verification; timeout-and-retain; fixed-slack admission with retain; resource-aware admission with retain. All receive identical safe storage, parser and deadlines. An ATEST-style cost controller can be adapted as a labelled comparator if its energy-specific objective cannot be reproduced; do not call an adaptation the original implementation.

Use randomized within-session execution order and multiple sessions; report cold-start and warm-service costs separately. Predeclare normal and safe pressure conditions without forcing OOM. Question and session blocks account for dependence. Ablate resource signal, predicted duration and cancellation handling. Primary Q_d; secondary valid delivery, harm, cancellation lag, p50/p95 latency, tokens and RAM. Pair confidence intervals by question, with session sensitivity; three sessions do not characterize all laptop workloads.

### 12–15. Budget and stop/go

**Zero-inference kill:** fault-inject timeouts and incomplete JSON into saved journals/mock streams. If ordinary timeout-and-retain provides the same observable guarantees, the proposed mechanism has no distinguishing value: stop before live work.

**Operational kill after separate approval:** 40–80 sequential calls over predeclared deadlines; measure whether cancellation actually stops resource use. Stop for unsafe RAM, orphaned backend work, no operational difference, or an effect attributable solely to an intentionally broken comparator. Do not claim accuracy from this sample.

If a distinct mechanism survives, budget 300–600 calls and 3–6 weeks, 5–30 active hours. Reserve fresh questions before collection; use known benchmark keys offline. Require a predeclared practically useful Q_d advantage, a paired interval consistent with improvement against timeout-and-retain, and no meaningful harm increase. Select a numeric margin using development service needs **before** evaluation; no generic 1-pp guarantee is feasible with this size. A second borrowed machine would strengthen external validity but is optional and separately disclosed.

Failure cases: no interruptions in realistic workloads; fixed cap is sufficient; correct answers already arrive before the deadline; cancellation is backend-inaccessible; state retention is the entire contribution; effects disappear after cache/order control. This backup has a higher risk of being merely good engineering than D1.

## 3. Weak fallback: Sentinel Adequacy for Frozen-Controller Protocol Transfer

### 1–5. Definition and closest work

**Working title:** *How Many Sentinel Questions Are Enough to Detect Harmful Protocol Changes in LLM Inference Controllers?*

Question: what can a small paired sentinel certify about the loss in utility after a fixed protocol change? Hypothesis: labelled paired sentinels detect some harmful changes missed by observable completion summaries, but cannot certify rare-harm invariance at tiny budgets. Conditional contribution: an independently validated detectability/failure boundary, not a new monitor. R14, R15 and R16 are strong direct competitors; R17/R18 undermine any claim that generic prompt sensitivity is novel.

### 6–7. Architecture and target

Freeze policy π and completion baseline π0. For protocol e define `U_e = E[correct(π,e) - correct(π0,e)]`; monitor `L = U_old - U_new`. Harm can be a separate endpoint rather than an arbitrary tuned weight. Store initial state and both policies' actions/outcomes for the same question under each protocol. One protocol changes at a time, with model digest/runtime fixed unless runtime change is the declared intervention.

A sentinel allocator supplies randomly selected questions with trusted offline outcomes to a **published** risk-monitoring method. The study measures its limits; no gate adaptation, threshold tuning or self-certified safety wrapper. Unlabelled feature alarms may indicate changed inputs, but cannot generally certify unchanged conditional correctness.

### 8–11. Experimental design

Baselines: no monitor; fixed uniform paired sentinel; completion-feature distribution alarm; supervised R14-style monitor; prediction-powered R15-style monitor. Give equal paid labels and all legally available metadata. A SHIFT-JEV-inspired comparator must be scoped to monitoring only; do not attribute full-system claims to a partial reimplementation.

Declare an output-cap change and a benign serialization/control change on development; do not hunt for whichever produces the largest evaluation loss. Confirm that input formatting controls preserve intended questions. Freeze development choices and sample independent evaluation groups. The current frozen gate is calibrated to the SEVRA protocol; a local protocol mismatch must be reported as an intentional transport stress, never silently bypassed in the working application. Prefer a separately specified heuristic controller for a compatible local setting if the production artifact refuses the protocol.

Primary: paid outcomes until detecting a **predeclared** loss at controlled false alarms. Secondary: missed changes, benign alarm rate, delay and inconclusive fraction. Use a published anytime-valid method with its assumptions; no claim of arbitrary-shift label-free safety. Vary sentinel count only in predeclared replay curves, not by picking a favorable final count. Holm/Bonferroni or a single primary shift avoids unreported multiple testing.

### 12–15. Budget, milestones and kill test

Week 1: closest-work reduction check and compatible protocol specification. A 30-question, two-protocol, initial+recheck pilot costs 120 calls, about 2–6 hours. **Kill** if the proposed question is already answered by R15/R16, if protocol variation cannot be isolated, or if there is no operational change distinguishable from ordinary randomness. A small pilot that sees no harm does not certify safety.

Only if justified, reserve 150 independent questions for a two-protocol comparison: 600 additional calls, about 10–30 hours, within a 3–6 week project. Larger multi-shift/multi-model claims exceed this minimal budget. Two independent annotators review any alleged parser/semantic discrepancies; keys stay outside prompts. No MATH without trustworthy keys.

The approximate paired-difference standard error scales as `sqrt((q - delta^2)/N)`, where q is discordance probability for a binary paired comparison. As an illustration, delta=0.05 and q=0.20 require roughly 620–630 independent questions for 80% power at a conventional two-sided 5% level, before multiplicity. A 150-question design will often be inconclusive. For zero independent harmful events, the approximate 95% upper bound is 3/N, not zero. These are planning calculations, not predictions of this project's effect.

This candidate is ranked third because the diagnostic question is falsifiable and the infrastructure is reusable, not because originality or statistical power is established. It should not displace a stronger externally supervised project if the leading direction fails.
