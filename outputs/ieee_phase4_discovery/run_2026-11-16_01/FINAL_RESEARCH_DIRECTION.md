# Final direction: evaluate inference policies more efficiently, rather than train another gate

**Decision: CONDITIONAL GO for a four-day, CPU-first feasibility investigation of D1. NO unconditional GO for a new paper or live collection.** No candidate's originality is established merely by this search. If a confirmed novel method is required before spending even four days, the honest answer is that none has yet met that threshold.

## My choice

**If this were my own research project and my goal were acceptance into a legitimate IEEE conference using an RTX 3050 laptop, I would investigate cost-aware, paired evaluation of fixed LLM inference policies.** I would give it a strict four-day kill gate, then abandon it if strong existing methods already solve the problem.

The research question becomes: **How much expensive generation is necessary to compare two policies' accuracy and harmful-revision rates reliably?** It is not another attempt to make the existing classifier beat completion. RecheckLab already has the useful ingredients: paired outcomes, recorded costs, repeated rollouts, group identities, frozen policies and a history of recognizing false-positive research directions.

Its appeal is scientific leverage per unit of compute. The first assumption can be tested without running an LLM. A useful result would generalize to evaluating policies other than RecheckLab and would report when adaptive evaluation fails, not just a favorable savings number. None of this establishes novelty by itself.

## Why this leads the ranking

| Rank | Candidate | Weighted score / 10 | Recommendation |
|---:|---|---:|---|
| 1 | D1 paired policy evaluation | 7.35 | Bounded conditional investigation |
| 2 | D2 interruption-safe answer delivery | 6.25 | Backup; must beat ordinary timeout-and-retain |
| 3 | D3 protocol-transfer sentinels | 5.95 | Weak backup; direct monitoring overlap and power risk |
| 4 | D4 oracle attainability | 5.40 | Useful clarification, not principal paper |
| 5 | D8 cache/order benchmarking | 5.10 | Required experimental control, weak standalone novelty |
| 6 | D6 executable counterexamples | 4.85 | Prior-art and coverage problems |
| 7 | D5 finalization contracts | 4.55 | Strong direct prior-art overlap |
| 8 | D7 verifier anchoring | 4.15 | Already substantially studied |

Scores are subjective triage, not acceptance probabilities. Originality and technical contribution carry 45% combined weight; reuse/hardware convenience only 10%. The other seven lose on direct prior-art overlap, a triviality risk, insufficient observable evidence, or needing substantial fresh inference before their key assumption can be tested.

## What is potentially different—and what is already known

The candidate focuses on **paired accuracy and harm differences**, known-zero agreement regions, clustered stochastic outcomes, and unequal generation costs, with valid uncertainty and an explicit inconclusive decision.

Active evaluation is established by [Kossen et al.](https://proceedings.mlr.press/v139/kossen21a.html); sequential LLM evaluation by [Hsu and Shekhar](https://arxiv.org/abs/2607.17409); rollout allocation by [Shen et al.](https://arxiv.org/abs/2609.28560); and safe sequential off-policy inference by [Karampatziakis et al.](https://proceedings.mlr.press/v139/karampatziakis21a.html). A new name or a formula obtained by substituting policy differences into one of these methods is not enough. The exact remaining gap is **plausible, not confirmed**.

A publishable contribution would need either a nontrivial method/validity result in this setting or an independently replicated, practically important boundary showing when common evaluation designs waste compute or misstate uncertainty. It must exceed standard stratified sampling, with all pilot and initial-response costs charged. A negative result can contribute only if it teaches something not already established.

## First experiment and kill criteria

After explicit approval, first write the mathematical reduction and compare assumptions with the four closest methods. If the proposed contribution is completely subsumed, **stop without implementation**.

If it survives, run a small CPU-only replay using predeclared policy pairs and whole question clusters. Compare uniform full-pool, uniform disagreement-only, fixed cost-adjusted stratification and a suitable existing sequential estimator. Count actual recorded tokens revealed by the evaluator and all common initial costs. Previously exposed records support a diagnostic of the evaluator, not fresh evidence that a gate generalizes.

**Go criterion for further investment:** validly justified uncertainty, no detectable coverage defect beyond simulation error, and a proposed practical reduction of at least 20% in paid recheck tokens versus the strongest simple comparator at the same joint precision on at least two predeclared replay libraries. No unexplained >10% regression on another library. These are prospective resource-allocation criteria, not predicted results. The exact simulation and multiplicity plan must be frozen before running it.

**Kill criteria:** a direct existing-method reduction with no new result; benefits only against wasteful full-pool sampling; pilot overhead erases savings; harm precision forces near-enumeration; failure of coverage; or apparent success dependent on cherry-picked pairs, future-token information or treating two saved rollouts as a known response distribution.

The existing ML/completion pair has essentially no disagreement. It is a sanity check, not the showcase for this method. The policy bank must include meaningful predeclared comparisons such as always versus completion, and report all pairs rather than select a winner afterwards.

## A realistic project boundary

Week 1: literature/derivation kill test and bounded CPU replay. Weeks 2–3 only after success: freeze a reproducible implementation and validate uncertainty. Weeks 3–5 only after separate approval and an operational preflight: a possible 300-question library with one initial plus two revisions each, 900 calls total. At the rough 60–180 seconds/call planning envelope this is 15–45 active hours, not a promise. Week 6: analysis and writing, with stopping allowed for inconclusive results.

Unused official training questions are locally available, but must be checked against all journals and near duplicates, then separated into development and genuinely new evaluation groups before generation. Existing test questions remain exposed. No benchmark download or larger model is necessary for the proposed first step. Tiny local completion pilots do not establish reliability for the later stage.

This plan can support a study of **evaluation efficiency**; 240 reserved evaluation questions do not certify a one-point answer-accuracy improvement or stringent rare-harm safety. Fresh exhaustive collection would validate simulated acquisition savings; it would not itself demonstrate live time saved by an adaptive acquisition run. That distinction must remain visible in the paper.

## Submission strategy

If a meaningful evaluation method and independent evidence emerge, consider a future IEEE BigComp research track or a suitably scoped IEEE Big Data research/workshop call; ICDM requires a stronger general methodological contribution. A real systems mechanism in D2 would fit ACM/IEEE SEC better. Official scope and track distinctions are documented in `RESEARCH_LANDSCAPE.md` and `REFERENCES.md`. No 2026 submission is recommended: the inspected deadlines are past. A demo, student forum or participation certificate is not a substitute for the requested research paper.

## If the leading idea fails

Do not automatically start another renamed gating project. Discuss a broader pivot with the professor toward an independently motivated evaluation or dependable-systems problem with a verified gap, reusing RecheckLab's infrastructure only where useful. The current record does not justify committing weeks to any of the eight candidates without its kill test. Stopping an unoriginal direction is better than manufacturing a contribution.

## Deliverables and preservation

The six requested files are complete: `RESEARCH_LANDSCAPE.md`, `EIGHT_CANDIDATES.md`, `CANDIDATE_RANKING.csv`, `TOP_THREE_BLUEPRINTS.md`, this report and `REFERENCES.md`. No candidate was implemented, no training or Ollama inference ran, no package/model/dataset was installed or downloaded, and no application or frozen artifact was changed. Final preservation verification reproduced the 357-file baseline digest recorded in the landscape report; all originals remain unchanged. Await explicit approval before implementation or any proposed kill experiment.
