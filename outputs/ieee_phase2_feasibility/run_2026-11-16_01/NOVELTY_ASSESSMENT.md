# Phase 2: novelty assessment

16 November 2026. Read alongside the unchanged project-root `IEEE_RESEARCH_AUDIT.md`. Targeted literature investigation, not an exhaustive systematic review. Publication status is limited to verified primary sources; absence of an exact match in this search does not establish novelty.

## Verdict on the proposed method

**Conservative residual gating is not a new general decision principle.** Baseline fallback under uncertainty, selective prediction, and cost-sensitive post-hoc routing are established. The precise combination of a completion default and a margin-controlled learned override for this particular replay was not verified as an already published identical algorithm. Nevertheless, the difference currently looks like an incremental parameterization, not a compelling new contribution.

Let `h(x)` be completion's recheck decision, `d = revised_correct − initial_correct`, and `c` be recheck tokens. The utility gain of reversing completion is

`R(x) = (1 − 2h(x)) [E(d | x) − lambda E(c | x)/1000]`.

If `h=0`, override when estimated recheck utility exceeds margin `m`. If `h=1`, override when estimated recheck utility is below `−m`. With `m=0`, this is ordinary expected-utility action selection. A positive margin creates an indifference region where the default is retained. **A margin is not automatically a confidence bound or safe-policy-improvement guarantee.** A method claiming safety needs assumptions, uncertainty control, a baseline-relative estimand, and independent evidence.

## Closest comparisons

| Source | Verified overlap | What remains different or unestablished |
|---|---|---|
| Dip, Zhou, Zhang, [Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning](https://arxiv.org/abs/2606.19808), 18 June 2026 preprint; [full text](https://arxiv.org/html/2606.19808v1), Sections 3–4, Appendices H/I/Y | SEVRA supplies the recorded data and already studies recoverability-aware gating, completion-state signals, correction/harm, cost, and longer-initial comparisons. | No explicit completion-default residual-margin algorithm was established in the inspected sections. Replacing its gate with a ridge override is insufficient differentiation. Baseline-relative uncertainty or an independently supported mechanism would need to add substantive evidence. |
| Zhang, [When Should LLMs Trust Their Own Revisions? A Risk-Aware Study of Intrinsic Self-Correction](https://arxiv.org/html/2609.35832v1), 23 September 2026 preprint, Section 5 and Appendix C | Initial-only help/harm prediction, thresholded selective revision, and comparison to unconditional policies already cover the risk-aware setting. | This paper explicitly limits its gate evidence: summary provenance is not fully linked to configurations; feature panels are not controlled ablations. Its fallback is not verified as the same per-example completion-default override. These limitations motivate better evidence, not claims that risk-aware gating is new. |
| Laroche, Trichelair, Tachet Des Combes, [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html), ICML 2019 | Retaining a baseline where evidence is weak is central to safe policy improvement. | The original setting is batch reinforcement learning with explicit assumptions, not these fully paired one-step revision records. RecheckLab cannot inherit its guarantees merely by borrowing the fallback idea. |
| Narasimhan et al., [Post-hoc estimators for learning to defer to an expert](https://papers.nips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html), NeurIPS 2022 | Post-hoc accuracy–cost decision learning and adaptive inference are established. | Revising with the same solver differs from sending to a separate expert, but the decision-theoretic structure is closely related. |
| Geifman and El-Yaniv, [Selective Classification for Deep Neural Networks](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), NeurIPS 2017 | Risk–coverage control and abstention have an extensive foundation. | Skipping revision retains an answer rather than abstaining from answering. Classification-risk guarantees do not directly bound harmful revisions or incremental utility under shift. |
| Ong et al., [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), 26 June 2024 preprint, revised February 2025 | Lightweight learned allocation under quality/cost trade-offs is well established. | Routing revision actions differs from routing among solvers; the distinction alone does not establish originality. |

### Direct answers

1. **Does essentially the same principle exist?** Yes: benefit-relative selection with fallback is established. An exactly identical implementation in this exact LLM setting was not verified.
2. **Has heuristic fallback plus learned override been evaluated for LLM verification?** The closest papers establish gated verification and fallback concepts, but this investigation did not verify the precise completion-default, confidence-certified override experiment. Do not turn that uncertainty into a “first” claim.
3. **Has risk-sensitive revision been studied in this setting?** Yes. HELP/HURT-aware intrinsic revision is directly covered by the two 2026 papers.
4. **Is this meaningfully different from SEVRA now?** Not yet. A new classifier or threshold formulation without held-out utility is a minor extension of its controller/data setting.
5. **What question remains worth asking?** How much apparent residual benefit survives scoring-format controls, and can an explicitly baseline-relative policy predict that remaining benefit under a completion-reliable protocol? This is a proposed narrower estimand, not an asserted gap in all literature.
6. **What would make it substantive?** Independent, protocol-matched evidence of incremental utility over strong completion/parse baselines; matched-cost comparisons; adequate event support; and evidence separating normalization recovery from semantic correction. A defensible uncertainty mechanism could be an additional contribution, but the current sample cannot support its safety calibration.

## Implication for the research title

Do not presently market “Conservative Residual Gating” as a new superior algorithm. It is a hypothesis tested here by one simple estimator. This feasibility run finds no useful policy improvement. A scientifically stronger next direction would study **measurement validity of residual correction value**, with protocol and grading assumptions made explicit. That would require an independently designed study, not relabeling this replay to produce a favorable score.

Local evidence: `rechecklab/predictor.py:297–340` already implements a harm-aware utility score and threshold selection; `features.py:88` defines the initial-only features. This run's `../feasibility.py:90–116` adds baseline-relative prediction, a training-derived cost estimate, and an empirically selected margin. These are modest changes, deliberately without a production artifact or deployment claim.
