# Redacted disagreement case studies

These cases were selected to illustrate distinct recorded mechanisms: an ML-only neutral recheck, a repeated HELP cluster missed by ML, a changed parsed answer without a correctness change, and rollout-level heterogeneity within a question. They are descriptive examples, not representative samples. Full canonical hashes identify the frozen records. Benchmark questions and answer values are withheld; mathematical/answer-bearing spans in the short model-response excerpts are redacted. No gold answer strings are included.

## Case 1 — Completed, correct initial answer; ML adds a neutral recheck

- **Cohort / canonical question hash:** MATH source test / `9fa5baacf1d04aa7d826cb1db4a6078a9083059bbc75060137bd59622828d366`
- **Rollouts in question:** 1 (rollout 0).
- **Initial status:** `stop`; parsed answer present; no base finalizer; initially correct.
- **Gate:** score 0.377368, above threshold 0.259681; ML says recheck, completion says skip.
- **Recorded outcome:** correct before and after; NO_CHANGE; parsed answer unchanged. The ML decision adds 1,718 replay tokens with no accuracy change.
- **Redacted text excerpts:** Initial: “To solve the problem, we analyze the given line in the diagram.” Revision: “To ensure the correctness of the attempted solution, we perform two candidate-specific checks as required.”
- **Interpretation:** A completed and apparently answerable response triggers the learned gate, while the completion rule skips it. This is an observed unnecessary cost, not evidence of harmful revision.

## Case 2 — MATH500: two HELP outcomes in one length/finalizer question

- **Cohort / canonical question hash:** MATH500 / `6084e17b0af018a13b2a3b25f06cdb8392c3bc145c7c676cf50e1fe275c81359`
- **Rollouts in question:** 2 (rollouts 0 and 1).
- **Initial status:** `length`; `base_finalizer_used=True`; stored parsed answer present; both initial outcomes are incorrect.
- **Gate:** score −0.384217, below threshold 0.259681; ML skips both recorded revisions while completion selects both.
- **Recorded outcome:** both rollouts are HELP: revisions are labeled correct. Parsed-answer-changed flags are true. The gate therefore loses one question-equivalent of final accuracy relative to completion. It avoids 5,312 raw rollout action tokens (2,656 question-equal mean tokens for this pair).
- **Redacted text excerpts:** Initial transcript begins, “Okay, let's try to solve the problem step by step.” One revision begins, “To determine the length of [math redacted] in the given right triangle [math redacted]…” The second revision response is 25 characters; its content is withheld.
- **Interpretation:** The logs establish that the stored initial outcome was wrong and both selected revisions were graded correct. They do not establish whether this was a semantic reasoning repair, answer finalization, or another measurement effect.

## Case 3 — MATH500: an answer changes, but both outcomes remain wrong

- **Cohort / canonical question hash:** MATH500 / `0a9f4b84d504e6a6e0500864e7802f402210c0a3f31c6cdbf3b8fa6eddac9e45`
- **Rollouts in question:** 2 (rollouts 0 and 1).
- **Initial status:** `length`; finalizer used; parsed answer present; both initial outcomes are incorrect.
- **Gate:** score −0.133219, below threshold; ML skips both revisions, completion selects both.
- **Recorded outcome:** both are NO_CHANGE (wrong-to-wrong). The parsed-answer-changed flag is false on one rollout and true on the other. Skipping these two actions avoids 16,404 raw rollout tokens (8,202 question-equal mean tokens) with no accuracy difference.
- **Redacted text excerpts:** Initial: “Okay, let's try to solve this step by step.” Revision: “Okay, let's try to figure out this problem step by step.”
- **Interpretation:** At least one parsed answer differs without a correctness transition. The record cannot say whether the answer changed semantically in a way unrelated to correctness, or whether formatting/extraction contributed.

## Case 4 — GSM8K: one question, mixed outcomes across two revisions

- **Cohort / canonical question hash:** GSM8K / `941ef392e2bd4cf2b6d44d23dfe2e1eaf6f3f0f24743c305f42cd715733e9cfe`
- **Rollouts in question:** 2 (rollouts 0 and 1).
- **Initial status:** `length`; finalizer used; parsed answer present; both initial outcomes are incorrect.
- **Gate:** score 0.240548, below threshold; ML skips both revisions, completion selects both.
- **Recorded outcome:** one rollout is HELP and the other NO_CHANGE. Both parsed-answer-changed flags are true. The completion choice improves the question-level mean accuracy by 0.5; skipping avoids 11,710 raw rollout action tokens (5,855 question-equal mean tokens).
- **Redacted text excerpts:** Initial: “Okay, let's try to solve this problem step by step.” Revision: “The problem says [math redacted] to his share and another [math redacted] and later tripled the amount left after that.”
- **Interpretation:** The two saved revisions for the same initial question do not have the same outcome. Treating these rollout rows as independent questions would overstate the evidence. The outcome labels do not resolve whether the difference arose from reasoning, answer finalization, or grading/extraction.

## Case 5 — GSM8K: terse revision changes the parsed answer but not correctness

- **Cohort / canonical question hash:** GSM8K / `2d9662b2f7bdeb70395a014c73dd7f27faf7c8a0ac02d36c0bd66f2498f332ef`
- **Rollouts in question:** 2.
- **Initial status:** `length`; finalizer used; parsed answer present; initially correct in both rollouts.
- **Gate:** score 0.188125, below threshold; ML skips while completion selects both.
- **Recorded outcome:** both transitions are NO_CHANGE, with correct initial and revised answers; both `changed_answer` flags are true. Each recorded revision response is only 15 characters. Skipping avoids 8,737 raw rollout tokens (4,368.5 question-equal mean tokens) without an accuracy change.
- **Text evidence:** The initial response is a long reasoning transcript; the revision field is a 15-character answer-only response. The answer content is withheld.
- **Interpretation:** This is compatible with an answer-finalization or formatting effect, but it does not prove one. Since correctness remains unchanged and gold keys are omitted, the record cannot adjudicate semantic equivalence or reasoning quality.

## Case-study boundary

All quoted passages are short excerpts from model-generated response fields and exclude answer-bearing strings. `base_answer`, `final_answer`, gold answers, and benchmark keys are not reproduced. These examples demonstrate why rollout-level HELP labels and parsed-answer changes should not be equated with a verified semantic correction.
