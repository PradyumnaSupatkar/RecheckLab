# Completion and frozen-gate failure analysis

## Answer extraction

Both candidates completed all six calls. Candidate A had 1/3 valid initial answers and 2/3 valid revisions; Candidate B had the same rates. Each had two initial calls ending at `length` and one ending at `stop`. Candidate B consumed the full 2,048 output-token allocation in two initial calls; in question 1431 its entire generated content was in the thinking field and its answer content was empty. Candidate A also exhausted 1,024 tokens on questions 1431 and 519. A `stop` reason alone did not ensure extraction: extraction is evaluated separately from completion reason.

The small sample does not establish that thinking mode or budget caused the differences. Candidate A's earlier disjoint five-question development check had 5/5 valid initial answers, while this fresh paired sample had 1/3. Question-specific answer length and reasoning are plausible contributors; the evidence cannot separate them.

## Frozen v0.2 gate (OOD only)

The canonical feature extractor accepted all six initial records, producing the frozen 19-feature schema. Gate actions:

- Candidate A: 0/3 rechecks; completion heuristic: 2/3.
- Candidate B: 1/3 rechecks; completion heuristic: 2/3.
- A and B gate actions differed on one question (`gsm8k:train:519`); completion-heuristic actions did not differ.
- For A, the gate missed both length-limited initial answers. For B, it missed one of two; it selected question 519 but not 1431.

The v0.2 artifact expects the historical SEVRA active-verify protocol. The live prompts, exact checkpoint lineage, thinking settings, output limits, and finalizer behavior do not establish protocol equivalence. These are OOD transfer scores, not deployment-validated predictions.

## Feature distribution

Relative to v0.2 source-training features, the mean absolute univariate standardized mean difference across the 19 features was 2.94 for A and 3.27 for B; five features had at least one observed value outside the source-training min/max. Largest shifts included `initial_answer_present` (23.07 SD for both), `initial_question_marks` (12.50 A, 15.68 B), and uncertainty/backtracking rates (about 3.7–4.8 SD). With only three questions per candidate, these figures are unstable and descriptive. They do not identify thinking mode, token budget, or any single protocol detail as the cause.

## Resource and study decision

Minimum post-call free RAM was 2.95 GiB; GPU peak reported use was about 2.6 GiB of 4 GiB. No call failed and the 2 GiB stop floor was not crossed. However, low extraction reliability blocks the larger study even though the resource run completed.
