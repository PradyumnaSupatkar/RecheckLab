# Phase 3 verdict: NO-GO for the broad measurement-artifact claim as the principal paper

**Date:** 16 November 2026. **Scope:** offline feasibility and descriptive measurement audit, not a new model or live experiment.

## Direct answer

**Does separating genuine reasoning correction from formatting, parsing and completion recovery provide a sufficiently original and scientifically meaningful direction for an IEEE conference paper?**

**Meaningful, yes; sufficiently original in the broad form proposed, no.** The August 2026 *Calibration Floor* preprint directly studies format repair masquerading as self-correction, including stronger interventions than our current replay permits. SEVRA already investigates completion-dependent recovery and longer-initial compute alternatives. See [NOVELTY_REVIEW.md](NOVELTY_REVIEW.md) for verified sources and limits of the comparison.

This is a research-investment decision, not a claim that measurement validity is solved or that a carefully differentiated replication could never be publishable.

## What the evidence supports

- Frozen HELP/HURT arithmetic, canonical-question weighting and token totals are internally consistent in the audited replay rows. This does not certify the publisher's mathematical grader.
- Training has 162 HELP and 15 HURT; 154 HELP follow a length-limited initial call. Development has 47 HELP, all after length-limited initial calls. Source test has 65 HELP, 62 length-associated. These are associations, not percentages of proven completion-only fixes.
- Six completed/nonempty training HELP examples explicitly assert mathematically plausible final results, with visible representation differences from the subsequently accepted strings. A seventh stopped example has an empty extracted answer despite a displayed final fraction. These are strong suspects requiring grading-provenance confirmation, not seven automatically corrected labels.
- Completion-controlled descriptive comparisons materially change the apparent benefit. Training stop/nonempty accuracy goes from 98.5839% to 97.8214% under always rechecking; development goes from 99.6721% to 98.3607%. Source-test stop/nonempty moves from 97.9522% to 98.2935%, a small exposed descriptive difference.
- Public finalizers are a major confound: 124/154 training length-associated HELP records already used an initial finalizer. A length flag is not equivalent to an absent final answer.
- No cases received a verified causal reasoning-repair label in this audit. This reflects missing adjudication, not proof that genuine corrections never occur. No historical grades changed.

## What the evidence cannot support

We cannot estimate the true fraction of semantic corrections, prove that a cheap finalizer would reproduce the observed gains, or infer a model's internal reasoning repair from final-answer agreement. Missing published MATH keys and unavailable pinned upstream grader behavior prevent an end-to-end correctness reconstruction. Seven selected cases do not establish general artifact prevalence. Existing test partitions are exposed; they cannot provide new independent confirmation.

The residual-gating NO-GO remains supported by its saved Phase 2 comparison: training-held-out completion and residual accuracy both 92.2435%, with nine extra neutral calls; development both 92.00% with no residual overrides. Those are archived results, not retrained in Phase 3. This audit independently recomputes the completion baselines, but does not rerun Phase 2 training.

## At most two alternatives

1. **A reproducible policy-ranking robustness audit.** Obtain the existing benchmark's authoritative MATH keys and pinned grader provenance through a separately approved step; do not substitute model revisions for gold. Two independent human annotators should assess a fixed, stratified set of initial/revised transcripts, including HURT and NO_CHANGE controls. Measure whether policy rankings survive admissible grading interpretations and unresolved-case bounds. First checkpoint: adequate agreement, reproducible evidence and a literature gap beyond the existing format/content decomposition. This is predominantly CPU and annotation work; it needs zero new LLM calls initially. It may be suitable as a replication/resource contribution if it exposes a general, actionable reproducibility failure, but one parser discrepancy is insufficient.

2. **Non-destructive finalization as a serving reliability question.** Investigate whether an answer-finalization operation preserves an already asserted answer, rather than whether it increases benchmark accuracy. A future study could contrast mechanical extraction, bounded finalization and full verification on the same saved initial responses, with equal token caps and measured realized costs. Only proceed after a focused novelty check confirms that answer-preservation/harm under this protocol remains unanswered. Do not present this as a novel method already established. On this laptop, first use existing records and human annotations; any later live pilot needs separate approval and memory safeguards.

Neither alternative warrants weeks of implementation now. A failed novelty check or inadequate adjudication is a stopping condition, not a reason to add model complexity.

## What would change the verdict

A documented gap relative to the direct prior art, independent annotation with trusted keys, a reproducible policy-relevant effect that survives ambiguity bounds, and a prospectively fixed evaluation would justify reconsideration. With a rare effect near 1 percentage point, a 50-question pilot cannot establish generality. Even for a 5-point paired accuracy difference and 20% discordant outcomes, a rough 80%-power normal approximation requires about 630 independent questions, before multiplicity or clustering inflation. A small pilot can establish operations or mechanisms, not such a final accuracy claim.

**No new inference plan is executed or prescribed as the immediate next action under this NO-GO.** Present the project as an auditable selective-compute prototype with a candid negative ML-versus-heuristic result and a useful measurement limitation. It is not yet ready to submit as a novel superior gating or reasoning-correction method.

## Preservation and reproducibility

Only this new run directory was written. `PRESERVATION_CHECK.json` verifies all 339 pre-existing non-cache project/data/report files against SHA-256 hashes. No source, Streamlit file, frozen protocol, model, old report or dataset was modified. No Ollama calls, model training, package installation or benchmark download occurred.

Primary outputs: [MEASUREMENT_AUDIT.md](MEASUREMENT_AUDIT.md), [CORRECTION_CASES.csv](CORRECTION_CASES.csv), [METRIC_SENSITIVITY.md](METRIC_SENSITIVITY.md). The CSV preserves all original grades and records unresolved interpretations explicitly.
