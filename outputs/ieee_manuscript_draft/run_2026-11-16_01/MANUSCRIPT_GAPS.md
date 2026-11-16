# Manuscript gaps before submission

This first draft is intentionally limited to what the saved analysis supports. The following issues are the main obstacles to a strong submission, ordered by impact.

1. **No fresh, prospectively collected evaluation.** The four cohorts reuse responses from one SEVRA release and the historical test sets have been inspected repeatedly. The MATH500 and GSM8K results are exploratory frozen-artifact replay diagnostics, not untouched replication. **Remedy:** freeze the protocol and analysis plan, then collect a new question-disjoint evaluation under the declared prompts, checkpoint, decoding settings, and gate. Keep the new evaluation sealed until all selection is finished.

2. **Incomplete response and grading provenance.** The local release omits gold answer strings and does not provide the exact upstream prompt bundle or checkpoint digest. The paper therefore inherits correctness transitions and cannot establish whether every apparent correction is semantically valid. **Remedy:** archive the exact prompt templates, model digest, generation configuration, answer extractor, grader/version, and benchmark-key provenance; regrade a blinded, question-level sample against official keys before drawing semantic-correction conclusions.

3. **Incremental learned-gate value is not demonstrated.** The gate ties completion on source test and CSQA, matches all CSQA decisions, and is slightly less accurate on the exploratory MATH500/GSM8K replays. A paper should present this as a negative result and avoid framing learned gating as a demonstrated contribution. **Remedy:** either narrow the contribution to the reproducible cross-domain comparison, or test a predeclared practical claim on new data (for example a justified accuracy non-inferiority margin at lower measured cost). Do not add another classifier search on the exposed cohorts.

4. **Token accounting is not a systems-efficiency measurement.** Replay token counts do not capture call startup, queueing, wall time, throughput, energy, memory pressure, or the classifier's online overhead. **Remedy:** in a future prospective run, record per-call end-to-end latency and token counts with fixed hardware, warm/cold state, concurrency, and failed-call accounting; measure the gate overhead and report paired uncertainty.

5. **Rare harmful events and exploratory multiplicity limit risk claims.** HURT support is small in the primary cohorts, and the gate has zero HURT recall on both historical test partitions. Many unadjusted policy comparisons and prior exposure further limit inferential language. **Remedy:** predeclare one primary contrast and a minimum practically relevant effect or non-inferiority margin; size a fresh evaluation for paired accuracy and harmful-flip endpoints, report cluster intervals, and treat zero observed harms as an upper-bound estimate rather than a guarantee.

## Draft controls

- The manuscript calls all recorded outcomes inherited and does not relabel them.
- The source MATH holdout is not described as official MATH Level 5; MATH500 remains a separate exploratory replay cohort.
- Oracle rows are marked hindsight-only and are not presented as deployable.
- No measured latency or energy benefit is claimed.
- The 2026 arXiv references are identified as version 1 preprints, not peer-reviewed publications.
