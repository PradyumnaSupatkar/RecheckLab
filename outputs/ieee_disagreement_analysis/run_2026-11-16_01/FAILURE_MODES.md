# Failure modes and completion/reasoning evidence

This report separates mechanically recorded fields from interpretations. The classifier and threshold are frozen. “HELP,” “HURT,” and “NO_CHANGE” are inherited correctness transitions; they are not human judgments about reasoning quality.

## Mechanical breakdown of every disagreement

| Cohort | Disagreement rows / question groups | Initial state | Stored parsed answer present | Initial finalizer used | Initial incorrect | HELP / NO_CHANGE / HURT rows | Parsed-answer-changed flag true |
|---|---:|---|---:|---:|---:|---:|---:|
| MATH source test | 2 / 2 | stop: 2 | 2/2 | 0/2 | 0/2 | 0 / 2 / 0 | 0/2 |
| CommonsenseQA target test | 0 / 0 | No disagreements | — | — | — | 0 / 0 / 0 | — |
| MATH500 | 18 / 9 | length: 18 | 18/18 | 18/18 | 18/18 | 4 / 14 / 0 | 15/18 |
| GSM8K | 10 / 5 | length: 10 | 10/10 | 10/10 | 8/10 | 3 / 7 / 0 | 10/10 |

### Missing, truncated, and completed answers

- There are **zero empty parsed `base_answer` fields** among discordant records. This is after any recorded base finalizer; it does not establish that the unfinalized reasoning generation had a complete answer.
- The 28 MATH500/GSM8K discordances all have `base_done_reason=length` and `base_finalizer_used=True`. They are length-limited solver attempts followed by a stored finalizer-produced answer. The two MATH source-test discordances are `stop`, parsed-answer-present, non-finalized, correct initial responses.
- There are **zero stop-completed but initially incorrect** discordant rows. All 18 MATH500 initial outcomes are wrong; eight of ten GSM8K outcomes are wrong and two are correct.
- Consequently the ML gate’s observed misses are concentrated in length/finalizer cases. Calling them “missing answer extraction” would be inaccurate at the stored-record level. Whether the finalizer rescued an incomplete answer, repaired formatting, or supplied a reasoning correction cannot be established from the completion flag alone.

### Transition and decision effects

- **MATH source test:** ML adds two rechecks to stop-completed correct answers. Both are NO_CHANGE, both parsed answers are unchanged, and accuracy does not change. These are costly false-positive gate decisions relative to completion, not harmful revisions.
- **MATH500:** ML skips all 18 length-selected records. Four rollout transitions are HELP (across three question groups), and 14 are NO_CHANGE (across eight groups). Thus the policy misses four recorded helpful fixes across three questions; it avoids no observed HURT event in these disagreements. The 14 NO_CHANGE rows are all wrong-to-wrong: the initial was incorrect and the revision remained incorrect.
- **GSM8K:** ML skips all ten length-selected records. Three HELP rows occur across two question groups; seven NO_CHANGE rows occur across four groups. There is no HURT. Within the NO_CHANGE rows, five are wrong-to-wrong and two are correct-to-correct.
- **CSQA:** the frozen gate and completion rule agree on all rows; there is no discordance to attribute.

At question level, five of 16 discordant groups have a negative ML-minus-completion accuracy difference (three MATH500, two GSM8K), and 11 have zero difference. No discordant question favors ML on accuracy. Repeated rows are not counted as independent questions.

### Parsed-answer change and possible finalization effects

The recorded `changed_answer` flag is true on 25 of 30 disagreement rows: 15 MATH500 and all ten GSM8K rows. Those 25 include seven HELP and 18 NO_CHANGE transitions; none is HURT. Thus answer change and correctness improvement are not interchangeable. In particular, the answer changed while correctness remained unchanged on 18 rows. This field records an answer-level change, not a verified semantic reasoning change.

Seven discordant revision strings are at most 25 characters long: one MATH500 record and six GSM8K records. These include three HELP and four NO_CHANGE transitions. Their terse form is mechanically observable and consistent with an answer-finalization style response, but short outputs do not prove that a correction was formatting-only. The gold keys and full grading trace needed for that conclusion are not available in this release.

### Cost failure modes

| Cohort | Gate action difference | Outcome mix | Signed action-token difference, raw rollout sum | Question-equal total-token difference vs completion |
|---|---|---|---:|---:|
| MATH source test | Adds 2 rechecks | 2 neutral | +4,162 | +10.405 tokens/question |
| MATH500 | Skips 18 completion rechecks | 4 HELP, 14 neutral | −129,228 | −129.228 tokens/question |
| GSM8K | Skips 10 completion rechecks | 3 HELP, 7 neutral | −48,175 | −18.262 tokens/question |

On MATH500, NO_CHANGE cases account for 111,415 of the 129,228 raw avoided revision tokens (86.2%); HELP cases account for 17,813. On GSM8K, NO_CHANGE cases account for 37,727 of 48,175 raw avoided revision tokens (78.3%); HELP cases account for 10,448. This is why the gate saves recorded tokens while losing some observed accuracy: most skipped cost lies in neutral revisions, but some skipped revisions are helpful. The token differences are replay accounting and not latency or energy measurements.

## What is established, and what remains speculative

**Mechanically established:** decision directions and scores; the frozen threshold; length/stop status; finalizer flags; stored parsed-answer presence; transition labels; parsed-answer-change flags; action-token counts; and paired replay accuracy/cost. See `tables/DISAGREEMENTS.csv` and `tables/FAILURE_MODE_CROSSTAB.csv`.

**Not established:** whether the HELP cases represent a better mathematical or commonsense reasoning process, whether a finalizer or format change alone explains an outcome, whether a concise revision is semantically justified, or whether the same pattern occurs under a new prompt/checkpoint. These require answer-key-aware blinded transcript adjudication and independent prospective data, neither of which is part of this analysis.
