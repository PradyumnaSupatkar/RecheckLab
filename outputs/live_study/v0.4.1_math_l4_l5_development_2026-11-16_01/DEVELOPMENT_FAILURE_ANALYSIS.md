# v0.4.1 MATH L4/L5 development failure analysis

## Decision

The 10-question reliability gate failed. The 60-question evaluation was not started. This is a protocol reliability result, not evidence about comparative policy efficacy on MATH.

## Answer extraction and completion

- Initial exact-answer extraction: 4/10 (40%). Revision extraction: 5/10 (50%). The required rate was at least 9/10 for both.
- All 11 parser failures were `missing_final_marker`. Six initial calls and five revision calls stopped at Ollama `done_reason=length`; five additional cases had both calls length-limited. There were no transport/model API call failures.
- The parser was deliberately strict: only a final `FINAL:` line with an exact integer or rational was accepted. The failure mechanism was generation ending at its 512-token output cap before a final marker, not a failed synthetic grader.
- All parseable answers were correct under exact Fraction comparison: initial 4/4; revision 5/5. Among the ten questions, outcomes were HELP=1, HURT=0, NO_CHANGE=9. The single HELP was extraction recovery (initial response had no parseable final answer; revision was parseable and correct), not a confirmed correction between two parseable mathematical answers. The five pairs with neither answer parseable are not interpretable as mathematical NO_CHANGE evidence.

## Calls, cost, latency, and resources

- Completed 20/20 calls sequentially: 10 initial and 10 revision.
- Mean prompt-plus-output tokens: initial 553.2, revision 1,011.5, combined 1,564.7 per question pair.
- Mean wall time per call: initial 20.06 s, revision 18.58 s; pair mean 38.64 s. Individual times and token counts are in `PAIRS.jsonl` and `RAW_CALLS.jsonl`.
- Minimum observed free RAM: 3.508 GiB; minimum GPU free memory: 1,466 MiB. Safety thresholds were met.

## Policies and frozen gate

- Completion heuristic selected 6/10 based on initial `done_reason=length`.
- Frozen v0.2 OOD gate selected 6/10. Its decisions matched the completion heuristic on all 10 development questions (0 disagreements); this small reliability check is not a useful estimate of policy accuracy.
- All 10 initial-only feature vectors were available and scored before their corresponding revisions. The gate artifact, feature extractor, and threshold remained unchanged.

## Selection audit note

The initial pre-inference eligibility pass mistakenly scanned its own downloaded source cache as a prior local cohort and reported zero eligible questions. This was corrected before any LLM call: the current experiment directory was excluded from the prior-cohort scan, while all previously existing project data remained included. The final audit found 1,623 eligible questions after excluding 188 MATH500 overlaps; the pinned source rows, final disjoint 10/60 selection, and answer keys were frozen before inference. This is recorded here to preserve the audit trail.

## Next step

Do not launch the 60-question study under this protocol. A separately declared development protocol would need to address length-limit final-answer loss before a new prospective freeze. Do not infer a beneficial ML-gate effect from this 10-question check.
## Development-only policy replay

Using the same stored pairs and strict exact-answer scoring (unparseable answers count as not correct):

| Policy | Correct | Rechecks | Mean tokens/question | Replayed seconds/question |
|---|---:|---:|---:|---:|
| Never recheck | 4/10 | 0 | 553.2 | 20.06 |
| Always recheck | 5/10 | 10 | 1,564.7 | 38.65 |
| Completion heuristic | 5/10 | 6 | 1,266.3 | 33.97 |
| Frozen v0.2 gate (OOD) | 5/10 | 6 | 1,266.3 | 33.97 |

The completion heuristic and gate made identical decisions on this selected development sample. All policies remain descriptive pilot diagnostics; no uncertainty or significance claim is warranted.
