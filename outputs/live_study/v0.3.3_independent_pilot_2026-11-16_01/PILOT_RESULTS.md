# Independent live pilot results

**Run:** `v0.3.3_independent_pilot_2026-11-16_01`  
**Status:** complete: 8/8 question pairs; 16/16 calls succeeded.

This pilot contains newly generated local Ollama responses for GSM8K questions absent from the prior planned/evaluated cohorts. It is separate from the historical SEVRA recorded-response replay. Results are descriptive only; eight questions cannot establish significance or generalization.

## Answer completion and transitions

- Parseable explicit final answers: initial 2/8 (25.0%; exact 95% CI 3.2%-65.1%); revision 6/8 (75.0%; exact CI 34.9%-96.8%). Eight of 16 calls failed extraction; completed unparseable answers are scored incorrect.
- `done_reason=length`: initial 5/8; revision 2/8.
- Initial call token accounting: 806 prompt + 6785 generated = 7591 total tokens. Revision calls: 8029 prompt + 5172 generated = 13201 total tokens.
- Correctness transitions: HELP 4/8 (50.0%; exact 95% CI 15.7%-84.3%); HURT 0/8 (0.0%; exact CI 0.0%-36.9%); NO_CHANGE 4/8 (50.0%; exact CI 15.7%-84.3%).

## Policies on the same recorded pairs

| Policy | Accuracy (correct/8; exact 95% CI) | Rechecks | Mean prompt+output tokens/question | Replayed latency/question | Accuracy difference vs never (paired bootstrap 95% CI) |
|---|---:|---:|---:|---:|---:|
| Never recheck | 2/8 (25.0%; 3.2%-65.1%) | 0/8 (0.0%) | 949 | 39.6 s | +0.0 pp (+0.0, +0.0) pp |
| Always recheck | 6/8 (75.0%; 34.9%-96.8%) | 8/8 (100.0%) | 2599 | 72.5 s | +50.0 pp (+12.5, +87.5) pp |
| Completion heuristic | 5/8 (62.5%; 24.5%-91.5%) | 5/8 (62.5%) | 2218 | 66.5 s | +37.5 pp (+12.5, +75.0) pp |
| Frozen ML gate | Not evaluated: protocol incompatible | N/A | N/A | N/A | N/A |

Selective-policy latency replays initial-call time plus observed revision time only for selected questions; it is not latency measured in a deployed gated workflow. All questions received one unconditional revision to make paired policy comparisons possible.

Intervals are question-level exact binomial intervals for rates and paired question-resampling bootstrap intervals (20,000 resamples, seed 20261116) for accuracy differences. No significance or generalization claim is made.

## Recommendation

The separate five-question development check with `thinking=false` produced 10/10 parseable answers, but this tiny unpaired sample does not establish reliability or a causal improvement. A 30-50-question independent validation under that newly frozen protocol is a reasonable next study after ensuring adequate RAM headroom; the evaluation pilot reached 0.39 GiB free system RAM. Keep the frozen ML gate out unless a matching compatibility contract is available. Do not launch the larger study automatically.

## Separate development-only completion check

A five-question follow-up tested one predeclared setting change (`thinking=false`) on different questions. It produced 10/10 parseable answers and is documented separately in `../v0.3.3_development_thinking_off_2026-11-16_01/DEVELOPMENT_RESULTS.md`. This is not pooled with the eight-question pilot and does not demonstrate a causal completion improvement.
