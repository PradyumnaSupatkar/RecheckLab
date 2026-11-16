# Historical inference-failure review

This review separates response-generation failures from answer-scoring outcomes. Counts below were recalculated from saved raw calls/pair records; frozen studies were not changed.

## Thinking checkpoint: v0.3.3 independent pilot

- Checkpoint digest `359d7dd4...` identifies Qwen3-4B-Thinking-2507, not Instruct. Historical manifest used 1,024 output tokens per call with thinking enabled.
- 7/16 calls ended at the 1,024-token limit while producing a nonempty `thinking` field; the reasoning channel consumed the available generation budget before a parseable final answer. These are reasoning-token exhaustion plus a `length` completion reason.
- 1 additional initial call ended with `stop` but no `FINAL:` marker. That is a completed generation with an extraction/prompt-format failure, not a length truncation.
- No completed response with a `FINAL:` marker had malformed numeric syntax in this sample. All parseable answers were correct: initial 2/8, revisions 6/8; zero valid-but-wrong answers were observed.

## Earlier thinking-disabled check on the same Thinking-only checkpoint

The separate five-question check recorded 5/5 parseable and correct initial answers and 5/5 parseable and correct revisions, with no `length` completions. Its raw log retained call events but not response bodies, so missing-marker versus malformed-answer subtypes cannot be audited from that run. This small disjoint result did not establish checkpoint support for a non-thinking mode.

## Later protocol comparison on the Thinking checkpoint

- Thinking-off at 1,024 tokens: three length-limit truncations without a final answer; parseable calls on the remaining questions were correct.
- Thinking-on at 2,048 tokens: three calls exhausted the budget in the thinking channel before a parseable final answer; observed calls ending in `stop` that parsed were correct.
- No malformed final numeric answers or valid-but-incorrect answers were observed among the saved parsed examples. This comparison is not used to select the new protocol.

## Failure taxonomy for the new Instruct run

The new protocol records `reasoning_token_exhaustion`, `length_limit_truncation`, `missing_final_marker`, `malformed_final_answer`, `parseable/correct`, and `parseable/incorrect` separately. The Instruct model is verified by `general.finetune=Instruct` and version `2507`; no `think` request field is sent.
