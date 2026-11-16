# Gate protocol compatibility

**Status: incompatible; the frozen gate is not applied in this pilot.**

The saved v0.2 artifact has 19 `pre-recheck-v2` features and threshold 0.25968145276947807, with score `p_HELP - 1.0 * p_HURT`; it rechecks only when score is strictly greater than the threshold. Its required fields are the original question, initial response, parsed initial answer, actual initial token count, initial completion reason, and finalizer flag.

The v0.2 contract is `qwen3:4b` / SEVRA `active_verify` with protocol ID `sevra:10411137fcaa02b3bd549a2285b43ffe315417f6:active_verify`. The checkpoint digest and full SEVRA prompts are unpublished. This pilot uses a newly fingerprinted local Ollama checkpoint and explicit prompts: one 1024-token initial call, one 1024-token recheck, 4096 context, thinking enabled, temperature 0, seeds 42/43, and no finalizer. SEVRA documents a 4096-token initial limit and a possible 512-token finalizer. Matching the model family/tag is insufficient. Although the pilot records all required pre-recheck features, protocol compatibility fails. The gate is therefore excluded from policy decisions and results; its identifiers or artifact contract are not altered.

## Existing independent live-call context

Earlier v0.3.0, v0.3.1, and v0.3.2 journals contain respectively 38/38, 105/105, and 22/22 successful calls; explicit `FINAL` answer extraction succeeded for 31, 77, and 14 of those calls. `done_reason=length` occurred 7, 28, and 8 times. These runs use other fixed cohorts and are not merged with this pilot.
