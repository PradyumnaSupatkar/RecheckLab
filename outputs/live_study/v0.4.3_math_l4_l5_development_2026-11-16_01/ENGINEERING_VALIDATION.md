# v0.4.3 engineering validation

## Result

The durable streaming and recovery fixtures passed. No v0.4.3 evaluation inference has been started.

## Checks executed

- Python compilation passed for `durable_stream.py`, `run_eval_v043.py`, and both test modules.
- Nine synthetic stream tests passed: natural stop, thinking/content chunk storage, interrupted-call partial reconstruction, new attempt after restart, disconnect without done, no-progress timeout classification, absent completion metadata, malformed raw JSON-line preservation, warning-only repetition diagnostics, final-answer parsing from reconstructed text, and truncated JSONL detection.
- Four read-only recovery tests passed: question/key files are byte-identical to v0.4.2 and retain ordering; the original first-question attempt reconstructs as incomplete with no invented output metadata; the other 59 questions have no attempts; context, output token setting, model digest, and gate checksum remain frozen.
- `--initialize` created the separate v0.4.3 manifest and journal without Ollama inference.

## Preservation

v0.4.1 and v0.4.2 experiment directories remain untouched. The v0.4.3 frozen evaluation selection and answer keys are exact byte copies. The original first call record is retained in the v0.4.2 folder and referenced by hash in the v0.4.3 journal. No classifier, feature code, prompts, generation settings, raw historical records, or manuscripts were changed.

## Current evaluation state

The v0.4.3 evaluation is not complete: zero new calls have been issued and the 60-question selection remains available. Question 1's v0.4.2 attempt 1 is documented as incomplete; any v0.4.3 retry must be recorded as a separate attempt. The recovery procedure can process the remaining 59 questions and later retry incomplete attempts only with the explicit `--retry-incomplete` option.

No response, answer, token count, or completion reason has been inferred for the interrupted v0.4.2 call. No policy comparison or evaluation claim is made.
