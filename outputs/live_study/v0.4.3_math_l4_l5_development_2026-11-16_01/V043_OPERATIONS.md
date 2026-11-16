# RecheckLab v0.4.3 operations

## Frozen scope

v0.4.3 changes delivery, journaling, diagnostics, and recovery only. It reuses the v0.4.2 frozen question order, question/key bytes, prompts, exact grader, Qwen3-4B-Instruct-2507 digest, 8,192 context, `num_predict=-1`, temperature, seed derivation, and frozen v0.2 gate. It does not change sampling, concurrency, model offload, or classifier behavior. Gate decisions remain OOD and are journaled before any revision request.

The prior v0.4.2 first question remains attempt 1 and incomplete. The original record contains no response chunks, token statistics, or completion reason. A new call for that question is attempt 2 and has a distinct call ID. Do not merge attempt 1 and attempt 2 into one call.

## Engineering validation

Run from the repository root:

```powershell
python outputs/live_study/v0.4.3_math_l4_l5_development_2026-11-16_01/test_durable_stream.py
python outputs/live_study/v0.4.3_math_l4_l5_development_2026-11-16_01/test_v043_recovery.py
```

These tests use synthetic stream fixtures and read-only frozen selection checks. They do not call Ollama or read evaluation keys into any gate decision.

## Recovery procedure

The v0.4.3 evaluation directory starts with the byte-identical frozen selection and key files plus `STREAM_JOURNAL.jsonl` documenting the v0.4.2 interruption. Its current state is summarized in the sibling evaluation directory's `RECOVERY_STATE.md`. To process untouched questions first, run:

```powershell
python outputs/live_study/v0.4.3_math_l4_l5_development_2026-11-16_01/run_eval_v043.py --run
```

This preserves the first question as incomplete attempt 1, records it as pending, and proceeds to the other 59 questions in original order. Completed calls and question pairs are reused from checkpoints on restart. An incomplete call is never retried implicitly.

When ready to explicitly retry incomplete calls, run:

```powershell
python outputs/live_study/v0.4.3_math_l4_l5_development_2026-11-16_01/run_eval_v043.py --run --retry-incomplete
```

The first question then uses attempt 2, with a separate call journal. If interrupted again, a later explicit retry creates attempt 3. Completed calls are not regenerated. A call is considered completed only when Ollama returns its `done` event; missing `done_reason` or token counts remain null and do not get inferred.

## Journal layout

- `STREAM_JOURNAL.jsonl`: append-only `call_start`, one `chunk` per received stream line, optional diagnostic warning events, and explicit complete/interrupted/error terminals. Each line is flushed and fsync'd. Each chunk stores the original line, parsed JSON, IDs, attempt, UTC time, elapsed monotonic time, accumulated response hashes/progress, repetition diagnostics, and cached hardware telemetry.
- `COMPLETED_CALLS.jsonl`: durable normalized checkpoint after each completed call.
- `GATE_DECISIONS.jsonl`: original v0.2 gate output, saved before revision calls.
- `QUESTION_CHECKPOINTS.jsonl`: paired question checkpoint after initial and revision calls.
- `RUN_STATUS.json`: latest resumable status snapshot.

`durable_stream.reconstruct_calls()` rebuilds content and thinking from saved chunks. Incomplete attempts remain incomplete. If the final JSONL line is malformed or truncated, the reader fails loudly and preserves the source file; it never silently truncates or repairs evidence. Preserve a copy and perform explicit recovery before continuing.

The 120-second socket read timeout detects a no-progress stall; the 45-minute watchdog remains the hard limit. Long-active and repetitive-output diagnostics are warnings only and never end generation. Resource telemetry is sampled at most every five seconds and attached to received chunks; no RAM or VRAM reserve cutoff is used.
