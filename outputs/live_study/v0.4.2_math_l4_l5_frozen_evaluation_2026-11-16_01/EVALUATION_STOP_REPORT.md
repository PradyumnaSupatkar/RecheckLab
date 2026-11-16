# v0.4.2 evaluation stop report

## Outcome

The 60-question evaluation did not complete. One initial call was attempted on the first preselected question. It remained open for approximately 15 minutes without returning a completion envelope. I disconnected the experiment client because this was an extreme runtime outlier. The configured 45-minute hard watchdog had not elapsed. Ollama remained installed and running; disconnecting the client stopped generation, and no unrelated process was killed. No further evaluation calls were launched.

The call returned no persisted response body, token counts, completion reason, parsed answer, or question outcome. The streaming client buffered chunks in memory and appended `RAW_CALLS.jsonl` only after the Ollama response completed. Its in-memory partial response was lost on interruption. `INTERRUPTED_CALL.json` records the exact request, question ID, protocol, and observations available; it does not invent unavailable output metadata. No ML decision or revision was recorded for this incomplete initial answer.

## Resource observations near interruption

- Available RAM: about 1.39 GiB.
- Pagefile use: about 6%, approximately 1.04 GiB; it remained stable during monitoring.
- RTX 3050: about 2,579 MiB used and 1,384 MiB free; temperature peaked at 71 C in the run.
- Ollama reported a 4.2 GB runner with 45% CPU / 55% GPU at context 8,192.
- The llama runner was actively consuming CPU for the long call. The client disconnect stopped generation; the server and model were left running normally.

The call's output token count and actual completion cause are unknown. It cannot be determined from saved evidence whether the model was producing unusually long reasoning, looping, or approaching an internal context limit. The completed development and context benchmark calls do show that `num_predict=-1` can end naturally, but they do not establish this evaluation response's behavior.

## Analysis status

The 10-question development rerun passed extraction: 10/10 initial and 10/10 revision answers parsed, all 20 calls completed with `done_reason=stop`, and all 10 frozen-gate feature vectors were compatible and scored before revision. It had 9/10 initial and revised accuracy, with HELP/HURT/NO_CHANGE = 0/0/10. The frozen gate selected no rechecks and agreed with completion on all 10; this sample does not estimate evaluation performance.

Because the evaluation produced no complete question pair, no four-policy accuracy, token-cost, latency, or uncertainty comparison is reported. The first selected question remains recorded as an attempted incomplete case; the untouched selection for the other 59 questions remains frozen. This run is not a completed 60-question study and supports no evaluation claim.

## Required engineering fix before any future run

Change the streaming collector to append each received chunk and resource heartbeat to a per-call durable journal as it arrives. Add a configurable but scientifically justified runaway detector (for example, repeated-output detection and a long-call watchdog), and include partial responses as incomplete cases. Any resumed experiment must use a new version and explicitly account for this attempted question; do not silently retry it or claim that all 60 remain untouched.
