# Operational preflight review

Exactly five existing CommonsenseQA official-training questions and 20 generation
calls ran. All five questions were selected by seed 46 from 9,541 eligible rows,
with no ID or canonical-question overlap with previous local-study manifests.
These preflight questions should be excluded from later development/evaluation.

All 20 calls completed normally with no Ollama error, retry, finalizer, download,
or settings change. The run used Ollama 0.40.13 and installed qwen3:4b digest
359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7.
Both phase budgets were 2,048, num_ctx was 4,096, thinking was enabled,
temperature was 0.6, top-p 0.95, top-k 20, initial seed 42 and recheck seed 43.
Exact prompts, template, settings and output/parser format are in protocol.json.

| Configuration | Valid FINAL | Correct | HELP / HURT | Pair oracle, diagnostic only | Avg actual tokens/call | Avg seconds/call |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 3/5 (60%) | 2/5 | N/A | N/A | 1,961.4 | 93.81 |
| A: existing generic | 4/5 (80%) | 2/5 | 0 / 0 | 2/5 | 1,251.8 | 48.84 |
| B: conservative commonsense | 3/5 (60%) | 2/5 | 0 / 0 | 2/5 | 1,416.8 | 56.14 |
| C: independent-first | 3/5 (60%) | 3/5 | 1 / 0 | 3/5 | 1,519.2 | 60.58 |

These correctness counts and paired oracles are diagnostic, not evidence of
general benchmark accuracy or a reliable prompt ranking. C corrected one valid
wrong initial answer. No configuration recovered an invalid initial answer into
a correct answer. A produced a valid but wrong answer on one invalid-initial case.

There were eight token-limit completions, seven of which lacked a valid FINAL
answer. Two of five initial answers were invalid; all three rechecks failed to
finish on one of those questions. Thus 2,048 tokens did not reliably produce
complete answers under the requested settings.

Context checks passed for every request. The largest conservative input bound
plus the full 2,048-token output allowance was 3,692/4,096. Actual prompt tokens
peaked at 248, leaving room for the full output allowance. The runner reported
the requested 4,096-token context throughout. Every A/B/C request used exactly
the same saved visible initial content; previous thinking and previous rechecks
were excluded. Trusted keys were stored separately and used only after inference.

The runner used 34% CPU / 66% GPU placement. Recorded between-call GPU free memory
was 1,258-1,294 MiB, but available system RAM fell to 0.591 GiB (96% memory load).
There was no allocation/OOM error, but this limited RAM headroom is a practical
reason not to scale the unchanged setup. These samples are not peak-memory
measurements and do not establish which running application caused the pressure.

Total consumption was 30,746 actual tokens (3,284 prompt + 27,462 generated).
Wall time was 1,301.30 seconds, approximately 21.69 minutes, including first-call
cold loading and between-call validation. Per-call latency includes loading and
backend caching as observed; the first call reported 10.63 seconds of loading.

Independent read-only verification reproduced selection, all 20 request histories,
visible-content parsing, trusted-key grading, tokens, 15 paired outcomes, summary
metrics and protocol/script hashes. All 229 protected project files were rehashed
and are unchanged; no new files exist outside outputs/generation_experiments/.

Recommendation: do not proceed unchanged to the 50-question study. First review
an explicitly approved small adjustment addressing answer completion and available
RAM. No further inference, gate training, Streamlit modification or deployment ran.
