# Initial-answer completion reliability experiment

Six new initial-only calls completed successfully on the installed qwen3:4b.
No rechecks, finalizers, retries, downloads, gate training, dashboard changes or
larger experiment ran. All 247 protected project files, including the historical
generation experiments and frozen protocols, are byte-for-byte unchanged.

## Diagnosis of the seven historical invalid answers

Every invalid response generated exactly 2,048 tokens and reported `length`.
Historical grades remain unchanged.

| Archived call | Thinking characters | Visible characters | Diagnosis |
|---|---:|---:|---|
| 03_initial | 8,475 | 0 | Output budget exhausted before any visible answer |
| 03_A | 8,696 | 0 | Output budget exhausted before any visible answer |
| 03_B | 8,530 | 0 | Output budget exhausted before any visible answer |
| 03_C | 8,287 | 389 | Visible option analysis began, but was truncated before FINAL |
| 05_initial | 9,144 | 0 | Output budget exhausted before any visible answer |
| 05_B | 9,368 | 0 | Output budget exhausted before any visible answer |
| 05_C | 9,158 | 0 | Output budget exhausted before any visible answer |

For the six empty-visible cases, all saved generated text is in the thinking
channel, consistent with remaining in thinking when the shared output allowance
ran out. Old nonstream responses do not expose the exact last token/channel.
03_C had demonstrably left thinking and begun visible output. Its partial
analysis includes an answer-like preference, but no declared FINAL. It remains
invalid, rather than being reinterpreted as a correct final answer.

No complete visible FINAL answer was rejected by the existing parser. The eighth
length completion, 04_initial, contains FINAL: C; it was correctly parsed and
correctly graded wrong. There is no evidence of malformed complete FINAL syntax
or a parser-only failure in the seven invalid responses.

The configured `<|im_start|>` and `<|im_end|>` stop strings match model defaults.
All seven failures say length rather than stop, so there is no evidence that
stop strings directly caused them. The output limit covers thinking and visible
generation together; a concise visible-answer instruction does not impose a
hard thinking-token limit on this thinking-only checkpoint.

## New completion experiment

Reused old questions 1, 3 and 5: one completed control and both invalid initial
cases. This is a completion-stress sample, not a random accuracy evaluation.
A is saved INITIAL output, not the historical generic recheck condition A.

Concise prompt for B and C:

> Solve the multiple-choice question. Avoid unnecessary explanation. Finish with exactly FINAL: X, where X is one A-E letter.

| Condition | Budget | Valid FINAL | Correct, diagnostic only | Length completions | Generated tokens, total | Actual tokens, total | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: saved original initial | 2,048 | 1/3 | 1/3 | 2 | 5,641 | 5,914 | 98.05 s |
| B: concise initial | 2,048 | 1/3 | 1/3 | 2 | 5,609 | 5,873 | 86.37 s |
| C: concise initial | 3,072 | 3/3 | 2/3 | 0 | 6,418 | 6,682 | 95.44 s |

Average generated/actual tokens per question: A 1,880.33/1,971.33;
B 1,869.67/1,957.67; C 2,139.33/2,227.33. Actual tokens equal prompt plus
generated tokens, including thinking. Latencies include observed loading and
caching; they are not isolated prompt-speed estimates.

The two formerly failing questions completed under C in 2,268 and 2,637 generated
tokens. One was correct; the other was a valid but wrong answer. The control
ended naturally at 1,513 tokens in both B and C. Thus C improved completion on
this stress sample, while the concise prompt at 2,048 alone did not.
For both failed B cases, B's thinking text is an exact prefix of C's thinking
text; the control's thinking and visible responses are identical across B/C.
This supports continued seeded generation beyond the smaller cap as the cause
of the observed completion improvement.

A used archived Ollama 0.40.13/nonstream outputs. New B/C used Ollama 0.40.14
and streaming, with the same installed model digest
359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7.
A-to-B differences cannot isolate prompt effects from the backend change.
B and C share the current frozen backend, prompt, seed 42, temperature 0.6,
top-p 0.95, top-k 20, min-p 0, repeat penalty 1, thinking enabled and original
stop strings. Only their output limits differ. The original parser and exact-key
grader were retained. Keys never entered model requests or the streaming worker.
Independent read-only verification reproduced all six raw streams, nine records,
key-based grades, condition summaries, context bounds, 1,091 RAM samples and all
247 protected-file hashes. New inference consumed 12,027 generated / 12,555
actual tokens over 545.44 seconds of call wall time.

## Resources, verification and recommendation

Context stayed at 4,096 throughout. Conservative full-request bounds for C were
3,621, 3,668 and 3,673; actual prompts were 81, 93 and 90 tokens. No request
needed a larger context or input truncation.

Available RAM was sampled every 0.5 seconds during loading and inference. New-run
minimum was **4.1015 GiB** (B minimum 4.2867, C minimum 4.1015). A's archived
post-call minimum on these three questions was 1.2849 GiB; that snapshot method
is not comparable to the new continuous samples. No resource abort, allocation
error, backend error or concurrent model occurred.

Declared guards require 5 GiB free before cold loading, 2 GiB before later B
calls and 2.5 GiB before each C call. C also requires B's observed minimum to
remain at least 2 GiB. Below 1.5 GiB, the parent terminates the active streaming
client and stops the entire run without another request. Mock no-inference
checks verified both critical-RAM cancellation and interruption cleanup.

Use C as the better-supported next Qwen completion setting, retaining the RAM
guards and 4,096 context. This three-question result does not establish a general
completion rate, justify 80% accuracy, or authorize a larger run. Preserve these
as operational diagnostics and validate any next step separately.

A similarly sized quantized non-thinking checkpoint, such as
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
is a practical future comparison: the official model is 4B and non-thinking-only,
so it avoids this hidden-thinking completion bottleneck. Its actual laptop fit,
latency and CommonsenseQA correctness remain unmeasured. None was installed.

Full evidence: protocol.json, raw_chunks_*.jsonl, response_*.json,
records.jsonl, ram_samples.jsonl, condition_comparison.csv and integrity.json.
