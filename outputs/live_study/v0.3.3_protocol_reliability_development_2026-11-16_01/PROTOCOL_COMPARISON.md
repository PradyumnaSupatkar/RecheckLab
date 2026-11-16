# Protocol comparison and recommendation

**Run:** `v0.3.3_protocol_reliability_development_2026-11-16_01`  
**Status:** 3 prospectively selected development questions; 12/12 calls completed. This check is not final evaluation evidence and does not establish reliability or generalization.

## Observed completion

| Candidate | Initial valid | Revision valid | Completion reasons (initial/revision) | Mean call latency | Mean initial/revision total tokens | Initial accuracy* | Revision accuracy* |
|---|---:|---:|---|---:|---:|---:|---:|
| A_thinking_off_1024 | 1/3 | 2/3 | length 2/1; stop 1/2 | 39.2s | 951 / 1822 | 33% | 67% |
| B_thinking_on_2048 | 1/3 | 2/3 | length 2/1; stop 1/2 | 63.3s | 1634 / 1472 | 33% | 67% |

*Accuracy is shown only as descriptive grading evidence and was not used to select a protocol. For both candidates, the three questions were the same. Candidate A produced 1/3 valid initial answers; Candidate B produced 1/3. Across both protocols, initial valid extraction was 2/6 and revision extraction was 4/6. Candidate A was faster on mean calls (39.2s vs 63.3s).

## Failure interpretation

The historical pilot had 5/8 initial calls stop at `length` with thinking enabled and a 1,024-token cap. The prior separate thinking-off development sample had 5/5 parseable initial answers at the same cap. In this newly selected paired sample, both A (thinking off, 1,024) and B (thinking on, 2,048) had only 1/3 valid initial answers. Two B initial calls exhausted 2,048 generated tokens; one had no final-channel content because all generated tokens were in the thinking field. A also exhausted 1,024 tokens on two initial calls. This is consistent with a response-length/budget issue and shows material question sensitivity, but the sample cannot attribute failures to thinking mode, token cap, or prompt alone.

## Frozen gate and feature compatibility

All six candidate/question initial records passed the canonical validation and yielded all 19 required `pre-recheck-v2` features. Inputs used only question text, full initial response (including the separate thinking channel where present), parsed initial answer, actual prompt-plus-output token count, actual completion reason, and explicit `finalizer_used=false`. Revised text, keys, correctness, and outcomes were excluded from gate prediction inputs. Candidate A and B gate actions disagreed on 1/3 questions; completion decisions disagreed on 0/3. Gate scores are OOD transfer diagnostics only: the v0.2 gate's SEVRA active-verify prompts, checkpoint identity, finalizer, and generation protocol do not match these Ollama protocols. No protocol identifiers were altered and no production validator was bypassed.

Univariate feature comparisons against frozen v0.2 source training are descriptive only (see `FEATURE_SHIFT.csv`). The protocol comparison does not identify any single cause of shifts.

## Resources

The pre-call baseline was 5.93 GiB available RAM, 266 MiB/4096 MiB GPU memory used, and no loaded Ollama model. During sequential collection, lowest recorded post-call free RAM was 2.95 GiB and the maximum system memory utilization was 80.9%. At completion, free RAM was 2.95 GiB; GPU telemetry was `NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 2582 MiB, 30 %`. All calls completed without hitting the 2.0 GiB stop floor. No unrelated processes were terminated and no system configuration was changed.

## Decision

**No final protocol qualifies yet.** Neither candidate showed reliable initial extraction on the fresh paired development questions, and B's larger 2,048-token budget did not reliably produce final answers. Candidate A was faster, but only 1/3 initial answers were valid here. The earlier 5/5 thinking-off result and this 1/3 result are both small, disjoint samples; they should not be pooled as a formal reliability estimate. Do not launch a 30â€“50-question evaluation yet.

The defensible next step is another separately versioned development check after freezing one explicit adjustment (for example, a larger thinking-off token budget) on new questions. That candidate must reach a predeclared initial and revision extraction criterion with acceptable RAM and latency before an evaluation protocol is frozen. Do not choose a setting based on accuracy, and do not change v0.2 weights or threshold.
