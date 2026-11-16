# Research evidence summaries

This directory contains selected summaries copied or distilled from frozen local outputs. Original experiment artifacts remain unchanged locally.

## Active v0.2 policy performance (recorded responses)

Source-test and target-test results replay saved SEVRA revision responses; these are not newly generated inference. The target test contains 733 questions and two saved revision rollouts per question (1,466 rollout rows). Initial-answer accuracy was 76.13%. Always rechecking reduced accuracy to 72.85% and produced 4.71% harmful flips. The frozen gate rechecked 3.41% and reached 76.67%, a 0.55 percentage-point gain over keeping the initial answer. This is a retrospective recorded-response comparison, not evidence of live generalization.

Source-test (400 questions): frozen gate 90.00% accuracy versus 74.75% keeping initial; it rechecked 27.25%. Results are from the fixed historical split.

## ML classification results

On the recorded-response target test, the three-class classifier reached 91.54% accuracy, macro-F1 0.394, weighted-F1 0.900, and log loss 0.322. NO_CHANGE-only accuracy was 93.86%; HURT precision was undefined because no HURT predictions were made. Therefore high accuracy is dominated by class imbalance and does not establish reliable HURT detection. Source-test macro-F1 was 0.542.

## Key negative findings

A bounded final accuracy audit reported NO VERIFIED IMPROVEMENT. It found no justified new policy improvement on held-out data. The leading length-or-missing rule matched the frozen gate on the adaptation confirmation and did not establish a meaningful gain. The existing classifier/rule pair had little disagreement. Do not present tuning or replay of exposed historical results as independent confirmation.

## Dataset provenance and limitations

The public replay data are the pinned SEVRA recovery-outcomes release documented in DATA_SOURCES.md. They contain published Qwen3-4B responses and annotations; this project did not generate those responses or reproduce the original authors' evaluation. The analysis uses selected records and inherited benchmark/source limitations and licenses. Public data terms remain applicable; this repository does not assign a new license to them.

Independent live collection is a separate study using GSM8K and CommonsenseQA questions and actual local Ollama calls. Its protocols and collection records live under outputs/live_study locally and are not included as raw bulk data in this summary. The live protocol differs from the recorded SEVRA protocol; the v0.2 gate has not been validated for it. Smoke checks and a handful of calls are functionality evidence only, not live predictive-accuracy evidence. CommonsenseQA validation subset is not an official leaderboard test set. See DATA_SOURCES.md and RESEARCH_PROTOCOL.md for pinned provenance and design.

## IEEE research audit conclusions

The audit concludes that current results are preliminary and do not support a strong conference claim of a validated policy improvement or live generalization. It recommends establishing a distinct methodological contribution, testing it against appropriate existing methods, using fresh independent evidence, reporting uncertainty and costs, and preserving negative/inconclusive outcomes. These summaries are not a substitute for the full audit and protocol.
