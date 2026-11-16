# v0.4.2 development rerun results

The fixed ten development questions were reused. All responses were fresh under the v0.4.2 8K uncapped protocol; no evaluation question or gold answer was used to select protocol settings.

- Calls completed: 20/20; initial and revision extraction: 10/10 each. Every call reported `done_reason=stop`; no length-limit completions.
- Initial accuracy: 9/10; revision accuracy: 9/10. HELP/HURT/NO_CHANGE: 0/0/10. One question remained incorrect after revision.
- Frozen-gate features were compatible for 10/10 and scored before revision. OOD gate rechecks: 0/10; completion rechecks: 0/10; disagreements: 0.
- Mean initial tokens: 127.8 prompt + 1,193.4 output = 1,321.2 total; mean initial wall latency 70.08 s; mean generation speed 18.95 output tokens/s.
- Mean revision tokens: 1,387.2 prompt + 416.1 output = 1,803.3 total; mean revision wall latency 29.80 s; mean generation speed 16.65 output tokens/s.
- Paired replay diagnostics: never/completion/gate retained initial answers at 9/10 and averaged 1,321.2 tokens and 70.08 s per question. Always-recheck was 9/10 and averaged 3,124.5 tokens and 99.88 s per question. These ten items are development data; do not interpret the equality as gate performance evidence.
- Maximum generated initial response was 2,879 tokens and ended naturally. Across development, minimum free RAM was 1.388 GiB, minimum free VRAM 1,371 MiB; pagefile use remained low. No fixed RAM or VRAM reserve was applied.

The separate two-question context benchmark completed all eight calls naturally. Both 8K and 16K produced 4/4 parseable outputs. Mean speed was 18.947 tokens/s at 8K versus 15.826 at 16K; minimum free RAM was 4.315 versus 2.976 GiB, and maximum measured GPU use was 2,551 versus 2,687 MiB. `ollama ps` showed 45% CPU / 55% GPU at 8K and 56% CPU / 44% GPU at 16K. This supported selecting 8K for reliability plus throughput/resource use, without using answer correctness.
