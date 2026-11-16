# Hardware and runtime

Evaluation collection ran from 2026-11-16T13:18:08.787Z through 2026-11-16T16:09:15.492Z; elapsed wall-clock duration is approximately 2.85 hours (10267 seconds). This includes the long calls and stop/recovery intervals between the first v0.4.3 inference and the final user cancellation. It excludes the earlier development validation.

Completed pairs: 43; completed inference calls: 86; incomplete attempted calls: 4. Configured model: qwen3:4b-instruct, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; context 8192; `num_predict=-1`. A 420-second runtime amendment was recorded separately; the original manifest remains unchanged.

Measured completed-call durations: n=86, mean 53.7s, median 35.3s, maximum 283.7s. Initial calls: mean 67.4s, median 51.0s, maximum 245.8s. Revisions: mean 42.5s, median 28.1s, maximum 285.4s.

Telemetry samples after timestamp deduplication: 1787.

| Metric | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| CPU utilization | 10.70 % | 55.04 % | 88.40 % |
| Available RAM | 0.31 GiB | 1.85 GiB | 6.99 GiB |
| Pagefile use | 0.81 GiB | 3.91 GiB | 6.49 GiB |
| GPU utilization | 0.00 % | 13.44 % | 31.00 % |
| VRAM used | 221.00 MiB | 2596.69 MiB | 2768.00 MiB |
| GPU temperature | 58.00 °C | 67.60 °C | 73.00 °C |

Hardware values are journal telemetry samples, not continuous laboratory instrumentation. No hardware configuration, Ollama parallelism, or generation settings were changed for this report.
