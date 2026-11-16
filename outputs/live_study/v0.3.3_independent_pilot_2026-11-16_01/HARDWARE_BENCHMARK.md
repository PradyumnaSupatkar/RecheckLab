# Hardware and inference performance

- CPU: AMD Ryzen 7 4800H, 8 cores / 16 logical processors.
- RAM: 16 GB installed (user supplied); free at protocol freeze 2.92 GiB; minimum sampled during collection 0.39 GiB over 116 samples.
- GPU: RTX 3050 Laptop, 4096 MiB VRAM. At freeze: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 3617 MiB, 0 %. During 5-second sampling, minimum free VRAM 1266 MiB, maximum used 2697 MiB, maximum utilization 84%. Sampling may miss brief peaks.
- Ollama 0.40.16; `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. Model was not loaded before collection; no model was downloaded.
- System Python 3.12.6 used without modifying the broken `.venv`. Packages: {"joblib": "1.5.3", "numpy": "2.3.2", "pandas": "2.3.3", "pyarrow": "24.0.0", "scikit-learn": "1.8.0"}. System pandas differs from the project pin.

## Measured throughput

- 16 successful calls; 0 failures; 8/8 pairs.
- End-to-end time 586.9 s (9.8 min); summed call wall time 580.0 s. Mean / median / p95 call time 36.3 / 41.4 / 51.8 s.
- Observed rate: 49.1 question pairs/hour.
- At this rate, an unconditional 30-50-question run (60-100 calls) would take about 36.7-61.1 minutes, before any protocol correction; this estimate is rough and based on eight questions.
- Selective-policy latency is replayed from observed per-call time, not measured in a real gated run.

## Separate thinking-off development check

The five-question development run completed 10 calls in 226.4 seconds (22.6 seconds/call, 79.5 pairs/hour). Linear estimate for 30-50 pairs is 22.6-37.7 minutes for 60-100 calls. This is based on only five questions and did not include RAM/VRAM sampling; it is not a guarantee. The initial 8-question pilot sampled free RAM down to 0.39 GiB, so verify adequate headroom before a longer run.
