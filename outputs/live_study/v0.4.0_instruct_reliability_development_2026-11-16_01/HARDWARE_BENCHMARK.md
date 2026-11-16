# Hardware and inference benchmark

**Checkpoint:** Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct`), digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; Ollama `ollama version is 0.40.16`. Sequential local inference only.

| Phase | Questions | Calls | Mean initial call | Mean revision call | Mean initial prompt+output tokens | Mean revision prompt+output tokens | Minimum free RAM | Minimum free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 10 | 20 | 5.68s | 5.21s | 178.9 | 315.5 | 4.11 GiB | 1487 MiB |
| Frozen evaluation | 40 | 80 | 5.22s | 5.27s | 181.2 | 318.9 | 2.32 GiB | 1458 MiB |

Pre-inference baseline: 5.89 GiB available RAM; GPU `{'name': 'NVIDIA GeForce RTX 3050 Laptop GPU', 'total_mib': 4096, 'used_mib': 271, 'free_mib': 3825, 'utilization_percent': 0}`. Across the evaluation, maximum swap use was 4.4%. The enforced stop floors were 2.0 GiB RAM and 512 MiB VRAM; neither floor was crossed. All 100 calls returned with recorded usage counts and actual client wall latency. The response thinking field was nonempty in 0/100 calls. No retry, model download, process termination, or system configuration change occurred.
