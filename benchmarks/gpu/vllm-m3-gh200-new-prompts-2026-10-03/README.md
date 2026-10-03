# vLLM with `--quantization glyd` on a GH200, a set of prompts for each pass (2026-10-03)

The `bench` and `big` steps of [`vllm-m3-gh200-2026-09-30`](../vllm-m3-gh200-2026-09-30) again: `vllm bench serve`, bf16 against Glyd, on Qwen3-8B (`bench/`) and Qwen3-32B
(`big/`), on one NVIDIA GH200, with each pass's prompts new to the server. The published run's passes all used `--seed 0`, so a later pass repeated an earlier
pass's prompts, and vLLM's prefix cache (on, as deployments run it) served those it still held: its servers' logs show a prefix cache hit rate of up to 46.1% (bf16) and
46.8% (Glyd) for Qwen3-8B, and 53.5% and 46.3% for Qwen3-32B. Here each pass's seed is its place in the list of rates (0, 1 and 2), the same for both modes, and the summary
prints each server's highest prefix cache hit rate.

## Setup

- **Machine:** NVIDIA GH200 480GB (97,871 MiB, 900 W, 1,980 MHz at most), driver 580.126.20. Neoverse-V2, 64 CPUs, 525 GB of memory, aarch64.
- **Software:** the published run's tree (766746a, v0.25.1 with the plugin); vLLM 0.30.0 from PyPI, aarch64 wheels (torch 2.13.0+cu130, transformers 5.18.0; the published run's was 5.17.0).
  The library was built there for sm_90a (C API 5; GPU code 90).
- **Run:** the published run's `bench` and `big` steps, one after the other on one instance. Each mode's server started twice, on an empty compile cache and then warm (the warm one
  measured); `--gpu-memory-utilization 0.9`, `--max-model-len 4096`, the random dataset of 1,024 tokens in and 256 out. Qwen3-8B: 64 prompts at 1 request a second, 128 at 4 and
  256 at once; Qwen3-32B: 32, 64 and 128.
- **Glyd's layout:** `mma12`.

## Qwen3-8B (`bench/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 489,344 tokens | 119.47x | 477,440 tokens |
| Glyd `mma12` | 12.53 GiB | 509,056 tokens (1.04x) | 124.28x | 497,168 tokens (1.04x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 0.9%, glyd 1.0%. Not flagged (the summary flags a run above 1%).

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.98 | 250.0 | 44 / 71 | 6.0 / 6.4 | 64 | 1980 MHz, 56 C |
| 1 | glyd | 0.98 | 249.9 | 46 / 67 | 6.1 / 6.6 | 64 | 1980 MHz, 58 C |
| 4 | bf16 | 3.81 | 975.4 | 50 / 94 | 6.9 / 7.9 | 128 | 1965 MHz, 57 C |
| 4 | glyd | 3.80 | 973.9 | 56 / 129 | 7.4 / 9.1 | 128 | 1965 MHz, 58 C |
| inf | bf16 | 21.16 | 5418.2 | 3678 / 6777 | 32.2 / 43.9 | 256 | 1605 MHz, 62 C |
| inf | glyd | 19.61 | 5020.2 | 3860 / 7070 | 35.2 / 47.4 | 256 | 1800 MHz, 62 C |

Every request completed. **Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.00x (0.98 to 0.98) | +6% (44 to 46 ms) | +1% (6.0 to 6.1 ms) | +1% |
| 4 | 1.00x (3.81 to 3.80) | +13% (50 to 56 ms) | +8% (6.9 to 7.4 ms) | +8% |
| inf | 0.93x (21.16 to 19.61) | +5% (3,678 to 3,860 ms) | +9% (32.2 to 35.2 ms) | +8% |

## Qwen3-32B (`big/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 61.03 GiB | 79,760 tokens | 19.47x | 73,456 tokens |
| Glyd `mma12` | 48.27 GiB | 132,144 tokens (1.66x) | 32.26x | 125,856 tokens (1.71x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 0.0%, glyd 0.0%. Not flagged.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.85 | 218.4 | 178 / 428 | 23.0 / 24.4 | 32 | 1980 MHz, 63 C |
| 1 | glyd | 0.86 | 221.4 | 226 / 512 | 21.7 / 23.3 | 32 | 1980 MHz, 60 C |
| 4 | bf16 | 2.90 | 741.1 | 247 / 524 | 31.8 / 37.7 | 64 | 1965 MHz, 62 C |
| 4 | glyd | 2.80 | 717.8 | 352 / 670 | 39.8 / 48.1 | 64 | 1965 MHz, 66 C |
| inf | bf16 | 4.36 | 1117.2 | 11111 / 22292 | 47.2 / 78.9 | 128 | 1905 MHz, 71 C |
| inf | glyd | 3.88 | 993.6 | 8440 / 14317 | 74.4 / 89.9 | 128 | 1485 MHz, 72 C |

Every request completed. **Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.01x (0.85 to 0.86) | +27% (178 to 226 ms) | -6% (23.0 to 21.7 ms) | -5% |
| 4 | 0.97x (2.90 to 2.80) | +42% (247 to 352 ms) | +25% (31.8 to 39.8 ms) | +25% |
| inf | 0.89x (4.36 to 3.88) | -24% (11,111 to 8,440 ms) | +58% (47.2 to 74.4 ms) | +18% |

## Against the published run

| Model | Saturated requests a second, Glyd against bf16: published | here |
| :--- | :--- | :--- |
| Qwen3-8B | 0.92x (27.60 to 25.43) | 0.93x (21.16 to 19.61) |
| Qwen3-32B | 0.88x (5.55 to 4.89) | 0.89x (4.36 to 3.88) |

The rate-1 pass is the first on its server, with no earlier prompts to repeat, and reproduces the published one: Qwen3-8B's first token +4% then (45 to 46 ms) and +6% here, each token +1% then (6.0 to
6.1 ms) and here; Qwen3-32B's first token +28% then (178 to 228 ms) and +27% here, each token -6% then (23.1 to 21.7 ms) and here.

## Files

- `vj-steps.txt`: each step's exit.
- `bench/` and `big/`: `env.txt`, `machine.txt`, `machine-short.txt`, `steps.txt` (each step's time and its prefix cache hit rate line), `summary.txt`, the console (`bench-MODEL.txt`) and `bench-MODEL/` (per mode and
  rate: vLLM's result JSON and console output, nvidia-smi's samples each second of a rate; each server's log and KV cache lines, cold and warm; `summary.txt`).
