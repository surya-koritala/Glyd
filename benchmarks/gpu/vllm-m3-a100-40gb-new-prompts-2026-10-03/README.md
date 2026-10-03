# vLLM with `--quantization glyd` on an A100 40 GB, a set of prompts for each pass (2026-10-03)

The `bench` and `big` steps of [`vllm-m3-a100-40gb-2026-09-30`](../vllm-m3-a100-40gb-2026-09-30) again: `vllm bench serve`, bf16 against Glyd, on Qwen3-8B (`bench/`) and Qwen3-14B
(`big/`), on one NVIDIA A100-SXM4-40GB, with each pass's prompts new to the server. The published run's passes all used `--seed 0`, so a later pass repeated an earlier
pass's prompts, and vLLM's prefix cache (on, as deployments run it) served those it still held: its servers' logs show a prefix cache hit rate of up to 43.1% (bf16) and
53.5% (Glyd) for Qwen3-8B, and 40.8% and 52.7% for Qwen3-14B. Here each pass's seed is its place in the list of rates (0, 1 and 2), the same for both modes, and the summary
prints each server's highest prefix cache hit rate.

## Setup

- **Machine:** NVIDIA A100-SXM4-40GB (400 W, 1,410 MHz), driver 580.126.20. AMD EPYC 7J13, 30 vCPUs, 216 GB of memory, x86_64.
- **Software:** the published run's tree (766746a, v0.25.1 with the plugin); vLLM 0.30.0 from PyPI (torch 2.13.0+cu130, transformers 5.18.0; the published run's was 5.17.0).
  The library was built there for sm_80 (C API 5; GPU code 80).
- **Run:** the published run's `bench` and `big` steps, one after the other on one instance. Each mode's server started twice, on an empty compile cache and then warm (the warm one
  measured); `--gpu-memory-utilization 0.9`, `--max-model-len 4096`, the random dataset of 1,024 tokens in and 256 out. Qwen3-8B: 64 prompts at 1 request a second, 128 at 4 and
  256 at once; Qwen3-14B: 32, 64 and 128.
- **Glyd's layout:** `mma12`.

## Qwen3-8B (`bench/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 141,040 tokens | 34.43x | 133,776 tokens |
| Glyd `mma12` | 12.53 GiB | 160,720 tokens (1.14x) | 39.24x | 153,472 tokens (1.15x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 1.3%, glyd 1.3%. The summary flags a run above 1% as not comparable, and flags this one; the share is
the same on both sides. In [`l4-vllm-kv-graphs-2026-10-01`](../l4-vllm-kv-graphs-2026-10-01) a run with one repeated prompt (hit rates 1.1% to 1.2%) and one with none (0.0%) gave
saturated rows within 1.4%.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.95 | 242.1 | 108 / 162 | 14.8 / 16.3 | 64 | 1410 MHz, 59 C |
| 1 | glyd | 0.95 | 243.6 | 129 / 223 | 13.2 / 15.3 | 64 | 1410 MHz, 57 C |
| 4 | bf16 | 3.56 | 911.6 | 150 / 403 | 21.5 / 26.5 | 128 | 1410 MHz, 63 C |
| 4 | glyd | 3.59 | 920.1 | 180 / 421 | 21.9 / 28.6 | 128 | 1395 MHz, 64 C |
| inf | bf16 | 5.85 | 1498.1 | 15466 / 39256 | 64.3 / 102.1 | 256 | 1320 MHz, 68 C |
| inf | glyd | 5.73 | 1467.0 | 16896 / 36487 | 75.4 / 111.2 | 256 | 1320 MHz, 68 C |

Every request completed. **Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.01x (0.95 to 0.95) | +19% (108 to 129 ms) | -10% (14.8 to 13.2 ms) | -10% |
| 4 | 1.01x (3.56 to 3.59) | +20% (150 to 180 ms) | +2% (21.5 to 21.9 ms) | +3% |
| inf | 0.98x (5.85 to 5.73) | +9% (15,466 to 16,896 ms) | +17% (64.3 to 75.4 ms) | +13% |

## Qwen3-14B (`big/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 27.52 GiB | 45,744 tokens | 11.17x | 37,728 tokens |
| Glyd `mma12` | 21.98 GiB | 80,960 tokens (1.77x) | 19.77x | 63,568 tokens (1.68x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 0.0%, glyd 0.0%. Not flagged.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.84 | 214.3 | 213 / 455 | 26.3 / 27.9 | 32 | 1410 MHz, 57 C |
| 1 | glyd | 0.86 | 220.0 | 253 / 561 | 22.8 / 25.1 | 32 | 1410 MHz, 58 C |
| 4 | bf16 | 2.44 | 623.4 | 737 / 2762 | 43.7 / 51.0 | 64 | 1410 MHz, 63 C |
| 4 | glyd | 2.76 | 706.0 | 503 / 952 | 45.3 / 62.2 | 64 | 1380 MHz, 63 C |
| inf | bf16 | 2.63 | 673.2 | 17956 / 42177 | 48.8 / 78.8 | 128 | 1410 MHz, 68 C |
| inf | glyd | 3.37 | 861.7 | 14109 / 30149 | 64.9 / 98.7 | 128 | 1305 MHz, 68 C |

Every request completed. **Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.03x (0.84 to 0.86) | +19% (213 to 253 ms) | -13% (26.3 to 22.8 ms) | -12% |
| 4 | 1.13x (2.44 to 2.76) | -32% (737 to 503 ms) | +4% (43.7 to 45.3 ms) | +2% |
| inf | 1.28x (2.63 to 3.37) | -21% (17,956 to 14,109 ms) | +33% (48.8 to 64.9 ms) | +1% |

## Against the published run

| Model | Saturated requests a second, Glyd against bf16: published | here |
| :--- | :--- | :--- |
| Qwen3-8B | 1.19x (6.36 to 7.55) | 0.98x (5.85 to 5.73) |
| Qwen3-14B | 1.65x (2.65 to 4.37) | 1.28x (2.63 to 3.37) |

The rate-1 pass is the first on its server, with no earlier prompts to repeat, and reproduces the published one: Qwen3-8B's first token +18% then (106 to 125 ms) and +19% here, each token -10% then (14.7 to
13.2 ms) and here; Qwen3-14B's first token +18% then (209 to 248 ms) and +19% here, each token -13% then and here.

## Files

- `vj-steps.txt`: each step's exit.
- `bench/` and `big/`: `env.txt`, `machine.txt`, `machine-short.txt`, `steps.txt` (each step's time and its prefix cache hit rate line), `summary.txt`, the console (`bench-MODEL.txt`) and `bench-MODEL/` (per mode and
  rate: vLLM's result JSON and console output, nvidia-smi's samples each second of a rate; each server's log and KV cache lines, cold and warm; `summary.txt`).
