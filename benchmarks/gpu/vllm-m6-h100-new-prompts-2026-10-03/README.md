# vLLM with `--quantization glyd` on an H100 SXM, a set of prompts for each pass (2026-10-03)

The `moebench` step of [`vllm-m6-h100-2026-10-01`](../vllm-m6-h100-2026-10-01) again: `vllm bench serve`, bf16 against Glyd, on Qwen3-30B-A3B, on one NVIDIA H100 SXM, with each pass's prompts
new to the server. The published run's passes all used `--seed 0`, so a later pass repeated an earlier pass's prompts, and vLLM's prefix cache (on, as deployments run it) served those it
still held: its servers' logs show a prefix cache hit rate of up to 46.1% (bf16) and 51.4% (Glyd). Here each pass's seed is its place in the list of rates (0, 1 and 2), the same for both
modes, and the summary prints each server's highest prefix cache hit rate.

## Setup

- **Machine:** NVIDIA H100 80GB HBM3, the SXM5 (81,559 MiB, power limit 700 W, 1,980 MHz at most), driver 580.126.20. Xeon Platinum 8480+ (26 CPUs), x86_64.
- **Software:** the published run's tree (7fe66a2, release-0.26.0, the v0.26.0 candidate); vLLM 0.30.0 from PyPI (torch 2.13.0+cu130, transformers 5.18.0). The library was built there for
  sm_90a (C API 7; GPU code 90).
- **Run:** each mode's server started twice, on an empty compile cache and then warm (the warm one measured); `--gpu-memory-utilization 0.9`, `--max-model-len 4096`, the random dataset of
  1,024 tokens in and 256 out: 64 prompts at 1 request a second, 128 at 4 and 256 at once.
- **Glyd's layout:** `mma12`.

## Qwen3-30B-A3B (`bench-Qwen3-30B-A3B/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 56.88 GiB | 124,720 tokens | 30.45x | 118,720 tokens |
| Glyd `mma12` | 44.27 GiB | 263,376 tokens (2.11x) | 64.30x | 257,392 tokens (2.17x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 1.1%, glyd 1.0%. The summary flags a run above 1% as not comparable, and flags this one (bf16 1.1%);
the share is about the same on both sides. In [`l4-vllm-kv-graphs-2026-10-01`](../l4-vllm-kv-graphs-2026-10-01) a run with one repeated prompt (hit rates 1.1% to 1.2%) and one with none (0.0%)
gave saturated rows within 1.4%.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.97 | 249.0 | 62 / 265 | 6.8 / 9.1 | 64 | 1980 MHz, 45 C |
| 1 | glyd | 0.97 | 248.7 | 84 / 125 | 7.2 / 9.7 | 64 | 1980 MHz, 45 C |
| 4 | bf16 | 3.74 | 957.5 | 67 / 114 | 11.8 / 13.9 | 128 | 1980 MHz, 51 C |
| 4 | glyd | 3.71 | 950.8 | 129 / 536 | 20.3 / 25.6 | 128 | 1980 MHz, 48 C |
| inf | bf16 | 12.03 | 3080.5 | 6737 / 16684 | 28.6 / 51.8 | 256 | 1980 MHz, 59 C |
| inf | glyd | 10.07 | 2577.5 | 4248 / 20028 | 64.5 / 70.7 | 256 | 1980 MHz, 56 C |

Every request completed. **Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.00x (0.97 to 0.97) | +35% (62 to 84 ms) | +5% (6.8 to 7.2 ms) | +6% |
| 4 | 0.99x (3.74 to 3.71) | +92% (67 to 129 ms) | +73% (11.8 to 20.3 ms) | +73% |
| inf | 0.84x (12.03 to 10.07) | -37% (6,737 to 4,248 ms) | +126% (28.6 to 64.5 ms) | +48% |

## Against the published run

| Saturated requests a second, Glyd against bf16 | published | here |
| :--- | :--- | :--- |
| Qwen3-30B-A3B | 0.95x (12.16 to 11.58) | 0.84x (12.03 to 10.07) |

The rate-1 pass is the first on its server, with no earlier prompts to repeat: Glyd's first token +39% then (60 to 82 ms) and +35% here (62 to 84 ms), each token +8% then (6.7 to 7.2 ms) and +5% here.

## Files

`env.txt`, `machine.txt`, `machine-short.txt`, `steps.txt` (each step's time and its prefix cache hit rate line), `summary.txt`, the console (`bench-Qwen3-30B-A3B.txt`) and `bench-Qwen3-30B-A3B/` (per mode and rate: vLLM's
result JSON and console output, nvidia-smi's samples each second of a rate; each server's log and KV cache lines, cold and warm; `summary.txt`).
