# vLLM with `--quantization glyd` on an A100 40 GB (M3, 2026-09-30)

M3's unattended job (`vllm_job.sh`) on one NVIDIA A100-SXM4-40GB: the M2 checks against vLLM's own bf16, then `vllm
bench serve` bf16 against Glyd on Qwen3-8B and on Qwen3-14B. Qwen3-32B does not fit 40 GB packed either: its Linears
are about 39 GiB in the smallest layout (`mma`) and 44 GiB in 12-bit, with 2.9 GiB of embeddings and LM head on top.

## Setup

- **Machine:** NVIDIA A100-SXM4-40GB (400 W, 1,410 MHz), driver 580.126.20. AMD EPYC 7J13, 30 vCPUs, 216 GB of
  memory, x86_64.
- **Software:** the job's tarball at 766746a, main's v0.25.1 with the plugin. vLLM 0.30.0 came from PyPI in the job
  (torch 2.13.0+cu130, transformers 5.17.0). The library was built there for sm_80 (C API 5; GPU code 80).
- **Run:** by the coordinator's `run.sh`, one job a step on one instance: check, bench, big.
  - vLLM and nvcc installed in 31 s, and the library built in 51 s.
  - Qwen3-8B (16.4 GB) downloaded in 22 s, and Qwen3-14B (29.6 GB) in 28 s.
- **Glyd's layout:** the one `best_layout` picks on an A100, 12-bit.

## Glyd against bf16, by GPU and load

Everything measured with `vllm bench serve` so far, M2's L4 and M3's A10, A100 and GH200. Each server was started warm,
on the compile cache its first (cold) start filled, at `--gpu-memory-utilization 0.9` for both. The random dataset was
1,024 tokens in and 256 out. A ratio or a percentage is Glyd's against bf16's.

| GPU (Glyd's layout) | Model | KV cache, warm | Load (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| L4 (M2, `mma`, v0.25.0) | Qwen3-8B | 1.94x | 0.25 | 1.03x (0.22 to 0.23) | +30% (458 to 594 ms) | -22% (69.9 to 54.8 ms) | -20% |
| L4 (M2, `mma`, v0.25.0) | Qwen3-8B | 1.94x | 1 | 1.15x (0.67 to 0.77) | -73% (3,213 to 870 ms) | +1% (99.3 to 100.1 ms) | -7% |
| L4 (M2, `mma`, v0.25.0) | Qwen3-8B | 1.94x | inf | 1.29x (0.77 to 0.99) | -23% (148,852 to 115,115 ms) | +48% (109.6 to 162.1 ms) | -12% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 1 | 1.04x (0.85 to 0.88) | +16% (384 to 445 ms) | -21% (50.9 to 40.4 ms) | -20% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 4 | 1.22x (1.24 to 1.51) | -39% (27,898 to 17,134 ms) | +32% (66.8 to 88.5 ms) | -12% |
| A10 (12-bit) | Qwen3-8B | 1.73x | inf | 1.31x (1.24 to 1.62) | -25% (93,052 to 70,196 ms) | +30% (66.1 to 85.7 ms) | -16% |
| **A100 40 GB (12-bit)** | Qwen3-8B | 1.14x | 1 | 1.01x (0.95 to 0.95) | +18% (106 to 125 ms) | -10% (14.7 to 13.2 ms) | -10% |
| **A100 40 GB (12-bit)** | Qwen3-8B | 1.14x | 4 | 1.01x (3.56 to 3.59) | +15% (97 to 112 ms) | -2% (19.4 to 18.9 ms) | -2% |
| **A100 40 GB (12-bit)** | Qwen3-8B | 1.14x | inf | 1.19x (6.36 to 7.55) | -32% (13,452 to 9,098 ms) | -2% (58.7 to 57.5 ms) | -16% |
| **A100 40 GB (12-bit)** | Qwen3-14B | 1.77x | 1 | 1.03x (0.84 to 0.86) | +18% (209 to 248 ms) | -13% (26.3 to 22.8 ms) | -12% |
| **A100 40 GB (12-bit)** | Qwen3-14B | 1.77x | 4 | 1.15x (2.38 to 2.73) | -30% (418 to 292 ms) | -12% (37.8 to 33.2 ms) | -13% |
| **A100 40 GB (12-bit)** | Qwen3-14B | 1.77x | inf | 1.65x (2.65 to 4.37) | -57% (17,726 to 7,636 ms) | +4% (48.4 to 50.2 ms) | -32% |
| GH200 (12-bit) | Qwen3-8B | 1.04x | 1 | 1.00x (0.98 to 0.98) | +4% (45 to 46 ms) | +1% (6.0 to 6.1 ms) | +1% |
| GH200 (12-bit) | Qwen3-8B | 1.04x | 4 | 1.00x (3.81 to 3.80) | +6% (37 to 39 ms) | +6% (6.7 to 7.0 ms) | +6% |
| GH200 (12-bit) | Qwen3-8B | 1.04x | inf | 0.92x (27.60 to 25.43) | +4% (1,448 to 1,501 ms) | +9% (29.7 to 32.5 ms) | +9% |
| GH200 (12-bit) | Qwen3-32B | 1.66x | 1 | 1.01x (0.85 to 0.86) | +28% (178 to 228 ms) | -6% (23.1 to 21.7 ms) | -5% |
| GH200 (12-bit) | Qwen3-32B | 1.66x | 4 | 0.96x (2.85 to 2.75) | +37% (180 to 246 ms) | +14% (28.8 to 32.7 ms) | +14% |
| GH200 (12-bit) | Qwen3-32B | 1.66x | inf | 0.88x (5.55 to 4.89) | -52% (6,342 to 3,047 ms) | +82% (37.7 to 68.4 ms) | +28% |

The runs are in `../l4-vllm-m2-2026-09-29`, `../vllm-m3-a10-2026-09-30`, `../vllm-m3-a100-40gb-2026-09-30` and
`../vllm-m3-gh200-2026-09-30`.

- **Wins:**
  - Capacity: 1.04-1.94x the KV cache. The low end is Qwen3-8B on the 96 GB GH200, which bf16 barely fills; Qwen3-32B
    there gets 1.66x.
  - Requests a second where bf16 runs short of KV cache (the L4, A10 and A100): 1.15-1.65x.
  - The first token once bf16 queues: 23-73% sooner on the L4, A10 and A100, and 52% on the GH200 with Qwen3-32B.
  - The time per output token below saturation: 2-22% less on the L4, A10 and A100, and 6% on the GH200 with
    Qwen3-32B at 1 request a second.
  - Each request's whole time on the L4, A10 and A100: 2-32% less at every load measured.
- **Losses:**
  - **The first token below saturation:** 15-18% later on the A10 and A100 (1 and 4 requests a second, where bf16 is
    not queueing), 30% on the L4 at 0.25. On the GH200: 4-6% later with Qwen3-8B, 28-37% with Qwen3-32B.
  - **On the GH200 at saturation:**
    - requests a second: 0.92x with Qwen3-8B and 0.88x with Qwen3-32B;
    - the time per output token: 9% and 82% more;
    - each request's whole time: 9% and 28% more;
    - the GPU's power the same (a median 659-667 W while it worked), and its SM clock 1,830 MHz against bf16's 1,575 with
      Qwen3-8B, 1,875 against 1,950 with Qwen3-32B.

    Hopper's gap is not profiled here.
  - **On the GH200 below saturation, Qwen3-8B:** parity in requests a second at 1 and 4 a second, with the time per
    output token 1-6% more.
  - **The time per output token at saturation on the other GPUs:** 30-32% more on the A10 and 48% on the L4, where
    each step carries more requests; within 4% on the A100.
  - **The inter-token p99** (the steps with a prompt in them): 13-25% more on the A10 and A100.

The SM clock in the tables is nvidia-smi's median over the samples while the GPU worked (`bench_summary.py`: above the
midpoint of the least and most power drawn), recomputed from the logs after the runs; the jobs' own `summary.txt` and
console files keep the median over every sample, idle ones included.

## Check: all 15 passed (`check/`, 1,057 s)

| Qwen3-8B, against vLLM's bf16 | Glyd `mma` | Glyd 12-bit |
| :--- | ---: | ---: |
| Packs unpacked to their weights, bit for bit | 144 of 144 | 144 of 144 |
| Worst layer against F.linear, 1-4,096 tokens | 3.69e-3 | 3.90e-3 |
| Top-1 / \|Δ\| on bf16's continuation (bf16 eager's: 0.9903 / 8.26e-3) | 0.9883 / 7.31e-3 | 0.9903 / 6.94e-3 |

- **Compile cache:** a graph each for bf16 and the two layouts; each layout started again loaded its own.
- **Exact eager:** bf16 eager's bits (8 of 8 prompts, and the continuation).
- **Exact compiled:** refused. With inductor's deterministic mode, compiled bf16's bits (8 of 8, and the
  continuation).
- **Default mode compiled in that mode:** the same bits across a restart on its graphs.

## Bench: Qwen3-8B (`bench/`, 738 s)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 141,040 tokens | 34.43x | 133,776 tokens |
| Glyd 12-bit | 12.53 GiB | 160,720 tokens (1.14x) | 39.24x | 153,472 tokens (1.15x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.95 | 242.1 | 106 / 166 | 14.7 / 16.2 | 13.9 / 65 | 1410 MHz, 56 C |
| 1 | glyd | 0.95 | 243.6 | 125 / 211 | 13.2 / 15.2 | 12.1 / 77 | 1410 MHz, 55 C |
| 4 | bf16 | 3.56 | 912.0 | 97 / 257 | 19.4 / 26.6 | 16.2 / 80 | 1410 MHz, 58 C |
| 4 | glyd | 3.59 | 919.8 | 112 / 367 | 18.9 / 28.2 | 14.6 / 98 | 1410 MHz, 60 C |
| inf | bf16 | 6.36 | 1627.0 | 13,452 / 35,863 | 58.7 / 103.4 | 32.1 / 186 | 1350 MHz, 64 C |
| inf | glyd | 7.55 | 1931.8 | 9,098 / 25,869 | 57.5 / 110.5 | 41.5 / 210 | 1395 MHz, 64 C |

The prompts: 64 at 1 request a second, 128 at 4, 256 at once. Every request completed. Neither mode queued at 1 and 4
requests a second: the KV cache holds about 110 requests of 1,280 tokens for bf16 and 125 for Glyd.

## Big: Qwen3-14B (`big/`, 702 s)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 27.52 GiB | 45,744 tokens | 11.17x | 37,728 tokens |
| Glyd 12-bit | 21.98 GiB | 80,960 tokens (1.77x) | 19.77x | 63,568 tokens (1.68x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.84 | 214.4 | 209 / 442 | 26.3 / 27.8 | 24.0 / 134 | 1410 MHz, 55 C |
| 1 | glyd | 0.86 | 220.1 | 248 / 540 | 22.8 / 25.0 | 20.0 / 164 | 1410 MHz, 56 C |
| 4 | bf16 | 2.38 | 609.4 | 418 / 3,763 | 37.8 / 47.0 | 30.9 / 227 | 1410 MHz, 59 C |
| 4 | glyd | 2.73 | 698.2 | 292 / 1,012 | 33.2 / 44.5 | 26.2 / 284 | 1410 MHz, 60 C |
| inf | bf16 | 2.65 | 679.7 | 17,726 / 41,718 | 48.4 / 78.2 | 32.8 / 250 | 1410 MHz, 64 C |
| inf | glyd | 4.37 | 1118.9 | 7,636 / 21,584 | 50.2 / 97.3 | 32.6 / 303 | 1410 MHz, 63 C |

The prompts: 32 at 1 request a second, 64 at 4, 128 at once. Every request completed. bf16's KV cache holds about 36
requests of 1,280 tokens, Glyd's about 63: at 4 a second bf16 began to queue (its TTFT p99 3.8 s against Glyd's 1.0),
and at saturation Glyd served 1.65x the requests a second.
