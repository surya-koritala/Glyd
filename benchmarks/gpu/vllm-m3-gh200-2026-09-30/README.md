# vLLM with `--quantization glyd` on a GH200 (M3, 2026-09-30)

M3's unattended job (`vllm_job.sh`) on one NVIDIA GH200 (Hopper, aarch64): the M2 checks against vLLM's own bf16,
then `vllm bench serve` bf16 against Glyd on Qwen3-8B and on Qwen3-32B.

## Setup

- **Machine:** NVIDIA GH200 480GB (97,871 MiB of HBM, 900 W, 1,980 MHz at most), driver 580.126.20. Neoverse-V2, 64
  CPUs, 525 GB of memory, aarch64. Region us-east-3; 57 minutes for the three jobs.
- **Software:** the job's tarball at 766746a, main's v0.25.1 with the plugin. vLLM 0.30.0 came from PyPI in the job,
  aarch64 wheels (torch 2.13.0+cu130, transformers 5.17.0). The library was built there for sm_90a (C API 5; GPU code
  90).
- **Run:** by the coordinator's `run.sh`, one job a step on one instance: check, bench, big.
  - vLLM and nvcc installed in 22 s, and the library built in 37 s.
  - Qwen3-8B (16.4 GB) downloaded in 19 s, and Qwen3-32B (65.5 GB) in 53 s.
- **Glyd's layout:** the one `best_layout` picks on Hopper, 12-bit.

## Glyd against bf16, by GPU and load

Everything measured with `vllm bench serve` so far, M2's L4 and M3's A10, A100 and GH200. Each server was started warm,
on the compile cache its first (cold) start filled, at `--gpu-memory-utilization 0.9` for both. The random dataset was
1,024 tokens in and 256 out. A ratio or a percentage is Glyd's against bf16's.

| GPU (Glyd's layout) | Model | KV cache, warm | Load (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| L4 (M2, the smallest layout (`mma`), v0.25.0) | Qwen3-8B | 1.94x | 0.25 | 1.03x (0.22 to 0.23) | +30% (458 to 594 ms) | -22% (69.9 to 54.8 ms) | -20% |
| L4 (M2, `mma`, v0.25.0) | Qwen3-8B | 1.94x | 1 | 1.15x (0.67 to 0.77) | -73% (3,213 to 870 ms) | +1% (99.3 to 100.1 ms) | -7% |
| L4 (M2, `mma`, v0.25.0) | Qwen3-8B | 1.94x | inf | 1.29x (0.77 to 0.99) | -23% (148,852 to 115,115 ms) | +48% (109.6 to 162.1 ms) | -12% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 1 | 1.04x (0.85 to 0.88) | +16% (384 to 445 ms) | -21% (50.9 to 40.4 ms) | -20% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 4 | 1.22x (1.24 to 1.51) | -39% (27,898 to 17,134 ms) | +32% (66.8 to 88.5 ms) | -12% |
| A10 (12-bit) | Qwen3-8B | 1.73x | inf | 1.31x (1.24 to 1.62) | -25% (93,052 to 70,196 ms) | +30% (66.1 to 85.7 ms) | -16% |
| A100 40 GB (12-bit) | Qwen3-8B | 1.14x | 1 | 1.01x (0.95 to 0.95) | +18% (106 to 125 ms) | -10% (14.7 to 13.2 ms) | -10% |
| A100 40 GB (12-bit) | Qwen3-8B | 1.14x | 4 | 1.01x (3.56 to 3.59) | +15% (97 to 112 ms) | -2% (19.4 to 18.9 ms) | -2% |
| A100 40 GB (12-bit) | Qwen3-8B | 1.14x | inf | 1.19x (6.36 to 7.55) | -32% (13,452 to 9,098 ms) | -2% (58.7 to 57.5 ms) | -16% |
| A100 40 GB (12-bit) | Qwen3-14B | 1.77x | 1 | 1.03x (0.84 to 0.86) | +18% (209 to 248 ms) | -13% (26.3 to 22.8 ms) | -12% |
| A100 40 GB (12-bit) | Qwen3-14B | 1.77x | 4 | 1.15x (2.38 to 2.73) | -30% (418 to 292 ms) | -12% (37.8 to 33.2 ms) | -13% |
| A100 40 GB (12-bit) | Qwen3-14B | 1.77x | inf | 1.65x (2.65 to 4.37) | -57% (17,726 to 7,636 ms) | +4% (48.4 to 50.2 ms) | -32% |
| **GH200 (12-bit)** | Qwen3-8B | 1.04x | 1 | 1.00x (0.98 to 0.98) | +4% (45 to 46 ms) | +1% (6.0 to 6.1 ms) | +1% |
| **GH200 (12-bit)** | Qwen3-8B | 1.04x | 4 | 1.00x (3.81 to 3.80) | +6% (37 to 39 ms) | +6% (6.7 to 7.0 ms) | +6% |
| **GH200 (12-bit)** | Qwen3-8B | 1.04x | inf | 0.92x (27.60 to 25.43) | +4% (1,448 to 1,501 ms) | +9% (29.7 to 32.5 ms) | +9% |
| **GH200 (12-bit)** | Qwen3-32B | 1.66x | 1 | 1.01x (0.85 to 0.86) | +28% (178 to 228 ms) | -6% (23.1 to 21.7 ms) | -5% |
| **GH200 (12-bit)** | Qwen3-32B | 1.66x | 4 | 0.96x (2.85 to 2.75) | +37% (180 to 246 ms) | +14% (28.8 to 32.7 ms) | +14% |
| **GH200 (12-bit)** | Qwen3-32B | 1.66x | inf | 0.88x (5.55 to 4.89) | -52% (6,342 to 3,047 ms) | +82% (37.7 to 68.4 ms) | +28% |

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

## Check: all 15 passed (`check/`, 1,093 s)

| Qwen3-8B, against vLLM's bf16 | Glyd `mma` | Glyd 12-bit |
| :--- | ---: | ---: |
| Packs unpacked to their weights, bit for bit | 144 of 144 | 144 of 144 |
| Worst layer against F.linear, 1-4,096 tokens | 3.78e-3 | 3.78e-3 |
| Top-1 / \|Δ\| on bf16's continuation (bf16 eager's: 0.9948 / 3.46e-3) | 0.9955 / 0.00e+00 | 0.9974 / 3.48e-3 |

- **Compile cache:** a graph each for bf16 and the two layouts; each layout started again loaded its own.
- **Exact eager:** bf16 eager's bits (8 of 8 prompts, and the continuation).
- **Exact compiled:** refused. With inductor's deterministic mode, compiled bf16's bits (8 of 8, and the
  continuation).
- **Default mode compiled in that mode:** the same bits across a restart on its graphs.

## Bench: Qwen3-8B (`bench/`, 698 s)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 489,328 tokens | 119.46x | 477,456 tokens |
| Glyd 12-bit | 12.53 GiB | 509,056 tokens (1.04x) | 124.28x | 497,200 tokens (1.04x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.98 | 249.9 | 45 / 71 | 6.0 / 6.4 | 5.9 / 6 | 1980 MHz, 53 C |
| 1 | glyd | 0.98 | 249.9 | 46 / 66 | 6.1 / 6.6 | 5.9 / 10 | 1980 MHz, 54 C |
| 4 | bf16 | 3.81 | 974.7 | 37 / 77 | 6.7 / 7.8 | 6.3 / 33 | 1980 MHz, 55 C |
| 4 | glyd | 3.80 | 973.4 | 39 / 76 | 7.0 / 8.4 | 6.5 / 30 | 1980 MHz, 57 C |
| inf | bf16 | 27.60 | 7066.5 | 1,448 / 3,952 | 29.7 / 33.4 | 21.8 / 212 | 1575 MHz, 60 C |
| inf | glyd | 25.43 | 6508.8 | 1,501 / 4,116 | 32.5 / 36.4 | 24.3 / 249 | 1830 MHz, 59 C |

The prompts: 64 at 1 request a second, 128 at 4, 256 at once. Every request completed. Neither mode was short of KV
cache: at once, at most 48% of bf16's was in use and 55% of Glyd's (the servers' logs, every 10 s). At saturation, while
it worked, the GPU drew a median of 659 W with bf16 and 661 W with Glyd, of its 900 W.

## Big: Qwen3-32B (`big/`, 1,063 s)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 61.03 GiB | 79,760 tokens | 19.47x | 73,456 tokens |
| Glyd 12-bit | 48.27 GiB | 132,144 tokens (1.66x) | 32.26x | 125,840 tokens (1.71x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.85 | 218.3 | 178 / 429 | 23.1 / 24.4 | 21.3 / 106 | 1980 MHz, 59 C |
| 1 | glyd | 0.86 | 221.4 | 228 / 547 | 21.7 / 23.4 | 19.3 / 129 | 1980 MHz, 58 C |
| 4 | bf16 | 2.85 | 730.5 | 180 / 560 | 28.8 / 35.9 | 23.5 / 206 | 1980 MHz, 60 C |
| 4 | glyd | 2.75 | 704.5 | 246 / 870 | 32.7 / 42.9 | 26.5 / 216 | 1980 MHz, 60 C |
| inf | bf16 | 5.55 | 1421.5 | 6,342 / 16,097 | 37.7 / 79.0 | 27.8 / 263 | 1950 MHz, 61 C |
| inf | glyd | 4.89 | 1252.5 | 3,047 / 7,541 | 68.4 / 76.1 | 40.4 / 889 | 1875 MHz, 61 C |

The prompts: 32 at 1 request a second, 64 at 4, 128 at once. Every request completed. bf16's KV cache holds about 62
requests of 1,280 tokens and Glyd's about 103. At once (the servers' logs, every 10 s), bf16 ran at most 69 requests
together with up to 59 waiting, and Glyd 107 with up to 23 waiting. Glyd's median inter-token latency, a step's time,
was 40.4 ms against 27.8. At saturation, while it worked, the GPU drew a median of 667 W with bf16 and 662 W with Glyd.
What Glyd's kernels cost there
is not profiled here.
