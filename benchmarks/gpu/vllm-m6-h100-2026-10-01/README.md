# vLLM with `--quantization glyd` on an H100 SXM (M6, 2026-10-01)

The plugin's checks against vLLM's own bf16 on Qwen3-8B, a mixture of experts' layer by tokens a step on Qwen3-30B-A3B,
and `vllm bench serve` bf16 against Glyd on Qwen3-30B-A3B, on one NVIDIA H100 SXM, from the v0.26.0 candidate. They are
three of the seven jobs of the session's 68 minutes; the others are in
`../option2-2026-09-29/h100-sxm-measure` (the long-prompt mode, Qwen3-14B),
`../l4-vllm-fraction-2026-09-30/h100-sxm` (`fraction`, Qwen3-32B) and `../repro-2026-09-30/h100-sxm` (bf16's
repeatability); the KV-cache job is in the `kv-study` branch's `research/kv-cache`. The Hopper row of M3 was a GH200;
this is the first run through vLLM on an H100.

## Setup

- **Machine:** NVIDIA H100 80GB HBM3, the SXM5 (81,559 MiB of HBM3, power limit 700 W, 1,980 MHz at most), driver
  580.126.20. Xeon Platinum 8480+ (26 CPUs), x86_64. Lambda `gpu_1x_h100_sxm5`; the session ran 00:27-01:33 UTC
  (`session/`).
- **Software:** the tree 7fe66a2, release-0.26.0 (the v0.26.0 candidate: C API 7, the plugin with `fraction`); vLLM 0.30.0
  from PyPI (torch 2.13.0+cu130, transformers 5.18.0, nvcc 13.0, in the job); the library built there for sm_90a (GPU code
  90: this tree had no H100 class). The plugin runs the library as v0.25.1's: the long-prompt mode is opt-in, and the plugin
  does not ask for it.
- **Run:** `session/hop4_all.sh` ran the session's jobs one after another; each of these three is `vllm_job.sh` (as it
  ran) with its steps in `VJ_STEPS`: `check` (Qwen3-8B), `moeroutes` and `moebench` (Qwen3-30B-A3B).
  - The first lines of each job's log show two shell errors, `line 98: 30: command not found` and `line 99: [: :
    integer expression expected`, from the script's disk-space test (its `NEED` sum): the test failed false and skipped
    nothing. No step was affected.
  - Qwen3-8B (16.4 GB) downloaded in 18 s, Qwen3-30B-A3B (61.1 GB) in 41 s.
- **Glyd's layout:** the 12-bit one (`best_layout` on Hopper); the check ran both layouts.

## Check: all 15 passed (`check/`, 696 s)

`check_vllm.py --quick` on Qwen3-8B (TP 1):

| Qwen3-8B, against vLLM's bf16 | Glyd, the smallest layout (`mma`) | Glyd 12-bit |
| :--- | ---: | ---: |
| Packs unpacked to their weights, bit for bit | 144 of 144 | 144 of 144 |
| Worst layer against F.linear, 1-4,096 tokens | 3.78e-3 | 3.78e-3 |
| Top-1 on bf16's continuation (bf16 eager's: 0.9922) | 0.9922 | 0.9935 |
| Mean \|Δ logprob\| (bf16 eager's against its graphs: 9.25e-3) | 8.12e-3 | 7.32e-3 |
| KV cache at `--gpu-memory-utilization 0.85` (bf16: 354,496 tokens) | 379,024 tokens (1.07x) | 374,224 tokens (1.06x) |

- **Compile cache:** a graph each for bf16 and the two layouts; each layout started again loaded its own.
- **Exact eager:** bf16 eager's tokens, logprobs and prompt_logprobs bit for bit (8 of 8 prompts, and the continuation).
- **Exact compiled:** refused, with Glyd's message. With inductor's deterministic mode, compiled bf16's bits (8 of 8, and
  the continuation).
- **Default mode compiled in that mode:** the same bits across a restart on its graphs (12-bit layout).

## A mixture of experts' layer by tokens a step (`moe-routes/`, 112 s)

`moe_routes.py` on Qwen3-30B-A3B's first MoE layer (128 experts, 8 a token, hidden 2,048, intermediate 768), one GPU:
T random tokens each routed to 8 experts at random, the GPU time of a call (the median of 20 after a warm-up), the Glyd
layer's two settings of `GLYD_MOE_DECODE_MIN` (`-1`: the first way to run the experts throughout, `1`: the second; the
plugin's default switches to the second at 1,152 tokens) against vLLM's own kernel on the bf16 experts.

| Tokens a step | Setting -1 (ms) | Setting 1 (ms) | Setting 1 against -1 | bf16's layer (ms) | Setting -1 against bf16's | Setting 1 against bf16's |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.075 | 0.275 | 3.68x | 0.207 | 0.36x | 1.33x |
| 2 | 0.097 | 0.264 | 2.73x | 0.200 | 0.48x | 1.32x |
| 4 | 0.148 | 0.396 | 2.67x | 0.233 | 0.63x | 1.70x |
| 8 | 0.207 | 0.648 | 3.13x | 0.320 | 0.65x | 2.03x |
| 16 | 0.279 | 0.957 | 3.43x | 0.402 | 0.69x | 2.38x |
| 32 | 0.535 | 1.339 | 2.50x | 0.518 | 1.03x | 2.59x |
| 64 | 0.709 | 1.453 | 2.05x | 0.558 | 1.27x | 2.61x |
| 128 | 0.743 | 1.502 | 2.02x | 0.590 | 1.26x | 2.55x |
| 256 | 0.800 | 1.525 | 1.90x | 0.597 | 1.34x | 2.55x |
| 512 | 0.917 | 1.569 | 1.71x | 0.612 | 1.50x | 2.56x |
| 1,024 | 1.452 | 1.668 | 1.15x | 0.651 | 2.23x | 2.56x |
| 2,048 | 1.948 | 1.896 | 0.97x | 0.749 | 2.60x | 2.53x |
| 4,096 | 3.567 | 2.551 | 0.72x | 1.030 | 3.46x | 2.48x |

- **Setting 1 is the faster from 2,048 tokens a step here**, between 1,024 (where it took 1.15x setting -1's time) and
  2,048 (0.97x), against the plugin's switch at 1,152 tokens, which an L4 and granite-3.1-3b-a800m-instruct set
  (`../l4-vllm-moe-routes-2026-09-30`). From 1,152 tokens to the crossing, which 1,152 and 1,536 were not run to place,
  the plugin takes the second way where the first is probably the faster.
- **Setting -1 against bf16's layer:** faster to 16 tokens a step (0.36-0.69x), the same at 32 (1.03x), slower from 64
  (1.27x, 3.46x at 4,096). Setting 1 never beats bf16's layer (2.5-2.6x from 32 tokens).
- **The two settings' outputs** differed by at most 5.6e-3 (relative).

## Bench: Qwen3-30B-A3B, bf16 and Glyd (`moe-bench/`, 649 s)

`vllm bench serve`, each server started twice, on an empty compile cache and then warm (the warm one measured), the same
`--gpu-memory-utilization 0.9` for both, the random dataset of 1,024 tokens in and 256 out: 64 prompts at 1 request a
second, 128 at 4 and 256 at once. One run each.

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 56.88 GiB | 124,720 tokens | 30.45x | 118,720 tokens |
| Glyd 12-bit | 44.27 GiB | 263,376 tokens (2.11x) | 64.30x | 257,392 tokens (2.17x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.97 | 249.2 | 60 / 258 | 6.7 / 8.6 | 1980 MHz, 40 C |
| 1 | glyd | 0.97 | 248.4 | 82 / 124 | 7.2 / 10.0 | 1980 MHz, 38 C |
| 4 | bf16 | 3.75 | 959.2 | 49 / 100 | 11.2 / 13.5 | 1980 MHz, 44 C |
| 4 | glyd | 3.70 | 946.4 | 102 / 746 | 17.6 / 28.9 | 1980 MHz, 44 C |
| inf | bf16 | 12.16 | 3113.7 | 6,772 / 16,912 | 28.6 / 51.9 | 1980 MHz, 55 C |
| inf | glyd | 11.58 | 2963.5 | 1,766 / 17,386 | 62.4 / 70.1 | 1980 MHz, 50 C |

Glyd against bf16 (each rate's means from the runs' JSON):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | A request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.00x (0.974 to 0.970) | +39% (60 to 82 ms) | +8% (6.7 to 7.2 ms) | +9% |
| 4 | 0.99x (3.75 to 3.70) | +106% (49 to 102 ms) | +58% (11.2 to 17.6 ms) | +58% |
| inf | 0.95x (12.16 to 11.58) | -74% (6,772 to 1,766 ms) | +118% (28.6 to 62.4 ms) | +26% |

Every request completed. At once (the servers' logs, every 10 s), bf16's KV cache filled: it ran 104 requests with 152
waiting, then 99 with 39 (its KV cache 99.5-100% used), and Glyd ran 174 with 82 waiting, then 220 with 36. At
saturation, while it worked, the GPU drew a median of 582 W with bf16 and 457 W with Glyd, of its 700 W, at 1,980 MHz in
both (nvidia-smi's samples above the midpoint of the least and the most power drawn, `bench_summary.py`'s way).

- **Capacity:** 2.11x the KV cache, 56.88 to 44.27 GiB of weights.
- **Saturated:** 0.95x bf16's requests a second although bf16 ran short of KV cache, the first token 74% sooner on
  average (p99 the same: 16.9 and 17.4 s) and each token 118% later, with up to 220 requests in a step against 104.
- **Below saturation:** the same requests a second at 1 a second with the first token 39% later and each token 8%; at 4 a
  second 0.99x, the first token and each token later by 106% and 58%.
- **What the layer's table says about it:** past 32 tokens a step a Glyd layer takes 1.03-3.5x bf16's layer time.
  Where the time goes in a step is not profiled here.

## Files

- `check/`: `check-Qwen3-8B.txt` (the console), `check-Qwen3-8B/` (`report.txt`, each run's JSON and log), `summary.txt`,
  `steps.txt`, `machine.txt`, `machine-short.txt`, `env.txt`, `job.log`, `log/` (the downloads, the environment).
- `moe-routes/`: `moeroutes/{glyd,bf16}.{txt,json}`, `summary.txt`, `steps.txt` and the rest as above.
- `moe-bench/`: `bench-Qwen3-30B-A3B.txt` (the console), `bench-Qwen3-30B-A3B/` (per mode and rate: the server's log
  `serve-MODE[-cold].txt`, `kv-MODE[-cold].txt`, `MODE-rateR.json`, `bench-MODE-rateR.txt`, `smi-MODE-rateR.csv`, then
  `summary.txt`), and the rest as above.
- `session/`: `hop4_all.sh` (the session's driver), `steps.txt` (each job's start and exit), `gpu.txt`.
- `vllm_job.sh`: the job as it ran.
