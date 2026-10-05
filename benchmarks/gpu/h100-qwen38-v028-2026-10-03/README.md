# Qwen3.8-27B on an NVIDIA H100 80GB HBM3: bf16 against Glyd 0.27.0 and Glyd 0.28, vLLM 0.30.0 (2026-10-03)

`vllm serve Qwen/Qwen3.8-27B` (text only: `--language-model-only`, the vision tower is not loaded) on one GPU, four servers one after another on the same machine: vLLM's own bf16,
Glyd 0.28 (`--quantization glyd`; the embedding and the output layer packed as well as the layers' Linears), Glyd 0.27.0 as `pip install "glyd[vllm]"` gives it, and bf16 again at the
end, so that a machine that drifts during the run shows as the two bf16 runs apart. Each server is measured with `vllm bench serve` and a quality probe.

## Setup

- **Machine:** NVIDIA H100 80GB HBM3 (81559 MiB, top SM clock 1980 MHz, 700.00 W power limit), driver 580.126.09; Intel(R) Xeon(R) Platinum 8480+, 224 vCPUs, 2015 GB of memory, x86_64.
- **Software:** vllm 0.30.0; torch 2.13.0+cu130 CUDA 13.0; transformers 5.18.0; flashinfer 0.6.18.post1; glyd 0.27.0; glyd-gpu 0.27.0; GPU NVIDIA H100 80GB HBM3; nvcc: Build cuda_13.0.r13.0/compiler.36424714_0.
  Glyd 0.27.0: `glyd[vllm]==0.27.0` from PyPI (its own library). Glyd 0.28: the code of 0.28.0rc1 (the version strings aside), built for this GPU.
- **Run:** `--gpu-memory-utilization 0.9`, `--max-model-len 4096`, vLLM's own defaults otherwise (prefix caching on); each server on a compile cache of its own (bf16 again starts on bf16's). `vllm bench serve`, the random dataset of 1,024 tokens in and 256 out (`--ignore-eos`), 64 prompts at 1 request a second and 256 prompts at once; a seed of its own for each pass, the same prompts for every server, none of them any earlier pass's on a server. The quality probe runs before a server's first pass.
- **Glyd's layout:** `mma12` (`auto` on this GPU). **KV cache:** `kv` auto, where Glyd's lossless KV cache is on for the GPUs it is measured on; for this model it stays off in both Glyd versions: `the KV cache stays vLLM's (kv auto): sliding window, linear attention and state-space layers are not held; head size 256: 128 only`.

## Can the servers be compared?

**NOT COMPARABLE: the passes' median clocks differ by 9% (1800 to 1980 MHz), over 5%**

The saturated Glyd passes drew more power (696 W against 664 and 676) and the GPU held them to its power limit at a lower
clock (1,800 and 1,852 MHz against bf16's 1,980); the numbers below are as measured.

The guard: not comparable where a saturated pass's median SM clock is under 86% of the GPU's top clock, the passes' medians differ by more than 5%, a pass was over 85 C, bf16's two runs differ by more than 5% in requests a second, or a server's prefix cache hit rate is above 1%; a GPU that throttles under 40 s of matmuls before the first server ends the run.

```
| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 |  | 1980 | 1230 | 72 | 493 | 132 |
| bf16 | inf | 4.72 | 1980 | 1335 | 79 | 664 | 53 |
| bf16-2 | 1 |  | 1980 | 1425 | 74 | 495 | 133 |
| bf16-2 | inf | 4.70 | 1980 | 1350 | 80 | 676 | 53 |
| v027 | 1 |  | 1980 | 1200 | 77 | 547 | 131 |
| v027 | inf | 5.05 | 1800 | 1380 | 79 | 696 | 50 |
| v028 | 1 |  | 1980 | 1380 | 78 | 549 | 131 |
| v028 | inf | 5.05 | 1852 | 1380 | 80 | 696 | 50 |
bf16-2: the driver's slowdown reasons, seconds active of 310: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 53, hardware power brake 0
bf16: the driver's slowdown reasons, seconds active of 359: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 55, hardware power brake 0
v027: the driver's slowdown reasons, seconds active of 395: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 77, hardware power brake 0
v028: the driver's slowdown reasons, seconds active of 395: hardware thermal slowdown 0, software thermal slowdown 0, software power cap 77, hardware power brake 0
```

## Weights and KV cache, as vLLM logs them

| Server | Weights (GiB) | against bf16 | KV cache (tokens) | against bf16 | Maximum concurrency at 4,096 tokens | GPU memory in use (GiB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 50.22 |  | 122,538 |  | 29.92x | 69.1 |
| Glyd 0.28 | 38.77 | 0.772x | 204,117 | 1.67x | 49.83x | 69.1 |
| Glyd 0.27.0 | 39.69 | 0.790x | 197,632 | 1.61x | 48.25x | 69.1 |
| bf16 again | 50.22 | 1.000x | 132,778 | 1.08x | 32.42x | 70.5 |

Whole-model weights, bf16 50.22 GiB; Glyd 0.28 38.77 GiB (-22.8%); Glyd 0.27.0 39.69 GiB (-21.0%).

## Requests, tokens and token times

Random prompts of 1,024 tokens, 256 out, 1 request a second (64 prompts):

| Server | Requests/s | against bf16 | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 0.92 |  | 234.8 | 199 / 326 | 23.4 / 25.3 | 64 | 1980 MHz, 72 C |
| Glyd 0.28 | 0.92 | 1.01x | 236.4 | 227 / 426 | 21.7 / 24.4 | 64 | 1980 MHz, 78 C |
| Glyd 0.27.0 | 0.92 | 1.01x | 236.6 | 225 / 421 | 21.6 / 24.4 | 64 | 1980 MHz, 77 C |
| bf16 again | 0.92 | 1.00x | 234.4 | 195 / 330 | 23.8 / 25.6 | 64 | 1980 MHz, 74 C |

Random prompts of 1,024 tokens, 256 out, every request at once (256 prompts):

| Server | Requests/s | against bf16 | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 4.72 |  | 1207.6 | 22972 / 47234 | 46.4 / 54.7 | 256 | 1980 MHz, 79 C |
| Glyd 0.28 | 5.05 | 1.07x | 1293.5 | 19394 / 44830 | 66.2 / 81.1 | 256 | 1860 MHz, 80 C |
| Glyd 0.27.0 | 5.05 | 1.07x | 1292.0 | 19784 / 44663 | 64.7 / 79.0 | 256 | 1800 MHz, 79 C |
| bf16 again | 4.70 | 1.00x | 1203.4 | 22147 / 47986 | 48.8 / 58.3 | 256 | 1980 MHz, 80 C |

Prefix cache hit rate, at most (each server's log, every 10 s): bf16 0.0%, Glyd 0.28 0.0%, Glyd 0.27.0 0.0%, bf16 again 0.0%

**Against bf16** (the runs' JSON means):

| Rate (req/s) | Server | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | Glyd 0.28 | 1.01x (0.92 to 0.92) | +14% (199 to 227 ms) | -8% (23.4 to 21.7 ms) | -7% |
| 1 | Glyd 0.27.0 | 1.01x (0.92 to 0.92) | +13% (199 to 225 ms) | -8% (23.4 to 21.6 ms) | -7% |
| 1 | bf16 again | 1.00x (0.92 to 0.92) | -2% (199 to 195 ms) | +2% (23.4 to 23.8 ms) | +1% |
| inf | Glyd 0.28 | 1.07x (4.72 to 5.05) | -16% (22,972 to 19,394 ms) | +43% (46.4 to 66.2 ms) | +4% |
| inf | Glyd 0.27.0 | 1.07x (4.72 to 5.05) | -14% (22,972 to 19,784 ms) | +39% (46.4 to 64.7 ms) | +4% |
| inf | bf16 again | 1.00x (4.72 to 4.70) | -4% (22,972 to 22,147 ms) | +5% (46.4 to 48.8 ms) | -1% |

bf16's two runs are the noise of the run: its saturated requests a second, first and again, are on the rows above.

## Quality sanity

Eight fixed prompts answered greedily, 384 tokens each; then bf16's continuations fed back to each server and scored (the share of tokens the server's top-1 is, and how far its log-probabilities are from bf16's). A greedy continuation can differ from another server's once two next tokens are within the arithmetic's rounding, which is why bf16 again is in the table.

| Mode | First token | Whole continuation | Tokens before the first difference (of 384) | bf16's continuation: top-1 | Mean \|logprob difference\| | Largest |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 8 of 8 | 8 of 8 | 384-384 | 99.19% | 0.0000 | 0.000 |
| Glyd 0.28 | 8 of 8 | 2 of 8 | 23-384 | 99.22% | 0.0003 | 0.129 |
| Glyd 0.27.0 | 8 of 8 | 2 of 8 | 23-384 | 99.22% | 0.0003 | 0.129 |
| bf16 again | 8 of 8 | 7 of 8 | 47-384 | 99.22% | 0.0013 | 0.126 |

## Files

- `machine.txt`, `env.txt`: the machine and the software.
- `serve/<server>/`: `vllm bench serve`'s results (`*-rate*.json`), the weights and KV cache lines of the server's log
  (`kv-*.txt`), the GPU's clock and temperature each second (`smi-*.csv`) and its slowdown reasons (`smi-reasons.csv`),
  the quality probe (`probe-*.json`), and Glyd's lines in the server's log (`glyd-notes.txt`).
- `serve/report.txt`, `serve/clocks.txt`, `serve/ratios.txt`: the tables above as the run wrote them.
