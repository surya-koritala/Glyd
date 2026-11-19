# The lossless KV cache in vLLM on an L4: size, values, serving (2026-09-30 to 2026-10-01)

`kv: lossless` holds vLLM 0.30's KV cache in 12 bits a value and reads every value back bit for bit
([gpu/vllm/README.md](../../../gpu/vllm/README.md#the-kv-cache-in-12-bits)). These are the runs of its first version on the dev L4:
the cache's size, the values through the write kernel and back, `compute-sanitizer`, vLLM's `bench serve`, the outputs
against vLLM's own spread, decode tokens a second, and the kernels' own timings.

## Setup

- **Machine:** AWS g6.4xlarge: NVIDIA L4 (23,034 MiB, 72 W cap), driver 595.91.07, 16 vCPUs.
- **Software:** vLLM 0.30.0 (torch 2.13.0+cu130) in its own venv; the library built for sm_89 (CUDA 13.0 nvcc); compute-sanitizer from CUDA 13.2. Model: Qwen3-8B, bf16 weights (`fraction` 0), eager unless a row says compiled.
- The figures below are those of the first version's final form; later records measure the later versions.

## The cache's size

vLLM's flags: eager, `--max-model-len 8704`, 8,704 tokens a step, prefix caching and chunked prefill off, 0.9.

| | tokens | blocks | against stock |
| :--- | ---: | ---: | ---: |
| stock | 25,232 | 1,577 | |
| kv lossless | 32,576 | 2,036 | 1.2911x |

At 0.9 and `--max-model-len 4096` (the bench's flags) the cache was 37,088 tokens against stock's 28,560 (1.2986x, eager) and 33,552 against 25,568 (1.3122x, compiled, warm); a larger step leaves less memory for it (16,384 tokens a step: 1.280x).

## The values (`check/`; `check_vllm.py --kv --study`)

Qwen3-8B unless a row says otherwise. Every row passes.

| # | what |
| ---: | :--- |
| 1 | every 16-bit pattern (NaN, infinity, denormals and zeros among them), K and V, through the write kernel and back: bit for bit |
| 2 | recycling and rewrites: 24 sequences in 190 steps of mixed sizes (161 rewrites, padding slots) into blocks of a previous owner's random bytes, every token the last value written: bit for bit |
| 3 | decode attention: 10 sequences of 1 to 2,049 tokens (the last block partly written, the bytes past it a previous owner's), 1, 2, 4, 5 and 8 query heads a KV head: within 0.0052 of an fp32 reference (at most 0.012 of the output's largest) |
| 4 | vLLM's keys and values of 10 real prompts (39,472 tokens, 2,910,191,616 values over 36 layers, a prompt in one step then a token a step, into blocks of a previous owner's bytes): 0 differ |
| 5 | verify in vLLM, 10 prompts of 113 to 8,192 tokens and 512 generated each, every write checked against a bf16 shadow of each layer's cache: 5,734,785,024 values, none differ (a difference stops the run) |
| 6 | KV cache 32,576 tokens against stock's 25,232 (1.2911x; at least 1.29x), eager, max-model-len 8704, 8704 tokens a step, 0.9, bf16 weights; (max-model-len 4096, 0.85: 22,560 against stock's 17,568, 1.2842x) |
| 7 | prompt_logprobs of a 1,542-token prompt (stock's own continuation fed back) bit for bit stock's |
| 8 | the 8 prompts' first generated token's logprob bit for bit stock's; their 64 greedy tokens the same as stock's from the start for [20, 64, 23, 64, 64, 44, 64, 51] tokens, mean abs logprob difference on the shared prefixes 0.0100 |
| 9 | vLLM's default mode (torch.compile, piecewise CUDA graphs around the attention layers, which run eager): the 8 prompts' first generated token's logprob bit for bit stock's in the same mode; their 64 greedy tokens the same as stock's from the start for [64, 64, 64, 64, 64, 64, 64, 64] tokens, mean abs logprob difference on the shared prefixes 0.0093 |

Row 9 compares two separate compiles (stock's and the plugin's) of the default mode. Inductor picks some kernels' variants by timing them, so the row passes when the two compiles land on the same variant and not otherwise: later runs on the same flags, [`l4-vllm-kv-extension-2026-10-01/compiled/`](../l4-vllm-kv-extension-2026-10-01), found stock's own compiles on empty caches in two sets of first-token logprobs, up to 0.053 apart. The check now compares the bits in inductor's deterministic mode, where they are equal.

`compute-sanitizer` memcheck, initcheck, racecheck and synccheck over the kernels' quick checks (3 and 16 KV heads; the same checks without the sanitizer also take 32): 0 errors (`san/report.txt`).

## `vllm bench serve` (`bench-eager/`, `bench-default/`; `bench_serve.sh`)

Random dataset, 1,024 tokens in and 256 out, `--ignore-eos`, 0.9, `--max-model-len 4096`, prefix caching and chunked prefill off on both sides, bf16 weights, one server a mode; 64 requests at 1 a second, 256 at once (saturated), and 8 one at a time (one user).

**Eager.** bf16: weights 15.27 GiB, KV cache 28,560 tokens, max concurrency 6.97x at 4,096 tokens; kv: weights 15.3 GiB, KV cache 37,088 tokens, max concurrency 9.05x at 4,096 tokens

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 15.27 | 28,560 | 0.66 | 3,150 | 103.4 |
| 1 | kv | 15.30 (1.00x) | 37,088 (1.30x) | 0.71 (1.08x) | 947 (0.30x) | 102.2 (0.99x) |
| inf | bf16 | 15.27 | 28,560 | 0.78 | 149,178 | 115.9 |
| inf | kv | 15.30 (1.00x) | 37,088 (1.30x) | 0.91 (1.17x) | 121,174 (0.81x) | 125.1 (1.08x) |
| one | bf16 | 15.27 | 28,560 | 0.06 | 311 | 61.5 |
| one | kv | 15.30 (1.00x) | 37,088 (1.30x) | 0.06 (1.00x) | 312 (1.00x) | 61.2 (1.00x) |

**vLLM's default mode (torch.compile, piecewise graphs), warm.** bf16: weights 15.27 GiB, KV cache 25,568 tokens, max concurrency 6.24x at 4,096 tokens (started cold, on an empty compile cache: weights 15.27 GiB, KV cache 25,568 tokens, max concurrency 6.24x at 4,096 tokens); kv: weights 15.3 GiB, KV cache 33,552 tokens, max concurrency 8.19x at 4,096 tokens (started cold, on an empty compile cache: weights 15.3 GiB, KV cache 25,552 tokens, max concurrency 6.24x at 4,096 tokens)

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 15.27 | 25,568 | 0.62 | 4,258 | 101.2 |
| 1 | kv | 15.30 (1.00x) | 33,552 (1.31x) | 0.70 (1.13x) | 1,646 (0.39x) | 101.6 (1.00x) |
| inf | bf16 | 15.27 | 25,568 | 0.74 | 160,042 | 110.6 |
| inf | kv | 15.30 (1.00x) | 33,552 (1.31x) | 0.89 (1.20x) | 127,493 (0.80x) | 118.1 (1.07x) |
| one | bf16 | 15.27 | 25,568 | 0.06 | 314 | 60.7 |
| one | kv | 15.30 (1.00x) | 33,552 (1.31x) | 0.06 (1.01x) | 318 (1.01x) | 60.6 (1.00x) |

## Agreement with vLLM's own outputs (`agree/`; `kv_agree.py`)

Eager; each run in a vLLM of its own; six prompts (614-4,100 tokens, 256 greedy tokens, one batch) and 64 windows of 64-2,048 tokens with stock's own 256-token continuation forced (a logits processor), 16,320 positions.

| measure | stock against stock | kv against stock |
| :--- | ---: | ---: |
| six prompts, 256 tokens: identical continuations | 1 of 6 (one at a time against the batch) | 1 of 6 |
| first divergence at token | 40, 118, 21, none, 48, 0 | 16, 65, 13, none, 50, 118 |
| chosen-token logprob difference on the common prefix: median / p99 / max | 3.2e-06 / 0.102 / 0.136 | 1.7e-06 / 0.085 / 0.174 |
| 64 prompts forced, 16,320 positions: top-1 agreement | 99.173% (8 sequences at a time against 64: 135 positions differ) | 99.216% against stock's 64 (128 differ), 99.099% against stock's 8 (147) |
| mean abs logprob difference of the forced token / p99 | 0.00993 / 0.0990 | 0.00953 / 0.0979 |

## Decode tokens a second, eager (`tps/`; `kv_tps.py`)

The same batches in vLLM's default mode (compiled, FULL decode graphs), which is how vLLM serves, are in [`l4-vllm-kv-graphs-2026-10-01`](../l4-vllm-kv-graphs-2026-10-01) (1.018x at 1 x 8,192 there; 1.045x, 1.039x, 1.047x at the same batches of 3 x 8,192, 24 x 1,024, 6 x 4,096). An eager step launches each layer's attention from Python, and FlashAttention's wrapper costs about 100 us of the host a call that the plugin's call does not: where the host is a step's time (a fast GPU), an eager number carries that cost and is not the kernel's gain.

Eager, bf16 weights, 128 new tokens, 0.96, 8,704 tokens a step, each side in a vLLM of its own, batches of B requests of L tokens in all (a prompt of L - 128 random tokens, then 128 new), the decode steps alone (a run of 129 tokens less a run of 1). Three rounds, the order alternating (stock first, kv first, stock first), each configuration's tokens a second the median of the rounds'. KV cache: stock 34,848 tokens, kv 45,184 (1.297x).

| batch x tokens | stock: tokens/s (ms a step) | kv: tokens/s (ms a step) | kv / stock | each round's |
| :--- | ---: | ---: | ---: | :--- |
| 4 x 8,192 | 47.8 (83.7) | 50.9 (78.6) | **1.065** | 1.068, 1.066, 1.065 |
| 32 x 1,024 | 356.4 (89.8) | 375.2 (85.3) | **1.053** | 1.055, 1.055, 1.047 |
| 1 x 8,192 | 15.3 (65.3) | 15.6 (64.1) | **1.018** | 1.019, 1.019, 1.014 |
| stock 4 x 8,192, kv 5 x 8,192 | 47.9 (83.4) | 61.0 (82.0) | **1.272** | one run (each side at its largest batch) |
| stock 34 x 1,024, kv 44 x 1,024 | 372.7 (91.2) | 484.1 (90.9) | **1.299** | one run (each side at its largest batch) |
| stock 8 x 4,096 | 95.5 (83.8) | 101.8 (78.6) | **1.066** | one run |
| stock 3 x 8,192 | 38.4 (78.1) | 40.3 (74.5) | **1.048** | one run |
| stock 1 x 1,024 | 16.3 (61.2) | 16.4 (60.9) | **1.005** | one run |

## The kernels alone, and a decode step (`kernels/`)

**One layer's decode attention and cache write, alone** (`kv_decode_bench.py`; Qwen3-8B's layer: 32 query heads over 8 KV heads; each shape in as many copies as make it larger than the L2; microseconds a call; FlashAttention 2 is vLLM's `flash_attn_varlen_func` on a bf16 cache of the same values, the write against vLLM's `reshape_and_cache_flash`, on a *cold* GPU: a 512 MB pass queued before each call).

| batch x tokens | attention | FlashAttention 2 | FlashAttention 2 / attention | cache write, cold | vLLM's write, cold |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 32 x 1,024 | 392.0 | 547.5 | 1.40x | 18.4 | 7.0 |
| 4 x 8,192 | 394.1 | 548.0 | 1.39x | 10.2 | 6.1 |
| 1 x 8,192 | 105.3 | 141.5 | 1.34x | 10.2 | 6.1 |
| 8 x 4,096 | 393.9 | 548.4 | 1.39x | 10.2 | 6.1 |
| 3 x 8,192 | 298.6 | 407.4 | 1.36x | 10.8 | 7.2 |
| 1 x 1,024 | 18.3 | 27.0 | 1.48x | 10.2 | 5.6 |
| 44 x 1,024 | 538.0 | 757.1 | 1.41x | 23.6 | 7.2 |
| 5 x 8,192 | 490.4 | 674.1 | 1.38x | 9.2 | 5.1 |

The attention columns are timed in a CUDA graph, as a decode step in vLLM's default mode has them (the GPU's time, [`l4-vllm-kv-graphs-2026-10-01`](../l4-vllm-kv-graphs-2026-10-01)). The first version of this table timed a loop of calls from Python (393.5 / 556.3 at 32 x 1,024 ... 19.7 / 101.9 at 1 x 1,024): a FlashAttention call costs about 100 us of the host before it launches anything, which was all of a small shape's time (1 x 1,024: 5.18x), not the kernel's.

**Where a decode step goes** (`kv_step_profile.py`: Qwen3-8B, eager, bf16 weights, 0.96, B requests of L tokens in all, the steps driven through the engine, 20 steps under torch's profiler; ms a step; wall is without the profiler).

| batch x tokens | | stock | kv |
| :--- | :--- | ---: | ---: |
| 32 x 1,024 | step, wall | 88.74 | 84.35 |
| | GPU busy | 88.68 | 84.27 |
| | attention (FlashAttention's kernels; the plugin's decode and combine) | 18.72 | 13.40 |
| | the cache write (the plugin's kernel; stock's is in *other*) | | 0.66 |
| | GEMMs | 68.56 | 69.22 |
| | other | 1.40 | 0.99 |
| 4 x 8,192 | step, wall | 82.71 | 77.71 |
| | GPU busy | 82.39 | 77.41 |
| | attention (FlashAttention's kernels; the plugin's decode and combine) | 19.55 | 14.07 |
| | the cache write (the plugin's kernel; stock's is in *other*) | | 0.51 |
| | GEMMs | 61.86 | 62.08 |
| | other | 0.99 | 0.75 |

## Files

- `bench-eager/`, `bench-default/`: `bench_serve.sh`'s results (`summary.txt`, the bench's JSON and text, nvidia-smi's clock and temperature each second).
- `agree/`, `tps/`: `kv_agree.py`'s and `kv_tps.py`'s results. `check/`: the checks' result files (JSON; stock's runs beside kv's). `san/`: the sanitizer's report. `kernels/`: `kv_decode_bench.py`'s results. `capacity/`: stock's run of the cache's size.
- The tools are in [`gpu/vllm/`](../../../gpu/vllm). Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record.
