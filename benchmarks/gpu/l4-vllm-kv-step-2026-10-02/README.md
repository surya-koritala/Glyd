# A decode step on the L4: the write, the attention and what they add up to (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, the dev L4 (24 GB), vLLM's default mode (torch.compile, FULL decode graphs) unless a table says eager; `kv: lossless` against vLLM's own cache.
This record is where a decode step's time goes with kv and with stock, the kv write and attention kernels timed alone in a CUDA graph, how well those predict the tokens a second, and what the changes to the decode step's write and attention did (the numbers before them are marked "before").
Code: kv-vllm 591a37bc. The same measurements on an A100: [`a100-vllm-kv-step-2026-10-02`](../a100-vllm-kv-step-2026-10-02).

- A decode step with kv is 1.005x to 1.053x faster than with vLLM's cache at the same batch (the profile, below): its attention is faster, its write is not.
- One layer launches 2 kernels where vLLM's launch 3 (a step: 385 kernels against 491).
- The kv write of a decode step costs 0.14 to 0.25 ms a step now, against 0.30 to 0.63 before; vLLM's own is 0.08 to 0.10.
- The kernels' graph times predict the tokens a second to within 0.3% at most (table below).

## A decode step (the profile)

Milliseconds a step (36 layers): the step's wall time, and the GPU's time by kind (`kv_step_profile.py --graphs`). Stock's attention is FlashAttention's split-KV kernel and its combine kernel; its cache write is vLLM's own.

| batch x tokens | stock a step | kv a step | kv / stock | attention: stock -> kv | cache write: stock -> kv |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 x 1,024 | 60.28 | 59.98 | 1.005x | 1.009 -> 0.664 | 0.085 -> 0.138 |
| 1 x 8,192 | 64.33 | 63.04 | 1.020x | 5.102 -> 3.721 | 0.082 -> 0.135 |
| 3 x 8,192 | 76.81 | 73.15 | 1.050x | 14.497 -> 10.623 | 0.085 -> 0.179 |
| 32 x 1,024 | 86.72 | 82.37 | 1.053x | 18.302 -> 13.408 | 0.099 -> 0.252 |

## The kernels alone: one layer, in a CUDA graph (microseconds)

The GPU's time as a step has it. A call timed from a Python loop (eager) is longer for FlashAttention by about 100 us of the host a call (its wrapper), which is not the kernel's: those times are in the JSON files, and not used here.

| batch x tokens | decode: kv | FlashAttention 2 | FlashAttention 2 / kv | write: kv | write: vLLM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 x 1,024 | 17.97 | 26.73 | 1.49x | 3.63 | 1.84 |
| 1 x 2,048 | 29.93 | 44.49 | 1.49x | 3.56 | 1.82 |
| 1 x 4,096 | 54.6 | 75.39 | 1.38x | 2.59 | 1.82 |
| 1 x 8,192 | 102.86 | 141.13 | 1.37x | 4.55 | 1.95 |
| 2 x 2,048 | 55.3 | 77.28 | 1.40x | 2.64 | 1.82 |
| 3 x 8,192 | 295.58 | 405.96 | 1.37x | 4.91 | 2.02 |
| 4 x 8,192 | 392.04 | 548.15 | 1.40x | 5.2 | 2.1 |
| 6 x 4,096 | 295.55 | 406.63 | 1.38x | 6.5 | 2.1 |
| 8 x 1,024 | 101.91 | 142.39 | 1.40x | 4.79 | 1.97 |
| 8 x 4,096 | 392.32 | 548.81 | 1.40x | 5.22 | 2.1 |
| 16 x 1,024 | 197.83 | 294.71 | 1.49x | 5.17 | 2.02 |
| 24 x 1,024 | 293.89 | 409.04 | 1.39x | 6.6 | 2.15 |
| 32 x 1,024 | 390.09 | 547.64 | 1.40x | 7.6 | 2.18 |
| 32 x 8,192 | 3083.37 | 4269.47 | 1.38x | 9.57 | 2.28 |
| 64 x 8,192 | 6161.28 | 8483.02 | 1.38x | 10.37 | 2.66 |
| 128 x 4,096 | 6162.1 | 8544.95 | 1.39x | 14.74 | 3.12 |


## The kv write, before and after the decode step's path

In a decode step (milliseconds a step, 36 layers; the profile above, and before the decode step's write path):

| batch x tokens | kv write before | kv write now | vLLM's write |
| :--- | ---: | ---: | ---: |
| 1 x 1,024 | 0.308 | 0.138 | 0.085 |
| 1 x 8,192 | 0.295 | 0.135 | 0.082 |
| 32 x 1,024 | 0.626 | 0.252 | 0.099 |

The write kernel alone (one layer, microseconds, in a CUDA graph; warm and after a clean 512 MB read):

| batch x tokens | before | now | before, cold | now, cold | vLLM's, cold |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 x 1,024 | 6.35 | 3.63 | 14.34 | 7.17 | 5.12 |
| 1 x 2,048 | 5.71 | 3.56 | 11.1 | 8.19 | 5.1 |
| 1 x 4,096 | 4.74 | 2.59 | 8.19 | 6.14 | 5.12 |
| 1 x 8,192 | 7.78 | 4.55 | 11.26 | 8.11 | 5.12 |
| 4 x 8,192 | 8.83 | 5.2 | 15.36 | 9.22 | 5.12 |
| 8 x 1,024 | 7.83 | 4.79 | 13.33 | 8.19 | 5.12 |
| 8 x 4,096 | 8.96 | 5.22 | 13.31 | 9.22 | 5.12 |
| 16 x 1,024 | 8.17 | 5.17 | 14.34 | 10.24 | 5.12 |
| 32 x 1,024 | 11.13 | 7.6 | 20.48 | 16.38 | 5.12 |

After a clean read of 512 MB before each call (the weights a step streams), the write takes 6.1 to 16.4 us for kv (8.2 to 25.6 before the change) against 5.1 for vLLM's, at the shapes of the table above.

## Decode tokens a second, predicted from the kernels and measured

128 new tokens a request, vLLM's default mode, three rounds with the two processes alternating (stock first, kv first, stock first), 0.96 of the GPU's memory so that both caches hold the batch. The prediction is the step's time less the kernels' difference (FlashAttention and vLLM's write against kv's attention and write, from the graph times above, 36 layers).

| batch x tokens | stock | kv | kv / stock, each round | median | predicted from the kernels |
| :--- | ---: | ---: | :--- | ---: | ---: |
| 1 x 8,192 | 15.5 (64.4 ms a step) | 15.9 (63.0 ms a step) | 1.022, 1.024, 1.023 | **1.023x** | 1.020x |
| 3 x 8,192 | 38.8 (77.3 ms a step) | 40.8 (73.5 ms a step) | 1.052, 1.046, 1.056 | **1.052x** | 1.053x |
| 24 x 1,024 | 289.5 (82.9 ms a step) | 302.4 (79.4 ms a step) | 1.049, 1.045, 1.047 | **1.047x** | 1.050x |
| 6 x 4,096 | 77.5 (77.4 ms a step) | 81.6 (73.6 ms a step) | 1.053, 1.048, 1.062 | **1.053x** | 1.052x |
| 1 x 1,024 | 16.6 (60.4 ms a step) | 16.7 (60.0 ms a step) | 1.007, 1.006, 1.008 | **1.007x** | 1.004x |

The KV cache at that setting: 32,208 tokens with vLLM's and 41,408 with kv (1.286x).

## Files

`decode/` (the kernels: the graph and eager timings of each shape, the largest difference of the two attention outputs; `-before`: the library before the decode step's write path; `-new`: after it; `-final`: kv-vllm 591a37bc; `cold-write-*`: the write after a clean 512 MB read), `profile/` (stock, kv before and kv now), `launches/` (the kernels a step launches, by kind, under torch's profiler), `tps/` (each round's result).
The tools are in [`gpu/vllm/`](../../../gpu/vllm): `kv_decode_bench.py`, `kv_step_profile.py`, `kv_tps.py`, `kv_launches.py`.

