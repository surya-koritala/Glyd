# The lossless KV cache's decode step on an L4 (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an NVIDIA L4 (24 GB), vLLM's default mode (torch.compile, FULL decode graphs); `kv: lossless` against vLLM's own cache ("stock" below).
The same measurements on an A100: [`a100-vllm-kv-step-2026-10-02`](../a100-vllm-kv-step-2026-10-02).

- A decode step with the lossless cache is 1.005x to 1.053x faster than with vLLM's cache at the same batch (the profile, below).
- The cache's write is the one part of a step it does slower than vLLM's; its attention takes less time than vLLM's, which more than pays for it at every shape measured.
- The KV cache at the setting of the last table holds 41,408 tokens against vLLM's 32,208 (1.286x).

## A decode step (the profile)

Milliseconds a step (36 layers): the step's wall time, and the GPU's time by kind.

| batch x tokens | stock a step | kv a step | kv / stock | attention: stock -> kv | cache write: stock -> kv |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 x 1,024 | 60.28 | 59.98 | 1.005x | 1.009 -> 0.664 | 0.085 -> 0.138 |
| 1 x 8,192 | 64.33 | 63.04 | 1.020x | 5.102 -> 3.721 | 0.082 -> 0.135 |
| 3 x 8,192 | 76.81 | 73.15 | 1.050x | 14.497 -> 10.623 | 0.085 -> 0.179 |
| 32 x 1,024 | 86.72 | 82.37 | 1.053x | 18.302 -> 13.408 | 0.099 -> 0.252 |

## Decode tokens a second

128 new tokens a request, vLLM's default mode, three rounds with the two processes alternating (stock first, kv first, stock first), 0.96 of the GPU's memory so that both caches hold the batch.

| batch x tokens | stock | kv | kv / stock, each round | median |
| :--- | ---: | ---: | :--- | ---: |
| 1 x 8,192 | 15.5 (64.4 ms a step) | 15.9 (63.0 ms a step) | 1.022, 1.024, 1.023 | **1.023x** |
| 3 x 8,192 | 38.8 (77.3 ms a step) | 40.8 (73.5 ms a step) | 1.052, 1.046, 1.056 | **1.052x** |
| 24 x 1,024 | 289.5 (82.9 ms a step) | 302.4 (79.4 ms a step) | 1.049, 1.045, 1.047 | **1.047x** |
| 6 x 4,096 | 77.5 (77.4 ms a step) | 81.6 (73.6 ms a step) | 1.053, 1.048, 1.062 | **1.053x** |
| 1 x 1,024 | 16.6 (60.4 ms a step) | 16.7 (60.0 ms a step) | 1.007, 1.006, 1.008 | **1.007x** |

The KV cache at that setting: 32,208 tokens with vLLM's and 41,408 with kv (1.286x).

## Files

`profile/` (stock and kv: the step's wall time and the GPU's time by kind), `tps/` (each round's result, the first pass and the table). Result files are the harness's own JSON; the lists of the kernels each step ran are not part of this record.
