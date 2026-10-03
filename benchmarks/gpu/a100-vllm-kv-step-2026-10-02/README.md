# The lossless KV cache's decode step on an A100 (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an A100 SXM4 40 GB (108 SMs; Lambda), vLLM's default mode (torch.compile, FULL decode graphs), `kv: lossless` against vLLM's own cache ("stock" below). The kernels' checks of the job: 81 of 81 pass here.
The KV cache holds 177,392 tokens against vLLM's 136,592 (1.299x) at the same memory utilization (0.9).
The same measurements on an L4: [`l4-vllm-kv-step-2026-10-02`](../l4-vllm-kv-step-2026-10-02).

## One request: decode tokens a second

128 new tokens a request, in three rounds with the two processes alternating (stock first, kv first, stock first).

| batch x tokens | stock | kv | kv / stock, each round | median |
| :--- | ---: | ---: | :--- | ---: |
| 1 x 8,192 | 71.4 (14.00 ms a step) | 73.7 (13.57 ms a step) | 1.031, 1.032, 1.032 | **1.032x** |
| 1 x 1,024 | 75.3 (13.27 ms a step) | 76.7 (13.04 ms a step) | 1.018, 1.018, 1.018 | **1.018x** |

## A decode step (the profile)

Milliseconds a step (36 layers): the step's wall time, and the GPU's time by kind, in the default mode.

| batch x tokens | stock a step | kv a step | kv / stock | attention: stock -> kv | cache write: stock -> kv |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 x 8,192 | 14.05 | 13.65 | 1.029x | 1.532 -> 1.031 | 0.112 -> 0.184 |
| 1 x 1,024 | 13.34 | 13.17 | 1.013x | 0.769 -> 0.534 | 0.112 -> 0.186 |
| 3 x 8,192 | 15.88 | 14.86 | 1.069x | 3.342 -> 2.330 | 0.108 -> 0.193 |
| 32 x 1,024 | 17.51 | 15.96 | 1.097x | 4.380 -> 2.779 | 0.116 -> 0.218 |

The write is the one part of the step that kv does slower; the attention more than pays for it at every shape.

## Files

`tps/tps-batch1.json` (each round's tokens a second), `profile/` (stock and kv). Prefix hits on this machine type: [`a100-vllm-kv-prefix-2026-10-02`](../a100-vllm-kv-prefix-2026-10-02).
