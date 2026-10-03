# The lossless KV cache on a GH200: decode, serving and the checks (2026-10-03)

Qwen3-8B, bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580), a GH200 480GB (compute capability 9.0; aarch64 host, 64 CPUs; Lambda); `kv: lossless` against vLLM's own cache ("stock" below), vLLM's default mode (torch.compile, FULL decode graphs).
One run of every step of the job (`kvg`): the plugin's tests, the kernels' checks, decode tokens a second, `vllm bench serve` with 1,024 and 8,192 tokens in, the engine's checks, and the prefix-caching, chunked-prefill and `exact` checks. The H100's record, with the same job: [`h100-vllm-kv-2026-10-02`](../h100-vllm-kv-2026-10-02).

## What to know

- **Decoding is faster at every batch measured**, in the default mode: 1.003x to 1.187x the tokens a second at the same batch (one run each: 1.003x for one request of 1,024 tokens, 1.011x for one of 8,192, 1.013x for 4 of 8,192, 1.021x for 32 of 1,024 and 1.187x for 32 of 8,192; in three rounds with the two alternating, 1.005x and 1.019x for the two single requests), and 1.305x with each cache at the largest batch it holds (stock 58 of 8,192, `kv` 76). The KV cache holds 1.305x the tokens (482,864 against 630,112).
- **Serving, 1,024 tokens in and 256 out** (`vllm bench serve`): 1.00x the requests a second at 1 a second and for one user, **1.06x saturated** (22.47 against 21.18), each generated token 0.91x the time saturated (29.3 and 32.1 ms), the first token 1.01x saturated (3,731 and 3,686 ms). At 1 a second the first token comes 3 ms later (47 against 44 ms), as on the H100; for one user it is the same (41 ms). The prefix cache's hit rate in the servers' logs is 0.0% for both: every pass draws prompts of its own.
- **Serving, 8,192 tokens in and 256 out, saturated: more requests a second, a faster first token and faster tokens**: 3.12 requests a second with `kv` against 2.78 with vLLM's cache (1.12x), the first token after 7,944 ms and 8,321 ms (0.95x), each following token 47.8 ms and 49.5 ms (0.97x). The prefix cache's hit rate is 0.0% for both. (On the H100 the same run's tokens take 1.12x as long.)
- **Every check passes**: the plugin's tests, the kernels' checks 81 of 81, the engine's checks 19 of 19 and the prefix-caching, chunked-prefill and `exact` checks 12 of 12 (`kvg/steps.txt`). A decode step's attention is not bit-equal to vLLM's, as in every record of this cache: where a check compares greedy tokens, they can part from vLLM's after some tokens.
- **`kv: auto`** holds the cache on a GH200 (by the name the driver gives, `NVIDIA GH200 480GB`) from this record on. **Not measured yet**: an H200 (also compute capability 9.0: `auto` leaves it off until it is), tensor parallel (not supported).

## Decode tokens a second, default mode (128 new tokens, 0.9 of the GPU's memory)

| batch x tokens | stock | kv | kv / stock |
| :--- | ---: | ---: | ---: |
| 4 x 8,192 | 576.6 | 584.0 | 1.013x |
| 32 x 1,024 | 4,566.2 | 4,661.5 | 1.021x |
| 1 x 8,192 | 166.4 | 168.2 | 1.011x |
| 1 x 1,024 | 176.6 | 177.2 | 1.003x |
| 32 x 8,192 | 2,051.2 | 2,435.0 | 1.187x |
| stock 58 x 8,192, kv 76 x 8,192 | 2,395.1 | 3,126.6 | 1.305x |

One run each; three rounds with the two alternating for the single requests (1 x 8,192 1.022, 1.017, 1.017x; 1 x 1,024 1.005, 1.005, 1.005x; `tps/batch1-table.txt`).

## Serving (`vllm bench serve`, random dataset, 256 output tokens, 0.9, prefix caching and chunked prefill as vLLM has them)

| prompt | load | stock | kv | kv / stock |
| :--- | :--- | ---: | ---: | ---: |
| 1,024 tokens | 1 request a second | 0.98 requests/s, 6.0 ms a token, first token 44 ms | 0.98, 6.0 ms, 47 ms | 1.00x, 1.00x, 1.06x |
| | saturated | 21.18 requests/s, 32.1 ms a token, first token 3,686 ms | 22.47, 29.3 ms, 3,731 ms | 1.06x, 0.91x, 1.01x |
| | one user | 0.66 requests/s, 5.8 ms a token, first token 41 ms | 0.66, 5.8 ms, 41 ms | 1.00x, 1.00x, 0.99x |
| 8,192 tokens | saturated | 2.78 requests/s, 49.5 ms a token, first token 8,321 ms | 3.12, 47.8 ms, 7,944 ms | 1.12x, 0.97x, 0.95x |

The prefix cache's hit rate in the servers' logs is 0.0% for both modes in both runs (`serve-1k/hit-*.txt`, `serve-8k/hit-*.txt`): the 1,024-token passes draw their prompts with seeds 4, 5 and 6, the 8,192-token pass with seed 0.

The KV cache at those flags: 489,344 tokens with vLLM's and 641,072 with `kv` (1.31x) for the 1,024-token runs, 477,456 and 625,040 (1.31x) for the 8,192-token ones; the model's memory is 1.09x (16.57 against 15.27 GiB, and 16.59 in the 8,192-token run).

## Files

`kvg/`: the job's `steps.txt` (the steps and their times, with the checks' results), `env.txt`, `tps/` (the tables and, for the single requests, each round's result), `serve-1k/` and `serve-8k/` (each run's `vllm bench serve` output and result, the prefix cache's hit-rate lines (`hit-*.txt`) and the summary table).
