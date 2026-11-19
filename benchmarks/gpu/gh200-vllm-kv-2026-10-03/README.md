# The KV cache in 12 bits on a GH200: decode, serving and the checks (2026-10-03)

Qwen3-8B, bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580), a GH200 480GB (96 GB of GPU memory, compute capability 9.0; aarch64 host, 64 CPUs; Lambda); `kv: lossless` against vLLM's own cache, vLLM's default mode (torch.compile, FULL decode graphs) unless a table says eager.
One run of every step of the job (`kvg`, kv-vllm d766885b): the plugin's tests, the kernels' checks, the decode attention, decode tokens a second, `vllm bench serve` with 1,024 and 8,192 tokens in, the engine's checks, and the prefix-caching, chunked-prefill and `exact` checks. The H100's record, with the same job: [`h100-vllm-kv-2026-10-02`](../h100-vllm-kv-2026-10-02).

## What to know

- **Decoding is faster at every batch measured**, in the default mode: 1.005x the tokens a second for one request of 1,024 tokens (1.003x in the run of the table; 1.005, 1.005 and 1.005x in three rounds with the two alternating), 1.019x for one of 8,192 (1.022, 1.017 and 1.017x), 1.013x for 4 of 8,192, 1.021x for 32 of 1,024, 1.187x for 32 of 8,192, and 1.305x with each cache at the largest batch it holds (stock 58 of 8,192, `kv` 76). The KV cache holds 1.305x the tokens (482,864 against 630,112), 1.3011x at the engine check's flags (eager, 8,704 tokens a step).
- **Serving, 1,024 tokens in and 256 out** (`vllm bench serve`): 1.00x the requests a second at 1 a second and for one user, **1.06x saturated** (22.47 against 21.18), each generated token 0.91x the time saturated (29.3 and 32.1 ms), the first token 1.01x saturated (3,731 and 3,686 ms). At 1 a second the first token comes 3 ms later (47 against 44 ms), as on the H100; for one user it is the same (41 ms). The prefix cache's hit rate in the servers' logs is 0.0% for both: every pass draws prompts of its own.
- **Serving, 8,192 tokens in and 256 out, saturated: more requests a second, a faster first token and faster tokens**: 3.12 requests a second with `kv` against 2.78 with vLLM's cache (1.12x), the first token after 7,944 ms and 8,321 ms (0.95x), each following token 47.8 ms and 49.5 ms (0.97x). The prefix cache's hit rate is 0.0% for both. (On the H100 the same run's tokens take 1.12x as long.)
- **Every check passes**: the plugin's 22 tests, the kernels 81 of 81, the engine's checks 19 of 19 and the prefix-caching, chunked-prefill and `exact` checks 12 of 12. As on the H100, one line is held to vLLM's bits only within the difference its own batch makes: four requests, each with its own 2,048-token cached prefix, in one batch (their rows run in two groups): the first token's logprob is vLLM's bit for bit for none of the four and the greedy tokens part from vLLM's after 40, 8, 2 and 33 tokens. FlashAttention 3 schedules a call by its batch, so a row's bits depend on the rows beside it. Control: vLLM's own with its batch changed to two requests a step differs from its own four-request batch the same way (the first token's logprob bit for bit in 0 of 4, the largest difference 0.0229), and `kv`'s largest difference from vLLM's four-request batch is the same 0.0229. Confirmation: with the four rows run together in one call, as vLLM's own does, the first generated token's logprob is vLLM's bit for bit for all 4 of 4 (3 of the 4 sequences of 64 greedy tokens are the same: a decode step's attention is not bit-equal to vLLM's, as in every record of this cache).
- **`kv: auto`** holds the cache on a GH200 (by the name the driver gives, `NVIDIA GH200 480GB`) from this record on. **Not measured yet**: an H200 (also compute capability 9.0: `auto` leaves it off until it is), tensor parallel (not supported).

## Decode attention against vLLM's FlashAttention 3, one layer, in a CUDA graph (microseconds)

| batch x tokens | kv | FlashAttention 3 | FlashAttention 3 / kv |
| :--- | ---: | ---: | ---: |
| 1 x 1,024 | 7.96 | 11.68 | 1.47x |
| 1 x 2,048 | 10.12 | 12.36 | 1.22x |
| 1 x 4,096 | 13.17 | 14.15 | 1.07x |
| 1 x 8,192 | 16.82 | 18.96 | 1.13x |
| 2 x 2,048 | 11.40 | 13.71 | 1.20x |
| 4 x 8,192 | 42.06 | 45.69 | 1.09x |
| 8 x 4,096 | 41.33 | 45.73 | 1.11x |
| 32 x 1,024 | 41.15 | 46.72 | 1.14x |
| 32 x 8,192 | 231.63 | 290.64 | 1.25x |
| 64 x 8,192 | 460.57 | 570.51 | 1.24x |
| 32 x 32,768 | 878.28 | 1,136.41 | 1.29x |
| 128 x 4,096 | 461.77 | 570.10 | 1.23x |
| 256 x 1,024 | 247.08 | 305.77 | 1.24x |

Writing a token's keys and values takes 2.8 to 11.7 us for `kv` and 1.8 to 2.3 us for vLLM's.

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
Eager (`tps/eager-table.txt`) the gains are 1.12x to 1.14x at the same batch, and 1.352x with each cache at the largest batch it holds (stock 60 of 8,192, `kv` 79): an eager step launches each layer's attention from Python, and it is 2.8 to 3.4 ms shorter with `kv` at every batch (78 to 94 us a layer: the host's cost of FlashAttention's wrapper, which the plugin's call does not have, as in the L4 and H100 records), so an eager number carries that cost and is not the kernel's gain.

## Serving (`vllm bench serve`, random dataset, 256 output tokens, 0.9, prefix caching and chunked prefill as vLLM has them)

| prompt | load | stock | kv | kv / stock |
| :--- | :--- | ---: | ---: | ---: |
| 1,024 tokens | 1 request a second | 0.98 requests/s, 6.0 ms a token, first token 44 ms | 0.98, 6.0 ms, 47 ms | 1.00x, 1.00x, 1.06x |
| | saturated | 21.18 requests/s, 32.1 ms a token, first token 3,686 ms | 22.47, 29.3 ms, 3,731 ms | 1.06x, 0.91x, 1.01x |
| | one user | 0.66 requests/s, 5.8 ms a token, first token 41 ms | 0.66, 5.8 ms, 41 ms | 1.00x, 1.00x, 0.99x |
| 8,192 tokens | saturated | 2.78 requests/s, 49.5 ms a token, first token 8,321 ms | 3.12, 47.8 ms, 7,944 ms | 1.12x, 0.97x, 0.95x |

The prefix cache's hit rate in the servers' logs is 0.0% for both modes in both runs (`serve-1k/hit-*.txt`, `serve-8k/hit-*.txt`): the 1,024-token passes draw their prompts with seeds 4, 5 and 6, the 8,192-token pass with seed 0.

The KV cache at those flags: 489,344 tokens with vLLM's and 641,072 with `kv` (1.31x) for the 1,024-token runs, 477,456 and 625,040 (1.31x) for the 8,192-token ones; the model's memory is 1.09x (16.57 against 15.27 GiB).

## Files

`kvg/`: the job's `steps.txt` (the steps and their times), `machine-short.txt`, `env.txt`, `test-vllm.txt` (the plugin's tests), `decode-table.txt` (the decode attention against FlashAttention 3 with the largest difference of the two outputs), `tps/` (the tables and, for the single requests, each round's result), `serve-1k/` and `serve-8k/` (each run's `vllm bench serve` output and result, the prefix cache's hit-rate lines (`hit-*.txt`) and the summary table). The tools are in [`gpu/vllm/`](../../../gpu/vllm): `kv_decode_bench.py`, `kv_tps.py`, `bench_serve.sh`, `check_vllm.py`.
