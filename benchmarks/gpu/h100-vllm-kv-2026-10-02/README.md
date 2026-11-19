# The KV cache in 12 bits on an H100: decode, serving and the checks, before and after FlashAttention 3 (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580), an H100 SXM 80 GB (Lambda); `kv: lossless` against vLLM's own cache, vLLM's default mode (torch.compile, FULL decode graphs) unless a table says eager.
Three runs of the same job on the same machine type: `kvh` (kv-vllm 591a37bc, where a prompt's attention ran FlashAttention 2 although vLLM's own runs FlashAttention 3 on Hopper), `kvh2` (kv-vllm ff8f227b, where it runs vLLM's own FlashAttention 3) and `kvh3` (kv-vllm f5c4ed25: the plugin's tests, the kernels' checks and the prefix-caching, chunked-prefill and `exact` checks, with a control and a confirmation added for the one line `kvh2` failed). A fourth, `kvs1h` (kv-vllm d766885b, 2026-10-03), is the 1,024-token serve alone, repeated with the benchmark's passes drawing prompts of their own.

## What to know

- **Decoding is faster at every batch measured**, in the default mode: 1.005x the tokens a second for one request of 1,024 tokens (1.004x in three rounds with the two alternating, on `kvh2`), 1.023x for one of 8,192, 1.057x for 32 of 1,024, 1.197x for 32 of 8,192, and 1.342x with each cache at the largest batch it holds (stock 46 of 8,192, `kv` 60). The KV cache holds 1.306x the tokens (382,448 against 499,312).
- **Serving, 1,024 tokens in and 256 out** (`vllm bench serve`, `kvs1h`): 1.00x the requests a second at 1 a second and for one user, **1.05x saturated** (21.68 against 20.58), each generated token 0.92x the time saturated (31.2 and 33.9 ms), the first token 1.02x (3,636 and 3,565 ms). The prefix cache's hit rate in the servers' logs is at most 0.7% for vLLM's cache and 1.4% for `kv`, and 0.3% over each whole run: one of the 328 prompts appears twice (the summary flags the `kv` run, whose highest sample is above 1%), which skips at most one of 328 prefills. The first run of this serve (`kvh`: 23.11 and 24.33 requests a second saturated, 1.05x) had its passes share one set of prompts and 29.6% hits for vLLM's cache and 21.6% for `kv`; `kvs1h` replaces it. The 8,192-token runs below had 0.0% hits on both sides.
- **Serving, 8,192 tokens in and 256 out, saturated: more requests a second and a faster first token, but each generated token takes longer.** With `kvh2`: 2.67 requests a second with vLLM's cache and 2.80 with `kv` (1.05x), the first token after 9,199 ms and 8,196 ms (0.89x), each following token 43.9 ms and 49.2 ms (1.12x as long). Before the prompts ran vLLM's own FlashAttention (`kvh`) the same run was 2.65 and 2.53 requests a second (0.95x).
- **Every check passes on `kvh3`**: the plugin's 22 tests, the kernels 81 of 81 (and the engine's checks 19 of 19 on `kvh` and `kvh2`), and the prefix-caching, chunked-prefill and `exact` checks 12 of 12 (on `kvh2` 10 of 11, on `kvh` 4 of 11: seven lines failed, `exact` among them, while a prompt's attention ran FlashAttention 2 where vLLM's own runs 3). The one line `kvh2` failed is four requests, each with its own 2,048-token cached prefix, in one batch: their rows run in two groups, the first token's logprob is vLLM's bit for bit for none of the four and the greedy tokens part from vLLM's after 40, 8, 2 and 33 tokens. FlashAttention 3 schedules a call by its batch, so a row's bits depend on the rows beside it, and `kvh3` shows it two ways. Control: vLLM's own with its batch changed to two requests a step also differs from its own four-request batch (the first token's logprob bit for bit in 0 of 4, the largest difference 0.0229), and `kv`'s largest difference from vLLM's four-request batch is the same 0.0229. Confirmation: with the four rows run together in one call, as vLLM's own does, the first generated token's logprob is vLLM's bit for bit for all 4 of 4 (3 of the 4 sequences of 64 greedy tokens are the same: a decode step's attention is not bit-equal to vLLM's, as in every record of this cache).
- **Since measured**: a GH200, with the same job ([record](../gh200-vllm-kv-2026-10-03): decode 1.003x to 1.187x at the same batch, serving 1.06x saturated with 1,024 tokens in). **Not measured yet**: an H200 (also compute capability 9.0), tensor parallel (not supported).

## Decode attention against vLLM's FlashAttention 3, one layer, in a CUDA graph (`kvh`, microseconds)

| batch x tokens | kv | FlashAttention 3 | FlashAttention 3 / kv |
| :--- | ---: | ---: | ---: |
| 1 x 1,024 | 7.98 | 12.01 | 1.51x |
| 1 x 2,048 | 10.20 | 12.61 | 1.24x |
| 1 x 4,096 | 13.49 | 15.17 | 1.12x |
| 1 x 8,192 | 16.65 | 20.59 | 1.24x |
| 2 x 2,048 | 11.82 | 14.54 | 1.23x |
| 4 x 8,192 | 45.50 | 52.45 | 1.15x |
| 8 x 4,096 | 44.62 | 52.47 | 1.18x |
| 32 x 1,024 | 44.77 | 53.59 | 1.20x |
| 32 x 8,192 | 268.82 | 346.56 | 1.29x |
| 64 x 8,192 | 540.06 | 683.30 | 1.27x |
| 32 x 32,768 | 1,034.16 | 1,356.33 | 1.31x |
| 128 x 4,096 | 543.08 | 681.85 | 1.26x |
| 256 x 1,024 | 284.91 | 356.37 | 1.25x |

Writing a token's keys and values takes 2.8 to 11 us for `kv` and 1.8 to 2.3 us for vLLM's.

## Decode tokens a second, default mode (128 new tokens, 0.9 of the GPU's memory)

| batch x tokens | stock | kv | kv / stock |
| :--- | ---: | ---: | ---: |
| 4 x 8,192 | 499.3 | 514.5 | 1.030x |
| 32 x 1,024 | 3,811.2 | 4,030.1 | 1.057x |
| 1 x 8,192 | 144.8 | 148.1 | 1.023x |
| 1 x 1,024 | 152.8 | 153.6 | 1.005x |
| 32 x 8,192 | 1,733.9 | 2,075.4 | 1.197x |
| stock 46 x 8,192, kv 60 x 8,192 | 1,950.6 | 2,618.4 | 1.342x |

One run each; three rounds with the two alternating for the single requests (`kvh`: 1 x 8,192 1.022, 1.025, 1.023x; 1 x 1,024 1.006, 1.004, 1.004x; `kvh2`: 1 x 1,024 1.004, 1.004, 1.005x).
Eager (`kvh`, `tps/eager-table.txt`) the gains are 1.19x to 1.21x at the same batch: an eager step launches each layer's attention from Python, and FlashAttention's wrapper costs about 100 us of the host a call that the plugin's call does not, so an eager number carries that cost and is not the kernel's gain.

## Serving (`vllm bench serve`, random dataset, 256 output tokens, 0.9, prefix caching and chunked prefill as vLLM has them)

| prompt | load | stock | kv | kv / stock |
| :--- | :--- | ---: | ---: | ---: |
| 1,024 tokens (`kvs1h`) | 1 request a second | 0.97 requests/s, 6.9 ms a token, first token 39 ms | 0.97, 6.9 ms, 42 ms | 1.00x, 1.00x, 1.08x |
| | saturated | 20.58 requests/s, 33.9 ms a token, first token 3,565 ms | 21.68, 31.2 ms, 3,636 ms | 1.05x, 0.92x, 1.02x |
| | one user | 0.58 requests/s, 6.6 ms a token, first token 33 ms | 0.58, 6.6 ms, 36 ms | 1.00x, 1.00x, 1.08x |
| 8,192 tokens, `kvh` | saturated | 2.65 requests/s, 44.2 ms a token, first token 9,289 ms | 2.53, 54.0 ms, 9,442 ms | 0.95x, 1.22x, 1.02x |
| 8,192 tokens, `kvh2` | saturated | 2.67 requests/s, 43.9 ms a token, first token 9,199 ms | 2.80, 49.2 ms, 8,196 ms | 1.05x, 1.12x, 0.89x |

The 1,024-token rows (`kvs1h`) have the prefix cache's hit rate at most 0.7% (vLLM's cache) and 1.4% (`kv`) in the servers' logs, 0.3% over each whole run (one repeated prompt of 328); on the L4 a run with one repeated prompt (1.1% to 1.2% at most) and one with none (0.0%) gave the same saturated rows within 1.4% ([record](../l4-vllm-kv-graphs-2026-10-01)). The first run's 1,024-token rows (`kvh`, shared prompts, 29.6% and 21.6% hits) are superseded; the 8,192-token runs give 0.0% for both.

The KV cache at those flags: 388,928 tokens with vLLM's and 510,288 with `kv` (1.31x) for the 1,024-token runs, 377,024 and 494,256 (1.31x) for the 8,192-token ones; the model's memory is 1.07x (16.32 against 15.27 GiB).

## Files

`kvs1h/`: the 1,024-token serve alone (`steps.txt`, `machine-short.txt`, `env.txt`, `serve-1k/` with the prefix cache's hit-rate lines, `hit-*.txt`). `kvh/`, `kvh2/` and `kvh3/`: the job's `steps.txt` (the steps and their times), `machine-short.txt`, `env.txt` (`kvh3/test-vllm.txt`: the plugin's tests), `tps/` (the tables and each round's result), `serve-1k/` and `serve-8k/` (each run's `vllm bench serve` output and result, and the summary table). The tools are in [`gpu/vllm/`](../../../gpu/vllm): `kv_decode_bench.py`, `kv_tps.py`, `bench_serve.sh`, `check_vllm.py`.
