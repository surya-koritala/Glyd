# The lossless KV cache on an H100: decode, serving and the checks (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580), an H100 SXM 80 GB (Lambda); `kv: lossless` against vLLM's own cache ("stock" below), vLLM's default mode (torch.compile, FULL decode graphs) unless a table says otherwise.
Three runs of the same job on the same machine type, `kvh`, `kvh2` and `kvh3`, and a fourth, `kvs1h` (2026-10-03), the 1,024-token serve alone; each table says which it is from.

## What to know

- **Decoding is faster at every batch measured**, in the default mode: 1.005x the tokens a second for one request of 1,024 tokens (1.004x in three rounds with the two alternating, on `kvh2`), 1.023x for one of 8,192, 1.057x for 32 of 1,024, 1.197x for 32 of 8,192, and 1.342x with each cache at the largest batch it holds (stock 46 of 8,192, `kv` 60). The KV cache holds 1.306x the tokens (382,448 against 499,312).
- **Serving, 1,024 tokens in and 256 out** (`vllm bench serve`, `kvs1h`): 1.00x the requests a second at 1 a second and for one user, **1.05x saturated** (21.68 against 20.58), each generated token 0.92x the time saturated (31.2 and 33.9 ms), the first token 1.02x (3,636 and 3,565 ms). Every pass draws prompts of its own. The prefix cache's hit rate in the servers' logs is at most 0.7% for vLLM's cache and 1.4% for `kv`, and 0.3% over each whole run: one of the 328 prompts appears twice (the summary flags the `kv` run, whose highest sample is above 1%), which skips at most one of 328 prefills.
- **Serving, 8,192 tokens in and 256 out, saturated: more requests a second and a faster first token, but each generated token takes longer.** With `kvh2`: 2.67 requests a second with vLLM's cache and 2.80 with `kv` (1.05x), the first token after 9,199 ms and 8,196 ms (0.89x), each following token 43.9 ms and 49.2 ms (1.12x as long).
- **Every check passes on `kvh3`**: the plugin's tests, the kernels' checks 81 of 81, and the prefix-caching, chunked-prefill and `exact` checks 12 of 12. A decode step's attention is not bit-equal to vLLM's, as in every record of this cache: where a check compares greedy tokens, they can part from vLLM's after some tokens.
- **Not measured yet**: a GH200 and an H200 (both compute capability 9.0), tensor parallel (not supported).

## Decode tokens a second, default mode (`kvh`: 128 new tokens, 0.9 of the GPU's memory)

| batch x tokens | stock | kv | kv / stock |
| :--- | ---: | ---: | ---: |
| 4 x 8,192 | 499.3 | 514.5 | 1.030x |
| 32 x 1,024 | 3,811.2 | 4,030.1 | 1.057x |
| 1 x 8,192 | 144.8 | 148.1 | 1.023x |
| 1 x 1,024 | 152.8 | 153.6 | 1.005x |
| 32 x 8,192 | 1,733.9 | 2,075.4 | 1.197x |
| stock 46 x 8,192, kv 60 x 8,192 | 1,950.6 | 2,618.4 | 1.342x |

One run each; three rounds with the two alternating for the single requests (`kvh`: 1 x 8,192 1.022, 1.025, 1.023x; 1 x 1,024 1.006, 1.004, 1.004x; `kvh2`: 1 x 1,024 1.004, 1.004, 1.005x).

## Serving (`vllm bench serve`, random dataset, 256 output tokens, 0.9, prefix caching and chunked prefill as vLLM has them)

| prompt | load | stock | kv | kv / stock |
| :--- | :--- | ---: | ---: | ---: |
| 1,024 tokens (`kvs1h`) | 1 request a second | 0.97 requests/s, 6.9 ms a token, first token 39 ms | 0.97, 6.9 ms, 42 ms | 1.00x, 1.00x, 1.08x |
| | saturated (256 at once) | 20.58 requests/s, 33.9 ms a token, first token 3,565 ms | 21.68, 31.2 ms, 3,636 ms | 1.05x, 0.92x, 1.02x |
| | one user | 0.58 requests/s, 6.6 ms a token, first token 33 ms | 0.58, 6.6 ms, 36 ms | 1.00x, 1.00x, 1.08x |
| 8,192 tokens (`kvh2`) | saturated (64 requests at once) | 2.67 requests/s, 43.9 ms a token, first token 9,199 ms | 2.80, 49.2 ms, 8,196 ms | 1.05x, 1.12x, 0.89x |

The 1,024-token rows (`kvs1h`) have the prefix cache's hit rate at most 0.7% (vLLM's cache) and 1.4% (`kv`) in the servers' logs, 0.3% over each whole run (one repeated prompt of 328); the 8,192-token run had 0.0% on both sides.
The KV cache at those flags: 388,928 tokens with vLLM's and 510,288 with `kv` (1.31x) in the 1,024-token runs, 377,024 and 494,256 (1.31x) in the 8,192-token one; the model's memory is 1.07x (16.32 against 15.27 GiB, and 16.34 in the 8,192-token run).

## Files

`kvs1h/` (the 1,024-token serve: `steps.txt`, `env.txt`, `serve-1k/` with each run's `vllm bench serve` output and result, the prefix cache's hit-rate lines (`hit-*.txt`) and the summary), `kvh/` (the decode tables and each round's result), `kvh2/` (the 8,192-token serving runs: each run's `vllm bench serve` output and result, and the summary table; the 1 x 1,024 rounds), `kvh3/` (`steps.txt`: the steps and the checks' results). Each has `env.txt`.
