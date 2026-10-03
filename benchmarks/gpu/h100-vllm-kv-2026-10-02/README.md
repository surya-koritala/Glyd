# The lossless KV cache on an H100: decode, serving and the checks (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580), an H100 SXM 80 GB (Lambda); `kv: lossless` against vLLM's own cache ("stock" below), vLLM's default mode (torch.compile, FULL decode graphs) unless a table says otherwise.
Three runs of the same job on the same machine type, `kvh`, `kvh2` and `kvh3`; each table says which it is from.

## What to know

- **Decoding is faster at every batch measured**, in the default mode: 1.005x the tokens a second for one request of 1,024 tokens (1.004x in three rounds with the two alternating, on `kvh2`), 1.023x for one of 8,192, 1.057x for 32 of 1,024, 1.197x for 32 of 8,192, and 1.342x with each cache at the largest batch it holds (stock 46 of 8,192, `kv` 60). The KV cache holds 1.306x the tokens (382,448 against 499,312).
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

## Serving (`kvh2`: `vllm bench serve`, random dataset, 256 output tokens, 0.9, prefix caching and chunked prefill as vLLM has them)

| prompt | load | stock | kv | kv / stock |
| :--- | :--- | ---: | ---: | ---: |
| 8,192 tokens | saturated (64 requests at once) | 2.67 requests/s, 43.9 ms a token, first token 9,199 ms | 2.80, 49.2 ms, 8,196 ms | 1.05x, 1.12x, 0.89x |

The runs had 0.0% prefix-cache hits on both sides.
The KV cache at those flags: 377,024 tokens with vLLM's and 494,256 with `kv` (1.31x); the model's memory is 1.07x (16.34 against 15.27 GiB).

## Files

`kvh/` (the decode tables and each round's result), `kvh2/` (the 8,192-token serving runs: each run's `vllm bench serve` output and result, and the summary table; the 1 x 1,024 rounds), `kvh3/` (`steps.txt`: the steps and the checks' results). Each has `env.txt`.
