# Prefix caching, chunked prefill and `exact` on the lossless KV cache

Qwen3-8B, bf16 weights, vLLM 0.30.0, the dev L4, 2026-10-01. With `kv: lossless`, the later steps of a chunked prompt and the new tokens after a prefix hit run on vLLM's installed FlashAttention with the arguments its own call has, so their outputs are vLLM's bit for bit; decode rows run Glyd's attention kernel. `exact` sends decode rows through the same call as vLLM's own, all rows in one.

## What was measured

| | measured |
| :--- | :--- |
| prompt logprobs and the first generated tokens against stock with chunked prefill (512 and 2,048 tokens), with prefix hits, and with both, one request at a time, eager | prompt logprobs at every position bit for bit in 10 of 10 prompts of up to 4,096 tokens with chunked prefill at 512 and at 2,048; the first generated token's logprob bit for bit in 10 of 10 in all four configurations and 4 of 4 for four requests that share a prefix in one batch (the 64 tokens after it come from the decode kernel, which rounds as a batch's change does: the same as vLLM's in 4-5 of 10 prompts; `exact`: all) |
| `exact`: logits against stock's, eager | the 8 prompts' 64 greedy tokens and their logprobs bit for bit: 8 of 8 |
| a shared 2,048-token prompt (`vllm bench serve`, 64 requests): TTFT against stock's | mean TTFT **0.99x** eager, **0.97x** default mode; one user 1.02x and 1.05x; requests a second 1.10x |
| KV tokens against the first version's | at the standard flags 32,560 tokens (the first version 32,576, -0.05%); at the bench's flags 1.30x and 1.31x stock's |

What is bit for bit is everything FlashAttention computes, the prompt and the first token, and with `exact` the 64 tokens after it too. The first 64 generated tokens are not bit-equal with the decode kernel, whose sums are in another order than FlashAttention's (as far from vLLM's as vLLM's own batch change).

### Per configuration (`ext/`, `check_vllm.py --kv --ext-only --study`; one request at a time, `--max-num-seqs 1`, eager; stock and kv in processes of their own with the same flags)

| configuration | prompts | prompt logprobs bit for bit | first token's logprob | 64 tokens the same as stock's | tokens cached (kv = stock) | KV cache tokens: kv / stock |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| chunked prefill at 512 tokens | 10 x up to 4,096 | 10 of 10 | 10 of 10 | 5 of 10 |  | 42,080 / 32,240 |
| chunked prefill at 2,048 tokens | 10 x up to 4,096 | 10 of 10 | 10 of 10 | 4 of 10 |  | 40,640 / 31,136 |
| prefix hits (a prompt after another that shares its first 80%) | 10 x up to 3,564 | not asked (a request that asks skips the cache read) | 10 of 10 | 5 of 10 | 80-3,264 (the same as vLLM's) | 38,256 / 29,328 |
| prefix hits and chunked prefill at 512 tokens (a shared 80%, then 1,300 new tokens) | 10 x up to 4,564 | not asked (a request that asks skips the cache read) | 10 of 10 | 4 of 10 | 80-3,264 (the same as vLLM's) | 42,048 / 32,240 |

The four-request batch: first token's logprob bit for bit 4 of 4, 1,600 tokens cached each, the 64 tokens the same in 3 of 4. Failures: 0.

### TTFT: the shared 2,048-token prompt (`ttft/`; `bench_serve.sh`, DATASET=prefix: 1 prefix of 2,048 tokens, 128 of each request's own, 128 out; 64 requests at once, and 8 one at a time; the servers' defaults: prefix caching and chunked prefill on)

Eager (`ttft/ttft-eager2`):

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| inf | bf16 | 15.27 | 30,032 | 4.03 | 2,219 | 106.4 |
| inf | kv | 15.34 (1.00x) | 39,120 (1.30x) | 4.44 (1.10x) | 2,193 (0.99x) | 95.1 (0.89x) |
| one | bf16 | 15.27 | 30,032 | 0.13 | 82 | 62.0 |
| one | kv | 15.34 (1.00x) | 39,120 (1.30x) | 0.13 (1.01x) | 83 (1.02x) | 61.6 (0.99x) |

Default mode (compiled, piecewise graphs; warm; `ttft/ttft-default`):

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| inf | bf16 | 15.27 | 27,024 | 4.07 | 2,250 | 105.0 |
| inf | kv | 15.34 (1.00x) | 35,504 (1.31x) | 4.48 (1.10x) | 2,183 (0.97x) | 94.0 (0.90x) |
| one | bf16 | 15.27 | 27,024 | 0.13 | 78 | 61.3 |
| one | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.13 (1.00x) | 82 (1.05x) | 61.0 (1.00x) |

### Prompts that share nothing: 3,500 tokens in, 16 out, chunked prefill (default mode, warm; `ttft/ttft-long`)

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 15.27 | 27,024 | 0.79 | 1,967 | 268.3 |
| 1 | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.80 (1.00x) | 1,934 (0.98x) | 258.8 (0.96x) |
| inf | bf16 | 15.27 | 27,024 | 0.87 | 18,319 | 376.0 |
| inf | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.92 (1.05x) | 17,588 (0.96x) | 500.5 (1.33x) |
| one | bf16 | 15.27 | 27,024 | 0.51 | 1,051 | 61.9 |
| one | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.50 (1.00x) | 1,065 (1.01x) | 61.5 (0.99x) |

One user's first token costs 1.01x; a request at a time (1 a second) 0.98x; 32 at once 0.96x. The decode steps at saturation are slower (TPOT 1.33x): the cache holds 1.31x the requests, so more decode rows share every step.

### Prefix caching and chunked prefill on, as vLLM has them by default: the random dataset, 1,024 tokens in and 256 out, 256 requests (default mode, warm; `ttft/bench-m2-default`)

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| inf | bf16 | 15.27 | 27,024 | 0.76 | 149,449 | 109.9 |
| inf | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.94 (1.23x) | 120,418 (0.81x) | 116.9 (1.06x) |
| one | bf16 | 15.27 | 27,024 | 0.06 | 312 | 60.6 |
| one | kv | 15.34 (1.00x) | 35,504 (1.31x) | 0.06 (1.00x) | 320 (1.02x) | 60.6 (1.00x) |

## Under vLLM's engine with prefix caching and chunked prefill on (`kv_stress.py engine --ext`, 1,024 tokens a step, 64 sequences)

The same kinds of prompts as the stress runs of [`l4-vllm-kv-nostop-2026-10-01`](../l4-vllm-kv-nostop-2026-10-01): no engine stop, with and without `--verify` (a bf16 shadow of every layer's cache; every write and every context read back compared with it, 700 blocks):

| workload | prompts | values compared (verify run) |
| :--- | ---: | ---: |
| sinks: chat prompts (the first tokens of every sequence), 16 new tokens | 147 | 2,031,132,672 |
| natural windows, 32 new tokens | 300 | 9,135,341,568 |
| text in fourteen languages, 64 new tokens | 126 | 14,642,970,624 |
| code, tables, hex, base64, 32 new tokens | 100 | 17,014,284,288 |
| random token ids over the embedding matrix, 32 new tokens | 200 | 21,759,197,184 |
| random 1,024-token prompts as vllm bench makes them, 128 new tokens (ignore EOS) | 48 | 32,900,014,080 |
| a phrase or a few characters repeated (480 tokens), 256 new tokens (its continuation repeats) | 5 | 33,815,420,928 |
| the cache filled entirely: 10 prompts of one token repeated 1,000 times, 16 new tokens | 10 | 34,722,570,240 |
| one token repeated 4,000 times, 64 new tokens | 3 | 37,027,012,608 |

## The kv check suite on the final tree

`check_vllm.py --kv --no-kernels --no-ext --study` (vLLM's keys and values of 10 real prompts through the write kernel and back, verify in vLLM, the cache's size against stock's at the standard flags, prompt logprobs bit for bit, eager and
compiled in inductor's deterministic mode, the default compiled mode within the compile's variation, the refusals): 11 passed, 0 failed (`full/`). The kernels' checks and the extension's (`ext/`, above) are runs of their own: a process that has run the kernels' checks holds 1.4 GiB of the GPU for the rest of its life.

## Compiled mode: what is bit for bit, and what vLLM's own compiles vary by (`compiled/`)

In vLLM's default mode (torch.compile) inductor picks some kernels' variants by timing them, so two compiles of one model on two empty caches need not give the same bits. The first generated token's logprob of the 8 prompts,
compared bit for bit across compiles (each run a process of its own, `VLLM_CACHE_ROOT` empty unless said), falls in sets; each set is a group of runs with the same 8 values:

| set | runs |
| :--- | :--- |
| A | stock, the suite's first run (vLLM's default cache); stock, empty cache 1; kv lossless, empty cache 2; stock, the final suite's run |
| B | stock, empty cache 2; kv lossless, the suite's first run; kv lossless, the final suite's run |
| C | kv lossless, empty cache 1 |
| D | stock, deterministic mode, empty cache 1; stock, deterministic mode, empty cache 2; kv lossless, deterministic mode, empty cache 1; kv lossless, deterministic mode, empty cache 2 |

The largest difference in a first-token logprob between two sets:

| | A | B | C | D |
| :--- | ---: | ---: | ---: | ---: |
| A |  | 0.053 | 0.091 | 0.009 |
| B | 0.053 |  | 0.096 | 0.055 |
| C | 0.091 | 0.096 |  | 0.091 |
| D | 0.009 | 0.055 | 0.091 |  |

Each of stock's default-mode sets (A, B) holds a run of the plugin too, bit for bit: a compile of the plugin has vLLM's own first-token logprobs whenever the two compile alike. The plugin's compiles also landed on a set (C) that
stock's four compiles did not. The default mode is therefore checked for the same first token and for the continuations' mean |logprob difference| (0.008 to 0.010 in every pair of runs; stock against its other compile 0.0100), and the bits are checked in
inductor's deterministic mode (the last set: `inductor_compile_config` `deterministic`, `combo_kernels`, no `benchmark_combo_kernel`), where compiling twice on empty caches gives the logprobs of all 64 tokens bit for bit the same,
stock's (8 of 8 prompts) and the plugin's (8 of 8). In that mode, the plugin against stock:

- the first generated token's logprob of the 8 prompts: the same bits in all four runs (8 of 8 prompts);
- the prompt logprobs of a 1,542-token prompt (stock's own continuation fed back), 1,541 positions: bit for bit;
- the 64 greedy tokens: the plugin's the same as stock's from the start for [30, 64, 64, 64, 52, 64, 57, 51] tokens (the decode kernel's sums are in another order than FlashAttention's: the same spread as eager).

## Decode tokens a second on this version (`decode/tps/`, `kv_tps.py`)

KV cache: stock 34,848 tokens, kv 45,152 (1.296x); decode tokens a second (128 new tokens, eager, bf16 weights, util 0.96; one run each)

| Batch x tokens | Stock | KV | KV / stock |
| :--- | ---: | ---: | ---: |
| stock 4 x 8192, kv 4 x 8192 | 47.9 (83.5 ms a step; 1455, 74) | 51.4 (77.9 ms a step; 1455, 73) | 1.072 |
| stock 32 x 1024, kv 32 x 1024 | 354.5 (90.3 ms a step; 1560, 76) | 375.0 (85.3 ms a step; 1545, 75) | 1.058 |

The first version's, three rounds alternating: 1.065x at 4 x 8,192 and 1.053x at 32 x 1,024, the cache 45,184 tokens (1.297x).

## Files

`ext/` (the extension checks' result files: tokens, logprobs, prompt logprobs, tokens cached), `ttft/` (each bench's results JSON, its logs and summary), `full/` (the suite: each run's result file), `compiled/` (the compiles: each run's result file), `decode/tps/`. The tools are in [`gpu/vllm/`](../../../gpu/vllm). Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record.
