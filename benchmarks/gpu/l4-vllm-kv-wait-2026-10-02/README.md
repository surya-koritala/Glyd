# The lossless KV cache on five models: the KV room, the values, and floods of rare-token prompts (2026-10-02)

vLLM 0.30.0, an NVIDIA L4 (24 GB), 2026-10-02. Ungated mirrors; `kv: lossless` against vLLM's own cache ("stock" below).
Rare-token prompts are prompts of a model's rare tokens (tokens whose embeddings were never trained: Llama-3.1-8B has 289, Mistral-7B-v0.3 902, the Qwen models none).

- No request that stock vLLM serves ends with an error. 40 prompts of 1,400 rare tokens at once with 8 ordinary ones: all 40 served on every one of the five models, and 24 concurrent over `vllm serve` on Llama-3.1-8B: all 24 served. Where there are many of them at once, the others wait their turn, as vLLM's own requests do when KV blocks are short.
- **Known limit:** prompts made mostly of rare tokens are served more slowly when many arrive at once: the flood ran at 0.88x (Llama-3.1-8B) and 0.77x (Mistral-7B) of stock's prompt tokens a second, and at 1.00x to 1.01x on the Qwen models, which have no such token.
- The 8 ordinary requests sent with a flood are served and are not delayed behind the flood's prompts: they finish in 8.1 s (Mistral-7B) and 10.5 s (Llama-3.1-8B) where stock's, queued behind the flood, take 19.0 s and 19.8 s (alone 3.8 s and 4.2 s); under `VLLM_BATCH_INVARIANT=1` their tokens and logprobs are the same, bit for bit, as without the flood (8 of 8 on every model measured; stock's own, the control: 8 of 8). Over `vllm serve` the 6 ordinary requests took 4.0 s in the flood against 0.95 s alone and stock's 9.4 s, with the same text; in a steady stream with bursts (510 requests) the ordinary requests' latency was stock's.
- The KV cache is 1.25x to 1.30x stock's on the five models (below).
- Every value read back is the value written: 0 differ in every run below (real prompts' keys and values written and read back, `verify` against a bf16 copy of each layer, the floods).

## KV cache per model

| Model | layers x KV heads | KV tokens, stock -> kv (8,704 tokens a request and a step) | kv / stock |
| :--- | ---: | ---: | ---: |
| Qwen3-4B | 36 x 8 | 83,008 -> 107,760 | 1.2982x |
| Qwen3-8B | 36 x 8 | 25,232 -> 32,384 | 1.2834x |
| Qwen2.5-7B | 28 x 4 | 77,152 -> 98,544 | 1.2773x |
| Mistral-7B-v0.3 | 32 x 8 | 43,520 -> 55,456 | 1.2743x |
| Llama-3.1-8B | 32 x 8 | 30,464 -> 38,160 | 1.2526x |

## Values

| Model | real prompts' keys and values written and read back: values / differ | verify: values / differ | the model's checks |
| :--- | ---: | ---: | ---: |
| Qwen3-4B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 13 of 13 |
| Qwen3-8B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 21 of 21 |
| Qwen2.5-7B | 770,129,920 / 0 | 2,461,405,184 / 0 | 13 of 13 |
| Mistral-7B-v0.3 | 1,845,166,080 / 0 | 5,982,126,080 / 0 | 21 of 21 |
| Llama-3.1-8B | 1,726,021,632 / 0 | 5,591,007,232 / 0 | 21 of 21 |

## Prompts of rare tokens, one at a time

| Model | prompts | verify: values / differ | prompt logprobs bit for bit stock's | first token bit for bit stock's |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 3 | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen3-8B | 3 | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen2.5-7B | 3 | 106,917,888 / 0 | 3 of 3 | 3 of 3 |
| Mistral-7B-v0.3 | 8 | 556,793,856 / 0 | 8 of 8 | 8 of 8 |
| Llama-3.1-8B | 13 | 872,677,376 / 0 | 13 of 13 | 13 of 13 |

## 40 prompts of 1,400 rare tokens at once with 8 ordinary ones, kv / stock

| Model | served | ended with an error or a 400 | wall seconds | prompt tokens a second | kv / stock throughput | the 8 ordinary prompts' latency alone, s (median) | in the flood, s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 40 of 40 / 40 of 40 | 0 / 0 | 11.25 / 11.3 | 4,985 / 4,964 | 1.00x | 2.41 / 2.40 | 11.25 / 11.30 |
| Qwen3-8B | 40 of 40 / 40 of 40 | 0 / 0 | 19.97 / 20.17 | 2,810 / 2,782 | 1.01x | 4.26 / 4.25 | 19.97 / 20.17 |
| Qwen2.5-7B | 40 of 40 / 40 of 40 | 0 / 0 | 17.34 / 17.42 | 3,236 / 3,220 | 1.00x | 3.89 / 3.88 | 17.34 / 17.42 |
| Mistral-7B-v0.3 | 40 of 40 / 40 of 40 | 0 / 0 | 24.55 / 19.03 | 2,286 / 2,950 | 0.77x | 3.82 / 3.80 | 8.06 / 19.03 |
| Llama-3.1-8B | 40 of 40 / 40 of 40 | 0 / 0 | 22.4 / 19.76 | 2,506 / 2,839 | 0.88x | 4.21 / 4.22 | 10.52 / 19.76 |

(Each cell is kv / stock.)

## The ordinary prompts' bits in the flood (`VLLM_BATCH_INVARIANT=1`)

| Model | flood served | kv: the 8 ordinary prompts' 64 greedy tokens and logprobs the same bits as alone | stock (the control) | ordinary latency alone / in the flood, s | flood wall seconds kv / stock |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 40 of 40 | 8 of 8 | 8 of 8 | 5.10 / 22.72 | 22.72 / 23.43 |
| Mistral-7B-v0.3 | 40 of 40 | 8 of 8 | 8 of 8 | 4.58 / 9.37 | 28.58 / 22.09 |
| Llama-3.1-8B | 40 of 40 | 8 of 8 | 8 of 8 | 4.97 / 12.33 | 26.24 / 22.72 |

## `vllm serve` (defaults, an empty set-up cache, `--max-model-len 4096 --max-num-seqs 32`), kv against stock

24 concurrent prompts of 1,400 tokens of the model's rare tokens (Llama-3.1-8B: 85 of its 289; Qwen3-8B, which has none: one repeated token, behind a token of each prompt's own), with 6 ordinary requests (16 tokens) sent 0.3 to 0.8 s after the flood started.

| Model | | start to serving, s (kv: the first start's set-up included) | the 24 rare-token prompts | the flood's seconds | prompt tokens a second | the ordinary requests (HTTP) | their latency in the flood, median, s | their text the same as alone | a prompt of 700 rare tokens after it | a prompt of 4,000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | kv lossless | 258 | 24 of 24 (HTTP 200) | 11.43 | 2,940 | 200 | 4.024 (0.953 alone) | yes | 200 | 200 (1.29 s) |
| Llama-3.1-8B | stock | 54 | 24 of 24 (HTTP 200) | 9.99 | 3,363 | 200 | 9.428 (0.952 alone) | yes | 200 | 200 (1.28 s) |
| Qwen3-8B | kv lossless | 159 | 24 of 24 (HTTP 200) | 10.17 | 3,304 | 200 | 9.603 (0.965 alone) | yes | 200 | 200 (1.36 s) |
| Qwen3-8B | stock | 66 | 24 of 24 (HTTP 200) | 10.86 | 3,095 | 200 | 10.291 (0.964 alone) | NO | 200 | 200 (1.36 s) |

### A steady stream with bursts: Llama-3.1-8B over `vllm serve`, 3 ordinary requests a second (32 tokens) for 150 s, a burst of 12 prompts of 1,400 rare tokens every 30 s

|  | ordinary requests | ordinary latency out of a burst, s (median / p95 / max) | during a burst | rare-token requests | their latency, s (median / max) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| kv lossless | 450 (200: 450) | 2.074 / 4.674 / 6.758 | 4.356 / 6.565 / 6.83 | 60 (200: 60) | 3.659 / 6.406 |
| stock | 450 (200: 450) | 2.068 / 6.605 / 6.851 | 4.456 / 6.775 / 6.943 | 60 (200: 60) | 4.908 / 5.428 |

## The kernels, the sanitizer and a stress run

- The kernels' checks: 64 checks, all passed.
- compute-sanitizer over the kernels' quick checks (`sanitizer/`): memcheck: 0 errors; initcheck: 0 errors; racecheck: 0 hazards displayed (0 errors, 0 warnings); synccheck: 0 errors.
- A stress run (Qwen3-8B, quick): 25,683,296,256 values written and read back: 0 differ, none lost. Through vLLM's engine, with `verify`: 12,986,523,648 and 13,988,044,800 values compared, none differ; with prefix caching and chunked prefill on: no engine stop.

## Llama-3.1-8B: the full run (34 of 35 checks pass)

The model's checks, the floods, the compiled and FULL-graph runs, batch invariance, prefix caching, chunked prefill and `exact`, against vLLM's own.

| | check |
| :--- | :--- |
| PASS | the first start's set-up (32,768 tokens, 96 s); 289 of the 128,256 tokens are rare ones (21 that the tokenizer makes from text, 268 only by token ids) |
| PASS | vLLM's own keys and values of 14 real prompts (26,337 tokens, 1,726,021,632 values over 32 layers) written and read back: 0 differ |
| PASS | `verify`, 14 prompts of 6 to 8,192 tokens and 512 generated each: 5,591,007,232 values, none differ |
| PASS | KV cache 38,160 tokens against stock's 30,464 at the standard flags (1.2526x; eager, `--max-model-len 8704`, 0.9); 27,232 against 21,872 (1.2451x) at `--max-model-len 4096`, 0.85 |
| PASS | prompt logprobs of a 1,543-token prompt bit for bit stock's; the 8 prompts' first generated token's logprob bit for bit stock's, their 64 greedy tokens the same from the start for 64, 41, 64, 19, 64, 57, 64 and 64 tokens |
| PASS | vLLM's default mode (compiled, piecewise graphs and FULL decode graphs, 1,718 captured): the first generated tokens stock's, their logprobs within 0.000 of stock's; the FULL decode graphs alone (35 captured) replay eager's tokens and logprobs bit for bit (8 of 8) |
| PASS | `VLLM_BATCH_INVARIANT=1`, eager and FULL graphs: each of 8 prompts' 64 tokens and logprobs the same bits alone and in a batch of 8 (8 of 8 each; vLLM's own the control: 8 of 8); compiled in inductor's deterministic mode, both: the first token's logprob bit for bit (8 of 8) and a 1,543-token prompt's logprobs |
| PASS | chunked prefill (512 and 2,048 tokens, 14 prompts of up to 4,096): prompt logprobs bit for bit stock's (14 of 14) and the first token's logprob (14 of 14); prefix hits with chunked prefill 14 of 14; four requests sharing 1,600 tokens 4 of 4; eight sharing a prefix 8 of 8; requests with prefixes of their own bit for bit |
| FAIL | prefix hits (14 prompts of up to 3,564 tokens, one request at a time): the first token's logprob bit for bit stock's in 13 of 14. The prompt that differs reads a cached prefix that holds 4 tokens an earlier request generated (the same scenario on Qwen3-8B, on an A100: [`a100-vllm-kv-prefix-2026-10-02`](../a100-vllm-kv-prefix-2026-10-02), where `exact` makes all 14 bit for bit); in a fresh engine all 32 layers of its prefill are bit for bit stock's; `verify` over the 14 prompts: no value differs |
| PASS | `exact`: 8 prompts' 64 tokens and logprobs bit for bit stock's (8 of 8) |
| PASS | 13 prompts of the rare tokens: every write compared with a bf16 copy (872,677,376 values) none differ; prompt logprobs bit for bit stock's (13 of 13), the first token (13 of 13) |
| PASS | the floods of this record, in the same run (tables above): 40 of 40 served, with and without prefix caching, under `VLLM_BATCH_INVARIANT=1` |

## Files

`models/<model>/` (each model's result files: the checks, the floods, the rare-token runs; JSON, stock beside kv), `serve-*` (each `vllm serve` run's result file), `sanitizer/`. Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record.
