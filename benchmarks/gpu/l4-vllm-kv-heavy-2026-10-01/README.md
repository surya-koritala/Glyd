# Prompts of rare tokens on five models, and Llama-3.1-8B served (2026-10-02)

vLLM 0.30.0, the dev L4 (24 GB). Ungated mirrors. This record measures the lossless KV cache on five models, including the tokens whose embeddings were never trained (rare tokens): a prompt of them is held in more than 12 bits, in a second reserve the cache keeps for it.

- Every value read back is the value written: 0 differ in every run below (real prompts through the write kernel and back, `verify` against a bf16 shadow of each layer, the stress runs).
- The engine never stopped, in any run, on any prompt.
- Llama-3.1-8B is served by default: 21 of its ordinary tokens have zero embedding rows, which an earlier version refused.
- The two reserves take 2.3% to 4.7% of the KV cache on these models (table below); the KV cache is 1.24x to 1.29x stock's. (The later version's, with smaller reserves: 1.25x to 1.30x, [`l4-vllm-kv-wait-2026-10-02`](../l4-vllm-kv-wait-2026-10-02).)

## KV cache per model

| Model | layers x KV heads | vocabulary swept | rare tokens: from text / ids only | first reserve | second reserve | KV tokens, stock -> kv (8,704 tokens a request and a step) | bits a value |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 36 x 8 | 151,936 (50 s) | 0 / 0 | 2.13% | 0.23% | 83,008 -> 107,264 (1.2922x) | 12.38 |
| Qwen3-8B | 36 x 8 | 151,936 (68 s) | 0 / 0 | 2.07% | 0.23% | 25,232 -> 32,224 (1.2771x) | 12.53 |
| Qwen2.5-7B | 28 x 4 | 152,064 (63 s) | 0 / 0 | 2.38% | 0.29% | 77,152 -> 97,968 (1.2698x) | 12.60 |
| Mistral-7B-v0.3 | 32 x 8 | 32,768 (14 s) | 0 / 902 | 1.82% | 1.55% | 43,520 -> 55,216 (1.2688x) | 12.61 |
| Llama-3.1-8B | 32 x 8 | 128,256 (56 s) | 21 / 268 | 3.19% | 1.55% | 30,464 -> 37,856 (1.2426x) | 12.88 |

## Values

| Model | real prompts through the write kernel and back: values / differ | verify: values / differ | the model's checks |
| :--- | ---: | ---: | ---: |
| Qwen3-4B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 14 of 14 |
| Qwen3-8B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 14 of 14 |
| Qwen2.5-7B | 770,129,920 / 0 | 2,461,405,184 / 0 | 14 of 14 |
| Mistral-7B-v0.3 | 1,845,166,080 / 0 | 5,982,126,080 / 0 | 14 of 14 |
| Llama-3.1-8B | 1,726,021,632 / 0 | 5,591,007,232 / 0 | 14 of 14 |

## Prompts of the rare tokens, one at a time

Each rare token alone x 700, drawn at random x 1,000, and in runs of 200 among text (the Qwen models have none: the token that takes the most, repeated).

| Model | prompts | verify: values / differ | prompt logprobs bit for bit stock's | first token bit for bit stock's |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 3 | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen3-8B | 3 | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen2.5-7B | 3 | 106,917,888 / 0 | 3 of 3 | 3 of 3 |
| Mistral-7B-v0.3 | 8 | 556,793,856 / 0 | 8 of 8 | 8 of 8 |
| Llama-3.1-8B | 13 | 872,677,376 / 0 | 13 of 13 | 13 of 13 |

## Llama-3.1-8B: the full run (28 of 29 checks pass)

The model's checks, the compiled and FULL-graph runs, batch invariance, prefix caching, chunked prefill and `exact`, against vLLM's own (`models/llama31-8b/`).

- Calibration (32,768 tokens, 97 s; the sweep of the 128,256-token vocabulary, 56 s): 289 tokens need the second reserve (21 that the tokenizer makes from text, 268 only by token ids).
- vLLM's keys and values of 14 real prompts (26,337 tokens, 1,726,021,632 values over 32 layers) through the write kernel and back: 0 differ. `verify`, 14 prompts of 6 to 8,192 tokens and 512 generated each: 5,591,007,232 values, none differ.
- KV cache 37,856 tokens against stock's 30,464 (1.2426x, 12.88 bits a value; at least 1.241x), eager, `--max-model-len 8704`, 8,704 tokens a step, 0.9, bf16 weights.
- Prompt logprobs of a 1,543-token prompt bit for bit stock's; the 8 prompts' first generated token's logprob bit for bit stock's; their 64 greedy tokens the same as stock's from the start for [64, 41, 64, 19, 64, 57, 64, 64] tokens, mean abs logprob difference on the shared prefixes 0.0087.
- vLLM's default mode (torch.compile, piecewise CUDA graphs): the 8 prompts' first generated token stock's but for prompt 2 (a near tie), its logprob within 0.092 of stock's (inductor picks kernels by timing them: compiles of stock's own differ too); FULL_AND_PIECEWISE, 1,718 graphs captured. The FULL decode graphs alone (35 captured) replay eager's tokens and logprobs bit for bit (8 of 8).
- Under `VLLM_BATCH_INVARIANT=1` (bf16 weights, kv lossless; eager, and FULL decode graphs with no torch.compile): each of 8 prompts' 64 greedy tokens and their logprobs the same bits alone and in a batch of 8 (8 of 8 each; vLLM's own bf16 under the same variable and mode, the control: 8 of 8). Compiled in inductor's deterministic mode (both): the first token's logprob bit for bit stock's (8 of 8) and a 1,543-token prompt's logprobs bit for bit.
- Chunked prefill (512 and 2,048 tokens, 14 prompts of up to 4,096 tokens one request at a time, eager): prompt logprobs bit for bit stock's (14 of 14 each), the first token's logprob bit for bit (14 of 14 each), the 64 greedy tokens the same as stock's in 11 of 14 and 10 of 14.
- Prefix hits (a prompt after another that shares its first 80%, 14 prompts of up to 3,564 tokens): the first token's logprob bit for bit stock's in **13 of 14** (the one miss, below); with chunked prefill at 512 tokens (a shared 80%, then 1,300 new tokens): 14 of 14. Four requests that share their first 1,600 tokens in one batch, requests (up to six) with a cached prefix of 2,048 tokens each of their own, eight requests that share a cached prefix of 1,600 tokens: the first generated token's logprob bit for bit stock's. Metadata built under torch's synchronization errors: none raised, the outputs the run without them's.
- `exact`: the 8 prompts' 64 greedy tokens and logprobs bit for bit stock's, 8 of 8 (tokens the same from the start for [64, 41, 64, 64, 64, 64, 64, 64]).
- 13 prompts of the model's rare tokens (21 made by the tokenizer from text, 268 only by token ids; each alone x 700, drawn at random x 1,000, runs of 200 among text): every write compared with a bf16 shadow of each layer: 872,677,376 values, none differ; prompt logprobs bit for bit stock's (13 of 13), the first token and its logprob bit for bit (13 of 13); the 24 greedy tokens the same as stock's from the start for [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 5, 9, 24] tokens.

### vLLM's default mode (compiled), each run on an empty compile cache: the first generated token of the 8 prompts (its logprob) (`llama-default/`)

| prompt | stock 1 | stock 2 | stock 3 (the checks' run) | kv 1 | kv 2 | kv 3 (the checks' run) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 304 (-1.0373) | 304 (-1.0373) | 304 (-1.0440) | 304 (-1.0373) | 304 (-1.0440) | 304 (-1.0373) |
| 1 | 262 (-0.0175) | 262 (-0.0175) | 262 (-0.0175) | 262 (-0.0175) | 262 (-0.0175) | 262 (-0.0175) |
| 2 | 220 (-1.5098) | 220 (-1.5098) | 578 (-1.4180) | 220 (-1.5098) | 578 (-1.4180) | 220 (-1.5098) |
| 3 | 69890 (-0.5517) | 69890 (-0.5517) | 69890 (-0.5645) | 69890 (-0.5517) | 69890 (-0.5645) | 69890 (-0.5517) |
| 4 | 38840 (-2.0119) | 38840 (-2.0119) | 38840 (-1.9650) | 38840 (-2.0119) | 38840 (-1.9650) | 38840 (-2.0119) |
| 5 | 279 (-1.3605) | 279 (-1.3605) | 279 (-1.3652) | 279 (-1.3605) | 279 (-1.3652) | 279 (-1.3605) |
| 6 | 10373 (-0.2796) | 10373 (-0.2796) | 10373 (-0.2754) | 10373 (-0.2796) | 10373 (-0.2754) | 10373 (-0.2796) |
| 7 | 1070 (-0.5646) | 1070 (-0.5646) | 1070 (-0.6015) | 1070 (-0.5646) | 1070 (-0.6015) | 1070 (-0.5646) |

## Kernels and sanitizer

- Kernel suite (`check_vllm.py --kv --kernels-only`): 63 checks, all passed.
- compute-sanitizer over the kernels' quick checks (`sanitizer/`): memcheck: 0 errors; initcheck: 0 errors; racecheck: 0 hazards displayed (0 errors, 0 warnings); synccheck: 0 errors.

## kv_stress (Qwen3-8B, quick)

- capture: 25,697,452,032 values written through the kernel and read back: 0 differ, none lost.
- engine, with `verify` (700 blocks): 13,988,044,800 values compared, no engine stop; engine, prefix caching and chunked prefill on: no engine stop.

## The prefix-hit prompt whose first token's logprob differs from stock's (Llama-3.1-8B, eager)

The repository's README.md (1,024 tokens) after 8 other prompts, then a request of its first 816 tokens and 300 of another prompt's (816 tokens cached in both engines).

| run | first generated token's logprob |
| :--- | ---: |
| stock, a fresh engine | -0.5760883 |
| kv, a fresh engine | -0.5760883 |
| stock, after 8 other prompts (either order) | -0.5738637 |
| kv, after 8 other prompts (either order) | -0.5870041 |

Per attention layer, the 300 x 4096 outputs of that request's prefill: stock against kv in a fresh engine, 32 of 32 layers bit for bit; stock in a fresh engine against stock after 8 prompts, layer 0 already differs (largest difference 1.95e-3, all 300 tokens), layer 2 1.2e-2; stock against kv after 8 prompts, layer 0 identical, layer 1 differs (2.4e-4), layer 3 9.8e-3.
`verify` (every write and every context read back against the shadow) over the 14 prompts of that scenario: no value differs. Repeating a run gives the same bits (stock twice, kv twice). What moves the prompt (it reads a cached prefix that holds tokens an earlier request generated, made by the decode kernel; vLLM's own FlashAttention with another split of the keys in its decode steps moves it the same way; with `exact` the scenario is bit for bit) is in [`l4-vllm-kv-prefix-2026-10-02`](../l4-vllm-kv-prefix-2026-10-02).

## Files

`models/<model>/` (each model's check results: JSON, stock beside kv), `llama-default/` (the default-mode runs above), `sanitizer/`. The tools are in [`gpu/vllm/`](../../../gpu/vllm). Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record.
