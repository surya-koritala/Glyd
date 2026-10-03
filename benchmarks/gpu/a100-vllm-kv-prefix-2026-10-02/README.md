# Prefix hits with the lossless KV cache on an A100, with controls (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an A100 SXM4 40 GB (Lambda), eager, prefix caching on; `kv: lossless` against vLLM's own cache ("stock" below).
14 prompts (8 short ones and 6 of this repository's documents), one request at a time: each whole prompt (one token generated), then a request of its first 80% (to a multiple of 16 tokens) and the first 300 tokens of the next prompt, 64 greedy tokens.
For every prefill step a hash of each of the 36 attention layers' query, key, value and output is kept, so that two runs are compared layer by layer.

## Result

| run | first token's logprob bit for bit stock's | the 64 greedy tokens identical to stock's | README.md: first token's logprob |
| :--- | ---: | ---: | ---: |
| stock | | | -0.745402 |
| kv | 13 of 14 | 8 of 14 | -0.684592 |
| kv, `exact` (decode steps through vLLM's own attention too) | **14 of 14** | **14 of 14** | -0.745402 |
| stock, its decode steps' keys in one split instead of FlashAttention's choice | 13 of 14 | 9 of 14 | -0.684007 |

- **The one first token that differs is README.md's**: its first request reads 304 cached tokens, 4 of them made by an earlier request's decode steps.
- **Every prefill step that reads a cached prefix written by prefill steps is bit for bit stock's in all 36 layers**: the second requests of the four prompts with 1,632, 3,264, 3,264 and 3,264 cached tokens (a fifth, 2,608 cached and 6 new tokens, has no per-layer hashes: its 64 tokens are stock's).
- **vLLM's own decode attention with one setting changed moves it the same way** (the control, the last row): the same prompt's first token's logprob differs by 0.0614 (kv: 0.0608), 13 of 14 first tokens are bit for bit and 9 of 14 sequences identical (kv: 13 and 8).
- **`exact` is bit for bit**: all 14 first tokens, all 14 sequences of 64 tokens, and every one of the 36 layers of README.md's first request.
- The sequences that part from stock's after a first token that is bit for bit: short prompts 5 and 7 (no cached prefix) at tokens 57 and 58, and the documents at tokens 6, 9 and 55 (the control: 7 at token 46, documents at 6, 13 and 55).

## Files

`runs/` (each run's result file: tokens, logprobs and cached tokens of every request, and the per-layer hashes of every prefill step), `compare/` (each run against stock's).
