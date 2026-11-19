# Prefix hits on an A100: the same scenario, per layer, with the controls (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an A100 SXM4 40 GB (Lambda), eager, prefix caching on; the scenario, the tool and its runs on the dev L4 are in [`l4-vllm-kv-prefix-2026-10-02`](../l4-vllm-kv-prefix-2026-10-02).
14 prompts (8 short ones and 6 of this repository's documents), one request at a time: each whole prompt (one token generated), then a request of its first 80% (to a multiple of 16 tokens) and the first 300 tokens of the next prompt, 64 greedy tokens.
For every prefill step a hash of each of the 36 attention layers' query, key, value and output is kept, so that two runs are compared layer by layer; for README.md's first request (1,024 tokens, 304 of them cached) the layer 1 and 2 tensors are compared value by value.

## Result

| run | first token's logprob bit for bit stock's | the 64 greedy tokens identical to stock's | README.md: first token's logprob | README.md's first request: first layer where its prefill differs from stock's |
| :--- | ---: | ---: | ---: | :--- |
| stock | | | -0.745402 | |
| kv | 13 of 14 | 8 of 14 | -0.684592 | layer 1, the attention output |
| kv, `exact` (decode steps through vLLM's own FlashAttention too) | **14 of 14** | **14 of 14** | -0.745402 | none: 36 layers bit for bit |
| stock, its decode steps' keys in one split instead of FlashAttention's choice | 13 of 14 | 9 of 14 | -0.684007 | layer 1, the attention output |

- **The one first token that differs is README.md's**, as on the L4: its first request reads 304 cached tokens, 4 of them keys and values that an earlier request's decode steps made. Query, key and value of layer 1 are bit for bit stock's; the attention output is not.
- **Every prefill step that reads a cached prefix written by prefill steps is bit for bit stock's in all 36 layers**: the second requests of the four prompts with 1,632, 3,264, 3,264 and 3,264 cached tokens (a fifth, 2,608 cached and 6 new tokens, has no per-layer hashes: its 64 tokens are stock's).
- **vLLM's own decode attention with one change moves it the same way**: with only the keys of its decode steps in one split (FlashAttention 2 against FlashAttention 2), the same prompt's first token's logprob differs by 0.0614 (kv: 0.0608), the first layer that differs is layer 1's attention output, 35 of 36 layers differ after it (kv: the same), 13 of 14 first tokens are bit for bit and 9 of 14 sequences identical (kv: 13 and 8).
- **`exact` is bit for bit**: all 14 first tokens, all 14 sequences of 64 tokens, and every one of the 36 layers of README.md's first request.
- The sequences that part from stock's after a first token that is bit for bit: short prompts 5 and 7 (no cached prefix) at tokens 57 and 58, and the documents at tokens 6, 9 and 55 (the control: 7 at token 46, documents at 6, 13 and 55).

How far README.md's first request's layers part (720 new tokens; elements that differ, the largest difference (the largest value)):

| layer | tensor | kv against stock | stock with its decode steps' keys in one split, against stock |
| ---: | :--- | :--- | :--- |
| 1 | query, key, value | 0 | 0 |
| 1 | attention output | 37,356 of 2,949,120; 9.8e-4 (0.53) | 31,266 of 2,949,120; 9.8e-4 (0.53) |
| 2 | query | 688,849 of 2,949,120; 9.4e-2 (14.56) | 654,255 of 2,949,120; 9.4e-2 (14.56) |
| 2 | key | 157,653 of 737,280; 1.3e-1 (21.75) | 150,657 of 737,280; 1.9e-1 (21.75) |
| 2 | value | 219,243 of 737,280; 3.9e-3 (0.78) | 209,558 of 737,280; 3.9e-3 (0.78) |
| 2 | attention output | 1,554,143 of 2,949,120; 3.9e-3 (0.77) | 1,510,301 of 2,949,120; 3.9e-3 (0.77) |

Same tree and tools as the L4 record (tree 32795af3; the job's `pfx` step).

## Files

`runs/` (each run's result file: tokens, logprobs and cached tokens of every request, and the per-layer hashes of every prefill step), `compare/` (`kv_prefix_control.py compare` and `dumps` of each run against stock's). The tool is [`gpu/vllm/kv_prefix_control.py`](../../../gpu/vllm/kv_prefix_control.py).
