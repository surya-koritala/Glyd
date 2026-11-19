# Prefix hits: the one prompt whose first token differs from vLLM's, and what moves it (2026-10-02)

Qwen3-8B, bf16 weights, vLLM 0.30.0, the dev L4, eager. The scenario is the extension checks' (`check_vllm.py --kv --ext-only`, "prefix hits"): 14 prompts (8 short ones and 6 of this repository's documents), one request at a time, prefix caching on.
For each prompt the whole prompt is run (one token generated), then a request of its first 80% (to a multiple of 16 tokens) and the first 300 tokens of the next prompt, 64 greedy tokens (README.md's second request: 816 tokens cached, 300 new).
[`kv_prefix_control.py`](../../../gpu/vllm/kv_prefix_control.py) runs it and keeps, for every prefill step, a hash of each of the 36 attention layers' query, key, value and output, so that two runs are compared layer by layer: the first layer where they differ is where they part.

On the A100 SXM4 40 GB (the job `kva`) this check had one first token that was not vLLM's bit for bit (13 of 14) and 8 of 14 sequences of 64 tokens identical. This record is what the prompt is, on the L4.

## Result

| run (the check's order of the prompts unless said) | first token's logprob bit for bit stock's | the 64 greedy tokens identical to stock's | README.md: first token's logprob | README.md's first request: first layer where its prefill differs from stock's |
| :--- | ---: | ---: | ---: | :--- |
| stock | | | -0.710262 | |
| stock, the 8 short prompts in the reverse order | 14 of 14 | 14 of 14 | -0.710262 | none: 36 layers bit for bit |
| kv | 13 of 14 | 10 of 14 | -0.746714 | layer 1, the attention output |
| kv, `exact` (decode steps through vLLM's own FlashAttention too) | **14 of 14** | **14 of 14** | -0.710262 | none: 36 layers bit for bit |
| stock, its decode steps' keys in one split instead of FlashAttention's choice | 13 of 14 | 12 of 14 | -0.744788 | layer 1, the attention output |
| README.md alone in a new engine: stock, against stock in the sequence | differs (-0.747907 against -0.710262) | the same for 43 of 64 tokens | -0.747907 | layer 0 (no cached prefix) |
| README.md alone in a new engine: kv, against stock alone | bit for bit | the same for 43 of 64 tokens | -0.747907 | none: 36 layers bit for bit |

- **Every prefill step that reads a cached prefix written by prefill steps is bit for bit vLLM's in all 36 layers**: the second requests of four prompts with 1,632, 3,264, 3,264 and 3,264 cached tokens (query, key, value and output of every layer, kv against stock; a fifth, 2,608 cached and 6 new tokens, has no per-layer hashes: its 64 tokens are stock's), and with `exact` the 14 prompts' every first token and every sequence of 64 tokens and their logprobs.
- **The prompt that differs is README.md.** Its first request reads 304 cached tokens, which are the 300 tokens of the request before it (its prompt) and the first 4 tokens that request generated. Those 4 tokens' keys and values were made by that run's decode steps. kv's decode attention sums in another order than FlashAttention does, so those keys and values differ from stock's in the last bits, and the first request that reads them differs from the first layer that attends to them: query, key and value of layer 1 are bit for bit stock's, the attention output is not. 16 and 16 tokens of two other prompts' first requests were cached from tokens that prefill steps wrote: both bit for bit.
- **vLLM's own decode attention with one change moves the same prompt the same way.** Stock with only the keys of its decode steps in one split (FlashAttention 2 against FlashAttention 2): README.md's first token differs by 0.0345 (kv: 0.0365), the first layer that differs is layer 1's attention output and 35 of 36 layers differ after it (kv: the same), 13 of 14 first tokens bit for bit and 12 of 14 sequences identical (kv: 13 and 10). Stock against itself with the short prompts in another order: all 14 and all 14.
- **The same scale without a prefix hit**: README.md alone in a new engine (no cached prefix, its 1,024 tokens computed in one step) differs from README.md in the sequence (304 cached) in stock itself, from layer 0: first token -0.747907 against -0.710262, 43 of 64 tokens the same. kv alone against stock alone: bit for bit in all layers, first token bit for bit.
- **Storage**: kv with `verify` over the same scenario (every write and every context read against a bf16 copy of what was written): 3,168,829,440 values, none differ, and its tokens and logprobs are the run without `verify`'s (14 of 14).

How far the first request's layers part (README.md's first request, 720 new tokens, layer outputs of 4,096 values a token; elements that differ of 2,949,120, the largest difference (the largest value)):

| layer | kv against stock | stock with its decode steps' keys in one split, against stock |
| ---: | :--- | :--- |
| 0 | 0 | 0 |
| 1 | 28,052; 1.2e-3 (0.53) | 27,778; 1.2e-3 (0.53) |
| 2 | 1,685,121; 5.9e-3 (0.77) | 1,674,116; 3.9e-3 (0.77) |
| 3 | 2,138,906; 1.7e-2 (0.66) | 2,135,611; 1.4e-2 (0.66) |

On the A100 (the job `kvs`; [`a100-vllm-kv-prefix-2026-10-02`](../a100-vllm-kv-prefix-2026-10-02)) the same scenario gives the same: kv 13 of 14 first tokens bit for bit and 8 of 14 sequences identical; `exact` 14 of 14 and 14 of 14; stock with its decode steps' keys in one split 13 of 14 and 9 of 14; README.md's first request first differs at layer 1's attention output in kv and in that control, in no layer with `exact`.

## What changed in the checks

The prefix-hit check no longer counts, among the first tokens that must be vLLM's bit for bit, the prompts whose first request read blocks an earlier request's decode steps wrote (it lists them, with whether they were bit for bit); and a second check runs the same prompts with `exact`: every first token and every sequence of 64 tokens must be vLLM's bit for bit (14 of 14).

## Files

`runs/` (each run's result file: tokens, logprobs and cached tokens of every request, and the per-layer hashes of every prefill step), `compare/` (`kv_prefix_control.py compare` of each run against its reference: for each prompt the cached tokens, the first token, the sequences' agreement and the first layer that differs). The tool is [`gpu/vllm/kv_prefix_control.py`](../../../gpu/vllm/kv_prefix_control.py); the runs are `run MODEL stock|kv OUT [--order fwd|rev|fresh:N] [--exact] [--verify] [--decode-splits 1]`.
