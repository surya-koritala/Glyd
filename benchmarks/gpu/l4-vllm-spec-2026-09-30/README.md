# Speculative decoding with Glyd in vLLM, on an L4 (2026-09-30)

One user, greedy, Qwen3-8B: bf16 and `--quantization glyd`, each without speculation, with vLLM's n-gram prompt lookup
and with an EAGLE-3 draft. Every run is `gpu/vllm/spec_decode.py`, a vLLM of its own.

## Setup

- **Machine:** the dev L4 (AWS g6.4xlarge, 24 GB, 72 W), vLLM 0.30.0, v0.25.1's library. The plugin is the
  vllm-plugin branch after review 1 (5513d94) with this task's fix for a packed draft.
- **Model:** Qwen/Qwen3-8B, Glyd's layout the L4's, the smallest layout (`mma`), `--gpu-memory-utilization 0.9`,
  `max_model_len` 4096. Default is compiled with CUDA graphs; runs named `eager` use `--enforce-eager`.
- **Speculation:**
  - n-gram: 5 tokens, lookup of 2 to 4;
  - EAGLE-3: `RedHatAI/Qwen3-8B-speculator.eagle3` (Apache-2.0, ungated, revision 08610ff, 2.0 GB, 3 tokens). vLLM
    reads its speculators config natively; no remote code runs.
- **Prompts (`spec_decode.py`),** each with thinking off and at most 400 tokens out:
  - "edit": 5 prompts that fix, annotate, convert, rewrite or summarize a given text, code or data;
  - "chat": 5 open questions and requests.
  - Each run sends two warm-up requests first, then one request at a time.
- **Measures:**
  - output tokens/s: a mix's tokens over its requests' whole times;
  - TTFT: vLLM's own time to the first token, a mean over the mix;
  - acceptance: vLLM's counters, accepted draft tokens over proposed ones, and 1 + accepted a draft (tokens a step).

## Speed (`summary.txt`)

| Run | Edit: tokens/s | Chat: tokens/s | Edit: TTFT | Chat: TTFT | Draft tokens accepted (edit, chat) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| bf16 | 16.6 | 16.7 | 150 ms | 82 ms | |
| bf16, n-gram | 27.1 | 16.7 | 107 ms | 71 ms | 40%, 12% |
| bf16, EAGLE-3 | did not fit | | | | |
| bf16, EAGLE-3, util 0.95, 2,048 tokens a step | 43.3 | 30.3 | 171 ms | 119 ms | 73%, 39% |
| Glyd | 21.2 | 21.5 | 178 ms | 61 ms | |
| Glyd, n-gram | 35.5 | 22.1 | 152 ms | 57 ms | 40%, 12% |
| Glyd, EAGLE-3 | 55.0 | 38.9 | 201 ms | 102 ms | 74%, 39% |
| Glyd, EAGLE-3, its draft packed too | 56.0 | 39.7 | 190 ms | 78 ms | 73%, 39% |

- **Glyd with EAGLE-3** made 3.3x bf16's tokens a second on the edit mix, and 2.3x on chat. Against bf16 with the
  same draft it made 1.27x and 1.28x.
- **bf16 with EAGLE-3 does not fit this L4** at vLLM's default memory settings: 15.3 GiB of weights and the 2.0 GB draft
  left no KV cache (`bf16-eagle3.log`: "0.47 GiB available", 0.58 GiB needed for one request of 4,096 tokens). With
  `--gpu-memory-utilization 0.95` and `--max-num-batched-tokens 2048` it fits; those settings alone do not change bf16's
  speed (`bf16-tight`: 16.6 and 16.7 tokens/s).
- **n-gram** helps where the answer repeats the prompt: 1.6-1.7x on the edit mix, about nothing on chat.
- **Time to the first token:** Glyd's is later on the edit mix's longer prompts, and sooner on chat's short ones. The
  EAGLE-3 draft adds its own prefill.

## Correctness: the same tokens, request by request

| Runs compared | Requests with the same tokens |
| :--- | :--- |
| Glyd exact, eager, n-gram, against bf16 eager, n-gram | 10 of 10 |
| Glyd exact, eager, EAGLE-3, against bf16 eager, EAGLE-3 | 10 of 10 |
| Glyd exact, eager, EAGLE-3 with its draft packed too, against bf16 eager, EAGLE-3 | 10 of 10 |
| Under `VLLM_BATCH_INVARIANT=1`, eager: bf16 with n-gram against bf16 without | 10 of 10 |
| Under `VLLM_BATCH_INVARIANT=1`, eager: Glyd exact with n-gram against bf16 without | 10 of 10 |
| Eager: bf16 with n-gram against bf16 without | 5 of 10 |
| Eager: bf16 with EAGLE-3 against bf16 without | 3 of 10 |
| Compiled: bf16 with n-gram against bf16 without | 5 of 10 |
| Compiled: Glyd with n-gram, with EAGLE-3, against Glyd without | 4 of 10, 5 of 10 |
| Compiled: bf16 run again with other memory settings, no speculation | 4 of 10 |
| Compiled: Glyd with its draft packed, run again on the same compile cache | 3 of 10 |

- **Exact mode with speculation gives bf16's tokens,** eager, with the n-gram and the EAGLE-3 drafts, and with the
  draft packed by Glyd too (its products then bf16's GEMM on its unpacked weights, as the target's).
- **Speculation's tokens are the plain decode's only where the products' bits do not depend on the batch.** vLLM's own
  bf16, eager, which gives the same tokens from one process to the next, gave other tokens with speculation in 5 of 10
  requests (n-gram) and 7 of 10 (EAGLE-3).
  - A verify step multiplies the step's token and its draft tokens (up to 1 + 5) in one pass; plain decoding multiplies
    one token a step.
  - vLLM's GEMMs and attention pick their kernels and splits by that shape, so the sums round differently, and a near
    tie between two tokens can go the other way. The parted requests part at tokens 18 to 330.
  - Under `VLLM_BATCH_INVARIANT=1` every product's bits are the same whatever the batch: there bf16 with n-gram gave
    bf16's own tokens, 10 of 10, and so did Glyd exact with n-gram.
- **Compiled runs vary regardless of speculation:** the same bf16 with other memory settings kept 4 of 10 requests'
  tokens, and Glyd run twice on one compile cache 3 of 10 (vLLM's default compile mode; see
  `../l4-vllm-m2-2026-09-29`).

## The draft under `--quantization glyd`

- **By default vLLM leaves an EAGLE-3 draft in bf16.** It applies the target's quantization only to MTP and DSpark
  drafts; an EAGLE-3 draft's is its speculative config's own, unset. So 0 of the draft's 5 Linears were packed.
- **With `"quantization": "glyd"` in `--speculative-config`,** the draft is packed too: 5 of 5 Linears.
  - Its packs join the process's digest: `1919f39e5d861b88` against the target alone's `e4acc9daa09858a0`.
  - The target's packs stay in the digest.
  - A second run on the same compile cache loaded both graphs ("Directly load AOT compilation") under the same key.
  - Speed was as with the bf16 draft, and exact mode's tokens bf16's.
- **Found and fixed:** at first that run was refused by the plugin's memory check: it sized the target model for the
  draft, and counted memory PyTorch had cached after the target's packing as used ("2.6 GiB ... the GPU has 1.4 GiB
  free"). Now each config sizes its own model, and `bindings/python/test_vllm.py` (`test_draft`) checks the sizing and
  the digest.

## Files

- `m6_spec.sh`, `m6_bi.sh`, `m6_tight.sh`, `m6_retry.sh` and `m6_eager.sh`, with their logs: the runs, in that order.
  `m6_retry.sh` ran again the runs refused before the fix.
- `runs/`: each run's JSON (its tokens, times and counters) and vLLM's log. `bf16-eagle3.log` is the run that did not
  fit.
- `summary.txt`: `gpu/vllm/spec_summary.py runs`.
