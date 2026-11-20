# Prompts of several requests in one pass, RTX 4080 SUPER, Qwen3-8B (2026-11-20)

Follow-up to [rtx4080s-engine-vs-vllm-2026-11-20](../rtx4080s-engine-vs-vllm-2026-11-20), which found the engine's first-token time 1.7x and 2.4x vLLM's with 4 and 8 short prompts at once and
4% to 8% behind at 2,000 tokens. **The RTX 4080 SUPER is still not in `GPUS`: the short-prompt gap is closed to 6% (4 users) and 21% (8 users), the 2,000-token gap is not touched
(4% to 9%).** Same machine, model, loads, scripts and vLLM runs as that record (`vllm1`, `vllm2`); the engine is main 444e2dcf plus this change, built by the same CI recipe.

## What limited it (measured, `GLYD_ENGINE_TRACE_SCHED=1` and `/stats`)

A scheduler round was one prompt chunk of the oldest waiting sequence, then one decode step of all the others. A short prompt's pass takes 19 to 20 ms (the weights are read once: 10.3 GiB
at 736 GB/s is 15 ms) and a decode step 17 ms whether 1 or 8 rows are in it, so the k-th of 8 short prompts waited about k x 37 ms, and the 8 together cost 8 weight passes where one would do (first
token: 21, 80 and 162 ms for 1, 4 and 8 at once; the model above gives 145 ms for 8, measured 154 to 160). At 2,000 tokens a pass is 372 ms of GPU time, 95% of it the dense products
(gate and up 168 ms, down 97, qkv 42, o 33; 88 TFLOPS of the card's ~100 at fp32 accumulation), attention 17 ms: compute-bound, one prompt at a time is right, and vLLM's 357 ms is 4% less.

## What changed

1. **After a short prompt pass for prompts that came in during the round, the round's decode step is skipped** (at most 4 rounds in a row, passes of up to 256 tokens), so the prompts that follow
   within a few milliseconds are taken in before the 17 ms step, not after it. Tokens are the same bit for bit (decode is unchanged).
2. **`GLYD_ENGINE_BATCH_PREFILL=1`: the prompts of up to 8 sequences (each of at most 64 tokens to join, 2,048 tokens in all) run in one pass**: embedding, products, norms, rope and the cache
   write on all their rows, the attention on each sequence's rows as before (`Engine::prefill_batch`, `Sched::prefill_step`). Off by default, see below.

## First-token time, engine against vLLM (median of 2 runs each; 3 rounds a load)

| load | before (record above) | decode skip only | batched passes (`=1`) | vLLM | batched / vLLM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| short, 1 at once | 0.021 s | 0.021 | 0.021 | 0.024 | 0.88x |
| short, 4 at once | 0.080 s | 0.051 | 0.053 | 0.048 | 1.09x |
| short, 8 at once | 0.162 s | 0.117 | 0.082 | 0.068 | 1.21x |
| ~2000 tokens, 1 / 4 / 8 | 0.370 / 0.949 / 1.718 s | the same | the same | 0.357 / 0.881 / 1.584 | 1.04x / 1.08x / 1.09x |
| ~4000 tokens, 1 / 4 / 8 | 0.696 / 1.761 / 3.188 s | the same | the same | 0.734 / 1.825 / 3.300 | 0.95x / 0.97x / 0.97x |

Decode tokens/s a user and in all are unchanged at every load (the table of the record above: 85% to 90% of the 66.5 tokens/s ceiling with short prompts, 0.99x to 1.04x vLLM's); total tokens/s with
8 short prompts 436.6 (skip only) and 450.8 (batched), vLLM 452.1 (it was 431.5). Peak GPU memory 14,796 MiB, as before. The full table of the batched runs:

| load | decode tok/s a user: engine (% of ceiling) | vLLM (%) | engine / vLLM | total tok/s: engine | vLLM | first token s: engine | vLLM | engine / vLLM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (~35 tokens), 1 at once | 59.9 (90%) | 60.0 (91%) | 1.00x | 59.9 | 60.0 | 0.021 | 0.024 | 0.88x |
| short (~35 tokens), 4 at once | 58.5 (88%) | 58.7 (89%) | 1.00x | 232.4 | 233.6 | 0.053 | 0.048 | 1.09x |
| short (~35 tokens), 8 at once | 57.1 (86%) | 57.0 (86%) | 1.00x | 450.8 | 452.1 | 0.082 | 0.068 | 1.21x |
| ~2000-token, 1 at once | 58.7 (88%) | 58.7 (89%) | 1.00x | 58.7 | 58.7 | 0.371 | 0.357 | 1.04x |
| ~2000-token, 4 at once | 48.8 (73%) | 48.6 (74%) | 1.00x | 174.8 | 176.3 | 0.949 | 0.881 | 1.08x |
| ~2000-token, 8 at once | 39.9 (60%) | 39.5 (60%) | 1.01x | 261.1 | 263.6 | 1.719 | 1.584 | 1.09x |
| ~4000-token, 1 at once | 57.7 (87%) | 56.8 (86%) | 1.02x | 57.7 | 56.8 | 0.697 | 0.734 | 0.95x |
| ~4000-token, 4 at once | 42.3 (64%) | 41.4 (63%) | 1.02x | 142.9 | 139.0 | 1.762 | 1.825 | 0.97x |
| ~4000-token, 8 at once | 32.6 (49%) | 31.4 (47%) | 1.04x | 136.6 | 130.5 | 3.190 | 3.300 | 0.97x |

## Where it still fails the bar (`scripts/gpu_bar.py`)

- **Short prompts, 4 and 8 at once, 6% to 21% behind:** the requests of a burst arrive over 40 to 60 ms (tokenizing and templating one after another), so a first pass takes the first one or two
  (20 ms) and the rest follow; a pass of more than about 128 rows takes the library's decode-ahead route (41 to 43 ms for 143 to 251 rows against 25 ms for 108), which is the cost of the last pass.
- **2,000-token prompts, 4% to 9% behind (one user too: 0.371 against 0.357 s):** the engine's own prompt products run at 88 TFLOPS against vLLM's 92 (`bench --profile-prefill`: 372 ms of GPU time for 2,000 tokens, 95% of it the four products). Closing it is kernel work on the gate-and-up and down products, not scheduling; no change here moves it.

## Replies

One request at a time is **bit for bit** what it was (10 prompts of 96 tokens, cold cache and with the prompts' pages found: tokens and logprobs identical to the engine of main, `compare.py`). With the
default (decode skip only), concurrent replies are the same bit for bit as alone and as main's (10 of 10, cold and warm, `def-together` against `old-alone`). With `GLYD_ENGINE_BATCH_PREFILL=1` a pass
of more than 64 rows takes the products' route its row count gives, not the one each prompt takes alone, so the logits differ in the last bits: in `glyd-engine batchcheck` (passes of 2 to 8 sequences,
with and without earlier pages) **140 of 140 sequences give the same token, 4 are bit for bit (passes of at most 64 rows), 136 have logits within 0.19 of alone's** (about 1 to 2 bf16 steps at the
logits' size); over 96 tokens, 8 of 10 (cold) and 7 of 10 (warm) replies are token for token the same as alone's, the others part at a near tie (token 36, 40 or 67, logprobs within 0.16). That is why it is
off by default: a reply would then depend, in rare cases, on what else the server was asked at the time. The owner decides whether it is the default.

## Files

`scripts/tokens_check.py`, `scripts/compare.py`, `scripts/run_new.sh` (the reply check); the load, `run.sh`, `until_clean.sh` and `summarize.py` are the earlier record's. `glyd-engine batchcheck MODEL`
(development build) is the engine-level check of passes against single prompts. Raw outputs are not in the tree.
