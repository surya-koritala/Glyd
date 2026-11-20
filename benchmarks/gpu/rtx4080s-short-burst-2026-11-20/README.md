# Short bursts, RTX 4080 SUPER, Qwen3-8B (2026-11-20)

Levers 1 to 3 of the follow-up to [rtx4080s-batched-prefill-2026-11-20](../rtx4080s-batched-prefill-2026-11-20): what stands between the engine and vLLM at 4 and 8 short prompts at once, after the prompt
tile pick ([rtx4080s-gemm-tiles-2026-11-20](../rtx4080s-gemm-tiles-2026-11-20), 2,000-token pass 372 to 353 ms). `scripts/timeline.py` joins the client's sends and first tokens (`load.py`) with the server's
submits and prompt passes (`GLYD_ENGINE_TRACE_SCHED=1`).

## The timeline of 8 short requests at once (batched passes on), ms from the first send

```
   0.0  client sends #0 ... #3 (the eight go out over 2.5 ms)
   1.1  server submit 3, 4        <- tokenized and queued 0.4 to 1 ms after the send
   1.2  prompt pass of 2 [3, 4] (73 tokens, 24.5 ms) ends at 25.7
  25.7  server submit 5 ... 10    <- the six that arrived during that pass: queued until it ended
  25.8  prompt pass of 6 (213 tokens, 42.3 ms) ends at 68.1
  26.0  first tokens of the first two (ttft 26 ms)       68.4 to 69.1: the other six (ttft 63 to 68 ms)
```

Terms, against what each could be: HTTP, tokenizing and the chat template 0.4 to 1 ms (floor: well under 1 ms); waiting for the pass in progress, 24 ms for the six that came in during the first
(the scheduler takes in new sequences between its passes; before this change the request's thread was started by the scheduler's own loop too, so a request waited out two passes: submits at 0.7, 21 and
41 ms, the last first token at 231 ms against 69 now); the pass of 6 sequences, 213 tokens, **42 ms against a floor of 15 ms of weight reads plus about 8 ms of arithmetic**: the term far above its floor.

## What changed (Part of #214)

1. **A thread of its own takes requests off the queue** (`server::work_with`): the request threads, which parse, tokenize and apply the chat template, start at once and not when the scheduler's loop is
   next between rounds. Submits of a burst 21 ms apart become one (the timeline above); bits unchanged.
2. The scheduler's trace names the sequences and times of each pass, and `timeline.py`.

## The bar after levers 2 and 3 (two runs each; vLLM's two runs of the earlier record)

| load | default: decode tok/s a user | first token | batched passes on (`GLYD_ENGINE_BATCH_PREFILL=1`): first token | vLLM |
| :--- | ---: | ---: | ---: | ---: |
| short, 1 at once | 59.9 (90% of ceiling) | 0.021 s | 0.021 | 0.024 |
| short, 4 at once | 58.3 (88%) | 0.051 | 0.046 | 0.048 |
| short, 8 at once | 56.3 (85%) | 0.116 | **0.065** | 0.068 |
| ~2000 tokens, 1 / 4 / 8 | 58.7 / 49.0 / 40.2 | 0.352 / 0.904 / 1.638 | 0.352 / 0.903 / 1.640 | 0.357 / 0.881 / 1.584 |
| ~4000 tokens, 1 / 4 / 8 | 57.7 / 42.6 / 32.8 | 0.677 / 1.704 / 3.115 | the same | 0.734 / 1.825 / 3.300 |

With the batched passes on, every load passes (decode 1.00x to 1.05x vLLM's, memory 14,796 against 15,188 MiB, first token at or under vLLM's) except 2,000 tokens with 8 at once, 3.5% behind: the decode step
between each pair of prompts (7 of 17 ms, 60 ms of the mean) against vLLM's decode tokens riding in the prompt chunks' own steps. `scripts/gpu_bar.py` allows 3%.

## Lever 1: batched passes that are bit for bit what each prompt gives alone

The batched pass is not: the products of more than 64 rows take the library's route for that many rows (BIG, with its own split of the steps), not the step kernel (route GEMM) each prompt of 17 to 64 tokens
takes alone, so the logits differ in the last bit (136 of 140 sequences in `glyd-engine batchcheck`; same token in all 140). Made exact (branch `lever1-exact-batch-wip`, not merged):
the step kernel's launch takes S slices of up to 64 rows side by side (`mma_gemm_kernel`'s `S`: block b is part b / S of the same split of the steps, for slice b % S, its own workspace and counters), the
engine asks for route GEMM for the batch's products (the six matrices in the 12-bit layout, 7.3% of the bytes, whose route for 17 to 64 rows is MID, in slices one after another), the sequences join a pass at 17 to
64 tokens (16 rows or fewer take the step kernel, whose sums differ). **`batchcheck`: 140 of 140 sequences bit for bit, also with earlier pages** (and one launch gives the bits of the same slices launched
one after another: 0 of 3,744 products differ). It is slower than the route it replaces: a pass of 6 sequences, 213 tokens, takes 70 ms (against 42), of 4, 143 tokens, 52 (41), of 2, 71 tokens, 35 (24): the slices
decode the same weights again for every 64 rows (the step kernel is bound by that decode, not by the weights' 15 ms of reads, so the slices beside one another in L2 do not make it one pass), and 8 short users
at once give a first token of 0.110 s, not 0.065. So exact batching needs a kernel that decodes a weight once for all the rows it multiplies, with the step kernel's order of sums: the prompt kernel
(`mma_gemm_big`/stream-K) has the decode once for 128 or 256 rows but splits the steps by the grid, as above. Until then the batched pass is opt-in.
