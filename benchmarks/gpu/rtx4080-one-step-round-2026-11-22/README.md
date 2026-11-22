# A round that is one step, RTX 4080, gemma-4-12B at wren (2026-11-22)

The scheduler's round is a prompt pass and then a decode step. `GLYD_ENGINE_ONE_STEP=1` (R4a, docs/design/runtime.md section 5) makes it one step: the rows that decode and the prompts' chunks in
one pass, with a budget of prompt tokens beside the rows (`GLYD_ENGINE_STEP_TOKENS`, the rows of the step in all). This record measures both rounds on loads where it matters: requests that begin one
after another, so that the later prompts come in while the earlier replies are being written (`bench`'s `KIND+GAPms` loads, new with this change).

**Result: at 256 rows a step the one-step round cuts the 99th percentile of the gap between a reply's tokens 3.5 to 5 times (289 to 82 ms with 1k prompts coming in, 478 to 97 ms with 4k) at the
same total tokens/s, and the first token of those prompts comes 8% to 10% later. At 64 rows a step it is far worse on the first token and on the total. It stays off by default; the default is
not decided by this record alone (see the end).**

Machine: a rented RTX 4080 (16 GB, driver 595.91.07, Ryzen 9 3900X), nothing else on the GPU. Engine: main 684fb473 with the changes of M2 batches 5 and 6 (the scheduler's one-step round is this record's own pull request), built by the GPU job's recipe
(`scripts/ci_gpu_build.sh`, kernels for sm_89). Model: `google/gemma-4-12B-it:wren` (6.75 GB of weights). `glyd-engine bench` started the server itself with 8 users, no draft model;
3 runs a load after a warm-up round, 128 new tokens a request, temperature 0. The two-pass round was measured twice, before and after the others (`twopass.json`, `twopass2.json`):
every median agrees within 0.3%.

## Requests that begin one after another, 8 of them

| load | figure | two passes | one step, 64 | one step, 128 | one step, 256 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| short, 100 ms apart | first token | 0.028 s | 0.034 | 0.031 | 0.034 |
| | gap between tokens, 99th percentile | 39 ms | 27 | 27 | 27 |
| | decode tokens/s a user | 60.8 | 62.7 | 62.5 | 62.6 |
| | tokens/s in all | 378.5 | 380.1 | 380.7 | 379.8 |
| 1k, 250 ms apart | first token | 0.387 s | 1.163 | 0.608 | 0.426 |
| | gap between tokens, 99th percentile | 289 ms | 43 | 54 | 82 |
| | decode tokens/s a user | 34.7 | 36.7 | 36.9 | 36.3 |
| | tokens/s in all | 215.9 | 167.9 | 201.1 | 214.8 |
| 4k, 500 ms apart | first token | 3.440 s | 7.661 | 4.891 | 3.719 |
| | gap between tokens, 99th percentile | 478 ms | 44 | 60 | 97 |
| | decode tokens/s a user | 18.6 | 25.7 | 20.6 | 20.1 |
| | tokens/s in all | 81.5 | 54.9 | 75.2 | 84.7 |

## Requests all at once (the gate's loads)

| load | figure | two passes | one step, 64 | one step, 128 | one step, 256 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| short, 1 | first token / decode tokens/s | 0.025 s / 86.5 | 0.024 / 86.4 | 0.025 / 86.4 | 0.025 / 86.4 |
| short, 8 | first token | 0.126 s | 0.100 | 0.083 | 0.093 |
| | decode tokens/s a user / in all | 61.4 / 468.3 | 60.6 / 461.2 | 60.8 / 468.8 | 61.2 / 471.6 |
| | gap between tokens, 99th percentile | 17 ms | 27 | 28 | 17 |
| 1k, 1 | first token / decode tokens/s | 0.274 s / 81.9 | 0.273 / 81.8 | 0.273 / 81.8 | 0.273 / 81.9 |
| 4k, 1 | first token / decode tokens/s | 1.126 s / 78.7 | 1.126 / 78.7 | 1.124 / 78.7 | 1.130 / 78.7 |

One request at a time is the same in every column, as it must be: with nothing else running a step is either a prompt's chunk or one row.

## What it says

- **The budget decides it.** A prompt of 935 tokens at 61 tokens a step is 16 steps, each a full read of the weights; at 248 it is 4. With 64 rows a step, prompts that come in every 250 ms queue
  behind each other (first token 1.16 s against 0.39 s) and the total falls 22% (33% with 4k prompts). With 256, the chunk's own size, the prompt runs as fast as in a pass of its own.
- **Where the one step wins is the gap between tokens.** In the round of two passes a reply stops for every prompt pass (289 ms with 1k prompts coming in, 478 ms with 4k); in one step its row
  rides in the prompt's pass, and the longest gaps are the step's own time: 82 and 97 ms at 256 rows, 43 and 44 ms at 64.
- **The first token of a prompt that comes in while others decode is 8% to 10% later at 256 rows** (0.426 against 0.387 s; 3.72 against 3.44 s): its chunks are cut at 256 less the rows, and the
  round of two passes lets the decode step wait for up to two long passes, which the one step never does.
- Eight short prompts at once reach their first token sooner in one step (0.083 to 0.100 s against 0.126 s); this figure moves between sessions on this bench (the gate's band for it is about
  40%), so it is not counted as a gain.

## What it does not say

- One GPU, one model, three runs a load. The MoE families, a second card, and 64 users are not measured.
- A budget past the chunk (512 rows) was not tried: the executor's pass holds a chunk.
- Whether a reply is the same tokens as alone. With more than 64 rows in a step a product takes another route (`decoded_way`), so a row that decodes beside a long chunk can differ in its last
  bits from the same row alone, which the round of two passes never does for a dense model (its decode steps are rows alone). The GPU job's usercheck at 64 rows: Qwen3.5's layout 4 of 4 replies
  the same as alone, Muse Spark 0 of 4, Kimi-K3 2 of 4, GLM-5.3-Flash 0 of 4 (random models, counted, not refused); at 256 rows it is not measured.

So the default is a choice between two properties: the shortest gaps between tokens under load at the same throughput (one step, 256 rows), or replies whose bits do not turn on what ran
beside them (two passes, or one step held to 64 rows, which costs the first token and the total above). A product route that does not depend on the row count (the deterministic mode, #206)
would give both.
