# A round that is one step, RTX 4080, gemma-4-12B at wren (2026-11-22)

The scheduler's round is a prompt pass and then a decode step. `GLYD_ENGINE_ONE_STEP=1` (R4a, docs/design/runtime.md section 5) makes it one step: the rows that decode and the prompts' chunks in
one pass, with a budget of prompt tokens beside the rows (`GLYD_ENGINE_STEP_TOKENS`, the rows of the step in all). This record measures both rounds on loads where it matters: requests that begin one
after another, so that the later prompts come in while the earlier replies are being written (`bench`'s `KIND+GAPms` loads, new with this change).

**Result: at 256 rows a step the one-step round cuts the 99th percentile of the gap between a reply's tokens 3.5 to 5 times (289 to 82 ms with 1k prompts coming in, 478 to 97 ms with 4k) at the
same total tokens/s, and the first token of those prompts comes 8% to 10% later. At 64 rows a step it is far worse on the first token and on the total. When this was measured it was off by
default. It was made the default at a chunk's rows (512 for this model) afterwards and measured again on a second machine, with what it does to a reply's logits: see "Afterwards" at the
end, which also corrects two statements of this record.**

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
  behind each other (first token 1.16 s against 0.39 s) and the total falls 22% (33% with 4k prompts). With 256 the prompt runs nearly as fast as in a pass of its own. (This line said
  256 was "the chunk's own size". It is not: Gemma 4's chunk is 512 rows, `GEMMA_CHUNK`.)
- **Where the one step wins is the gap between tokens.** In the round of two passes a reply stops for every prompt pass (289 ms with 1k prompts coming in, 478 ms with 4k); in one step its row
  rides in the prompt's pass, and the longest gaps are the step's own time: 82 and 97 ms at 256 rows, 43 and 44 ms at 64.
- **The first token of a prompt that comes in while others decode is 8% to 10% later at 256 rows** (0.426 against 0.387 s; 3.72 against 3.44 s): its chunks are cut at 256 less the rows, and the
  round of two passes lets the decode step wait for up to two long passes, which the one step never does.
- Eight short prompts at once reach their first token sooner in one step (0.083 to 0.100 s against 0.126 s); this figure moves between sessions on this bench (the gate's band for it is about
  40%), so it is not counted as a gain.

## What it does not say

- One GPU, one model, three runs a load. The MoE families, a second card, and 64 users are not measured.
- The chunk's own rows (512) were not tried in this session. (This line said 512 was past the chunk. The chunk is 512; it is measured under "Afterwards".)
- Whether a reply is the same as alone. This record guessed that a step of 64 rows or fewer keeps a reply's bits and that past it only the last bits move. Both were wrong; the
  measurement is under "Afterwards".

## Afterwards

The round that is one step was made the default the same day, at a chunk's rows (512 for Gemma 4; `GLYD_ENGINE_ONE_STEP=0` is the round of two passes, `GLYD_ENGINE_STEP_TOKENS=N` a step of
N rows). It was then measured as the default on a second rented RTX 4080 (16 GB, driver 595.99.02, Threadripper PRO 3955WX), same model, same settings, main a7f0f1d5 with M2 batch 8
(`b-one-step-default.json`, `b-twopass.json`). The two-pass column agrees with the first machine's within 3% on the gaps and the totals; the 1k and 4k prompts' first tokens are 4% to 9% sooner here.

| load | figure | two passes | one step, 512 rows (the default) |
| :--- | :--- | ---: | ---: |
| short, 100 ms apart, 8 | first token | 0.029 s | 0.029 |
| | gap between tokens, 99th percentile | 38 ms | 27 |
| | decode tokens/s a user / in all | 61.4 / 382.0 | 63.3 / 383.9 |
| 1k, 250 ms apart, 8 | first token | 0.353 s | 0.330 |
| | gap between tokens, 99th percentile | 282 ms | 144 |
| | decode tokens/s a user / in all | 35.2 / 219.5 | 36.1 / 222.6 |
| 4k, 500 ms apart, 8 | first token | 3.295 s | 3.288 |
| | gap between tokens, 99th percentile | 465 ms | 168 |
| | decode tokens/s a user / in all | 19.0 / 83.2 | 20.0 / 86.3 |
| short, 8 at once | first token | 0.122 s | 0.103 |
| | decode tokens/s a user / in all | 62.1 / 473.9 | 62.3 / 482.4 |
| short, 1k, 4k, 1 at a time | first token; decode tokens/s | 0.024, 0.263, 1.091 s; 86.6, 82.1, 79.0 | 0.024, 0.263, 1.088; 86.6, 82.1, 79.0 |

At a chunk's rows the one step gives up nothing on these loads: the first token and the totals are the same or slightly better, and the longest gaps are 2 to 2.8 times shorter with 1k and 4k prompts coming in (1.4 times with short ones). At 256 rows the gaps were
shorter still (82 and 97 ms on the first machine) for a first token 8% to 10% later.

### What it does to a reply's logits

`glyd-engine roundcheck` (new, src/logits.rs) gives one sequence fixed tokens and compares its logits, step by step, when its row is alone in the step and when other rows are in it. On
this model at wren, with a text's tokens (the bench's passage: 40 as the prompt, the next 32 given one a step), on the second machine:

| the step holds | logits that differ from the row alone | largest difference | KL, mean / most | largest logit the same token's |
| :--- | ---: | ---: | ---: | ---: |
| the row and 1, 7 or 15 other sequences' rows | 0 | 0 | 0 | 32 of 32 steps |
| the row and a prompt's chunk of 7 or 15 | 0 | 0 | 0 | every step |
| the row and 31, 47, 62 or 63 other rows (other sequences' rows, or a prompt's chunk: the same figures) | 91% | 1.25 | 0.0047 / 0.042 | 31 of 32 |
| the row and a prompt's chunk of 127 | 91% | 1.12 | 0.0033 / 0.024 | 30 of 32 |
| the row and a prompt's chunk of 255 | 91% | 0.93 | 0.0037 / 0.020 | 18 of 18 |
| the row and a prompt's chunk of 511 | 90% | 0.69 | 0.0020 / 0.0066 | 9 of 9 |

- **What the other sequences hold never changes a row.** At every size, the row's logits beside one prompt and beside another of the same length are the same bits (0 of up to 8.4 million
  logits differ), on this model and on the random models of all five families. A row that read another sequence's keys or state would not pass this.
- **How many rows the step has does**, because a product's kernel is picked by them. In a step of 16 rows or fewer every logit is the row's own pass's, bit for bit (every family). In a
  larger one the logits differ as the table shows, and the same whether the other rows are a prompt's chunk or other sequences that decode. So this is not new with the one step: a decode
  step of more than 16 sequences already did it, and so does cutting a prompt into other chunks (this passage alone, in chunks of 7 to 127 against one pass: a KL of 0.0008 to 0.05 after
  its last token, and 0.36 at one place where it ends).
- **How large it is.** A KL of 0.002 to 0.005 from the row alone on average: the size of the rounding of the engine's bf16 values, which is also what its logits are held to against an
  f32 reference (tests/logits: a mean KL of 0.008 for Kimi-K3 and 0.012 for Muse Spark on random models). The token with the largest logit changed in 1 or 2 steps of 32, where the two
  largest were tied or nearly. A greedy reply can therefore differ from the same request alone once a prompt has run beside it.
- The prompt's side: this passage's logits after its last chunk are the same bits beside the row and alone, at every chunk size it reaches (7 to 127).
- Random models at the lossless level, the row in a step of 32 to 64 rows against alone: Qwen3.5's layout a KL of at most 0.0001, Muse Spark 0.0003, Gemma 4 0.001, Kimi-K3 none (one
  logit of 32,768 by 0.001 beside 31 others' rows), GLM-5.3-Flash (experts at a lossy level, the two largest logits 0.09 apart) 0.17.

**The choice the default makes.** One step: the gaps above, and a reply whose last bits (and, at a near tie, whose token) can turn on whether a prompt ran beside it. Two passes with
`GLYD_ENGINE_BATCH_PREFILL=0` (prompts one a pass): every reply is the reply alone, bit for bit, while 16 sequences or fewer decode, with the longer gaps. Two passes alone do not give that:
prompts that come together are run together, which changes their chunks' rows. A product whose result does not depend on the step's rows (the deterministic mode, #206) would give both.

### Also measured on the second machine

- **Conversations** (`b-chat.json`, the bench's new `1k*4` loads: 4 turns, each resending the conversation). One conversation: first token 0.263 s on the first turn and 0.340 s on the later
  ones; eight at once: 1.21 s and 0.70 s; a 4k one: 1.09 s and 1.17 s. The server found 0% of the prompt tokens in a cache: the description executor keeps nothing of a finished sequence, so
  every turn is run from its first token. That is R4b and R4c's work, and this is the load that will show it.
- **Many requests** (64 new tokens each, 2 runs). A server of 64 sequences (`b-64-sequences.json`): 64 at once, none failed, first token 0.47 s, 15.8 tokens/s a user, 905 in all; 256 at
  once, none failed, 7.1 s, 903 in all. A server of 8 sequences (`b-8-sequences-many.json`): 64 at once, none failed, first token 3.9 s, 457 tokens/s in all; 256 at once, none failed,
  17.3 s, 454 in all (the rest wait their turn in the scheduler's queue).
