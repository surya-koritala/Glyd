# glyd-engine for a few users at once (2026-10-03)

glyd-engine serves up to 8 users at once on one GPU (`--users N`, default 4): their replies are decoded together, a prompt that begins as an earlier one reuses what was computed for it, and where the GPU's memory for the replies
runs out the newest one waits and is run again from its own text, its stream going on where it was. This record measures it on an NVIDIA L4 (AWS g6.4xlarge) with Qwen3-8B at 1, 4 and 8 users, against vLLM 0.30.0 with Glyd's weights
and Glyd's lossless KV cache (the same weights and the same cache), and checks that the cache stays lossless as the memory of finished replies is reused and as it runs short.

## What was run

The final runs and every check below are of the engine at commit f468e8c9 of the engine-mvp branch (the build between is 5c52dcbb; the first measurement is the build before the scheduler's first-token and sampling changes).

`vllm bench serve` as the same client against both servers: random prompts of 1,024 tokens, 256 tokens out, `--ignore-eos`, `--max-concurrency` 1, 4 and 8 (a user sends the next request when its reply is complete) with 8, 24 and
32 requests; a warm-up pass first, then a new set of prompts for every pass (`--seed` per pass, the same sets for both servers), so no pass finds a prompt a server has seen: the prefix cache's hit rate is printed for every pass
(the engine's own count, vLLM's log) and is 0.00 in every one of them. The client sends no sampling settings: each server answers with the sampling the model's own configuration asks for (temperature 0.6, top-k 20, top-p 0.95 for
Qwen3-8B). Both servers had the GPU to themselves, a context of 4,096 tokens, and a start from 50 C or less where it cooled in time; the L4 slows as it heats at its 72 W cap. The final runs alternated (vLLM, engine, vLLM, engine); the first
table has every one of them, the second their means; the tables after them are the runs of the earlier builds.

| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine | 1 | engineC | 8 | 21.2 | 21.8 | 372, 379 | 45.9, 46.1 | 0.00 |
| glyd-engine | 1 | engineD | 8 | 21.1 | 21.7 | 377, 389 | 46.0, 46.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglydA | 8 | 20.6 | 21.2 | 419, 430 | 47.2, 47.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglydB | 8 | 20.7 | 21.3 | 407, 425 | 47.0, 47.2 | 0.00 |
| glyd-engine | 4 | engineC | 24 | 73.7 | 19.7 | 953, 1526 | 50.7, 53.2 | 0.00 |
| glyd-engine | 4 | engineD | 24 | 73.7 | 19.7 | 961, 1569 | 50.7, 52.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglydA | 24 | 71.9 | 19.4 | 1075, 1630 | 51.6, 54.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglydB | 24 | 71.8 | 19.4 | 1082, 1668 | 51.6, 54.3 | 0.00 |
| glyd-engine | 8 | engineC | 32 | 125.5 | 17.0 | 1322, 3362 | 58.7, 62.2 | 0.00 |
| glyd-engine | 8 | engineD | 32 | 124.2 | 16.9 | 1335, 3233 | 59.2, 63.1 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglydA | 32 | 125.8 | 17.6 | 1730, 3020 | 56.9, 61.6 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglydB | 32 | 127.2 | 17.7 | 1669, 2877 | 56.5, 61.3 | 0.00 |

| server | users | runs | output tok/s (mean) | per user tok/s (mean) | time to first token ms (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| glyd-engine | 1 | 2 | 21.2 | 21.8 | 374 |
| vLLM with Glyd (weights and KV cache) | 1 | 2 | 20.6 | 21.2 | 413 |
| glyd-engine | 4 | 2 | 73.7 | 19.7 | 957 |
| vLLM with Glyd (weights and KV cache) | 4 | 2 | 71.8 | 19.4 | 1078 |
| glyd-engine | 8 | 2 | 124.9 | 17.0 | 1329 |
| vLLM with Glyd (weights and KV cache) | 8 | 2 | 126.5 | 17.6 | 1700 |

The engine against vLLM with Glyd, the means of the final runs:

- 1 user: 21.2 tokens a second against 20.6 (+2.6%; the runs of each within 0.3% and 0.5% of each other), the first token after 374 ms against 413 (-9%)
- 4 users: 73.7 tokens a second against 71.8 (+2.6%; the runs of each within 0.0% and 0.1% of each other), the first token after 957 ms against 1078 (-11%)
- 8 users: 124.9 tokens a second against 126.5 (-1.3%; the runs of each within 1.0% and 1.1% of each other), the first token after 1329 ms against 1700 (-22%)

The build between (the engine thread's work after each step moved to the GPU, a request's first token still sent after its round's decode step; one run, the vLLM run beside it is kvglydA):

| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 1 | engineA | 8 | 21.1 | 21.8 | 421, 435 | 45.8, 46.1 | 0.00 |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 4 | engineA | 24 | 72.3 | 19.5 | 1043, 1620 | 51.4, 53.9 | 0.00 |
| glyd-engine (sampling on the GPU, first token not yet sent early) | 8 | engineA | 32 | 122.5 | 16.7 | 1411, 3459 | 59.9, 63.4 | 0.00 |

The first measurement, before the engine thread's work after each step was cut (same client, same prompts):

| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (first measurement) | 1 | engine1 | 8 | 20.5 | 21.1 | 376, 386 | 47.5, 47.8 | 0.00 |
| glyd-engine (first measurement) | 1 | engine2 | 8 | 20.4 | 21.0 | 384, 393 | 47.6, 47.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglyd1 | 8 | 20.6 | 21.3 | 411, 422 | 47.0, 47.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 1 | kvglyd2 | 8 | 20.6 | 21.2 | 417, 432 | 47.2, 47.3 | 0.00 |
| glyd-engine (first measurement) | 4 | engine1 | 24 | 65.1 | 17.3 | 997, 1577 | 57.7, 60.0 | 0.00 |
| glyd-engine (first measurement) | 4 | engine2 | 24 | 64.9 | 17.3 | 1001, 1584 | 57.9, 60.2 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglyd1 | 24 | 72.1 | 19.4 | 1059, 1604 | 51.5, 54.0 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 4 | kvglyd2 | 24 | 71.8 | 19.4 | 1073, 1624 | 51.6, 54.3 | 0.00 |
| glyd-engine (first measurement) | 8 | engine1 | 32 | 102.6 | 13.7 | 1362, 3380 | 72.7, 76.1 | 0.00 |
| glyd-engine (first measurement) | 8 | engine2 | 32 | 102.5 | 13.7 | 1359, 3381 | 72.9, 76.3 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglyd1 | 32 | 126.1 | 17.6 | 1719, 2920 | 56.8, 61.8 | 0.00 |
| vLLM with Glyd (weights and KV cache) | 8 | kvglyd2 | 32 | 125.7 | 17.6 | 1734, 3073 | 56.9, 61.8 | 0.00 |

| server | users | runs | output tok/s (mean) | per user tok/s (mean) | time to first token ms (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| glyd-engine (first measurement) | 1 | 2 | 20.5 | 21.0 | 380 |
| vLLM with Glyd (weights and KV cache) | 1 | 2 | 20.6 | 21.2 | 414 |
| glyd-engine (first measurement) | 4 | 2 | 65.0 | 17.3 | 999 |
| vLLM with Glyd (weights and KV cache) | 4 | 2 | 72.0 | 19.4 | 1066 |
| glyd-engine (first measurement) | 8 | 2 | 102.5 | 13.7 | 1360 |
| vLLM with Glyd (weights and KV cache) | 8 | 2 | 125.9 | 17.6 | 1727 |

Per user tok/s is the inverse of the mean time per token. In the final build a request's first token is sent as soon as its prompt is done (in the build between it was sent after the decode step of the same round, a decode step late: 46 ms at one user).
The engine's own counts for each pass (`stats-engine-cN.json` of each run, each less the one before it): decode steps, rows in a step, the GPU's own time for a step (milliseconds), the time the engine thread spends outside the GPU's work for a step (milliseconds), the GPU's time for a prompt (milliseconds), and the share of the pass in which the GPU was at work on a decode step or a prompt:

| run | users | decode steps | rows a step | GPU ms a step | engine thread's other work ms a step | prompt ms | GPU at work % of the pass |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| engineC | 1 | 2040 | 1.00 | 45.9 | 0.04 | 367 | 99.9 |
| engineC | 4 | 1543 | 3.97 | 48.1 | 0.06 | 376 | 100.0 |
| engineC | 8 | 1033 | 7.90 | 51.0 | 0.11 | 389 | 100.0 |
| engineD | 1 | 2040 | 1.00 | 46.0 | 0.05 | 372 | 99.9 |
| engineD | 4 | 1543 | 3.97 | 48.0 | 0.06 | 379 | 100.0 |
| engineD | 8 | 1033 | 7.90 | 51.4 | 0.12 | 396 | 100.0 |


## Where the engine thread's time went at 8 users

The same build, 8 users, 16 requests each time, alternated (host, GPU, host, GPU: `sampling/`): the draw for each row of a step made on the host from the row's logits (about 1.7 ms a row), and made from the 64 largest logits the step's own work finds on the GPU. The engine thread's time per step outside the GPU's work:

| sampling on | run | output tok/s | time per token ms | rows a step | GPU ms a step | engine thread outside the GPU's work, ms a step: before the step | after the step | the rows' work after it |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the host | host-015412 | 103.6 | 71.0 | 6.56 | 48.6 | 0.07 | 0.00 | 12.63 |
| the GPU | gpu-015539 | 123.2 | 58.6 | 6.56 | 50.7 | 0.08 | 0.00 | 0.01 |
| the host | host-015657 | 103.2 | 71.2 | 6.56 | 48.6 | 0.07 | 0.00 | 12.63 |
| the GPU | gpu-015825 | 123.8 | 58.3 | 6.56 | 50.5 | 0.08 | 0.00 | 0.01 |

## The cache stays lossless

Every check ran on the L4 with Qwen3-8B and the KV tables the engine made for it; the outputs are in `checks/`. A value the cache holds is the bits that were written, and where the cache has no room for what a request needs, the
request is never answered from what it could not hold.

- **A known answer** (`checks/kvtest.out`, `checks/kvtest-first-tables.out`). bf16 values made on the host, of every kind a cache is given (inside the range its layer's windows hold and outside it, both zeros, subnormals,
  infinities, NaNs of several payloads, the largest and the smallest numbers, blocks with few, many and more exceptions than a block can keep in its own room, and the 65,536 bit patterns once each) were written into 15 blocks of 16 tokens of each of
  the 36 layers, as one run of tokens, token by token and in pieces, read back and compared bit for bit; then written again from the first token and again from the middle, and compared again. 53,084,160 values compared
  (319 zeros, 317 negative zeros, 16,928 subnormals, 631 infinities, 17,452 NaNs in each of the three passes, a fifth of the values outside their windows): **0 differ**. With every block released the cache's extra memory
  is empty (0 of 28196 and 0 of 256 in use), and 0 blocks were lost.
- **Every write compared with a bf16 copy of what was written, at the step that makes it** (a copy of every layer's cache; each step's tokens are decoded back and compared before the next step runs): requests of natural text and of
  random tokens, 4 at a time on a cache of 128 to 144 blocks, so that finished prompts are reused, kept for later prompts and found again, and requests wait and are run again (the table below; `checks/churn-*.out`,
  `checks/extra-memory-*.out`, `checks/write-check-*.out`). In the server (`checks/churn-server-natural.out`): 300 short requests of natural text from 4 clients on a cache of 128 blocks, 0 wrong, 226,579 prompt tokens,
  5,680 found in the cache, 59 sent back to wait; 32,469,073,920 values compared, **0 differ**.
- **Where the cache has no room for what a request needs** (extra memory cut to 16 and 2, and to 8 and 1 units, so that random tokens run it out; `checks/extra-memory-*.out`): the step in which a block could not be stored exactly is named
  at once, the output of that step is taken back from the requests that hold the block (no token is made from it), and each such request waits until the ones running with it have ended, then runs again from its own text; one that is
  taken back twice runs alone; one that is lost alone with nothing kept for later prompts ends with an error in words. A request that ends so is run again alone, to see that it does not fit alone: those that made no token end so again; one that had made tokens before it ended may be served, its text run again as a prompt being computed another way than the same tokens were by steps (its keys and values then differ in their last bit) and so needing other extra memory (the table says how many end so again). The values of blocks named lost are left out of
  the comparison (counted in the table), every other value is compared. The few requests that part from their run alone do so after being run again, at a place where the model's two best choices are near-equal: a prompt run again is
  computed in another way than the same tokens were by steps (`checks/run-again.out`), so its keys and values differ in their last bit from the ones the steps made. In the server (`checks/server-extra-memory.out`): 6 checks ok, 0 failed: a request the cache cannot hold is answered with the reason in words, one it can hold
  is served whole, and the one after the failures gets the text it got before them.
- **After many rounds of churn** (`checks/rowcheck-fresh.out`, `checks/rowcheck-churn-300.out`): one sequence's logits with 0 to 7 others in the step, at every number of rows and in either row, on a fresh cache and after
  300 rounds of 8 sequences of random lengths on the same memory (100,407,214,080 values compared, 0 differ): the same logits bit for bit in every one.

The runs of requests against a bf16 copy (each request's tokens against the same request run alone on a cache with room, where that was made):

| run | requests | users | cache tokens | values compared | differing | values in lost blocks (not compared) | steps taken back | sent back to wait | ended with an error | same as alone / part from it without / after being sent back |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| natural text, 300 requests, 128 blocks | 300 natural | 4 | 2048 | 24,889,761,792 | 0 | 0 | 0 | 15 | 0 | (not compared) |
| random tokens, 300 requests, 128 blocks | 300 random | 4 | 2048 | 17,595,113,472 | 0 | 0 | 0 | 0 | 0 | (not compared) |
| natural text and random tokens, 300 requests, 128 blocks, little extra memory (16 and 2 units) | 300 mixed | 4 | 2048 | 23,068,385,280 | 0 | 133,963,776 | 129 | 130 | 0 | (not compared) |
| random tokens, extra memory 16 and 2 units | 100 random | 4 | 2304 | 6,843,727,872 | 0 | 44,015,616 | 46 | 46 | 0 | 97 / 0 / 3 |
| random tokens, extra memory 8 and 1 units | 100 random | 4 | 2304 | 8,600,223,744 | 0 | 120,766,464 | 128 | 128 | 0 | 91 / 0 / 9 |
| random tokens, extra memory 3 and 0 units (requests that do not fit alone end with an error) | 100 random | 4 | 2304 | 9,862,963,200 | 0 | 323,665,920 | 249 | 215 | 34 (alone, 33 end so again) | 61 / 0 / 5 |
| natural text, 100 requests, room for all of them | 100 natural | 4 | 2304 | 8,624,111,616 | 0 | 0 | 0 | 0 | 0 | 100 / 0 / 0 |
| random tokens, 100 requests, room for all of them | 100 random | 4 | 2304 | 5,946,900,480 | 0 | 0 | 0 | 0 | 0 | 100 / 0 / 0 |

## Other checks

- Several clients at once (`checks/clients-at-once.out`): 4 of 4 read what they read alone; 4 clients together 6.2 s (one at a time 23.5 s)
- Memory too small for the clients (`checks/clients-memory-too-small.out`): 3 of 3 read what they read alone; 3 clients together 43.6 s (one at a time 96.9 s) stats: {'prompt_tokens': 2919, 'prefix_hit_tokens': 1376, 'preemptions': 1, 'steps': 2999, 'rows': 4193} since the start: {'preemptions': 1, 'steps': 2999, 'rows': 4193}
- A client that leaves midway leaves nothing behind (`checks/clients-at-once.out`): after a client left midway: 0 running, 0 waiting
- The API's check of itself (`checks/api-smoke.out`): 19 checks ok, 0 failed; the soak (`checks/api-soak.out`): 129 rounds in 126 s: {'ok': 124, 'refused': 16, 'left': 17}; /health 200; GPU memory 13650 -> 13650 MiB
- Sampling on the GPU (`checks/sampling-candidates.out`, `checks/sampling-seeded-*.out`): the 64 largest logits of every row of a step are found on the GPU and the draw is made from them (temperature, top-k, top-p and min-p as the host's own sampler applies them);
  with a seed, 8 texts came out the same with the sampling on the host and on the GPU, asked one at a time and four at a time on a cold cache each time (the same 8 texts in all four runs).
- The layout default on this GPU (`checks/layout-default.out`): the smallest layout, as measured as fast as the 12-bit one on an Ada GPU; `--layout mma12` as asked.
- A decode step on the GPU takes 46.6 ms for one sequence of 1,100 tokens, 48.6 ms for 4 and 51.2 ms for 8 (`checks/step-profile.out`).

A reply can in general depend on how many replies are decoded with it; on this GPU the logits of a sequence were the same bit for bit at every number of rows from 1 to 8 (`checks/rows-bits.out`).
