# CUDA graphs, batch invariance and serving on the lossless KV cache

Qwen3-8B, vLLM 0.30.0, the dev L4, 2026-10-01; the serving rows (`vllm bench serve`, below) were re-measured on 2026-10-03. With `kv: lossless`, vLLM's FULL decode graphs capture a decode step (`UNIFORM_SINGLE_TOKEN_DECODE`: vLLM's default `FULL_AND_PIECEWISE` replays a decode step as one graph and runs the rest piecewise),
a sequence's output does not depend on the other rows nor on the batch's padding, and no step's metadata waits for the GPU on the host.

## What was measured

| | measured |
| :--- | :--- |
| FULL decode graphs captured and replayed, a replay's tokens against eager's | kv lossless's FULL decode graphs (35 captured, cudagraph mode FULL_DECODE_ONLY) replayed: the 8 prompts' 64 greedy tokens and their logprobs bit for bit those of eager (8 of 8 prompts) |
| `vllm bench serve` (random 1,024 in / 256 out; saturated and 1 request a second; servers warm; bf16 weights and Glyd weights) | bf16 weights: saturated 0.93 against 0.76 requests a second (**1.23x**), 1 a second 1.08x to 1.11x (two seeds; each token 0.99x to 1.01x), one user's token 60.2 against 60.6 ms (0.993x) and first token 333 against 331 ms (1.00x); Glyd weights (`kv+glyd` against `glyd@1`, Glyd's own weights with vLLM's cache): saturated 1.20 against 1.05 (**1.14x**), 1 a second each token 0.94x, one user's token 47.6 against 47.9 ms (0.993x) and first token 432 against 431 ms (1.00x); the prefix cache's hit rate 0.0% on all four servers (`bench/`, below) |
| under `VLLM_BATCH_INVARIANT=1`, a sequence's decode outputs at batch 1 and 8 | under VLLM_BATCH_INVARIANT=1 (bf16 weights, kv lossless, eager) each of 8 prompts' 64 greedy tokens and their logprobs the same bits alone and in a batch of 8 (8 of 8 prompts; vLLM's own bf16 under the same variable and mode, the control: 8 of 8) / under VLLM_BATCH_INVARIANT=1 (bf16 weights, kv lossless, FULL decode graphs, no torch.compile) each of 8 prompts' 64 greedy tokens and their logprobs the same bits alone and in a batch of 8 (8 of 8 prompts; vLLM's own bf16 under the same variable and mode, the control: 8 of 8) |
| kernel launches a layer in decode | **3.00**; vLLM's own 3.00 (its cache write and two FlashAttention kernels); 420 kernels a step against 491 |
| host synchronisation in a step's metadata | the steps with a cached prefix and chunked prefill (at 2,048 tokens, a prefix of 1,600 tokens cached) build their metadata with the host not waiting for the GPU: torch's synchronization errors on for every build raised none, and the outputs are the run without them's |

## Checks on the final tree

- Kernels (`check_vllm.py --kv --kernels-only`): 34 passed, 0 failed. The new ones: a sequence's decode output the same bits alone, in a batch and in a batch padded with rows of no tokens (40 outputs compared); rows of no tokens give zeros;
  the decode kernel in a CUDA graph replayed on four steps' lengths, block tables and queries (rows shuffled, shorter, padded) gives the bits of a call, and the write kernel in a graph replayed on three steps' tokens and slots
  (the last ones -1) reads back bit for bit; attention within 0.0014 of an fp32 reference at every batch setting, and for every group size 1 to 8 query heads a KV head; torch's synchronization errors raise
  where the host waits and at none of the step's metadata operations.
- `ext/` (`--ext-only`): 8 passed, 0 failed: prefix caching, chunked prefill and `exact` against vLLM's own as in [`l4-vllm-kv-extension-2026-10-01`](../l4-vllm-kv-extension-2026-10-01) (prompt logprobs bit for bit at every position, the first token bit for bit, `exact` 8 of 8), and the run with torch's synchronization errors on for every step's metadata.
- `san/` (`--sanitize`): 4 passed, 0 failed: memcheck, initcheck, racecheck and synccheck over the kernels' quick checks: 0 errors each.
- `full/` (`--no-kernels --no-ext`): 15 passed, 0 failed.
- Batches with cached prefixes (`distinct/`, one prompt at a time through `check_vllm.py --child`, against vLLM's own, eager):
  - four requests with a cached prefix of 1,600 tokens each of their own in one batch: the first token's logprob bit for bit vLLM's in 4 of 4, tokens cached the same (1600 each), the 64 greedy tokens the same in 1 of 4 (from the start for [56, 8, 41, 64]).
  - six requests with a cached prefix of 2,048 tokens each of their own: the first token's logprob bit for bit vLLM's in 6 of 6, tokens cached the same (2048 each), the 64 greedy tokens the same in 6 of 6 (from the start for [64, 64, 64, 64, 64, 64]).
  - eight requests that share a cached prefix of 1,600 tokens: the first token's logprob bit for bit vLLM's in 8 of 8, tokens cached the same (1600 each), the 64 greedy tokens the same in 3 of 8 (from the start for [64, 17, 51, 31, 7, 64, 21, 64]).

## `vllm bench serve` (`bench/`; `bench_serve.sh` with `SEED=4`: random 1,024 in, 256 out, `--ignore-eos`, 0.9, `--max-model-len 4096`, prefix caching and chunked prefill on, async scheduling on (vLLM's defaults), V2 model runner; one server a mode, started twice, the second measured; 64 requests at 1 a second, 256 at once, 8 one at a time)

Re-measured on 2026-10-03 through the compiled package (kv-vllm merged on gpu-package-v027, 7f40a75a; vLLM 0.30.0), with the benchmark as it is now: every pass draws prompts of its own (the seed plus the pass's number: 4, 5 and 6) and the servers' logs give the prefix cache's hit rate. **The hit rate is 0.0% on all four servers, in each of the 195 lines their logs give** (`bench/hit-*.txt`, `bench/summary.txt`). This run replaces the rows of 2026-10-01 (0.94 against 0.77 requests a second, and 1.25 against 1.08 with Glyd's weights), whose passes shared one set of prompts (the hit rate was not read then); the package's Glyd weights are 11.38 GiB (11.83 then).

Modes: `bf16` stock; `kv` bf16 weights and the cache lossless; `glyd@1` Glyd's weights (tiered, 12-bit) with vLLM's own cache; `kv+glyd` both. The second table is each against `bf16`.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.65 | 166.4 | 5861 / 16710 | 100.7 / 112.9 | 64 | 1500 MHz, 74 C |
| 1 | glyd@1 | 0.78 | 200.3 | 698 / 1126 | 102.8 / 132.4 | 64 | 1215 MHz, 72 C |
| 1 | kv | 0.72 | 184.6 | 2171 / 8144 | 101.4 / 114.3 | 64 | 1500 MHz, 74 C |
| 1 | kv+glyd | 0.80 | 203.7 | 680 / 1133 | 96.7 / 124.9 | 64 | 1185 MHz, 76 C |
| inf | bf16 | 0.76 | 193.5 | 151326 / 310359 | 111.0 / 182.8 | 256 | 1380 MHz, 83 C |
| inf | glyd@1 | 1.05 | 269.6 | 105717 / 224786 | 156.3 / 253.4 | 256 | 1080 MHz, 84 C |
| inf | kv | 0.93 | 237.8 | 122692 / 254981 | 117.7 / 198.9 | 256 | 1410 MHz, 83 C |
| inf | kv+glyd | 1.20 | 307.9 | 87962 / 197773 | 172.4 / 268.1 | 256 | 1140 MHz, 83 C |
| one | bf16 | 0.06 | 16.2 | 331 / 339 | 60.6 / 60.7 | 8 |  |
| one | glyd@1 | 0.08 | 20.3 | 431 / 438 | 47.9 / 48.0 | 8 |  |
| one | kv | 0.06 | 16.3 | 333 / 344 | 60.2 / 60.3 | 8 |  |
| one | kv+glyd | 0.08 | 20.4 | 432 / 441 | 47.6 / 47.7 | 8 |  |

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 15.27 | 27,024 | 0.65 | 5,861 | 100.7 |
| 1 | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 0.78 (1.20x) | 698 (0.12x) | 102.8 (1.02x) |
| 1 | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.72 (1.11x) | 2,171 (0.37x) | 101.4 (1.01x) |
| 1 | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 0.80 (1.22x) | 680 (0.12x) | 96.7 (0.96x) |
| inf | bf16 | 15.27 | 27,024 | 0.76 | 151,326 | 111.0 |
| inf | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 1.05 (1.39x) | 105,717 (0.70x) | 156.3 (1.41x) |
| inf | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.93 (1.23x) | 122,692 (0.81x) | 117.7 (1.06x) |
| inf | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 1.20 (1.59x) | 87,962 (0.58x) | 172.4 (1.55x) |
| one | bf16 | 15.27 | 27,024 | 0.06 | 331 | 60.6 |
| one | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 0.08 (1.25x) | 431 (1.30x) | 47.9 (0.79x) |
| one | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.06 (1.01x) | 333 (1.00x) | 60.2 (0.99x) |
| one | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 0.08 (1.26x) | 432 (1.30x) | 47.6 (0.78x) |

The 1-a-second pass follows the arrival times, so it moves between seeds. The first run on the same box and package (seed 0, `bench/seed0/`: one of its 328 prompts appears twice, so the servers' logs show hits for it, at most 1.1% to 1.2% and 0.3% over the whole run, which the summary flags) gave `kv` 0.72 against 0.67 requests a second (1.08x) and a mean first token of 3,195 ms for bf16 where this run's is 5,861 ms. Its saturated rows are this run's within 1.4% and its one-user rows within 0.7%: `kv` 0.93 against 0.76 requests a second saturated, `kv+glyd` 1.19 against `glyd@1`'s 1.05 (1.13x), the one user's token 60.2, 60.6, 47.8 and 47.9 ms. The "started cold" lines of this run's `summary.txt` are not cold starts: vLLM's compile cache had been filled by the first run, so a mode's first and second starts show the same cache sizes (the first run's lines are cold).

## The shared 2,048-token prompt (`ttft/`; `bench_serve.sh`, DATASET=prefix: 1 prefix of 2,048 tokens, 128 of each request's own, 128 out; 64 requests at once, and 8 one at a time; the servers' defaults)

Eager (`--enforce-eager`):

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| inf | bf16 | 15.27 | 30,032 | 4.11 | 2,147 | 104.5 |
| inf | kv | 15.34 (1.00x) | 39,104 (1.30x) | 4.48 (1.09x) | 2,130 (0.99x) | 94.3 (0.90x) |
| one | bf16 | 15.27 | 30,032 | 0.13 | 82 | 61.8 |
| one | kv | 15.34 (1.00x) | 39,104 (1.30x) | 0.13 (1.01x) | 84 (1.03x) | 61.4 (0.99x) |

Default mode (compiled, FULL decode graphs; servers warm):

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| inf | bf16 | 15.27 | 27,024 | 4.15 | 2,187 | 103.2 |
| inf | kv | 15.34 (1.00x) | 35,104 (1.30x) | 4.54 (1.09x) | 2,189 (1.00x) | 92.8 (0.90x) |
| one | bf16 | 15.27 | 27,024 | 0.13 | 78 | 61.1 |
| one | kv | 15.34 (1.00x) | 35,104 (1.30x) | 0.13 (1.00x) | 83 (1.07x) | 60.8 (0.99x) |

The TTFT is 0.99x and 1.00x of vLLM's, as in [`l4-vllm-kv-extension-2026-10-01`](../l4-vllm-kv-extension-2026-10-01) (0.99x eager, 0.97x default mode).

## Decode kernel and tokens a second (`decode/`)

One layer's decode attention against vLLM's FlashAttention 2 on the same values, microseconds a call, in a CUDA graph, as a step has it (`kv_decode_bench.py`, `decode/decbench-graph.json`; the calls cycle through copies of the pages larger than the L2 cache; FlashAttention 2 is `flash_attn_varlen_func`, vLLM's own cache write is `reshape_and_cache_flash`):

| B x L | kv_decode | FlashAttention 2 | FA2 / kv | kv write | vLLM's write | the same calls as a loop from Python: kv_decode, FA2 | max difference of the outputs (of the largest output) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 x 1024 | 392.0 | 547.5 | 1.40 | 10.4 | 2.7 | 395.4, 561.0 | 0.00195 (53.25) |
| 4 x 8192 | 394.1 | 548.0 | 1.39 | 9.1 | 2.4 | 397.8, 562.9 | 0.00049 (95) |
| 1 x 8192 | 105.3 | 141.5 | 1.34 | 6.7 | 2.1 | 108.1, 148.2 | 0.00024 (0.25781) |
| 8 x 4096 | 393.9 | 548.4 | 1.39 | 8.8 | 2.3 | 398.7, 563.6 | 0.00049 (95) |
| 3 x 8192 | 298.6 | 407.4 | 1.36 | 7.7 | 2.3 | 304.0, 427.5 | 0.00049 (159) |
| 1 x 1024 | 18.3 | 27.0 | 1.48 | 6.6 | 2.2 | 37.0, 145.2 | 0.00098 (0.30664) |
| 44 x 1024 | 538.0 | 757.1 | 1.41 | 11.0 | 2.6 | 544.3, 783.8 | 0.00195 (99.5) |
| 5 x 8192 | 490.4 | 674.1 | 1.38 | 8.3 | 2.5 | 496.6, 693.9 | 0.0625 (9.1875) |

Timed as a loop of calls from Python (the last but one column; the first version of this table, `decode/decbench.txt`), a FlashAttention call costs about 100 us of the host's before it launches anything, which is all of a small shape's time (1 x 1,024: 99.5 us a call in the first version of this table, against 27.0 us in a graph): the same calls in a CUDA graph have none of it, and the ratios at the small shapes fall to the large ones'. vLLM's cache write is 2 to 3 us in a graph where Glyd's is 7 to 11 us, which 36 layers add up to 0.16 to 0.3 ms of a step. The outputs differ by at most one bf16 step (0.0625 where the largest output is 9.2); the inputs are finite (the first version's timing data had outliers that overflow a score, and its difference was NaN).

Decode tokens a second, vLLM's default mode (compiled, FULL decode graphs), 128 new tokens, one run each. Each side at the largest batch its cache holds (0.96, `decode/tps-graphs/`):

KV cache: stock 28,656 tokens, kv 36,848 (1.286x); decode tokens a second (128 new tokens, default mode: compiled, FULL decode graphs, bf16 weights, util 0.96; one run each)

| Batch x tokens | Stock | KV | KV / stock |
| :--- | ---: | ---: | ---: |
| stock 4 x 8192, kv 4 x 8192 | does not fit |  |  |
| stock 32 x 1024, kv 32 x 1024 | does not fit |  |  |
| stock 1 x 8192, kv 1 x 8192 | 15.5 (64.4 ms a step; 1485, 75) | 15.8 (63.3 ms a step; 1455, 80) | 1.018 |
| stock 3 x 8192, kv 4 x 8192 | 38.8 (77.4 ms a step; 1410, 79) | 51.7 (77.3 ms a step; 1425, 78) | 1.334 |
| stock 27 x 1024, kv 35 x 1024 | 318.3 (84.8 ms a step; 1425, 79) | 411.4 (85.1 ms a step; 1485, 80) | 1.293 |

At batches both hold (0.96, `decode/tps-graphs-same-batch/`):

KV cache: stock 32,208 tokens, kv 41,664 (1.294x); decode tokens a second (128 new tokens, default mode: compiled, FULL decode graphs, bf16 weights, util 0.96; one run each)

| Batch x tokens | Stock | KV | KV / stock |
| :--- | ---: | ---: | ---: |
| stock 3 x 8192, kv 3 x 8192 | 38.8 (77.3 ms a step; 1455, 77) | 40.6 (74.0 ms a step; 1410, 80) | 1.045 |
| stock 24 x 1024, kv 24 x 1024 | 289.4 (82.9 ms a step; 1485, 80) | 300.7 (79.8 ms a step; 1485, 80) | 1.039 |
| stock 6 x 4096, kv 6 x 4096 | 77.6 (77.4 ms a step; 1560, 80) | 81.2 (73.9 ms a step; 1575, 80) | 1.047 |
| stock 1 x 8192, kv 1 x 8192 | 15.5 (64.5 ms a step; 1440, 79) | 15.8 (63.3 ms a step; 1425, 79) | 1.019 |

Eager, the same batches (0.9, `decode/tps-eager-same-batch/`):

KV cache: stock 25,232 tokens, kv 32,576 (1.291x); decode tokens a second (128 new tokens, eager, bf16 weights, util 0.9; one run each)

| Batch x tokens | Stock | KV | KV / stock |
| :--- | ---: | ---: | ---: |
| stock 3 x 8192, kv 3 x 8192 | 38.5 (78.0 ms a step; 1485, 76) | 40.2 (74.6 ms a step; 1455, 78) | 1.046 |
| stock 24 x 1024, kv 24 x 1024 | 284.4 (84.4 ms a step; 1560, 76) | 294.5 (81.5 ms a step; 1530, 81) | 1.036 |
| stock 6 x 4096, kv 6 x 4096 | 76.5 (78.4 ms a step; 1575, 77) | 80.3 (74.7 ms a step; 1590, 79) | 1.049 |
| stock 1 x 8192, kv 1 x 8192 | 15.3 (65.3 ms a step; 1485, 79) | 15.6 (64.1 ms a step; 1470, 76) | 1.020 |

KV cache tokens at the standard flags (eager, `--max-model-len 8704`, 8,704 tokens a step, 0.9; `capacity/`): stock 25,232, `kv` 32,576, **1.2911x**; in vLLM's default mode (1,922 graphs captured by each): 22,576 and 29,072, 1.2877x.

## Files

`ext/`, `san/`, `full/`, `distinct/` (the batches), `bench/` (each mode's results JSON, the lines of its server's log with its sizes and its prefix cache hit rate (`hit-*.txt`), nvidia-smi's clock and temperature, and the summary; `seed0/`: the first run's JSON, hit-rate lines and summary), `ttft/`, `decode/`, `launches/` (`kv_launches.py`: the kernels of 8 decode steps of 8 requests of 1,024 tokens under torch's profiler, default mode), `capacity/`,
`mrv1/` (vLLM's previous model runner, `VLLM_USE_V2_MODEL_RUNNER=0`: FULL graphs ran), `invariance/` (earlier standalone runs of the invariance pair). The tools are in [`gpu/vllm/`](../../../gpu/vllm). Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record. Logs are `.txt`.
