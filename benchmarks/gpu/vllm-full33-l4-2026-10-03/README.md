# vLLM with the embedding and the LM head packed too (v0.28's branch full33), on an AWS L4, 2026-10-03 and 04

`vllm serve --quantization glyd` on the branch `full33` (from v0.27.0, d5454a7e; private main's MoE work merged in), against
v0.27.0's plugin (d5454a7e's sources, its library built from them on the same box) and against vLLM's own bf16, on one
NVIDIA L4. The embedding and the LM head are packed in the layout of the Linears; lookups return the checkpoint's rows bit
for bit; the LM head's product is the library's own, within the usual product tolerance (`exact` keeps the LM head as vLLM
runs it).

## Setup

- **Machine:** AWS g6.4xlarge, NVIDIA L4 (23,034 MiB, sm_89), driver 595.91.07, AMD EPYC 7R13 (16 CPUs), Linux 7.0.
- **Software:** vLLM 0.30.0 from PyPI, torch 2.13.0+cu130, transformers 5.18.0, nvcc 13.0.88; the library built for
  sm_89 with `gpu/build_lib.sh`'s flags, one architecture (C API 14, unchanged); the plugin from the sources.
- **Runs:** every one under the box's lock (`flock ~/.glyd-box.lock`), one at a time, each vLLM in a process of its own,
  compiled with CUDA graphs (vLLM's default), `kv` off, `--gpu-memory-utilization 0.9` (Qwen3-4B 0.85, as its earlier
  starts), 2,048 tokens of context.

## Weights and KV cache, as vLLM logs them

"Model loading took" and "GPU KV cache size" in each run's log (`log/weights_kv_lines.txt` has every line); the same flags for the three
columns of a model. The first start of v0.27.0's plugin on the two 8B models (an empty compile cache) logs a lower KV
cache than its later starts (on Qwen3-8B vLLM's memory profile of that start counts 1.14 GiB more non-torch memory and 0.53 GiB more
activations; neither 8B model's first start on this branch shows it), so both are given.

| | bf16 | v0.27.0's plugin | this branch |
| :--- | ---: | ---: | ---: |
| Qwen3-8B, weights GiB | 15.27 | 11.38 (-25.5%) | **10.45 (-31.6%)** |
| Qwen3-8B, KV cache tokens, later starts (first start) | 22,928 | 50,368 (38,224) | **56,816** (56,816) |
| Qwen3-4B (tied), weights GiB, 85% of the GPU | 7.56 | 5.48 (-27.5%) | **5.15 (-31.9%)** |
| Qwen3-4B, KV cache tokens | 70,800 | 84,880 | **88,832** |
| Llama-3.1-8B, weights GiB | 15.0 | 11.05 (-26.3%) | **10.16 (-32.3%)** |
| Llama-3.1-8B, KV cache tokens, later starts (first start) | 25,232 | 59,184 (49,264) | **66,096** (66,096) |

The two small Qwen3 checkpoints whose config ties the tables and whose file holds an lm_head too (vLLM builds them apart; the second table is the first's bits, so one pack),
from `check_vllm.py --quick`'s own runs (85% of the GPU, the glyd run without `verify`):

| | bf16 | this branch |
| :--- | ---: | ---: |
| Qwen3-1.7B, weights GiB / KV cache tokens | 3.22 / 127,984 | **2.24 (-30.4%)** / **143,872** |
| Qwen3-0.6B, weights GiB / KV cache tokens | 1.12 / 156,320 | **0.79 (-29.5%)** / **159,504** |

## One user's tokens a second (`tps_ab.py`)

Qwen3-8B, one request of a random 1,024-token prompt decoded for 256 tokens (the time of the run less the time of a
run of one token, over the new tokens; the median of 3 requests), vLLM's default mode (compiled, CUDA graphs), `kv` off, a
vLLM of its own for each of 3 rounds of each plugin, the order of a round alternating (so the L4's clock, which falls as it
heats at its power cap, goes both ways: 1,155-1,470 MHz, 69-78 C over the runs).

| | tokens a second (rounds) | weights GiB | KV cache tokens (last round's start) |
| :--- | ---: | ---: | ---: |
| v0.27.0's plugin | 21.06 (21.10, 21.06, 21.04) | 11.38 | 50,368 |
| this branch | **21.68** (21.74, 21.68, 21.48) | 10.45 | 56,816 |

This branch against v0.27.0's plugin: **1.029x** (each round 1.030, 1.029, 1.021): the LM head's product is the one memory-bound product
that was bf16 (3.50 ms packed against 4.88 ms for Qwen3-8B's head at one row, `head_sweep.txt`), 1.4 ms of a 47 ms step.

## Saturated serving (`bench_serve.sh`, `vllm bench serve`)

Qwen3-8B, `vllm serve` as a user runs it (the engine in a process of its own, compiled, CUDA graphs, `--gpu-memory-utilization
0.9`, `--max-model-len 4096`), the random dataset, 1,024 tokens in and 256 out (`--ignore-eos`), 256 prompts sent at once
(`--request-rate inf`); a seed of its own for each pass (10 and 20), the same prompts for both plugins in a round; the measured
server is each plugin's second start of the round (the first compiled; its graphs and compile cache loaded), the order of the two
plugins alternating between rounds. The prefix cache is vLLM's default (on): its highest hit rate, from the server's log, was 0.0% in round 1 and 0.9% in round 2 for
both plugins (under the 1% above which a run is flagged). At this load the L4 held 1,080-1,125 MHz at 81-83 C (its power cap).

| | v0.27.0's plugin | this branch |
| :--- | ---: | ---: |
| KV cache tokens (max concurrency at 4,096 tokens) | 54,464 (13.3x) | **60,192 (14.7x)** |
| Output tokens a second, rounds 1 and 2 | 269.2, 268.5 | **278.7, 280.8** |
| Requests a second | 1.05, 1.05 | **1.09, 1.10** |
| Time to first token, mean (p99) s | 105.0 (224.8), 105.2 (225.9) | **100.5 (204.5), 100.6 (205.0)** |
| Time per output token, mean (p99) ms | 156.6 (253.2), 158.0 (256.7) | 168.3 (267.1), 168.5 (267.6) |

This branch's output tokens a second against v0.27.0's plugin: **1.035x and 1.046x**. Its time per output token is higher because
the larger KV cache keeps about 10% more sequences in each step (the step is longer, the run shorter): no loss at saturation, so the
chunked decode-then-GEMM alternative for the LM head was not tried.

## Where Qwen3-8B's weights are (`acct.py`: every block the allocator holds before vLLM's profile run, by owner)

The tiered layout, eager, one start, the branch before its decode buffer change; "Model loading took" is the allocated bytes at the end of the load less those
at its start (0.164 GiB, vLLM's own: a 150 MiB block and some small ones, the same under v0.27.0's plugin).

| GiB | v0.27.0's plugin | this branch |
| :--- | ---: | ---: |
| Linears' packs (36 layers) | 8.8644 | 8.8648 |
| Embedding and LM head: bf16 / packed | 2.3203 (bf16) | 1.5694 (packed) |
| Rotary cache, norm weights | 0.0098, 0.0006 | 0.0098, 0.0006 |
| The model's own tensors | 11.1951 | 10.4446 |
| The plugin's decode buffer (gate_up, 24,576 x 4,096 bf16: 192 MiB) | 0.1875 | 0.1875 |
| Model loading took (vLLM's log) | 11.38 | 10.63 |

The decode buffer is where a prompt of 896 tokens or more (2,560 for the 12-bit layout) is decoded for cuBLAS on the L4; now it is made for the call
(`_unpack`) where no CUDA graph can capture the decode, so a load takes 10.45 GiB and the KV cache is 56,816 tokens, not 55,904: the part of the buffer
that coexists with vLLM's profile run's peak (0.06 GiB of activation) is the difference from the 1,365 tokens the whole buffer would be (`sc-q8-*.log`).
Against bf16's 15.26 GiB of tensors: the model's own tensors -31.6% (10.435 GiB without the rotary cache), as vLLM reports it -31.6% now (15.27 to 10.45;
-30.4% with the decode buffer held), the pack alone 10.94 bits a weight.

The 12-bit layout, the same accounting on the final tree (the decode buffer not held): Linears' packs 9.7373 GiB, embedding and LM head 1.7473 GiB, the model's own tensors 11.495 GiB, vLLM's
"Model loading took" 11.50: **-24.7%** against bf16's 15.27 (`log/accounting.txt`).

## The LM head's product and the embedding's lookup (`head_sweep.py`, Qwen3-8B's head, 151,936 x 4,096)

CUDA events over 20 calls, the median of 3; bf16 is `F.linear` on the matrix (what vLLM runs without Glyd). The last column is the other way to a product
of many rows, the matrix decoded in pieces of at most 64 MB, then cuBLAS on each: slower than the library's own product from 1 to 1,024 rows in both
layouts and better only for the smallest layout from 2,048 rows (1.28x of bf16's time against 1.35x; 1.20x against 1.33x at 4,096), so the plugin does not use it.
Faster than bf16 where the product is memory-bound (up to 64 rows: the decode steps of any batch the L4 holds), about equal from 96 to 256 rows (0.96-1.06x tiered, 0.80-0.98x 12-bit),
slower from 384 rows (steps of 384 or more sequences, or prompt logprobs, per call): by 1.18-1.39x (smallest layout) and 1.00-1.12x (12-bit).


**mma** (10.77 bits a weight, 0.780 GiB (bf16 1.159 GiB))

| rows | route | packed ms | bf16 ms | packed / bf16 | in 64 MB pieces + cuBLAS / bf16 |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | gemm | 3.500 | 4.879 | **0.72** | 2.44 |
| 8 | gemm | 3.704 | 4.956 | **0.75** | 2.44 |
| 16 | gemm | 3.872 | 4.993 | **0.78** | 2.43 |
| 32 | gemm | 4.244 | 5.434 | **0.78** | 2.37 |
| 48 | gemm | 5.027 | 5.581 | **0.90** | 2.49 |
| 64 | gemm | 5.353 | 5.614 | **0.95** | 2.60 |
| 96 | prompt kernel | 5.860 | 5.645 | **1.04** | 2.44 |
| 128 | prompt kernel | 6.102 | 5.732 | **1.06** | 2.43 |
| 256 | prompt kernel | 8.892 | 9.286 | **0.96** | 1.83 |
| 384 | prompt kernel | 15.368 | 11.722 | **1.31** | 1.81 |
| 512 | prompt kernel | 16.529 | 14.026 | **1.18** | 1.80 |
| 768 | prompt kernel | 24.082 | 18.307 | **1.32** | 1.83 |
| 1024 | decode | 32.148 | 23.180 | **1.39** | 1.75 |
| 2048 | decode | 62.257 | 46.001 | **1.35** | 1.28 |
| 4096 | decode | 124.250 | 93.080 | **1.33** | 1.20 |

The embedding's lookup (ms a call; ids drawn uniformly over the table, Zipf-distributed ones are faster: `head_sweep.txt`), bf16's `F.embedding` beside it:

| tokens | packed | bf16 |
| ---: | ---: | ---: |
| 1 | 0.0127 | 0.0107 |
| 8 | 0.0120 | 0.0089 |
| 64 | 0.0374 | 0.0136 |
| 256 | 0.3436 | 0.0151 |
| 1024 | 1.3688 | 0.0176 |
| 4096 | 5.9473 | 0.2469 |
| 8192 | 12.1610 | 0.5590 |


**mma12** (12.03 bits a weight, 0.872 GiB (bf16 1.159 GiB))

| rows | route | packed ms | bf16 ms | packed / bf16 | in 64 MB pieces + cuBLAS / bf16 |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | gemm | 3.625 | 4.875 | **0.74** | 2.53 |
| 8 | gemm | 3.676 | 4.959 | **0.74** | 2.51 |
| 16 | gemm | 3.719 | 4.995 | **0.74** | 2.50 |
| 32 | x2 | 3.748 | 5.440 | **0.69** | 2.47 |
| 48 | x2 | 4.181 | 5.754 | **0.73** | 2.45 |
| 64 | x2 | 4.348 | 5.859 | **0.74** | 2.51 |
| 96 | prompt kernel | 4.734 | 5.971 | **0.79** | 2.44 |
| 128 | prompt kernel | 5.061 | 6.111 | **0.83** | 2.38 |
| 256 | prompt kernel | 8.962 | 9.105 | **0.98** | 2.05 |
| 384 | prompt kernel | 12.851 | 12.228 | **1.05** | 1.86 |
| 512 | prompt kernel | 14.837 | 14.900 | **1.00** | 1.76 |
| 768 | prompt kernel | 21.213 | 19.517 | **1.09** | 1.79 |
| 1024 | prompt kernel | 27.422 | 24.505 | **1.12** | 1.71 |
| 2048 | prompt kernel | 51.152 | 46.876 | **1.09** | 1.26 |
| 4096 | decode | 99.158 | 91.877 | **1.08** | 1.18 |

The embedding's lookup (ms a call; ids drawn uniformly over the table, Zipf-distributed ones are faster: `head_sweep.txt`), bf16's `F.embedding` beside it:

| tokens | packed | bf16 |
| ---: | ---: | ---: |
| 1 | 0.0125 | 0.0090 |
| 8 | 0.0122 | 0.0095 |
| 64 | 0.0337 | 0.0140 |
| 256 | 0.3698 | 0.0136 |
| 1024 | 1.5234 | 0.0139 |
| 4096 | 6.1012 | 0.2454 |
| 8192 | 12.2234 | 0.5663 |

## The lossless KV cache with the tables packed (`kv` auto and lossless)

- `vllm serve --quantization glyd` at its defaults (`kv` auto, which is lossless on an L4 for these models), `--max-model-len 4096`, 85% of the GPU, the engine in a process of
  its own: Qwen3-4B (tied) "the KV cache lossless ... the embedding and the LM head packed too (tied: one table)", 118,944 KV cache tokens; Qwen3-8B 68,592; each answered a
  completion over HTTP (`log/kv_combination.txt`).
- `kv` lossless against `kv` off on the same packs (eager, 4,096 tokens, 85% of the GPU, tiny_run's four prompts of 24 tokens): Qwen3-4B 119,200 KV cache tokens against 91,712
  (+30%), Qwen3-8B 67,696 against 51,648 (+31%); the prompt logprobs bit for bit kv off's (both models); the generated tokens the same for 3 of 4 prompts on Qwen3-4B and Qwen3-8B,
  the other diverging at its 17th and 10th token (the lossless backend's own attention kernel rounds as vLLM's does not). v0.27.0's plugin on Qwen3-8B at the same flags: KV cache 59,344 tokens against 45,440
  with `kv` off, the generated tokens equal for 24, 24, 10 and 24 tokens of the four prompts and the prompt logprobs bit for bit: the same four as this branch's (`log/kv_combination.txt`).
- A first start makes the model's KV tables (a child process runs the model on a bundled text: a few minutes). That calibration read the embedding's matrix for the norms
  of its rows, which the plugin no longer holds (an empty weight: it crashed): it decodes them from the pack now (`kvcal._norms_of`; test in `test_embedding_ops`).
  Found by this check; the tests before it had KV tables made already or `kv` off.
- `check_vllm.py --kv --quick` (the kernels' checks through the library): all passed (67 checks).

## Checks

- **Plugin tests** (`test_vllm.py`, with the GPU): 27 ok from the sources and 27 from the compiled wheel on the branch's last tree, among them the embedding's ops (lookups
  bit for bit over padded tables in both layouts, ids outside the table, the head's product within 1e-2 of `F.linear` for 1 to 1,000 rows with and without a bias, a tied
  head holding the embedding's buffers, a config that ties them with a copy in the checkpoint taking the first pack if the bits are the same) and the decode buffer's rule;
  `test_gpu.py` 39 ok; the glyd-gpu crate's tests (its declarations are the header's).
- **The library:** every row of the real tables bit for bit, in order and shuffled, both layouts (`rows_every_row.txt`: Qwen3-8B's and Llama-3.1-8B's
  embeddings and heads, Qwen3-4B's embedding), on the last tree; `check_capi.py` through both hosts bit for bit (`mma_rows`, `mma12_rows`: 32 calls each and 8 refused each; its prompt
  kernel's and its decoded routes' names, stale since v0.27.0's renaming, are corrected); the release gate (`check_wheel.py`) on the library and on the wheel built from
  this tree: 0 findings.
- **vLLM, tiny random Qwen3-architecture models** (`tables_check.py`, each in a vLLM of its own, compiled with CUDA graphs): tied (32,064 rows), untied (32,000),
  tied and untied with a vocabulary of 32,001 (vLLM pads it to 32,064: the weight loads fewer rows than the layer holds), the config tied with an lm_head in the checkpoint
  (equal to the embedding, and not): every table packed (one pack where tied or equal), every lookup the checkpoint's row bit for bit, each product within 1e-2 and the same bits
  twice, the first token of every prompt bf16's where bf16's margin is clear, prompt logprobs within 0.007 on average of bf16's, and `exact` (eager) bf16 eager's tokens,
  logprobs and prompt logprobs bit for bit with the tied models as vLLM runs them and the untied ones' embedding packed: 24 of 24 (`tables_check_6_models.txt` for the four models
  that are tied or untied outright, `tables_check_tc_eq_tc_ne.txt` for the other two; the first run's one failure was this check's own expectation of `exact` for a config that ties them,
  corrected and run again).
- **glyd saves** of a tied and an untied tiny model, in both layouts, loaded by vLLM in both layouts (eager): the same bits as the bf16 checkpoint packed at load, 8 of 8 (`saves.txt`).
- **`check_vllm.py --quick`** (`check_vllm/`): Qwen3-8B (untied) and Llama-3.1-8B (untied): all 18 passed each, on the tree with the decode buffer made for the call: the embedding and the LM head packed in both
  layouts (Qwen3-8B: 2 packs, 1,605 MiB tiered and 1,786 MiB 12-bit; Llama: 1,350 and 1,512), every lookup the checkpoint's row bit for bit, each product within 1e-2 (3.5e-3), every pack (144; Llama 128)
  decoded to its weights bit for bit, each layer's product within 1e-2 of `F.linear` at 1-4,096 tokens (4.7e-3 tiered, 3.7e-3 12-bit), top-1 agreement with bf16 on its continuation 0.9935 and 0.9922
  (Qwen3-8B; bf16 eager's own 0.9909) and 0.9942 and 0.9955 (Llama; 0.9948), the mean |logprob difference| within twice bf16's noise floor, `exact` (eager) bf16 eager's tokens, logprobs and prompt
  logprobs bit for bit (the untied models' embedding packed, the head as vLLM runs it), the compile caches. Qwen3-4B (tied): all 18 passed on the branch before the decode buffer's change and the later merges
  (`check_vllm/cv-q4-report.txt`: tied, one pack of 558 MiB tiered, `exact`: the one table as vLLM runs it); its `--brief` run in the cloud job's dry run (7 checks) passed on the last tree.
  The weights `check_vllm` prints for its glyd runs are of a run with `verify` on (the loading peak decodes the largest table again), so the weights above are of the plain runs.
- **Qwen3-1.7B and Qwen3-0.6B** (the config ties the tables, the checkpoint holds an lm_head equal to the embedding: vLLM builds them apart; `check_vllm/cv-q17-report.txt`, `cv-q06-report.txt`): all 18 passed each;
  one pack for the two tables (398 MiB and 199 MiB tiered: "lm_head holds the same table as the embedding or LM head packed before it: one pack"), every lookup the checkpoint's row bit for bit, `exact`: both as vLLM runs them.
- **compute-sanitizer** (memcheck, initcheck, racecheck, synccheck) over the rows kernels (`mma_rows`, `mma12_rows`: ends, repeats, zeros, inf and NaN rows, ids outside the table, K of 48, 64 and 128): 0 errors, 0 hazards (`log/compute_sanitizer.txt`).
- **KV cache:** see its section above.
- **The cloud job's shell** (`tp2_job.sh`, a dry run on this L4 with `TJ_TP=1`, vLLM's venv and the library given): every step ran and passed, 9 min 43 s (`cloud_job_dry_run/`).

## Files

- `log/`: `head_sweep.txt`, `accounting.txt` (acct.py), `weights_kv_lines.txt` (every start's weights and KV lines), `test_vllm_sources.txt`, `test_vllm_compiled_wheel.txt`, `test_gpu.txt`,
  `check_capi.txt`, `rows_every_row.txt`, `tables_check_6_models.txt` and `tables_check_tc_eq_tc_ne.txt`, `saves.txt`, `kv_combination.txt`, `machine.txt`, `compute_sanitizer.txt`, `driver_log.txt` (the order the holds of the box's lock ran in; the `exit 0` lines at 22:18 and 02:42 are of holds I killed and queued again).
- `check_vllm/`: each model's report (`*-report.txt`) and log (`*.txt`).
- `one_user/`: `tps_ab.txt` and each round's JSON. `saturated_serve/`: each pass's summary, `vllm bench serve` output and result JSON, server log lines and GPU clock and temperature each second.
- `cloud_job_dry_run/`: `tp2_job.sh` run on this L4 (one GPU, its venv and library given): its summary and steps.
- The scripts: `gpu/vllm/head_sweep.py`, `tables_check.py`, `tps_ab.py`, `bench_serve.sh`, `check_vllm.py`, `acct.py`, `routes.py`, `tp2_job.sh`; the library's rows kernel in `gpu/glyd_gpu.cu` (`mma_rows_kernel`).
