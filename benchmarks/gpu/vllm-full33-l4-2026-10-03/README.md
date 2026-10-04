# vLLM with the embedding and the output layer packed too, on an AWS L4 (2026-10-03 and 04)

`vllm serve --quantization glyd` with the embedding and the LM head (the output layer) packed as the Linears are, against v0.27.0's plugin and against vLLM's own bf16, on one NVIDIA L4. Every lookup of the embedding returns the checkpoint's row bit for bit; the output layer's product is the library's own, within the usual product tolerance (`exact` keeps the output layer as vLLM runs it, so its logits are bf16's bit for bit). A model whose config ties the two has one packed copy.
This is the code released in v0.28.0 as of the run's day; [vllm-v028-l4-2026-10-04](../vllm-v028-l4-2026-10-04) repeats the measurements on the release's final tree.

## Setup

- **Machine:** AWS g6.4xlarge, NVIDIA L4 (23,034 MiB, sm_89), driver 595.91.07, AMD EPYC 7R13 (16 CPUs), Linux 7.0.
- **Software:** vLLM 0.30.0 from PyPI, torch 2.13.0+cu130, transformers 5.18.0, nvcc 13.0.88; v0.27.0's plugin and this code, each with the library built for sm_89 on the same box.
- **Runs:** every one at a time on the box, each vLLM in a process of its own, compiled with CUDA graphs (vLLM's default), `kv` off, `--gpu-memory-utilization 0.9` (Qwen3-4B 0.85), 2,048 tokens of context.

## Weights and KV cache, as vLLM logs them

"Model loading took" and "GPU KV cache size" in each run's log (`log/weights_kv_lines.txt` has the lines); the same flags for the three columns of a model. The first start of v0.27.0's plugin on the two 8B models (an empty compile cache) logs a lower KV cache than its later starts, so both are given; neither 8B model's first start on this code shows it.

| | bf16 | v0.27.0's plugin | this code |
| :--- | ---: | ---: | ---: |
| Qwen3-8B, weights GiB | 15.27 | 11.38 (-25.5%) | **10.45 (-31.6%)** |
| Qwen3-8B, KV cache tokens, later starts (first start) | 22,928 | 50,368 (38,224) | **56,816** (56,816) |
| Qwen3-4B (tied), weights GiB, 85% of the GPU | 7.56 | 5.48 (-27.5%) | **5.15 (-31.9%)** |
| Qwen3-4B, KV cache tokens | 70,800 | 84,880 | **88,832** |
| Llama-3.1-8B, weights GiB | 15.0 | 11.05 (-26.3%) | **10.16 (-32.3%)** |
| Llama-3.1-8B, KV cache tokens, later starts (first start) | 25,232 | 59,184 (49,264) | **66,096** (66,096) |

The two small Qwen3 checkpoints whose config ties the two tables and whose file holds an lm_head too (vLLM builds them apart; the second table is the first's bits, so one pack), from `check_vllm.py --quick`'s own runs (85% of the GPU):

| | bf16 | this code |
| :--- | ---: | ---: |
| Qwen3-1.7B, weights GiB / KV cache tokens | 3.22 / 127,984 | **2.24 (-30.4%)** / **143,872** |
| Qwen3-0.6B, weights GiB / KV cache tokens | 1.12 / 156,320 | **0.79 (-29.5%)** / **159,504** |

Qwen3-8B's embedding and output layer take 2.3203 GiB in v0.27.0's plugin (bf16) and 1.5694 GiB packed. With `mma12`, the faster layout, the whole model takes 11.50 GiB in vLLM's log: **-24.7%** against bf16's 15.27.

## One user's tokens a second (`tps_ab.py`)

Qwen3-8B, one request of a random 1,024-token prompt decoded for 256 tokens (the time of the run less the time of a run of one token, over the new tokens; the median of 3 requests), vLLM's default mode (compiled, CUDA graphs), `kv` off, a vLLM of its own for each of 3 rounds of each plugin, the order of a round alternating (so the L4's clock, which falls as it heats at its power cap, goes both ways: 1,155-1,470 MHz, 69-78 C over the runs). `base` is v0.27.0's plugin, `new` this code.

| | tokens a second (rounds) | weights GiB | KV cache tokens (last round's start) |
| :--- | ---: | ---: | ---: |
| v0.27.0's plugin | 21.06 (21.10, 21.06, 21.04) | 11.38 | 50,368 |
| this code | **21.68** (21.74, 21.68, 21.48) | 10.45 | 56,816 |

This code against v0.27.0's plugin: **1.029x** (each round 1.030, 1.029, 1.021): the output layer's product is the one memory-bound product that was bf16 (3.50 ms packed against 4.88 ms for Qwen3-8B's head at one row), 1.4 ms of a 47 ms step.

## Saturated serving (`vllm bench serve`)

Qwen3-8B, `vllm serve` as a user runs it (the engine in a process of its own, compiled, CUDA graphs, `--gpu-memory-utilization 0.9`, `--max-model-len 4096`), the random dataset, 1,024 tokens in and 256 out (`--ignore-eos`), 256 prompts sent at once (`--request-rate inf`); a seed of its own for each pass (10 and 20), the same prompts for both plugins in a round; the measured server is each plugin's second start of the round (the first compiled), the order of the two plugins alternating between rounds. The prefix cache is vLLM's default (on): its highest hit rate, from the server's log, was 0.0% in round 1 and 0.9% in round 2 for both plugins (under the 1% above which a run is flagged). At this load the L4 held 1,080-1,125 MHz at 81-83 C (its power cap).

| | v0.27.0's plugin | this code |
| :--- | ---: | ---: |
| KV cache tokens (max concurrency at 4,096 tokens) | 54,464 (13.3x) | **60,192 (14.7x)** |
| Output tokens a second, rounds 1 and 2 | 269.2, 268.5 | **278.7, 280.8** |
| Requests a second | 1.05, 1.05 | **1.09, 1.10** |
| Time to first token, mean (p99) s | 105.0 (224.8), 105.2 (225.9) | **100.5 (204.5), 100.6 (205.0)** |
| Time per output token, mean (p99) ms | 156.6 (253.2), 158.0 (256.7) | 168.3 (267.1), 168.5 (267.6) |

This code's output tokens a second against v0.27.0's plugin: **1.035x and 1.046x**. Its time per output token is higher because the larger KV cache keeps about 10% more sequences in each step (the step is longer, the run shorter): no loss at saturation.

## The output layer's product and the embedding's lookup (`head_sweep.py`, Qwen3-8B's head, 151,936 x 4,096)

CUDA events over 20 calls, the median of 3; bf16 is `F.linear` on the matrix (what vLLM runs without Glyd). Faster than bf16 where the product is memory-bound (up to 64 rows: the decode steps of any batch the L4 holds), about equal from 96 to 256 rows (0.96-1.06x with `mma`, 0.79-0.98x with `mma12`), slower from 384 rows (steps of 384 or more sequences, or prompt logprobs, per call): by 1.18-1.39x (`mma`) and 1.00-1.12x (`mma12`).

**mma** (the matrix 0.780 GiB, bf16 1.159 GiB)

| rows | packed ms | bf16 ms | packed / bf16 |
| ---: | ---: | ---: | ---: |
| 1 | 3.500 | 4.879 | **0.72** |
| 8 | 3.704 | 4.956 | **0.75** |
| 16 | 3.872 | 4.993 | **0.78** |
| 32 | 4.244 | 5.434 | **0.78** |
| 48 | 5.027 | 5.581 | **0.90** |
| 64 | 5.353 | 5.614 | **0.95** |
| 96 | 5.860 | 5.645 | **1.04** |
| 128 | 6.102 | 5.732 | **1.06** |
| 256 | 8.892 | 9.286 | **0.96** |
| 384 | 15.368 | 11.722 | **1.31** |
| 512 | 16.529 | 14.026 | **1.18** |
| 768 | 24.082 | 18.307 | **1.32** |
| 1024 | 32.148 | 23.180 | **1.39** |
| 2048 | 62.257 | 46.001 | **1.35** |
| 4096 | 124.250 | 93.080 | **1.33** |

The embedding's lookup (ms a call; ids drawn uniformly over the table, Zipf-distributed ones are faster), bf16's `F.embedding` beside it:

| tokens | packed | bf16 |
| ---: | ---: | ---: |
| 1 | 0.0127 | 0.0107 |
| 8 | 0.0120 | 0.0089 |
| 64 | 0.0374 | 0.0136 |
| 256 | 0.3436 | 0.0151 |
| 1024 | 1.3688 | 0.0176 |
| 4096 | 5.9473 | 0.2469 |
| 8192 | 12.1610 | 0.5590 |

**mma12** (the matrix 0.872 GiB, bf16 1.159 GiB)

| rows | packed ms | bf16 ms | packed / bf16 |
| ---: | ---: | ---: | ---: |
| 1 | 3.625 | 4.875 | **0.74** |
| 8 | 3.676 | 4.959 | **0.74** |
| 16 | 3.719 | 4.995 | **0.74** |
| 32 | 3.748 | 5.440 | **0.69** |
| 48 | 4.181 | 5.754 | **0.73** |
| 64 | 4.348 | 5.859 | **0.74** |
| 96 | 4.734 | 5.971 | **0.79** |
| 128 | 5.061 | 6.111 | **0.83** |
| 256 | 8.962 | 9.105 | **0.98** |
| 384 | 12.851 | 12.228 | **1.05** |
| 512 | 14.837 | 14.900 | **1.00** |
| 768 | 21.213 | 19.517 | **1.09** |
| 1024 | 27.422 | 24.505 | **1.12** |
| 2048 | 51.152 | 46.876 | **1.09** |
| 4096 | 99.158 | 91.877 | **1.08** |

The embedding's lookup with `mma12` (ms a call): 0.0125 for 1 token (bf16 0.0090), 0.0122 for 8 (0.0095), 0.0337 for 64 (0.0140), 0.3698 for 256 (0.0136), 1.5234 for 1024 (0.0139), 6.1012 for 4096 (0.2454) and 12.2234 for 8192 (0.5663).

## The lossless KV cache with the tables packed (`kv` auto and lossless)

- `vllm serve --quantization glyd` at its defaults (`kv` auto, which is lossless on an L4 for these models), `--max-model-len 4096`, 85% of the GPU, the engine in a process of its own: Qwen3-4B (tied, one table) 118,944 KV cache tokens; Qwen3-8B 68,592; each answered a completion over HTTP.
- `kv` lossless against `kv` off on the same packs (eager, 4,096 tokens, 85% of the GPU, four prompts of 24 tokens): Qwen3-4B 119,200 KV cache tokens against 91,712 (+30%), Qwen3-8B 67,696 against 51,648 (+31%); the prompt logprobs bit for bit `kv` off's (both models); the generated tokens the same for 3 of 4 prompts on Qwen3-4B and on Qwen3-8B, the other diverging at its 17th and 10th token (a decode step's attention is not bit-equal to vLLM's, as in every record of this cache). v0.27.0's plugin on Qwen3-8B at the same flags: KV cache 59,344 tokens against 45,440 with `kv` off.
- `check_vllm.py --kv --quick` (the kernels' checks through the library): all passed (67 checks) (`check_vllm/cv-kvq-report.txt`).

## Checks

- **Library:** every row of the real tables bit for bit, in order and shuffled, in both layouts: Qwen3-8B's and Llama-3.1-8B's embeddings and heads and Qwen3-4B's embedding, 0 failures (`log/rows_every_row.txt`).
- **`check_vllm.py --quick`** (`check_vllm/`): Qwen3-8B (untied) and Llama-3.1-8B (untied): all 18 passed each: the embedding and the LM head packed in both layouts (Qwen3-8B: 2 packs, 1,605 MiB `mma` and 1,786 MiB `mma12`; Llama: 1,350 and 1,512), every lookup the checkpoint's row bit for bit, each product within 1e-2 (3.5e-3), every pack (144; Llama 128) decoded to its weights bit for bit, each layer's product within 1e-2 of `F.linear` at 1-4,096 tokens (4.7e-3 `mma`, 3.7e-3 `mma12`), top-1 agreement with bf16 on its continuation 0.9935 and 0.9922 (Qwen3-8B; bf16 eager's own 0.9909) and 0.9942 and 0.9955 (Llama; 0.9948), the mean |logprob difference| within twice bf16's noise floor, `exact` (eager) bf16 eager's tokens, logprobs and prompt logprobs bit for bit (the untied models' embedding packed, the head as vLLM runs it), the compile caches. Qwen3-4B (tied): all 18 passed (`check_vllm/cv-q4-report.txt`: tied, one pack of 558 MiB `mma`, `exact`: the one table as vLLM runs it).
- **Qwen3-1.7B and Qwen3-0.6B** (`check_vllm/cv-q17-report.txt`, `cv-q06-report.txt`): all 18 passed each; one pack for the two tables (398 MiB and 199 MiB `mma`), every lookup the checkpoint's row bit for bit, `exact`: both as vLLM runs them.
- **A glyd save** of a tied and an untied tiny model, in both layouts, loaded by vLLM in both layouts (eager): the same bits as the bf16 checkpoint packed at load, 8 of 8.
- **compute-sanitizer** (memcheck, initcheck, racecheck, synccheck) over the library's rows lookup: 0 errors, 0 hazards.

The weights `check_vllm` prints for its Glyd runs are of a run with `verify` on, so the weights above are of the plain runs.

## Files

- `log/`: `weights_kv_lines.txt` (every start's weights and KV lines; `base` is v0.27.0's plugin, `src` and `sc` this code, `q8`, `q4` and `llama` the models; a start whose line says "compilation: 86 s" is a first start on an empty compile cache, "2 s" a later one), `rows_every_row.txt`.
- `check_vllm/`: each model's report (`*-report.txt`).
- `one_user/`: `tps_ab.txt` and each round's JSON. `saturated_serve/`: each pass's summary, `vllm bench serve` output and result JSON, the server's weights and KV lines, and GPU clock and temperature each second.
