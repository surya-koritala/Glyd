# vLLM on an L4: v0.28.0 against v0.27.0, three models (2026-10-04)

`vllm serve --quantization glyd` on one NVIDIA L4: glyd-gpu 0.27.0 from PyPI (v0.27.0) against the glyd-gpu wheel built from the release's tree (v0.28.0), the same vLLM 0.30.0 for both, in one session, alternated. The embedding and the output layer (the LM head) are packed in v0.28.0 and are vLLM's bf16 tensors in v0.27.0; the smallest layout (`mma`, the L4's default) is faster at 1 to 16 tokens a step in v0.28.0. Nothing was set by hand: every option at its default (`kv` auto, which is lossless on an L4).
This is the release's code; [vllm-full33-l4-2026-10-03](../vllm-full33-l4-2026-10-03) is the whole-model packing as of an earlier day, before the smallest layout's later changes.

## Setup

- **Machine:** AWS g6.4xlarge, NVIDIA L4 (23,034 MiB, sm_89, 72 W cap, 2,040 MHz at most), driver 595.91.07, AMD EPYC 7R13 (16 CPUs), Linux 7.0 (`log/machine.txt`).
- **Software:** vLLM 0.30.0 from PyPI, torch 2.13.0+cu130, transformers 5.18.0; one environment for both sides. Each side is a directory put first on `PYTHONPATH`: v0.27.0 is `glyd_gpu-0.27.0-cp310-abi3-manylinux_2_28_x86_64.whl` from PyPI (sha256 1158aad78cfffda69f51bea985042cad7a31527271c3ea6af4e732e54485ad7d), v0.28.0 the wheel built from the release's tree (sha256 526779afde59fe38352907a3a441a412516ddf8ef1e686f216399b6e2fbf28ec; its version string is 0.27.0 until the release's version bump). `log/side-v027.txt` and `log/side-v028.txt` give each side's module path, version and plugin entry point.
- **Models:** Qwen/Qwen3-8B, NousResearch/Meta-Llama-3.1-8B and Qwen/Qwen3-4B (tied embedding), bf16 checkpoints from the Hugging Face Hub.
- **Weights and KV cache:** `vllm serve` as a user runs it (the engine in a process of its own, compiled, CUDA graphs), `--max-model-len 4096`, `--gpu-memory-utilization 0.9`, `kv` off (vLLM's own KV cache); each side's first start on an empty compile cache, then later starts on it.
- **One user:** `tps_ab.py`: a vLLM of its own for each of 3 rounds of each side, the order alternating, one request of a random 1,024-token prompt decoded for 256 tokens (the time of the run less the time of a run of one token, over the new tokens; the median of 3 requests), vLLM's default mode, `kv` off, 0.9 of the GPU, a context of 1,344 tokens.
- **Serving:** `vllm bench serve`, the random dataset, 1,024 tokens in and 256 out (`--ignore-eos`): 64 prompts at 1 request a second, then 256 at once; `kv` auto as users get it (lossless on an L4), `--max-model-len 4096`, 0.9 of the GPU; a seed of its own for each pass (round 1: 10 and 11, round 2: 20 and 21), the same prompts for both sides in a round; the prefix cache is vLLM's default and each server's highest hit rate, from its log, is printed in each summary. Round 1 starts each side's server twice (the first on an empty compile cache and with the KV cache's tables made at that first start, the second measured); round 2 starts each side's server once on round 1's caches. The order of the two sides alternates between rounds, and the GPU is let down to 55 C (at most 300 s) before each measured start.
- Every run was one at a time on the machine.

## Weights and KV cache (`weights_kv/`, kv off)

Qwen3-8B: first start / later start, KV cache tokens: bf16 19,760 / 27,024; v0.27.0 38,912 / 54,464; v0.28.0 60,736 / 60,736.
Llama-3.1-8B: first start / later start, KV cache tokens: bf16 26,112 / 32,592; v0.27.0 50,032 / 64,368; v0.28.0 70,112 / 70,112.
Qwen3-4B: first start / later start, KV cache tokens: bf16 73,968 / 83,408; v0.27.0 88,048 / 97,488; v0.28.0 99,120 / 99,312.

| | bf16 | v0.27.0 | v0.28.0 |
| :--- | ---: | ---: | ---: |
| Qwen3-8B, weights GiB | 15.27 | 11.38 (−25.5%) | **10.38 (−32.0%)** |
| Qwen3-8B, KV cache tokens, later start (first start) | 27,024 (19,760) | 54,464 (38,912) | **60,736 (60,736)** |
| Llama-3.1-8B, weights GiB | 15.0 | 11.05 (−26.3%) | **10.16 (−32.3%)** |
| Llama-3.1-8B, KV cache tokens, later start (first start) | 32,592 (26,112) | 64,368 (50,032) | **70,112 (70,112)** |
| Qwen3-4B, weights GiB | 7.56 | 5.48 (−27.5%) | **5.16 (−31.7%)** |
| Qwen3-4B, KV cache tokens, later start (first start) | 83,408 (73,968) | 97,488 (88,048) | **99,312 (99,120)** |

## One user's tokens a second (`one_user/`)

| | v0.27.0 | v0.28.0 | v0.28.0 against v0.27.0 |
| :--- | ---: | ---: | ---: |
| Qwen3-8B, tokens a second (rounds) | 21.16 (21.31, 21.15, 21.16) | **22.54** (22.57, 22.50, 22.54) | **1.065x** (each round 1.059, 1.064, 1.065) |
| Llama-3.1-8B, tokens a second (rounds) | 21.92 (22.00, 21.92, 21.77) | **23.01** (23.10, 23.01, 22.94) | **1.050x** (each round 1.050, 1.050, 1.054) |
| Qwen3-4B, tokens a second (rounds) | 37.54 (37.75, 37.54, 37.42) | **39.62** (39.74, 39.62, 39.44) | **1.055x** (each round 1.053, 1.055, 1.054) |

**The smallest layout's faster steps alone** (`one_user/Qwen3-8B/layout_alone/`: the same protocol, v0.27.0 against a build of the release's tree with the embedding and the LM head left as vLLM runs them): 21.19 against 22.07 tokens a second, **1.042x** (each round 1.041, 1.043, 1.036); with the embedding and the LM head packed too, the table above.

## Serving (`serve/`, `vllm bench serve`, kv auto)

Prefix cache hit rate in the servers' logs, the highest of any run: 0.4%.

**Saturated** (256 requests at once)

| All 256 requests at once | round | Requests/s | Output tokens/s | First token mean (p99) ms | Each token mean (p99) ms | KV cache tokens (first start) | SM clock, hottest |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, v0.27.0 | 1 | 1.21 | 309.9 | 87470 (196486) | 171.6 (268.2) | 71,248 (50,608) | 1095 MHz, 83 C |
| Qwen3-8B, v0.28.0 | 1 | 1.33 | 339.5 | 81904 (171966) | 176.9 (273.3) | 79,600 (79,600) | 1065 MHz, 83 C |
| Qwen3-8B, v0.27.0 | 2 | 1.20 | 307.4 | 88085 (198236) | 173.0 (269.9) | 71,248 | 1155 MHz, 83 C |
| Qwen3-8B, v0.28.0 | 2 | 1.31 | 336.5 | 83147 (173696) | 178.4 (276.6) | 79,600 | 1020 MHz, 83 C |
| Llama-3.1-8B, v0.27.0 | 1 | 1.32 | 338.0 | 81325 (173088) | 181.0 (277.7) | 81,792 (62,784) | 1065 MHz, 83 C |
| Llama-3.1-8B, v0.28.0 | 1 | 1.37 | 351.4 | 75873 (170643) | 187.6 (285.3) | 90,384 (90,384) | 1005 MHz, 83 C |
| Llama-3.1-8B, v0.27.0 | 2 | 1.31 | 335.5 | 81793 (174387) | 181.8 (278.4) | 81,792 | 1035 MHz, 83 C |
| Llama-3.1-8B, v0.28.0 | 2 | 1.36 | 349.3 | 76185 (171838) | 188.3 (290.5) | 90,384 | 1020 MHz, 83 C |
| Qwen3-4B, v0.27.0 | 1 | 2.26 | 577.7 | 42008 (99117) | 153.9 (252.5) | 127,152 (114,656) | 1170 MHz, 79 C |
| Qwen3-4B, v0.28.0 | 1 | 2.33 | 595.8 | 40851 (96913) | 151.2 (246.8) | 129,344 (129,072) | 1155 MHz, 80 C |
| Qwen3-4B, v0.27.0 | 2 | 2.25 | 576.0 | 42143 (99394) | 154.3 (253.4) | 127,152 | 1140 MHz, 81 C |
| Qwen3-4B, v0.28.0 | 2 | 2.33 | 597.3 | 40708 (96605) | 150.7 (245.9) | 129,344 | 1155 MHz, 80 C |

**1 request a second** (64 requests)

| 1 request a second, 64 requests | round | Requests/s | Output tokens/s | First token mean (p99) ms | Each token mean (p99) ms | KV cache tokens (first start) | SM clock, hottest |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, v0.27.0 | 1 | 0.82 | 210.1 | 659 (1142) | 88.4 (106.2) | 71,248 (50,608) | 1200 MHz, 73 C |
| Qwen3-8B, v0.28.0 | 1 | 0.83 | 212.2 | 633 (1112) | 83.8 (98.6) | 79,600 (79,600) | 1215 MHz, 75 C |
| Qwen3-8B, v0.27.0 | 2 | 0.81 | 206.7 | 608 (1007) | 91.0 (112.4) | 71,248 | 1200 MHz, 76 C |
| Qwen3-8B, v0.28.0 | 2 | 0.82 | 209.4 | 593 (976) | 85.3 (104.7) | 79,600 | 1215 MHz, 75 C |
| Llama-3.1-8B, v0.27.0 | 1 | 0.83 | 212.0 | 639 (1124) | 83.6 (98.6) | 81,792 (62,784) | 1170 MHz, 73 C |
| Llama-3.1-8B, v0.28.0 | 1 | 0.83 | 213.7 | 638 (1122) | 79.8 (93.7) | 90,384 (90,384) | 1155 MHz, 75 C |
| Llama-3.1-8B, v0.27.0 | 2 | 0.82 | 209.3 | 601 (990) | 86.0 (107.1) | 81,792 | 1140 MHz, 74 C |
| Llama-3.1-8B, v0.28.0 | 2 | 0.83 | 211.3 | 588 (1004) | 82.1 (101.3) | 90,384 | 1170 MHz, 76 C |
| Qwen3-4B, v0.27.0 | 1 | 0.89 | 228.8 | 300 (462) | 38.8 (43.3) | 127,152 (114,656) | 1395 MHz, 72 C |
| Qwen3-4B, v0.28.0 | 1 | 0.90 | 229.9 | 298 (462) | 38.1 (42.9) | 129,344 (129,072) | 1485 MHz, 73 C |
| Qwen3-4B, v0.27.0 | 2 | 0.89 | 227.7 | 295 (491) | 39.6 (47.1) | 127,152 | 1365 MHz, 73 C |
| Qwen3-4B, v0.28.0 | 2 | 0.89 | 228.6 | 291 (484) | 38.5 (46.2) | 129,344 | 1470 MHz, 73 C |

**v0.28.0 against v0.27.0**

| Against v0.27.0, v0.28.0's | round | Requests/s | First token mean | Each token mean | Output tokens/s | KV cache tokens |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, 1 request a second | 1 | **1.010x** (0.82 to 0.83) | −3.9% (659 to 633 ms) | −5.2% (88.4 to 83.8 ms) | 1.010x | 71,248 to 79,600 |
| Qwen3-8B, 1 request a second | 2 | **1.013x** (0.81 to 0.82) | −2.5% (608 to 593 ms) | −6.2% (91.0 to 85.3 ms) | 1.013x | 71,248 to 79,600 |
| Llama-3.1-8B, 1 request a second | 1 | **1.008x** (0.83 to 0.83) | −0.2% (639 to 638 ms) | −4.6% (83.6 to 79.8 ms) | 1.008x | 81,792 to 90,384 |
| Llama-3.1-8B, 1 request a second | 2 | **1.009x** (0.82 to 0.83) | −2.1% (601 to 588 ms) | −4.5% (86.0 to 82.1 ms) | 1.009x | 81,792 to 90,384 |
| Qwen3-4B, 1 request a second | 1 | **1.005x** (0.89 to 0.90) | −0.7% (300 to 298 ms) | −2.0% (38.8 to 38.1 ms) | 1.005x | 127,152 to 129,344 |
| Qwen3-4B, 1 request a second | 2 | **1.004x** (0.89 to 0.89) | −1.3% (295 to 291 ms) | −2.7% (39.6 to 38.5 ms) | 1.004x | 127,152 to 129,344 |
| Qwen3-8B, all 256 at once | 1 | **1.095x** (1.21 to 1.33) | −6.4% (87470 to 81904 ms) | +3.1% (171.6 to 176.9 ms) | 1.095x | 71,248 to 79,600 |
| Qwen3-8B, all 256 at once | 2 | **1.095x** (1.20 to 1.31) | −5.6% (88085 to 83147 ms) | +3.1% (173.0 to 178.4 ms) | 1.095x | 71,248 to 79,600 |
| Llama-3.1-8B, all 256 at once | 1 | **1.040x** (1.32 to 1.37) | −6.7% (81325 to 75873 ms) | +3.7% (181.0 to 187.6 ms) | 1.040x | 81,792 to 90,384 |
| Llama-3.1-8B, all 256 at once | 2 | **1.041x** (1.31 to 1.36) | −6.9% (81793 to 76185 ms) | +3.6% (181.8 to 188.3 ms) | 1.041x | 81,792 to 90,384 |
| Qwen3-4B, all 256 at once | 1 | **1.031x** (2.26 to 2.33) | −2.8% (42008 to 40851 ms) | −1.7% (153.9 to 151.2 ms) | 1.031x | 127,152 to 129,344 |
| Qwen3-4B, all 256 at once | 2 | **1.037x** (2.25 to 2.33) | −3.4% (42143 to 40708 ms) | −2.3% (154.3 to 150.7 ms) | 1.037x | 127,152 to 129,344 |

## Checks

- The plugin's tests with the GPU (`test_vllm.py`): 27 ok from the built wheel (exit 0) and 27 ok from the sources (exit 0).
- `check_vllm.py --quick` on Qwen3-8B, the built wheel: 18 passed, 0 failed (the embedding and the LM head packed in both layouts, every lookup the checkpoint's row bit for bit, every pack decoded to its weights bit for bit, each product within 1e-2 of `F.linear`, top-1 agreement with bf16 on its continuation, `exact` bf16 eager's tokens, logprobs and prompt logprobs bit for bit).
- `check_vllm.py --quick` on Llama-3.1-8B, the built wheel: 18 passed, 0 failed.
- `check_vllm.py --quick` on Qwen3-4B, the built wheel: 18 passed, 0 failed.
- `check_vllm.py --kv --quick` (the lossless KV cache's checks through the library) on Qwen3-8B, the built wheel: 67 passed, 0 failed.

## Files

- `log/`: `machine.txt`, `side-v027.txt`, `side-v028.txt`. `weights_kv/weights_kv_lines.txt`: vLLM's own weights, KV cache and compile lines of every start.
- `one_user/`: each model's `tps_ab.txt` and rounds' JSON; `layout_alone/` beside Qwen3-8B's. `serve/MODEL/SIDE-roundN/`: each pass's `vllm bench serve` output and result JSON, the server's weights and KV lines, nvidia-smi's GPU clock and temperature each second and the summary. `check_vllm/`: each model's report and the KV check's result (`kv_quick.txt`); `checks.txt`: the plugin's tests' results.
