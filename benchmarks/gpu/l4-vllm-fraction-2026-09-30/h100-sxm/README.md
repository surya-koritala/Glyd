# `fraction` on an H100 SXM, Qwen3-32B served saturated (2026-10-01)

The plugin's `fraction` sweep (`budget_job.sh`) on one NVIDIA H100 SXM from the v0.26.0 candidate: Qwen3-32B served by
vLLM's bf16 and by `--quantization glyd` at fractions 0, 0.25, 0.5, 0.75 and 1, every request sent at once. This is the
first Hopper run of the option; the GH200, where fraction 1 served 0.88x bf16's requests a second (M3), has not run it.
The job was the last of the session's seven (1,373 s; the others: `../../vllm-m6-h100-2026-10-01/session/`).

## Setup

- **Machine:** NVIDIA H100 80GB HBM3, the SXM5 (81,559 MiB of HBM3, power limit 700 W, 1,980 MHz at most), driver
  580.126.20. Xeon Platinum 8480+ (26 CPUs), x86_64. Lambda `gpu_1x_h100_sxm5`.
- **Software:** the tree 7fe66a2 (release-0.26.0, the v0.26.0 candidate, with `fraction`; C API 7, GPU code 90); vLLM
  0.30.0 from PyPI (torch 2.13.0+cu130, transformers 5.18.0, nvcc 13.0, in the job); the library built there for sm_90a
  in 45 s. The plugin does not ask for the opt-in long-prompt mode (`GLYD_GPU_WITH_SPLIT`), so the library behaves as v0.25.1's.
- **Model:** Qwen3-32B (64 layers, 65.5 GB), downloaded in 35 s. Glyd's layout: the 12-bit one.
- **Run:** one mode at a time, in the order bf16, glyd@1, glyd@0.5, glyd@0.25, glyd@0.75, glyd@0 (every one ran, 175-235 s
  each). Each: a `vllm serve` started cold on an empty compile cache
  with `--max-model-len 4096 --max-num-seqs 128 --compilation-config '{"max_cudagraph_capture_size":128}'` and
  `--gpu-memory-utilization 0.9193`, then `vllm bench serve` with the random dataset, 192 prompts of 1,024 tokens in and
  256 out (`--ignore-eos`), all sent at once. 0.9193 is 0.9 plus a cold start's 1.54 GiB of compile (measured on the
  GH200 in M3, Qwen3-32B) as a share of this GPU's memory, to give a cold server about the KV cache a warm one gets at
  0.9 (not checked on this GPU); graphs are captured to 128 requests, the most the bench has in flight. One run each.
- **Not M3's bench:** M3's ran warm servers at 0.9 with 32, 64 and 128 prompts; this sweep's modes compare with one
  another and with bf16 in the same job, not digit for digit with M3's.

## Qwen3-32B, saturated, against bf16 (`bench-Qwen3-32B/`, `summary.txt`)

| Mode | Layers packed | Weights | KV cache | Requests/s | TTFT mean | TPOT mean | A request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | | 61.03 GiB | 32,320 tokens | 2.51 | 32,958 ms | 36.6 ms | 42,293 ms |
| fraction 0 | 0 of 64 | 61.03 GiB (1.00x) | 32,320 tokens (1.00x) | 2.51 (1.00x) | 32,982 ms (1.00x) | 36.6 ms (1.00x) | 42,321 ms (1.00x) |
| fraction 0.25 | 16 of 64 | 58.20 GiB (0.95x) | 43,872 tokens (1.36x) | 2.85 (1.14x) | 27,116 ms (0.82x) | 41.7 ms (1.14x) | 37,750 ms (0.89x) |
| fraction 0.5 | 32 of 64 | 54.89 GiB (0.90x) | 57,392 tokens (1.78x) | 3.54 (1.41x) | 23,198 ms (0.70x) | 46.2 ms (1.26x) | 34,991 ms (0.83x) |
| fraction 0.75 | 48 of 64 | 51.59 GiB (0.85x) | 70,912 tokens (2.19x) | 3.50 (1.40x) | 20,883 ms (0.63x) | 52.8 ms (1.44x) | 34,356 ms (0.81x) |
| fraction 1 | 64 of 64 | 48.27 GiB (0.79x) | 84,528 tokens (2.62x) | 3.91 (1.56x) | 20,014 ms (0.61x) | 56.8 ms (1.55x) | 34,501 ms (0.82x) |

On an 80 GB H100, Qwen3-32B's bf16 weights leave room for only 32,320 tokens of KV cache, and every packed fraction
served more requests a second than bf16: 1.14x at 0.25, 1.41x at 0.5, 1.40x at 0.75 and 1.56x with every layer packed.

- **bf16 was short of KV cache.** At once (the servers' logs, every 10 s) it ran at most 25 requests at a time, and up
  to 175 waited; fraction 1 ran at most 81 at a time, and up to 111 waited (fractions 0 to 1, the most running: 25, 29, 40,
  53, 68 and 81; the most waiting: 175, 165, 158, 144, 124 and 111).
- **Fraction 0 is bf16:** the same weights and KV cache, 2.506 against 2.507 requests a second, each token 36.6 ms in
  both.
- **The more layers packed, the more requests a second**, but for fractions 0.5 and 0.75, which are within 1.3% of each
  other (3.544 and 3.498; one run each). The first token comes sooner at every packed fraction, 33.0 s on average for bf16 and
  20.0 s at fraction 1 (p99 65.5 and 41.7 s); a request took 42.3 s from start to end in bf16 and 34.4-35.0 s at 0.5-1.
- **Each token takes longer the more is packed**, 36.6 ms in bf16 and 56.8 ms at fraction 1 (p99 65.5 and 102.8 ms),
  with more requests in a step (at most 81 against 25) and each packed layer's weights rebuilt at every step (what that
  costs is not profiled here). The throughput is the larger: 3.91 against 2.51 requests a second, 1,001 against 642
  output tokens a second.
- **The GPU's power and clock** (nvidia-smi's samples above the midpoint of the least and the most power drawn while it
  worked, `bench_summary.py`'s way): a median of 576, 575, 619, 665, 685 and 698 W of the 700 W limit for bf16 and
  fractions 0 to 1, at 1,980, 1,980, 1,980, 1,965, 1,950 and 1,770 MHz: at fraction 1 the GPU ran at its power limit's edge.

## What it shows and does not

- **One GPU, one model, one load:** an H100 SXM with 80 GB, Qwen3-32B, every request at once (192 of them). Low load
  (the first token at 1 request a second) was not measured here. The load where bf16 is short of KV cache is where the
  packs' memory pays: on the GH200, with 96 GB, bf16 held 79,760 tokens of KV cache for the same model and Glyd served 0.88x
  its requests a second (`../../vllm-m3-gh200-2026-09-30`, a different bench: warm servers, 128 prompts at once).
- **Not measured:** other fractions, a fraction on a GH200, an H100 PCIe, an H200, another model, `exact` with a fraction,
  and the layers' speed apart from the memory (each packed layer's extra time per step is not profiled).

## Files

`bench-Qwen3-32B.txt` (the console), `bench-Qwen3-32B/` (per mode: the server's log `serve-MODE.txt`, `kv-MODE.txt` (its
weights and KV cache and the plugin's line), `MODE-rateinf.json`, `bench-MODE-rateinf.txt`, `smi-MODE-rateinf.csv`;
then `summary.txt`), `summary.txt`, `steps.txt`, `job.log`, `machine.txt`, `machine-short.txt`, `env.txt`, `log/`,
`budget_job.sh` (the job as it ran).
