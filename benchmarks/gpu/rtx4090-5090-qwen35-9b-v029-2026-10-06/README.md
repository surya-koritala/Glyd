# Qwen3.5-9B on an RTX 4090 and an RTX 5090: bf16 against Glyd 0.29 under vLLM 0.30.0 (2026-10-06)

Qwen/Qwen3.5-9B (bf16) served by vLLM 0.30.0 on one GPU, one pod per GPU (Runpod), the same job on both: vLLM's own bf16
(`vllm serve Qwen/Qwen3.5-9B --max-model-len 4096`, twice), and Glyd 0.29 (the wheels CI built for v0.29.0, before the
release) as `glyd run Qwen/Qwen3.5-9B`, `glyd serve Qwen/Qwen3.5-9B` and `glyd serve Qwen/Qwen3.5-9B -- --no-enforce-eager`
(compiled). Glyd's settings are the ones `glyd run` and `glyd serve` work out from the GPU.

- **Machines:** [machine-rtx4090.txt](machine-rtx4090.txt) (NVIDIA GeForce RTX 4090, 24,564 MiB, driver 595.91.07) and
  [machine-rtx5090.txt](machine-rtx5090.txt) (NVIDIA GeForce RTX 5090, 32,607 MiB, driver 595.91.07); Ubuntu 24.04.3, x86_64.
- **Logs:** [serve/](serve): each server's settings line and vLLM's own log lines for its arguments, the weights, the KV
  cache and any out-of-memory error, as logged.

## Weights, as vLLM logs them

| GPU | bf16 | Glyd, eager | Glyd, compiled |
| :--- | ---: | ---: | ---: |
| RTX 4090 | 17.66 GiB | **12.68 GiB** (−28.2%) | **12.69 GiB** (−28.1%) |
| RTX 5090 | 17.66 GiB | **13.85 GiB** (−21.6%) | **13.89 GiB** (−21.3%) |

## Context and KV cache

| GPU | bf16 (`--max-model-len 4096`) | Glyd (its settings) |
| :--- | :--- | :--- |
| RTX 4090 | out of memory at start, both times, with the context capped at 4,096 tokens ([serve/rtx4090-bf16.log](serve/rtx4090-bf16.log)) | compiled: a 189,440-token context, KV cache 197,225 tokens; eager: 211,968 tokens, KV cache 212,487 |
| RTX 5090 | KV cache 152,137 tokens | compiled: the model's whole 262,144 tokens, KV cache 389,307 tokens; eager: KV cache 404,942 |

The bf16 and Glyd servers ran at different context settings, so their KV caches are not compared here.
