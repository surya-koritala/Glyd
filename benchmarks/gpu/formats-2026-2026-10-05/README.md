# Lossless room in 2026 checkpoints as released (Runpod, 2026-10-05)

How much a lossless code could take from 13 2026 checkpoints in the formats they are published in (bf16, FP8, MXFP8, NVFP4, MXFP4
and their scales), sampled by HTTP range (no full download): the share of each format, the order-0 floors, what Glyd's two GPU
layouts take today (bf16 tensors only, packed with `glyd-gpu` 0.28.0 from PyPI and checked bit for bit on the GPU), and zstd -19 and
`glyd --max` on disk. `v2-rtx4090/summary.md` is the table; `v1-rtx4000ada/` is the first run, whose alphabetical sampling was
wrong (kept for the record, not for use).

| Model | GB as released | saved at the order-0 floor | Glyd today (smallest layout, bf16 only) | Glyd today (12-bit) |
| :--- | ---: | ---: | ---: | ---: |
| Qwen/Qwen3.8-27B | 56 | 33.9% | 32.7% | 24.8% |
| Qwen/Qwen3.8-27B-FP8 | 31 | 21.1% | 6.5% | 4.9% |
| google/gemma-4-26B-A4B-it | 52 | 33.9% | 32.7% | 24.7% |
| zai-org/GLM-5.3-Flash-BF16 | 643 | 34.1% | 33.1% | 24.8% |
| zai-org/GLM-5.3-Flash | 328 | 20.4% | 1.4% | 1.0% |
| MiniMaxAI/MiniMax-M3 | 854 | 34.0% | 33.0% | 24.8% |
| MiniMaxAI/MiniMax-M3-MXFP8 | 444 | 21.1% | 0.5% | 0.4% |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | 1121 | 33.2% | 30.2% | 21.7% |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | 352 | 13.3% | 3.1% | 2.3% |
| deepseek-ai/DeepSeek-V4.1-Flash | 510 | 13.0% | 0.3% | 0.2% |
| moonshotai/Kimi-K3 | 1561 | 12.9% | 2.4% | 1.8% |
| mistralai/Mistral-Small-4-119B-2603 | 121 | 18.5% | 0.8% | 0.6% |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | 566 | 11.2% | 1.2% | 0.9% |

```sh
# job/launch.sh: the first GPU with capacity of a list (the v2 run got an RTX 4090; 36 min)
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=90 NAME=fmt2026 MIN_DRIVER=550 RP_CUDA=12.8 RP_DISK=60 UPLOAD="fmt_measure.py" bash rp_run.sh fmt_job.sh
```

Two pods, about 58 minutes, about $0.55 together (launch ledger). Files: `v2-rtx4090/` (one JSON a repo, `summary.md`, `job.log`,
`gpu.txt`), `v1-rtx4000ada/`, `job/` (`fmt_job.sh`, `fmt_measure.py`, `probe_repos.py`, `names.py`, `launch.sh`, its output `launch.log`).
