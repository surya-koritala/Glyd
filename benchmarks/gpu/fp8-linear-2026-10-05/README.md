# The fused FP8 decode-and-product (f8_linear) against vLLM's FP8 GEMMs (Runpod, 2026-10-05)

The GPU runs behind the handoff's "f8_linear is 1.0 to 2.3x vLLM's time on the 4090, 1.5 to 5.7x on the H100" (branch
`fp8-lossless`, Task 5 of the FP8 plan). Exact (W8A16), bit for bit on Ada and Hopper; **the speed gate failed**: ours at most vLLM
0.30's FP8 GEMM at every shape of the five 2026 FP8 models and M = 1, 8, 32, 64 held on 46 of 356 products on the RTX 4090 (close at
1-8 tokens, 1.35-2.34x at 32-64) and 15 of 356 on the H100 (1.45-5.71x). Tables: `tables_4090.md`, `tables_h100.md`, `tables_all.md`;
report: `report.md` (the worker's).

| # | run | GPU | min | what it answered |
| :--- | :--- | :--- | ---: | :--- |
| 1 | `f8f-h1` | H100 80GB HBM3 | 5 | vLLM's times |
| 2 | `f8f-h2` | H100 | 5 | first fused kernel, checks; bench stopped at FlashInfer's link |
| 3 | `f8f-h3` | H100 | 9 | the kept kernel: checks, full bench with numerics (the H100 table) |
| 4 | `f8f-h4` | H100 | 9 | pipelined variant: slower |
| 5 | `f8f-h5` | H100 | 10 | staged decode + W8A8: slower |
| 6 | `f8f-a1` | L40S | 7 | Ada launch bug found; partial |
| 7 | `f8f-a2` | RTX 4090 | 21 | every check bit for bit, the full table (the 4090 table), the plan variant |

```sh
(cd f8f && ./mktar.sh)     # f8f_src.tar, COMMIT
GPU="NVIDIA H100 80GB HBM3" MAX_MIN=45 NAME=f8f-h3 UPLOAD="f8f_job.sh f8f_src.tar" bash rp_run.sh job_h3.sh
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=45 NAME=f8f-a2 UPLOAD="f8f_job.sh f8f_src.tar ncu_f8.py" bash rp_run.sh job_a2.sh
```

About 66 pod minutes. Files: `f8f-*/` (each run's results), `job/` (`f8f_job.sh`, the `job_*.sh` settings, the `run_*.sh` and
`rp_run_a*.sh` launch lines as run, `ncu_f8.py`, `mktar.sh`, `COMMIT`, the launcher's output `rp_run-*.log`).
