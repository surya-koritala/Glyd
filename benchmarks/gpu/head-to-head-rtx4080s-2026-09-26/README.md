# Glyd against DFloat11 and ZipServ, RTX 4080 SUPER, 2026-09-26

Qwen3-8B (16.38 GB in bf16: more than the card holds).

- `dfloat11-qwen3-8b.txt`: `dfl_bench.py` on `DFloat11/Qwen3-8B-DF11` (dfloat11 with transformers 4.57, its own environment), e2e.py's measures.
- `glyd-qwen3-8b-mma.txt`, `glyd-qwen3-8b-mma12.txt`: `gpu/e2e.py MODEL --format mma|mma12 --fused --tokens 64 --batch 1,8,32,64 --profile 16 --ppl enwik8`.
- `glyd-kernels-qwen3-8b.txt`: `h2h_glyd.py MODEL OUT 18`: layer 18's matrices dumped as bf16 for ZipServ's test, and cuBLAS and Glyd's products timed as ZipServ times itself (the L2 cache flushed before every call, CUDA events, 20 calls).
- `zipserv-kernels-qwen3-8b.txt`: ZipServ's `kernel_benchmark/test_mm M K N SplitK` (HPMLL/ZipServ_ASPLOS26 at its main branch) on the same matrices (a six-line patch reading A from `ZS_WEIGHT`), the best time over SplitK 1, 2, 4, 8, and its compression ratio. Built with CUDA 13.0 (`-std=c++17`, `crt/math_functions.h`'s `rsqrt`/`rsqrtf` declared `__THROW` for glibc 2.43).
