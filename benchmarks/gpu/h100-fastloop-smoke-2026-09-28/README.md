# H100 PCIe smoke run of gpu-fastloop at 7bfac10 (2026-09-29 UTC)

One Lambda gpu_1x_h100_pcie (114 SMs, CUDA 13.0, torch 2.14.0+cu130, transformers 5.17.0), job `h100_fast.sh`
(the library built from 7bfac10's sources, sm_90a), 13 minutes:

| Step | Result |
| :--- | :--- |
| build (sm_90a) | exit 0 |
| self-test | exit 0: 18 lines within 1e-2, 9 of them mma_gemm_wg's at 1-2100 tokens |
| check_capi | exit 0: 6200 calls, all bit for bit |
| test_gpu | exit 0: 15 ok |
| check_api dense | Qwen3-0.6B passed every check (compiled `generate()` by default, its eager cases, threads, a returned cache continued, compress, `exact=True` bit-identical to bf16, save, verify, reload); the job's 420 s timeout ended it while Qwen3-1.7B loaded (exit 124), no check failed |
| generate() tokens/s | not run: past the job's time budget |

The files: `summary.txt`, `steps.txt`, `versions.txt` and each step's output as the job wrote it.
