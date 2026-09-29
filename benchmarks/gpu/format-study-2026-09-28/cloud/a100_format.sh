#!/usr/bin/env bash
# gpu-format's confirming run on an A100 SXM4 40 GB (format-report-1.md section 4): the split-byte 12-bit layout
# against the 12-bit layout in the same kernels (the step kernel to 64 tokens, the prompt kernel past: variant 3 from
# 129) and cuBLAS, per layer of Qwen3-8B, 14B and 32B at 1-768 tokens, and check.py on Qwen3-8B. The A100's own 17-128
# kernel (mma12_ws_kernel) is not in the prototype: read 1-16 and 129-768 as main's routes.
#   bash ~/a100_format.sh   (in ~: this script, format_job.sh, format_src.tar, format_summary.py; ~/gpuenv/cuda.sh)
# Results in ~/results/format alone (summary.txt after every step, END last); within 25 minutes, the models' fetch
# included. Refuses any GPU but compute capability 8.0 (GPU= for a smoke test elsewhere).
GPU=${GPU-8.0} MS=${MS:-1,8,16,32,64,128,256,512,768} exec bash "$(dirname "${BASH_SOURCE[0]}")/format_job.sh"
