#!/usr/bin/env bash
# gpu-format's confirming run on an H100 SXM (format-report-1.md section 4), after gpu-hopper2's job in the same session:
# the split-byte 12-bit layout against the 12-bit layout in the same kernels (the TMA kernel at 17-512 tokens, the
# step kernel to 64) and cuBLAS, per layer of Qwen3-8B, 14B and 32B at 1-512 tokens, and check.py on Qwen3-8B.
#   bash ~/h100_format.sh   (in ~: this script, format_job.sh, format_src.tar, format_summary.py; ~/gpuenv/cuda.sh)
# Results in ~/results/format alone (summary.txt after every step, END last); within 25 minutes, the models' fetch
# included. Refuses any GPU but compute capability 9.0 (GPU= for a smoke test elsewhere).
GPU=${GPU-9.0} MS=${MS:-1,8,16,32,64,128,256,512} exec bash "$(dirname "${BASH_SOURCE[0]}")/format_job.sh"
