#!/usr/bin/env bash
# MINUTES: 38 at most (an A100 SXM4 40 GB of Lambda's gpu-base-24-04): glyd-engine's check on it, see engine_gpu_job.sh for what runs and what it needs (GLYD_GPU_LIB).
export GPU=a100 MAX_MIN="${MAX_MIN:-38}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/engine_gpu_job.sh" "$@"
