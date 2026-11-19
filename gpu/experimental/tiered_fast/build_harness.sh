#!/usr/bin/env bash
# build_harness.sh [OUT=./harness] [ARCH=89]: the phase-1 harness (harness.cu, v2.cuh) against ../../glyd_gpu.cu.
#   ARCH 80 (A100), 89 (L4, RTX 40), 90a (H100): -gencode arch=compute_$ARCH,code=sm_$ARCH. Static cudart: the binary needs only the driver.
#   Needs nvcc (CUDA 12.8 or newer for 90a; the box builds with 13.x). Writes OUT.ptxas (registers and spills of every kernel).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=$(cd "$HERE/../.." && pwd)
OUT=${1:-./harness}
A=${2:-89}
CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
FL="-O3 -lineinfo -std=c++20 --expt-relaxed-constexpr -isystem $CUDA_HOME/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_$A,code=sm_$A"
nvcc $FL -I"$GPU" -I"$HERE" -Xptxas -v ${EXTRA:-} -c "$HERE/harness.cu" -o "$OUT.o" 2> "$OUT.ptxas" || { grep -E "error" "$OUT.ptxas" | head -20; exit 1; }
nvcc $FL -cudart static -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib" "$OUT.o" -o "$OUT"
echo "built $OUT (sm_$A)"
