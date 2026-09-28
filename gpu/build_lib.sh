#!/usr/bin/env bash
# glyd_gpu.cu's kernels as a library behind its C API (no PyTorch in it), for
# glyd_gpu.py to load through glyd/gpu/_lib.py where nvcc is not at hand. The
# CUDA runtime is linked in, so it needs only the driver; code for Ampere
# (sm_80, sm_86), Ada (sm_89), Hopper (sm_90a) and Blackwell (sm_100, sm_120,
# where nvcc has them: CUDA 12.8 on), and compute_80 PTX for the GPUs after them
# (all but the TMA kernel, which is Hopper's: sm_90a alone). The name carries
# the CUDA major version: libglyd_gpu_cuda13.so.
#   bash gpu/build_lib.sh [OUT_DIR]      (default: next to glyd_gpu.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE}"
NVCC="$(command -v nvcc)"
CUDA="$(dirname "$NVCC")/.."
MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
mkdir -p "$OUT"
ARCHS="$("$NVCC" --list-gpu-arch)"
BLACKWELL=""
for a in 100 120; do
    if grep -qx "compute_$a" <<< "$ARCHS"; then BLACKWELL="$BLACKWELL -gencode arch=compute_$a,code=sm_$a"; fi
done
# The kernels' flags as PyTorch's extension build has them (the same code; CUDA's headers as system
# headers, as there: glibc 2.43 declares rsqrt too); only the C API exported, the runtime's symbols kept
# inside; lib/ for the pip packages' libcudart_static.a (a toolkit's is in lib64/).
"$NVCC" -O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CUDA/include" \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
    -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90a,code=sm_90a $BLACKWELL \
    -gencode arch=compute_80,code=compute_80 \
    --threads 0 -shared -Xcompiler -fPIC,-fvisibility=hidden -Xlinker --exclude-libs,ALL \
    -cudart static -L"$CUDA/lib" \
    -o "$OUT/libglyd_gpu_cuda$MAJOR.so" "$HERE/glyd_gpu.cu"
echo "$OUT/libglyd_gpu_cuda$MAJOR.so"
