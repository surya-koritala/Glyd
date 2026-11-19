#!/usr/bin/env bash
# glyd_gpu.cu's kernels as a library behind its C API (no PyTorch in it), for
# glyd_gpu_dev.py to load through glyd_gpu/_lib.py where nvcc is not at hand. The
# CUDA runtime is linked in, so it needs only the driver; code for Ampere
# (sm_80, sm_86), Ada (sm_89), Hopper (sm_90a) and Blackwell (sm_100, sm_120,
# where nvcc has them: CUDA 12.8 on), and compute_80 PTX for the GPUs after them
# (all but the TMA kernel, which is Hopper's: sm_90a alone). The name carries
# the CUDA major version: libglyd_gpu_cuda13.so.
#   bash gpu/build_lib.sh [OUT_DIR]      (default: next to glyd_gpu_dev.py)
# GLYD_GPU_NVCC_THREADS=N: nvcc's threads for the compile (default 0: one an architecture; a CI runner of 2 cores and 7 GB takes 2).
# GLYD_GPU_PYTHON=PATH: the Python that runs gen_names.py in a release build (default python3).
# GLYD_GPU_RELEASE=1 (the wheel's build, glyd-gpu-py/build.sh): the machine code alone for those architectures (no PTX), no line
# or debug information, the library stripped of its symbols and exporting the public header's eight functions and glyd_gpu_table
# (the package's way to the rest) alone, every kernel, device function and type of it under a neutral name (gen_names.py: the
# header it makes holds them, and GLYD_GPU_NAMES_MAP names a private file for the map from the new names to the old), the
# tuning variables not read (GLYD_RELEASE in glyd_gpu.cu), and an nvcc without Blackwell's architectures refused.
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
    if grep -qx "compute_$a" <<< "$ARCHS"; then BLACKWELL="$BLACKWELL -gencode arch=compute_$a,code=sm_$a"
    elif [ "${GLYD_GPU_RELEASE:-0}" = 1 ]; then echo "glyd_gpu: this nvcc has no compute_$a (CUDA 12.8 on): a release library needs it" >&2; exit 1; fi
done
PTX=(-gencode arch=compute_80,code=compute_80)
STRIP=()
RELEASE=()
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if [ "${GLYD_GPU_RELEASE:-0}" = 1 ]; then
    PTX=()
    STRIP=(-Xlinker --strip-all)
    "${GLYD_GPU_PYTHON:-python3}" "$HERE/gen_names.py" "$HERE/glyd_gpu.cu" "$TMP/glyd_gpu_names.h" ${GLYD_GPU_NAMES_MAP:+"$GLYD_GPU_NAMES_MAP"}
    RELEASE=(-DGLYD_RELEASE -I"$TMP")  # (no -g, -G or -lineinfo anywhere: the machine code carries no line, and the host code no debug, information)
fi
# The kernels' flags as PyTorch's extension build has them (the same code; CUDA's headers as system
# headers, as there: glibc 2.43 declares rsqrt too); only the C API exported, the runtime's symbols kept
# inside; lib/ for the pip packages' libcudart_static.a (a toolkit's is in lib64/).
FLAGS=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CUDA/include"
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
    -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
    -gencode arch=compute_90a,code=sm_90a $BLACKWELL
    ${PTX[@]+"${PTX[@]}"}
    ${RELEASE[@]+"${RELEASE[@]}"}
    -Xcompiler -fPIC,-fvisibility=hidden)
# Compiled with a thread an architecture, then linked apart: in one command with --threads, nvcc's device link
# runs an nvlink an architecture in parallel, each writing the one _dlink.reg.c (nvcc --dryrun), and CUDA 12.8's
# release build failed on it ("nvlink fatal: Could not read file ..._dlink.reg.c (target: sm_100)", x86_64,
# 2026-09-28).
if [ "${GLYD_GPU_RELEASE:-0}" = 1 ]; then  # (the C++ library's inline functions would be exported too, whatever -fvisibility says)
    printf '{ global: glyd_gpu_api_version; glyd_gpu_error_string; glyd_gpu_mma_linear_workspace; glyd_gpu_mma_linear; glyd_gpu_mma12_linear_workspace; glyd_gpu_mma12_linear; glyd_gpu_mma_unpack; glyd_gpu_mma12_unpack; glyd_gpu_table; local: *; };\n' > "$TMP/exports.map"
    STRIP+=(-Xlinker --version-script="$TMP/exports.map")
fi
"$NVCC" "${FLAGS[@]}" --threads "${GLYD_GPU_NVCC_THREADS:-0}" -c -o "$TMP/glyd_gpu.o" "$HERE/glyd_gpu.cu"
"$NVCC" "${FLAGS[@]}" -shared -Xlinker --exclude-libs,ALL ${STRIP[@]+"${STRIP[@]}"} -cudart static -L"$CUDA/lib" \
    -o "$OUT/libglyd_gpu_cuda$MAJOR.so" "$TMP/glyd_gpu.o"
echo "$OUT/libglyd_gpu_cuda$MAJOR.so"
