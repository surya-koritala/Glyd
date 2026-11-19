#!/usr/bin/env bash
# glyd-gpu's wheel in glyd-gpu-py/dist: the compiled package (setup.py), the BUSL license, the public C header (include/), and the GPU libraries
# (libglyd_gpu_cuda12.so, libglyd_gpu_cuda13.so) where they are given or this machine can build them:
#   GLYD_GPU_LIBS=DIR   prebuilt libraries (libglyd_gpu_cuda*.so) to take: a CI job's, one a CUDA version
#   GLYD_GPU_ENGINE=PATH  the engine's program (glyd-engine, built and checked by release.yml's engine job: cargo build --release --no-default-features
#                       in glyd-engine, GLYD_ENGINE_RELEASE=1): it goes in the package beside the libraries, executable
#   NVCC12=PATH NVCC13=PATH   build the library for that CUDA's nvcc (gpu/build_lib.sh, SASS for sm_80 to sm_120, no PTX, stripped)
#   neither: the nvcc on PATH builds the one for its CUDA; no nvcc: none (a wheel for the compiled modules alone, as on a Mac)
#   GLYD_GPU_JOBS=N     compiler jobs (default: the cores)
#   GLYD_GPU_BUILD_ARGS arguments for python -m build (--no-isolation: the build tools of this Python, not a fresh environment's)
#   GLYD_GPU_ABI3=0     a wheel for the CPython that builds it alone (default: cp310-abi3, the stable ABI: one for every CPython from 3.10 on)
# Linux without a library in the wheel is an error, unless GLYD_GPU_NO_LIBS=1; with the libraries and no engine, unless GLYD_GPU_NO_ENGINE=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PKG="$HERE/glyd_gpu"
cp "$ROOT/gpu/LICENSE" "$HERE/LICENSE"
cp "$HERE/include/glyd_gpu.h" "$PKG/glyd_gpu.h"  # the public header: gpu/glyd_gpu.h stays in the repository
rm -f "$PKG"/libglyd_gpu_cuda*.so "$PKG/glyd-engine"
if [ -n "${GLYD_GPU_LIBS:-}" ]; then
    cp "$GLYD_GPU_LIBS"/libglyd_gpu_cuda*.so "$PKG/"
elif [ -n "${NVCC12:-}${NVCC13:-}" ]; then
    for nvcc in ${NVCC12:-} ${NVCC13:-}; do
        PATH="$(dirname "$nvcc"):$PATH" GLYD_GPU_RELEASE=1 bash "$ROOT/gpu/build_lib.sh" "$PKG"
    done
elif command -v nvcc > /dev/null; then
    GLYD_GPU_RELEASE=1 bash "$ROOT/gpu/build_lib.sh" "$PKG"
fi
if [ -n "${GLYD_GPU_ENGINE:-}" ]; then
    cp "$GLYD_GPU_ENGINE" "$PKG/glyd-engine"
    chmod 755 "$PKG/glyd-engine"
fi
if [ "$(uname -s)" = Linux ] && [ -z "${GLYD_GPU_NO_LIBS:-}" ] && ! ls "$PKG"/libglyd_gpu_cuda*.so > /dev/null 2>&1; then
    echo "glyd-gpu: no GPU library for the wheel (GLYD_GPU_LIBS, NVCC12/NVCC13 or nvcc on PATH; GLYD_GPU_NO_LIBS=1 to build without)" >&2
    exit 1
fi
if [ "$(uname -s)" = Linux ] && [ -z "${GLYD_GPU_NO_LIBS:-}" ] && [ -z "${GLYD_GPU_NO_ENGINE:-}" ] && [ ! -x "$PKG/glyd-engine" ]; then
    echo "glyd-gpu: no engine program for the wheel (GLYD_GPU_ENGINE; GLYD_GPU_NO_ENGINE=1 or GLYD_GPU_NO_LIBS=1 to build without)" >&2
    exit 1
fi
rm -rf "$HERE/build" "$HERE/dist"
cd "$HERE" && python -m build --wheel ${GLYD_GPU_BUILD_ARGS:-} --outdir "$HERE/dist" .
ls -l "$HERE/dist"
