#!/usr/bin/env bash
# glyd_gpu.cu as a library behind its C API for AMD GPUs through HIP, which the engine's AMD build (GLYD_HIP=1) loads (GLYD_GPU_LIB, or
# libglyd_gpu_hip.so beside the executable): code objects for RDNA 3 and 4 in wave32 (GLYD_HIP_ARCHS, default gfx1100 gfx1101 gfx1200 gfx1201),
# hip.h in place of CUDA's headers. On AMD the packed layouts' products run by the step product, 32 tokens at a time (mma_linear_run), and
# unpack as on CUDA; the kernels that need Hopper's TMA, named barriers, WMMA or e4m3 products trap there and their host paths answer not
# supported (the AMD notes in glyd_gpu.cu). A development build: no release mode yet.
#   bash gpu/build_lib_hip.sh [OUT_DIR]      (default: next to this script)
# HIPCC: hipcc (default), or a script that runs a clang with HIP as hipcc would.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE}"
ARCHS=()
for a in ${GLYD_HIP_ARCHS:-gfx1100 gfx1101 gfx1200 gfx1201}; do ARCHS+=(--offload-arch="$a"); done
mkdir -p "$OUT"
# (only the C API exported, as the CUDA library; HIP 7 marks its calls nodiscard, which the host code's attribute queries ignore by design)
"${HIPCC:-hipcc}" -x hip -O3 -std=c++20 -fPIC -fvisibility=hidden -shared -mno-wavefrontsize64 -Wno-pass-failed -Wno-unused-result -Wno-unused-value \
    "${ARCHS[@]}" -o "$OUT/libglyd_gpu_hip.so" "$HERE/glyd_gpu.cu" -lamdhip64
echo "$OUT/libglyd_gpu_hip.so"
