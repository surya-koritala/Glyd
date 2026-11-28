#!/usr/bin/env bash
# The engine and the GPU library for the CI runner's GPU (gpu-tests.yml), reusing what an earlier run built where the sources are the same:
# cargo's target directory persists (CARGO_TARGET_DIR; cargo xtask ci-restore-mtimes keeps a fresh checkout from looking changed to it), and
# the GPU library, which is nvcc's alone and takes minutes, is kept by the hash of what makes it. Needs GLYD_ENGINE_ARCHS, NVCC and CARGO_TARGET_DIR.
# Appends GLYD_GPU_LIB_BUILT (the library's path) and E (the engine) to $GITHUB_ENV: the engine loads no library unless GLYD_GPU_LIB names one, which
# the steps that compare with the library or check it (foldcheck, packcheck --gpu, glyd-gpu verify) set from GLYD_GPU_LIB_BUILT.
# The two compiles of kernels (kernels/engine.cu in cargo's build, gpu/glyd_gpu.cu behind the library) are each one nvcc and do not wait for one another
# here: the library is built beside cargo's build where the runner has the processors (8 or more), and each compiles its architectures at once (nvcc
# --threads; an architecture is one process from the first line to its machine code). The library is built for the engine's architectures alone
# (GLYD_GPU_ARCHS: the runner's GPU, not the six a release library carries).
set -euo pipefail
export GLYD_ENGINE_NVCC_THREADS="${GLYD_ENGINE_NVCC_THREADS:-0}" GLYD_GPU_ARCHS="${GLYD_GPU_ARCHS:-$GLYD_ENGINE_ARCHS}"
key=$({ cat gpu/*.cu gpu/*.h gpu/build_lib.sh; "$NVCC" --version; echo "$GLYD_GPU_ARCHS"; } | sha256sum | cut -c1-16)
dir="$CARGO_TARGET_DIR/gpu-$key"
log="$CARGO_TARGET_DIR/library-build.log"
library() { PATH="$(dirname "$NVCC"):$PATH" bash gpu/build_lib.sh "$dir.tmp" > "$log" 2>&1; }
beside=""
if ls "$dir"/libglyd_gpu_cuda*.so > /dev/null 2>&1; then
    echo "the GPU library of $key is kept"
    log=""
elif [ "$(nproc 2> /dev/null || echo 1)" -ge 8 ]; then
    mkdir -p "$CARGO_TARGET_DIR"
    library &
    beside=$!
fi
engine=0
cargo build --release --locked -p glyd-engine || engine=$?
if [ -n "$log" ]; then
    # (a library that built is kept even where the engine did not: the next run finds it)
    if [ -n "$beside" ]; then wait "$beside" && built=0 || built=$?; else library && built=0 || built=$?; fi
    if [ "$built" != 0 ]; then
        cat "$log"
        echo "the GPU library did not build" >&2
        exit 1
    fi
    tail -n 1 "$log"
    rm -rf "$dir" && mv "$dir.tmp" "$dir"
fi
[ "$engine" = 0 ] || exit "$engine"
touch "$dir"
# (the three newest kept: a pull request's library and main's)
ls -dt "$CARGO_TARGET_DIR"/gpu-* 2> /dev/null | tail -n +4 | xargs -r rm -rf
echo "GLYD_GPU_LIB_BUILT=$(ls "$dir"/libglyd_gpu_cuda*.so | head -1)" >> "${GITHUB_ENV:-/dev/null}"
echo "E=$CARGO_TARGET_DIR/release/glyd-engine" >> "${GITHUB_ENV:-/dev/null}"
