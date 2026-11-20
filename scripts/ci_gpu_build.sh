#!/usr/bin/env bash
# The engine and the GPU library for the CI runner's GPU (gpu-tests.yml), reusing what an earlier run built where the sources are the same:
# cargo's target directory persists (CARGO_TARGET_DIR; scripts/ci_restore_mtimes.py keeps a fresh checkout from looking changed to it), and
# the GPU library, which is nvcc's alone and takes minutes, is kept by the hash of what makes it. Needs GLYD_ENGINE_ARCHS, NVCC and CARGO_TARGET_DIR.
# Appends GLYD_GPU_LIB and E (the engine) to $GITHUB_ENV.
set -euo pipefail
cargo build --release --locked -p glyd-engine
key=$({ cat gpu/*.cu gpu/*.h gpu/build_lib.sh; "$NVCC" --version; } | sha256sum | cut -c1-16)
dir="$CARGO_TARGET_DIR/gpu-$key"
if ls "$dir"/libglyd_gpu_cuda*.so > /dev/null 2>&1; then
    echo "the GPU library of $key is kept"
else
    PATH="$(dirname "$NVCC"):$PATH" bash gpu/build_lib.sh "$dir.tmp"
    rm -rf "$dir" && mv "$dir.tmp" "$dir"
fi
touch "$dir"
# (the three newest kept: a pull request's library and main's)
ls -dt "$CARGO_TARGET_DIR"/gpu-* 2> /dev/null | tail -n +4 | xargs -r rm -rf
echo "GLYD_GPU_LIB=$(ls "$dir"/libglyd_gpu_cuda*.so | head -1)" >> "${GITHUB_ENV:-/dev/null}"
echo "E=$CARGO_TARGET_DIR/release/glyd-engine" >> "${GITHUB_ENV:-/dev/null}"
