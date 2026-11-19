#!/bin/bash
# The release build on Ubuntu 22.04 (glibc 2.35), so it runs there and on anything newer: the same steps as the PC's rel_final.sh.
set -u
B=$HOME/b; cd $B/src
export CUDA_HOME=$B/cuda NVCC=$B/cuda/bin/nvcc PATH=$B/cuda/bin:$HOME/.cargo/bin:$PATH LD_LIBRARY_PATH=$B/cuda/lib:$B/cuda/lib64
echo "lib: $(date +%T)"
[ -f $B/out/libglyd_gpu_cuda13.so ] || { GLYD_GPU_RELEASE=1 GLYD_GPU_NVCC_THREADS=8 bash gpu/build_lib.sh $B/out > $B/lib.log 2>&1 || { echo LIB FAILED; tail -20 $B/lib.log; exit 1; }; }
echo "engine: $(date +%T)"
GLYD_ENGINE_RELEASE=1 RUSTFLAGS="--remap-path-prefix=$HOME=/home --remap-path-prefix=$B=/b --remap-path-prefix=$B/src=/src --remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$HOME/.rustup=/rustup" CARGO_TARGET_DIR=$B/target cargo build --release --no-default-features -p glyd-engine > $B/engine.log 2>&1 || { echo ENGINE FAILED; tail -30 $B/engine.log; exit 1; }
cp $B/target/release/glyd-engine $B/out/ && strip --strip-all $B/out/glyd-engine
echo "gate: $(date +%T)"
python3 glyd-gpu-py/check_wheel.py --engine $B/out/glyd-engine --engine-source glyd-engine --table-header gpu/glyd_gpu.h --summary > $B/gate.txt 2>&1; echo "engine gate exit $?"
python3 glyd-gpu-py/check_wheel.py --lib $B/out/libglyd_gpu_cuda13.so --header include/glyd_gpu.h --table-header gpu/glyd_gpu.h --summary > $B/gate_lib.txt 2>&1; echo "lib gate exit $?"
for f in glyd-engine libglyd_gpu_cuda13.so; do echo "$f needs: $(objdump -T $B/out/$f | grep -o 'GLIBC_[0-9.]*' | sort -V -u | tail -1)"; done
tail -4 $B/gate.txt; tail -3 $B/gate_lib.txt; ls -la $B/out; echo REL-DONE
