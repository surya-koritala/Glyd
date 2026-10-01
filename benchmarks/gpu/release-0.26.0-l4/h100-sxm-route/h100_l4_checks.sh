#!/usr/bin/env bash
# The route SPLIT's extension to an H100 SXM (the class GLYD_GPU_H100, code 7090, 132 SMs) on the dev L4, under the box's
# lock: release-0.26.0 at 889f4e3 with the extension's code patch (code.patch) on top; the library for sm_89
# (build_lib.sh's flags); check_capi.py, test_gpu.py whole, split_stress.py --quick, the glyd-gpu crate's tests. The L4's own
# code takes the routes without SPLIT (the route's fallback path); the H100 SXM's rule is checked on the host through both
# libraries (check_capi's pins and the host route check, no Hopper GPU).
set -u
H=~/h100chk R=~/h100chk/results W=~/h100chk/w
rm -rf "$R" && mkdir -p "$R/log"
exec > >(tee -a "$R/job.log") 2>&1
step() { echo "== $(date -u +%T) $*"; echo "$(date -u +%T) $*" >> "$R/steps.txt"; }
run() { local n=$1 t=$2; shift 2; timeout "$t" "$@" > "$R/$n.txt" 2>&1; local e=$?; echo "exit $e" >> "$R/$n.txt"; echo "$n: exit $e"; }
source ~/gpuenv/cuda.sh; unset LD_LIBRARY_PATH
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TMPDIR=$H/tmp
mkdir -p "$H/tmp"
echo "$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader), $(nvcc --version | tail -1), torch $(python -c 'import torch; print(torch.__version__)'); the tree $(cat "$W/COMMIT")" | tee "$R/machine-short.txt"
step "the library for sm_89"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden)
mkdir -p "$W/lib"
{ nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/libglyd_gpu_cuda13.so" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
echo "exit $?" >> "$R/log/build.txt"; tail -1 "$R/log/build.txt"
LIB=$W/lib/libglyd_gpu_cuda13.so; [ -f "$LIB" ] || { echo "FAIL: no library"; touch "$R/DONE"; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
step "check_capi.py"; ( cd "$W/src/gpu" && MAX_JOBS=8 run check_capi 1800 python -u check_capi.py "$LIB" )
step "test_gpu.py"; ( cd "$W/src/bindings/python" && run test_gpu 1800 python -u test_gpu.py )
step "split_stress.py --quick"; ( cd "$W/src/gpu" && run split_stress 1200 python -u split_stress.py "$LIB" --quick )
step "cargo test"; ( cd "$W/src/glyd-gpu" && source ~/.cargo/env 2> /dev/null; run cargo_test 1200 cargo test --release -- --test-threads 1 )
step "done"
touch "$R/DONE"
