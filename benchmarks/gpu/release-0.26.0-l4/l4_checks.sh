#!/usr/bin/env bash
# v0.26.0's release candidate on the dev L4, under the box's lock: the library for sm_89 (build_lib.sh's flags);
# check_capi.py, test_gpu.py whole, split_stress.py --quick, the crate's tests; check_vllm.py --quick on Qwen3-8B
# (both layouts) and granite-3.1-3b-a800m-instruct with the vLLM venv; respond.py's four modes on Qwen3-0.6B.
set -u
H=~/rc26 R=~/rc26/results W=~/rc26/w
rm -rf "$R" "$W" && mkdir -p "$R/log" "$W" "$H/tmp"
exec > >(tee -a "$R/job.log") 2>&1
step() { echo "== $(date -u +%T) $*"; echo "$(date -u +%T) $*" >> "$R/steps.txt"; }
run() { local n=$1 t=$2; shift 2; timeout "$t" "$@" > "$R/$n.txt" 2>&1; local e=$?; echo "exit $e" >> "$R/$n.txt"; echo "$n: exit $e"; }
source ~/gpuenv/cuda.sh; unset LD_LIBRARY_PATH
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TMPDIR=$H/tmp
tar -C "$W" -xf "$H/rc26_src.tar"
echo "$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader); the tree $(cat "$W/COMMIT")" | tee "$R/machine-short.txt"
step "the library for sm_89"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden)
mkdir -p "$W/lib"
{ nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/libglyd_gpu_cuda13.so" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
echo "exit $?" >> "$R/log/build.txt"; tail -1 "$R/log/build.txt"
LIB=$W/lib/libglyd_gpu_cuda13.so; [ -f "$LIB" ] || { echo "FAIL: no library"; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/bindings/python
step "check_capi.py"; ( cd "$W/gpu" && MAX_JOBS=8 run check_capi 1800 python -u check_capi.py "$LIB" )
step "test_gpu.py"; ( cd "$W/bindings/python" && run test_gpu 1800 python -u test_gpu.py )
step "split_stress.py --quick"; ( cd "$W/gpu" && run split_stress 1200 python -u split_stress.py "$LIB" --quick )
step "cargo test"; ( cd "$W/glyd-gpu" && source ~/.cargo/env 2> /dev/null; run cargo_test 1200 cargo test --release -- --test-threads 1 )
V=~/vllm-work/venv/bin/python
for m in Qwen/Qwen3-8B ibm-granite/granite-3.1-3b-a800m-instruct; do t=$(basename $m)
  step "check_vllm.py --quick $m"; ( cd "$R" && PYTHONSAFEPATH=1 run check_vllm-$t 5400 $V -u "$W/gpu/vllm/check_vllm.py" --quick --out "$R/check_vllm-$t" $m )
done
step "respond.py, Qwen3-0.6B, each mode"
M=$(ls -d ~/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/* | head -1)
for mode in bf16 bf16c glyd exact; do ( cd "$W/gpu" && run respond-$mode 600 python -u respond.py "$M" --mode $mode --out "$R/respond-$mode.json" --reps-ttft 2 --reps-rate 1 --reps-mix 1 ); done
step "done"
