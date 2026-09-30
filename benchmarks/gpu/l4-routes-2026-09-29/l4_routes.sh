#!/usr/bin/env bash
# A prompt's routes on an L4 (the AWS dev machine), end to end: route_e2e.py on Qwen3-8B and Qwen3-4B-Instruct-2507,
# Glyd in the tiered and the 12-bit layout (each a process: fused, decoded, decoded ahead in turn at each length) and
# bf16 (a process), nvidia-smi every 100 ms beside them (SM clock, power, the clock's limiting reasons); the models
# from the machine's cache. In the directory FILES: this, route_e2e.py, route_summary.py and the tree's tar
# (l4_src.tar: gpu/, bindings/python, COMMIT). Results in R, route_summary.py's tables in R/summary.txt; DONE last.
set -u
FILES=${FILES:-$(cd "$(dirname "$0")" && pwd)} R=${R:-$FILES/results} W=${W:-$FILES/w}
MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507"} LAYOUTS=${LAYOUTS:-"mma mma12"}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
step() { echo "== $(date -u +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
unset LD_LIBRARY_PATH
source "$HOME/gpuenv/cuda.sh"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TZ=UTC
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; lscpu | grep "Model name"; nproc; date -u
  python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"; } > "$R/machine.txt" 2>&1
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/l4_src.tar"
step "the library (build_lib.sh's flags), the tree $(cat "$W/src/COMMIT")"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden)
mkdir -p "$W/src/lib"
nvcc "${F[@]}" -c -o "$W/src/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" > "$R/log/build.txt" 2>&1 &&
  nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/src/lib/libglyd_gpu_cuda$MAJOR.so" "$W/src/lib/glyd_gpu.o" >> "$R/log/build.txt" 2>&1
echo "build exit $?"
export GLYD_GPU_LIB=$W/src/lib/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src/bindings/python
snap() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*']))"; }
for m in $MODELS; do
  d=$(snap "$m") || { echo "no $m"; continue; }
  n=$(basename "$m")
  for mode in $LAYOUTS bf16; do
    step "$n $mode"
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -lms 100 > "$R/smi-$n-$mode.csv" 2> /dev/null & S=$!
    if [ "$mode" = bf16 ]; then args=(--mode bf16); else args=(--mode glyd --layout "$mode"); fi
    timeout 1800 python -u "$FILES/route_e2e.py" "$d" "${args[@]}" --out "$R/route-$n-$mode.json" ${ROUTE_ARGS:-} > "$R/log/route-$n-$mode.txt" 2>&1
    echo "$n $mode: exit $?"; kill $S
    grep -E "tokens," "$R/log/route-$n-$mode.txt" | tail -3
  done
done
python "$FILES/route_summary.py" "$R" > "$R/summary.txt" 2> "$R/log/summary.txt"; echo "summary exit $?"
step "done"
