#!/usr/bin/env bash
# An L40S's prompt routes (route_e2e.py), unattended on one AWS g6e.xlarge (NVIDIA L40S, AWS Deep Learning AMI, Ubuntu
# 24.04, x86_64): Qwen3-8B in the tiered and the 12-bit layout, fused against decoded first against decoded ahead at
# 512-8192 tokens, and bf16, each a process of its own, nvidia-smi every 100 ms beside them (SM clock, power, the
# clock's limiting reasons); route_summary.py's tables in results/summary.txt. The L40S shares the L4's compute
# capability (its code 89, no class: fused prompts today) and its bandwidth a FLOP, with 350 W for 864 GB/s where the
# L4 has 72 W for 300: this says whether it wants the L4's decode for cuBLAS, the A10's decode ahead, or neither.
# In ~ (scratchpad/aws_gpu.sh uploads them, this as job.sh): route_e2e.py, route_summary.py and l40s_src.tar (the tree:
# gpu/, bindings/python, COMMIT). Each run starts from about the GPU's idle temperature. results/DONE from the exit
# trap however the job ends; no step starts past BUDGET
# (1260 s) and none runs past END (1440 s), counted from the job's start: the environment (PyTorch 2.14.0 for CUDA 13
# and transformers 5.17.0, made with uv; nvcc the AMI's CUDA 13.0), the library, the model's download meanwhile.
#   TYPE=g6e.xlarge MAX_MIN=35 NAME=ceiling/results-l40s UPLOAD="route_e2e.py route_summary.py l40s_src.tar" bash aws_gpu.sh l40s_job.sh
set -u
R=$HOME/results W=$HOME/l40sw
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s) BUDGET=${BUDGET:-1260} END=${END:-1440}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }
fail() { echo "FAIL: $*" | tee "$R/summary.txt"; exit 1; }

step "machine"
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,power.default_limit --format=csv
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; lscpu | grep -E "Model name|Architecture"; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u
  echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; ls -d /usr/local/cuda*; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH  # the AMI's CUDA stacks never ahead of the environment's own libraries (r1_job.sh: its cuDNN)
export TZ=UTC HF_HOME=$HOME/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1; }
IDLE=$(( $(temp) + 5 ))
cool() {  # each run from about the GPU's idle temperature (at most 60 s' wait): a GPU at its power cap slows as it heats
  local i
  for i in $(seq 0 30); do [ "$(temp)" -le "$IDLE" ] && break; sleep 2; done
  echo "the GPU at $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | head -1) after $((i * 2)) s (idle $IDLE C)"
}
echo "$(nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,power.limit --format=csv,noheader | head -1); $(uname -m), $(nproc) CPUs" | tee "$R/machine-short.txt"

step "environment: uv, PyTorch 2.14.0 (CUDA 13), transformers 5.17.0"
( set -e
  mkdir -p "$W/uv"
  curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"
  "$W/uv/uv" venv -q --python 3.12 "$W/env"
  "$W/uv/uv" pip install -q --python "$W/env/bin/python" "torch==2.14.0" "transformers==5.17.0" accelerate safetensors numpy huggingface_hub hf_transfer hf_xet
) > "$R/log/env.txt" 2>&1 || fail "no environment: $(tail -3 "$R/log/env.txt" | tr '\n' ' ')"
PY=$W/env/bin/python
"$PY" -c "import torch, transformers; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '|', torch.cuda.get_device_name())" | tee "$R/env.txt"

step "Qwen3-8B, in the background"
( timeout "$(tmo 900)" "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*']))" > "$W/model.dir" 2> "$R/log/dl.txt"; echo "exit $? at +$(el) s" >> "$R/log/dl.txt" ) &
DL=$!

step "the library for sm_89 (build_lib.sh's flags, the AMI's nvcc 13.0)"
rm -rf "$W/src" && mkdir -p "$W/src/lib" && tar -C "$W/src" -xf "$HOME/l40s_src.tar" || fail "no l40s_src.tar in ~"
CU=/usr/local/cuda-13.0
[ -x "$CU/bin/nvcc" ] || CU=$(ls -d /usr/local/cuda-13* 2> /dev/null | head -1)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden)
{ timeout "$(tmo 600)" "$CU/bin/nvcc" "${F[@]}" -c -o "$W/src/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
  timeout 300 "$CU/bin/nvcc" "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib64" -o "$W/src/lib/libglyd_gpu_cuda13.so" "$W/src/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
echo "build exit $? ($CU, the tree $(cat "$W/src/COMMIT"))" | tee -a "$R/log/build.txt"
[ -f "$W/src/lib/libglyd_gpu_cuda13.so" ] || fail "the library did not build: $(grep -m3 -i error "$R/log/build.txt" | tr '\n' ' ')"
export GLYD_GPU_LIB=$W/src/lib/libglyd_gpu_cuda13.so PYTHONPATH=$W/src/bindings/python

step "waiting for the download"
wait $DL
D=$(tail -1 "$W/model.dir" 2> /dev/null)
[ -f "$D/config.json" ] || fail "no model: $(tail -3 "$R/log/dl.txt" | tr '\n' ' ')"
for mode in mma mma12 bf16; do
  [ "$(el)" -lt "$BUDGET" ] || { echo "$mode: over the budget, not run"; continue; }
  step "route_e2e.py, Qwen3-8B, $mode"
  cool
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -lms 100 > "$R/smi-Qwen3-8B-$mode.csv" 2> /dev/null & S=$!
  if [ $mode = bf16 ]; then a=(--mode bf16); else a=(--mode glyd --layout $mode --routes fused,decoded,ahead); fi
  timeout "$(tmo 600)" "$PY" -u "$HOME/route_e2e.py" "$D" "${a[@]}" --lengths 512,768,1024,1536,2048,3072,4096,8192 --out "$R/route-Qwen3-8B-$mode.json" > "$R/log/route-Qwen3-8B-$mode.txt" 2>&1
  echo "$mode: exit $?"; kill $S
  grep "tokens," "$R/log/route-Qwen3-8B-$mode.txt" | tail -3
  "$PY" "$HOME/route_summary.py" "$R" > "$R/summary.txt" 2> "$R/log/summary.txt"
done
step "done in $(el) s"
