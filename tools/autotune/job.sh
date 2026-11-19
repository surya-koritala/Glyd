#!/usr/bin/env bash
# job.sh: the route and plan sweep (sweep.py) on one Runpod GPU, unattended. In ~: this and at_src.tar (the tree's gpu/, glyd-gpu-py/,
# tools/autotune/ and COMMIT). Steps: env (uv venv with vLLM 0.30.0 for its PyTorch, nvcc of PyTorch's CUDA), build (the tree's library
# for this GPU alone, a development build: its tuning variables read), checks (the self-test, every bf16 bit pattern in both layouts
# bit for bit, and check_step.py), fetch (2026 models' matrices by byte range, beside the build), warm (the clocks under load),
# sweep (sweep.py over every file), smi (nvidia-smi's clocks for the whole sweep). results/DONE from the exit trap; results/ALLDONE
# when it ran to its end. AT_END (seconds, default 2700) is the hard stop.
set -u
R=$HOME/results; W=$HOME/atw
mkdir -p "$R/log" "$W/tmp"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
note() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; echo "   $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep "${AT_END:-2700}"; echo "hard stop at +$(el) s" >> "$R/steps.txt"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute $CC (sm_$ARCH)" | tee "$R/gpu-short.txt"
cd "$W" && rm -rf src && mkdir src && tar -C src -xf "$HOME/at_src.tar" || { echo "no at_src.tar"; exit 1; }
cp src/COMMIT "$R/COMMIT"; cat src/COMMIT

step "env: uv, vLLM 0.30.0 (its PyTorch), nvcc"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-vllm.txt" 2>&1 || { note "FAIL: vLLM did not install"; tail -5 "$R/log/env-vllm.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "FAIL: nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
"$PY" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee "$R/env.txt"
echo "nvcc: $(nvcc --version | tail -1)" | tee -a "$R/env.txt"
note "env"

mkdir -p "$W/m"
( cd "$W/src/gpu"
  P=model.language_model.layers
  q35() { "$PY" fetch_tensors.py "$1" "$2" gate_up=$P.3.mlp.gate_proj.weight+$P.3.mlp.up_proj.weight down=$P.3.mlp.down_proj.weight \
    qkv=$P.3.self_attn.q_proj.weight+$P.3.self_attn.k_proj.weight+$P.3.self_attn.v_proj.weight o=$P.3.self_attn.o_proj.weight \
    gdn_in=$P.0.linear_attn.in_proj_qkv.weight+$P.0.linear_attn.in_proj_z.weight gdn_out=$P.0.linear_attn.out_proj.weight; }
  g4() { "$PY" fetch_tensors.py "$1" "$2" gate_up=$P.0.mlp.gate_proj.weight+$P.0.mlp.up_proj.weight down=$P.0.mlp.down_proj.weight \
    qkv=$P.0.self_attn.q_proj.weight+$P.0.self_attn.k_proj.weight+$P.0.self_attn.v_proj.weight o=$P.0.self_attn.o_proj.weight \
    qk_global=$P.5.self_attn.q_proj.weight+$P.5.self_attn.k_proj.weight o_global=$P.5.self_attn.o_proj.weight; }
  q35 Qwen/Qwen3.5-9B "$W/m/Qwen3.5-9B.safetensors" &
  q35 Qwen/Qwen3.5-27B "$W/m/Qwen3.5-27B.safetensors" &
  q35 Qwen/Qwen3.8-27B "$W/m/Qwen3.8-27B.safetensors" &
  g4 google/gemma-4-12B-it "$W/m/gemma-4-12B-it.safetensors" &
  g4 google/gemma-4-31B-it "$W/m/gemma-4-31B-it.safetensors" &
  "$PY" fetch_tensors.py Qwen/Qwen3.5-35B-A3B "$W/m/Qwen3.5-35B-A3B.safetensors" \
    qkv=$P.3.self_attn.q_proj.weight+$P.3.self_attn.k_proj.weight+$P.3.self_attn.v_proj.weight o=$P.3.self_attn.o_proj.weight \
    gdn_in=$P.0.linear_attn.in_proj_qkv.weight+$P.0.linear_attn.in_proj_z.weight gdn_out=$P.0.linear_attn.out_proj.weight \
    shared_gate_up=$P.3.mlp.shared_expert.gate_proj.weight+$P.3.mlp.shared_expert.up_proj.weight shared_down=$P.3.mlp.shared_expert.down_proj.weight &
  G=model.layers
  "$PY" fetch_tensors.py ibm-granite/granite-4.2-8b "$W/m/granite-4.2-8b.safetensors" gate_up=$G.3.mlp.gate_proj.weight+$G.3.mlp.up_proj.weight \
    down=$G.3.mlp.down_proj.weight qkv=$G.3.self_attn.q_proj.weight+$G.3.self_attn.k_proj.weight+$G.3.self_attn.v_proj.weight o=$G.3.self_attn.o_proj.weight &
  "$PY" fetch_tensors.py Qwen/Qwen3.5-9B "$W/m/head-Qwen3.5-9B.safetensors" lm_head=lm_head.weight &
  wait ) > "$R/log/fetch.txt" 2>&1 &
FETCH=$!

step "build: the library for sm_$ARCH (a development build)"
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
mkdir -p "$W/lib"
{ time { nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1
[ -f "$LIB" ] || { note "FAIL: the library did not build (log/build.txt)"; tail -20 "$R/log/build.txt"; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/glyd-gpu-py
"$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" 2>&1 | tee "$R/log/load.txt"
note "build: $(grep real "$R/log/build.txt"); $(tail -1 "$R/log/load.txt")"

cd "$W/src/gpu"
step "checks: the self-test, check_step.py"
timeout 900 "$PY" glyd_gpu_dev.py > "$R/check-selftest.txt" 2>&1; note "self-test exit $? ($(tail -1 "$R/check-selftest.txt" | cut -c1-160))"
timeout 900 "$PY" check_step.py > "$R/check-step.txt" 2>&1; note "check_step exit $? ($(tail -1 "$R/check-step.txt" | cut -c1-160))"

step "fetch: wait for the matrices"
wait "$FETCH"; note "fetch: $(grep -c ' from ' "$R/log/fetch.txt") matrices"; ls -la "$W/m" >> "$R/log/fetch.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 1 > "$R/smi.csv" 2> /dev/null & SMI=$!

step "warm: the clocks under load (20 s of bf16 GEMMs, untimed)"
"$PY" -c "
import time, torch
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device='cuda'); t = time.time()
while time.time() - t < 20: a @ a
torch.cuda.synchronize()"

step "sweep"
FILES=$(for m in Qwen3.5-9B Qwen3.5-27B Qwen3.8-27B Qwen3.5-35B-A3B gemma-4-12B-it gemma-4-31B-it granite-4.2-8b head-Qwen3.5-9B; do [ -f "$W/m/$m.safetensors" ] && echo "$W/m/$m.safetensors"; done)
timeout $(( ${AT_END:-2700} - $(el) - 60 )) "$PY" "$W/src/tools/autotune/sweep.py" $FILES --json "$R/sweep.json" > "$R/sweep.txt" 2>&1
note "sweep exit $? ($(tail -1 "$R/sweep.txt" | cut -c1-200))"
kill "$SMI" 2> /dev/null
step "done in $(el) s"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
