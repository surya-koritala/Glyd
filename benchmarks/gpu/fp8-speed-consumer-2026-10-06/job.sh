#!/usr/bin/env bash
# fp8-speed-consumer, pod 2 (an RTX 5090, sm_120): this branch's library (srcA.tar: f8_linear of the fp8-speed branch, the channel kind and the W8A8 product
# selectable by GLYD_F8_LIN_A8), vLLM 0.30.0 in its own venv (with ninja and nvcc for FlashInfer's JIT). check_f8.py (the GPU, the real samples of the seven
# 2026 FP8 repos); bench_f8.py at the four checkpoints' shapes (Qwen3.6-35B-A3B-FP8, Qwen3.8-27B-FP8, gemma-4-26B-A4B-it-FP8-dynamic, Mistral-Small-4-119B-2603),
# TP 1 and 2, M = 1, 8, 32, 64: vLLM's kernel as its own selection picks it, ours (default), ours with each variant (W8A8, ahead, dense), the decode alone,
# the route (decode into a scratch + vLLM's kernel), numerics of each against an exact product; then (lib B: srcB.tar, the W8A8 kernels at two CTAs an SM)
# ours alone with W8A8. results/DONE at the end.
set -u
R=$HOME/results; W=$HOME/f8w
mkdir -p "$R/log" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep ${F8_HARD:-4800}; echo "hard stop at +$(el) s"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 VLLM_LOGGING_LEVEL=INFO
step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
sed -n '/name, compute_cap/,+1p' "$R/machine.txt"
for v in A B; do [ -f "$HOME/src$v.tar" ] || continue; mkdir -p "$W/src$v" && tar -C "$W/src$v" -xf "$HOME/src$v.tar" || { echo "bad src$v.tar"; exit 1; }; echo "$v: $(cat "$W/src$v/COMMIT")"; done
step "uv; vLLM 0.30.0 (background); nvcc"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
export UV_CACHE_DIR=$W/uvcache TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
( "$UV" venv -q --python 3.12 "$W/venv"; for t in 1 2 3; do UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" numpy ninja && break; echo "(try $t failed)"; sleep 10; done; echo "vllm exit $?" ) > "$R/log/env-vllm.txt" 2>&1 & VLLM=$!
"$UV" venv -q --python 3.12 "$W/cvenv" > "$R/log/env-nvcc.txt" 2>&1
CPY=$W/cvenv/bin/python
UV_HTTP_TIMEOUT=300 "$UV" pip install -q --python "$CPY" numpy "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-runtime==13.0.*" >> "$R/log/env-nvcc.txt" 2>&1 || { echo "nvcc did not install"; tail -5 "$R/log/env-nvcc.txt"; exit 1; }
CU=$("$CPY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
echo "nvcc: $(nvcc --version | tail -1)" | tee "$R/env.txt"
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
step "samples (background)"
REPOS="zai-org/GLM-5.3-Flash Qwen/Qwen3.8-27B-FP8 MiniMaxAI/MiniMax-M3-MXFP8 mistralai/Mistral-Small-4-119B-2603 deepseek-ai/DeepSeek-V4.1-Flash Qwen/Qwen3.6-35B-A3B-FP8 RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
( cd "$W/srcA" && timeout 900 "$CPY" gpu/fp8/fetch_samples.py "$W/samples" $REPOS > "$R/log/fetch.txt" 2>&1; echo "fetch exit $?" >> "$R/log/fetch.txt" ) & FETCH=$!
step "libraries A, B"
for v in A B; do
  [ -d "$W/src$v" ] || continue
  ( cd "$W/src$v" && time bash gpu/build_lib.sh "$W/lib$v" ) > "$R/log/build$v.txt" 2>&1; BX=$?; echo "build $v exit $BX"; tail -3 "$R/log/build$v.txt"
  [ "$BX" = 0 ] || { [ "$v" = A ] && exit 1; }
done
step "wait for vLLM"
wait "$VLLM"; tail -2 "$R/log/env-vllm.txt"
PY=$W/venv/bin/python
"$PY" -c "import torch, vllm, numpy" 2> /dev/null || { echo "no vLLM: the job stops"; exit 1; }
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)
VCU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))" 2> /dev/null)
[ -n "$NVRTC" ] && [ -n "$VCU" ] && mkdir -p "$VCU/lib64" && for d in "$VCU/lib" "$VCU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export PATH=$W/venv/bin:$PATH  # (ninja for FlashInfer's JIT; nvcc from the CUDA 13.0 wheels above)
"$PY" -c "
import importlib.metadata as md, torch
def v(p):
    try: return md.version(p)
    except Exception: return '-'
print('vllm', v('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| flashinfer', v('flashinfer-python'), '| ninja', v('ninja'), '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee -a "$R/env.txt"
step "wait for the samples"
wait "$FETCH"; tail -8 "$R/log/fetch.txt"
SAMP=(); [ -f "$W/samples/manifest.json" ] && SAMP=(--samples "$W/samples")
MODELS="Qwen3.6-35B-A3B-FP8,Qwen3.8-27B-FP8,gemma-4-26B-A4B-it-FP8-dynamic,Mistral-Small-4-119B-2603"
run() {  # run V CMD...: with library V
  local v=$1; shift
  [ -f "$W/lib$v/libglyd_gpu_cuda$MAJOR.so" ] || { echo "no library $v"; return; }
  export GLYD_GPU_LIB=$W/lib$v/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src$v/glyd-gpu-py GLYD_FP8_SAMPLES=$W/samples
  "$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('$v: C API', _lib._lib.glyd_gpu_api_version())" 2>&1 | tail -1
  "$@"
}
cd "$W/srcA"
step "check A"; run A "$PY" "$W/srcA/gpu/fp8/check_f8.py" "${SAMP[@]}" > "$R/check_A.log" 2>&1; echo "check A exit $?" | tee -a "$R/check_A.log"; tail -14 "$R/check_A.log"
step "bench A (vLLM, ours and its variants, the decode, the route, numerics)"
run A "$PY" gpu/fp8/bench_f8.py "${SAMP[@]}" --models "$MODELS" --tp 1,2 --numerics --variant GLYD_F8_LIN_A8=1 --variant GLYD_F8_LIN_MIN=2 --variant GLYD_F8_LIN_MIN=4 --variant GLYD_F8_LIN_A8=1,GLYD_F8_LIN_MIN=2 \
  --variant GLYD_F8_LIN_A8=1,GLYD_F8_LIN_MIN=4 --variant GLYD_F8_LIN_AHEAD=1 --variant GLYD_F8_LIN_DENSE=1 \
  --out "$R/bench_A.json" > "$R/bench_A.log" 2>&1; echo "bench A exit $?" | tee -a "$R/bench_A.log"; grep -E "^gate|^ours at most" "$R/bench_A.log"
if [ -d "$W/srcB" ]; then
  step "check B (W8A8 cases)"; run B "$PY" "$W/srcB/gpu/fp8/check_f8.py" "${SAMP[@]}" --no-bw --only case_linear > "$R/check_B.log" 2>&1; echo "check B exit $?" | tee -a "$R/check_B.log"; tail -6 "$R/check_B.log"
  step "bench B (ours, W8A8)"; GLYD_F8_LIN_A8=1 run B "$PY" gpu/fp8/bench_f8.py "${SAMP[@]}" --models "$MODELS" --tp 1,2 --no-vllm --no-route --no-unpack --variant GLYD_F8_LIN_MIN=2 --variant GLYD_F8_LIN_MIN=4 --out "$R/bench_B.json" > "$R/bench_B.log" 2>&1; echo "bench B exit $?" | tee -a "$R/bench_B.log"
fi
step "emulation (case_linear, W8A8 and the channel kind among the plans)"
( cd "$W/srcA/gpu/fp8" && PYTHONPATH=$W/srcA/glyd-gpu-py timeout 1200 "$CPY" check_f8.py --emu --only case_linear > "$R/check_emu.log" 2>&1; echo "emu exit $?" | tee -a "$R/check_emu.log" ); tail -4 "$R/check_emu.log"
step "done"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
