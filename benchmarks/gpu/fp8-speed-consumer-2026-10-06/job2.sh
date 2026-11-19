#!/usr/bin/env bash
# fp8-speed-consumer, round 2 (after review), one pod: an RTX 5090 (sm_120) or an RTX 4090 (sm_89). The PR head's library (src.tar, COMMIT in it),
# vLLM 0.30.0 in its own venv. check_f8.py (the GPU, the real samples of seven 2026 FP8 repos); compute-sanitizer memcheck and racecheck on
# check_f8.py's case_lin_quick; bench_f8.py with the library's default only (no variants): vLLM's kernel as its own selection picks it, f8_linear,
# the decode route (decode into a scratch + vLLM's kernel, one graph), numerics; nvidia-smi clocks, power and temperature every second during it.
# 5090: the four checkpoints' shapes of round 1, tensor-parallel 1 and 2. 4090: the seven checkpoints' shapes,
# tensor-parallel 1, 2, 4, 8. cuRAND's and cuBLAS's headers beside nvcc (FlashInfer's JIT of vLLM's per-tensor FP8 kernel on the 5090 failed without them). Uploaded files CHECK_ONLY (no bench) and EMU: the CPU emulation's case_linear too (with CHECK_ONLY always). results/DONE at the end.
set -u
R=$HOME/results; W=$HOME/f8w
mkdir -p "$R/log" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep ${F8_HARD:-4200}; echo "hard stop at +$(el) s"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 VLLM_LOGGING_LEVEL=INFO
step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
sed -n '/name, compute_cap/,+1p' "$R/machine.txt"
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/src.tar" || { echo "bad src.tar"; exit 1; }; echo "src: $(cat "$W/src/COMMIT")"
step "uv; vLLM 0.30.0 (background); nvcc and compute-sanitizer"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
export UV_CACHE_DIR=$W/uvcache TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
( "$UV" venv -q --python 3.12 "$W/venv"; for t in 1 2 3; do UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" numpy ninja && break; echo "(try $t failed)"; sleep 10; done; echo "vllm exit $?" ) > "$R/log/env-vllm.txt" 2>&1 & VLLM=$!
"$UV" venv -q --python 3.12 "$W/cvenv" > "$R/log/env-nvcc.txt" 2>&1
CPY=$W/cvenv/bin/python
UV_HTTP_TIMEOUT=300 "$UV" pip install -q --python "$CPY" numpy "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-sanitizer-api==13.0.*" "nvidia-curand==10.4.0.*" "nvidia-cublas==13.0.*" >> "$R/log/env-nvcc.txt" 2>&1 || { echo "nvcc did not install"; tail -5 "$R/log/env-nvcc.txt"; exit 1; }
CU=$("$CPY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
echo "nvcc: $(nvcc --version | tail -1); compute-sanitizer: $(compute-sanitizer --version 2>&1 | tail -1)" | tee "$R/env.txt"
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
step "samples (background)"
REPOS="zai-org/GLM-5.3-Flash Qwen/Qwen3.8-27B-FP8 MiniMaxAI/MiniMax-M3-MXFP8 mistralai/Mistral-Small-4-119B-2603 deepseek-ai/DeepSeek-V4.1-Flash Qwen/Qwen3.6-35B-A3B-FP8 RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
( cd "$W/src" && timeout 900 "$CPY" gpu/fp8/fetch_samples.py "$W/samples" $REPOS > "$R/log/fetch.txt" 2>&1; echo "fetch exit $?" >> "$R/log/fetch.txt" ) & FETCH=$!
step "library"
( cd "$W/src" && time bash gpu/build_lib.sh "$W/lib" ) > "$R/log/build.txt" 2>&1; BX=$?; echo "build exit $BX"; tail -3 "$R/log/build.txt"
[ "$BX" = 0 ] || exit 1
step "wait for vLLM"
wait "$VLLM"; tail -2 "$R/log/env-vllm.txt"
PY=$W/venv/bin/python
"$PY" -c "import torch, vllm, numpy" 2> /dev/null || { echo "no vLLM: the job stops"; exit 1; }
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)
VCU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))" 2> /dev/null)
[ -n "$NVRTC" ] && [ -n "$VCU" ] && mkdir -p "$VCU/lib64" && for d in "$VCU/lib" "$VCU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export PATH=$W/venv/bin:$PATH:$CU/bin  # (ninja for FlashInfer's JIT; nvcc and compute-sanitizer from the CUDA 13.0 wheels above)
"$PY" -c "
import importlib.metadata as md, torch
def v(p):
    try: return md.version(p)
    except Exception: return '-'
print('vllm', v('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| flashinfer', v('flashinfer-python'), '| ninja', v('ninja'), '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee -a "$R/env.txt"
step "wait for the samples"
wait "$FETCH"; tail -8 "$R/log/fetch.txt"
SAMP=(); [ -f "$W/samples/manifest.json" ] && SAMP=(--samples "$W/samples")
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src/glyd-gpu-py GLYD_FP8_SAMPLES=$W/samples
"$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('C API', _lib._lib.glyd_gpu_api_version())" 2>&1 | tail -1
cd "$W/src"
step "check_f8"; "$PY" gpu/fp8/check_f8.py "${SAMP[@]}" > "$R/check_f8.log" 2>&1; echo "check exit $?" | tee -a "$R/check_f8.log"; tail -16 "$R/check_f8.log"
CS=$W/cs; mkdir -p "$CS" && cp "$CU/bin/compute-sanitizer" "$CU/bin/libsanitizer-collection.so" "$CU"/lib/libInterceptorInjectionTarget.so "$CU"/lib/libTreeLauncher*.so "$CU/lib/libsanitizer-public.so" "$CS/"  # (the toolkit's layout: the tool and its injection libraries in one directory)
SAN=""; i=0  # (the first sanitizer that runs a one-line torch script: the wheel's in the toolkit's layout, then the image's toolkit's)
for cs in "$CS/compute-sanitizer" /usr/local/cuda/compute-sanitizer/compute-sanitizer /usr/local/cuda/bin/compute-sanitizer; do
  [ -x "$cs" ] || continue; i=$((i + 1))
  { "$cs" --version | tail -1; timeout 300 "$cs" --tool memcheck --target-processes all "$PY" -c "import torch; print('torch ran:', torch.zeros(4, device='cuda').sum().item())"; echo "exit $?"; } > "$R/sanitizer_smoke$i.log" 2>&1
  echo "smoke $i ($cs): $(grep -E 'torch ran|ERROR SUMMARY|exit' "$R/sanitizer_smoke$i.log" | tr '\n' ' ')"
  grep -q "torch ran" "$R/sanitizer_smoke$i.log" && grep -q "ERROR SUMMARY: 0 errors" "$R/sanitizer_smoke$i.log" && { SAN=$cs; break; }
done
for t in memcheck racecheck; do
  [ -n "$SAN" ] || { echo "no sanitizer ran torch: $t not run"; break; }
  step "compute-sanitizer $t (case_lin_quick)"
  timeout 1200 "$SAN" --tool $t --target-processes all --error-exitcode 9 "$PY" gpu/fp8/check_f8.py --no-bw --only case_lin_quick > "$R/sanitizer_$t.log" 2>&1; echo "$t exit $?" | tee -a "$R/sanitizer_$t.log"; tail -4 "$R/sanitizer_$t.log"
done
case "$GPUNAME" in
  *5090*) MODELS="Qwen3.6-35B-A3B-FP8,Qwen3.8-27B-FP8,gemma-4-26B-A4B-it-FP8-dynamic,Mistral-Small-4-119B-2603"; TP=1,2;;
  *) MODELS="GLM-5.3-Flash,Qwen3.8-27B-FP8,MiniMax-M3-MXFP8,Mistral-Small-4-119B-2603,DeepSeek-V4.1-Flash,Qwen3.6-35B-A3B-FP8,gemma-4-26B-A4B-it-FP8-dynamic"; TP=1,2,4,8;;
esac
if [ -f "$HOME/CHECK_ONLY" ]; then MODELS=""; echo "CHECK_ONLY: no bench"; fi  # (an uploaded file: the checks alone)
[ -n "$MODELS" ] && step "bench_f8 ($MODELS; tp $TP; vLLM, f8_linear default, the route, numerics)"
if [ -n "$MODELS" ]; then
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 1000 > "$R/clocks.csv" 2>&1 & CLK=$!
  "$PY" gpu/fp8/bench_f8.py "${SAMP[@]}" --models "$MODELS" --tp "$TP" --numerics --no-unpack --out "$R/bench.json" > "$R/bench.log" 2>&1; echo "bench exit $?" | tee -a "$R/bench.log"; grep -E "^gate|^ours at most" "$R/bench.log"
  kill $CLK 2> /dev/null
fi
if [ -f "$HOME/CHECK_ONLY" ] || [ -f "$HOME/EMU" ]; then  # (uploaded files: the CPU emulation on this pod)
  step "emulation (case_linear)"
  ( cd "$W/src/gpu/fp8" && PYTHONPATH=$W/src/glyd-gpu-py timeout 1500 "$CPY" check_f8.py --emu --only case_linear > "$R/check_emu.log" 2>&1; echo "emu exit $?" | tee -a "$R/check_emu.log" ); tail -4 "$R/check_emu.log"
fi
step "done"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
