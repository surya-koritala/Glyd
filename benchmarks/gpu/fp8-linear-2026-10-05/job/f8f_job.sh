#!/usr/bin/env bash
# f8f_job.sh: Task 5 on one Runpod pod: vLLM 0.30.0 (uv venv), this tree's library (gpu/build_lib.sh) for the pod's GPU, the real samples, then
# F8_STEPS (default "check bench"): check = gpu/fp8/check_f8.py (F8_CHECK_ARGS), bench = gpu/fp8/bench_f8.py (F8_BENCH_ARGS). results/DONE at the end.
set -u
R=$HOME/results; W=$HOME/f8w
mkdir -p "$R/log" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep ${F8_HARD:-2500}; echo "hard stop at +$(el) s"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 VLLM_LOGGING_LEVEL=INFO
step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
sed -n '/name, compute_cap/,+1p' "$R/machine.txt"
cd "$W" && rm -rf src && mkdir src && tar -C src -xf "$HOME/f8f_src.tar" || { echo "no f8f_src.tar"; exit 1; }
cat "$W/src/COMMIT"
step "uv, vLLM 0.30.0, nvcc"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
export UV_CACHE_DIR=$W/uvcache TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-vllm.txt" 2>&1 || { echo "vllm did not install"; tail -5 "$R/log/env-vllm.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install -q --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { echo "nvcc did not install"; tail -5 "$R/log/env-nvcc.txt"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)  # (FlashInfer's and DeepGEMM's JIT link -lnvrtc)
[ -n "$NVRTC" ] && for d in "$CU/lib" "$CU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
"$PY" -c "
import importlib.metadata as md, torch
def v(p):
    try: return md.version(p)
    except Exception: return '-'
print('vllm', v('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| flashinfer', v('flashinfer-python'), '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee "$R/env.txt"
echo "nvcc: $(nvcc --version | tail -1)" | tee -a "$R/env.txt"
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
step "samples (background)"
( cd "$W/src" && timeout 900 "$PY" gpu/fp8/fetch_samples.py "$W/samples" > "$R/log/fetch.txt" 2>&1; echo "fetch exit $?" >> "$R/log/fetch.txt" ) & FETCH=$!
step "library"
( cd "$W/src" && time bash gpu/build_lib.sh "$W/lib" ) > "$R/log/build.txt" 2>&1; echo "build exit $?"
tail -3 "$R/log/build.txt"
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src/glyd-gpu-py
"$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('C API', _lib._lib.glyd_gpu_api_version())" 2>&1 | tee "$R/log/load.txt"
step "wait for the samples"
wait "$FETCH"; tail -3 "$R/log/fetch.txt"
SAMP=(); [ -f "$W/samples/manifest.json" ] && SAMP=(--samples "$W/samples")
cd "$W/src"
for st in ${F8_STEPS:-check bench}; do
  case $st in
    check) step "check_f8.py"; "$PY" gpu/fp8/check_f8.py "${SAMP[@]}" ${F8_CHECK_ARGS:-} > "$R/check_f8.log" 2>&1; echo "check exit $?" | tee -a "$R/check_f8.log"; tail -30 "$R/check_f8.log";;
    bench) step "bench_f8.py"; "$PY" gpu/fp8/bench_f8.py "${SAMP[@]}" --out "$R/bench_f8.json" ${F8_BENCH_ARGS:-} > "$R/bench_f8.log" 2>&1; echo "bench exit $?" | tee -a "$R/bench_f8.log"; tail -15 "$R/bench_f8.log";;
    ncu) step "ncu"; NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2> /dev/null | head -1); echo "ncu: ${NCU:-none}"
         [ -z "$NCU" ] && { timeout 300 apt-get install -y -q --no-install-recommends cuda-nsight-compute-13-0 > "$R/log/ncu-install.txt" 2>&1; NCU=$(ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2> /dev/null | head -1); echo "ncu after apt: ${NCU:-none}"; }
         if [ -n "$NCU" ]; then for shp in "34816 5120 1 block" "34816 5120 64 block" "4096 2048 1 block"; do set -- $shp
           timeout 300 "$NCU" --kernel-name regex:f8_linear --launch-skip 2 --launch-count 1 --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section InstructionStats "$PY" ~/ncu_f8.py $1 $2 $3 $4 > "$R/ncu_$1_$2_$3_$4.log" 2>&1; echo "ncu $shp exit $?"; done; fi;;
  esac
done
step "done"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
