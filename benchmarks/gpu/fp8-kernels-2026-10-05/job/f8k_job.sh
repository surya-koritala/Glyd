#!/usr/bin/env bash
# f8k_job.sh: Task 2's checks of the lossless FP8 kernels on one Runpod pod (rp_run.sh uploads this with f8k_src.tar, the worktree's gpu/ and glyd-gpu-py/):
# uv, a venv with PyTorch and pip's nvcc, the library built with gpu/build_lib.sh for the pod's GPU, gpu/fp8/fetch_samples.py's real samples fetched meanwhile,
# then gpu/fp8/check_f8.py (every case of test_fp8ref.py through the GPU's pack and unpack, strides, lists, CUDA graphs, the real samples, the bandwidth line).
# results/DONE from the exit trap however the job ends. Env: F8_SAMPLES (1: fetch the samples and check them; 0 not), F8_SAN (1: compute-sanitizer on two
# small cases where it is installed), F8_ONLY (check_f8 --only TEXT).
set -u
R=$HOME/results; W=$HOME/f8w
mkdir -p "$R/log" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep 2100; echo "hard stop at +$(el) s"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | sed -n '/name, compute_cap/,+1p'
nproc
cd "$W" && rm -rf src && mkdir src && tar -C src -xf "$HOME/f8k_src.tar" || { echo "no f8k_src.tar"; exit 1; }
cat "$W/src/COMMIT" 2> /dev/null

step "uv, the venv, PyTorch, nvcc"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
export UV_CACHE_DIR=$W/uvcache TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install -q --python "$W/venv/bin/python" torch numpy; } > "$R/log/env-torch.txt" 2>&1 || { echo "torch did not install"; tail -5 "$R/log/env-torch.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install -q --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { echo "nvcc did not install"; tail -5 "$R/log/env-nvcc.txt"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
"$PY" -c "import torch, numpy; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| numpy', numpy.__version__, '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee "$R/env.txt"
echo "nvcc: $(nvcc --version | tail -1)" | tee -a "$R/env.txt"
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')

if [ "${F8_SAMPLES:-1}" = 1 ]; then
  step "samples (in the background)"
  ( cd "$W/src" && timeout 1100 "$PY" gpu/fp8/fetch_samples.py "$W/samples" > "$R/log/fetch.txt" 2>&1; echo "fetch exit $?" >> "$R/log/fetch.txt" ) &
  FETCH=$!
fi

step "library: gpu/build_lib.sh"
( cd "$W/src" && time bash gpu/build_lib.sh "$W/lib" ) > "$R/log/build.txt" 2>&1; echo "build exit $?"
tail -4 "$R/log/build.txt"
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
[ -f "$LIB" ] || { echo "no library"; exit 1; }
ls -l "$LIB"
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/glyd-gpu-py

step "the library loads"
"$PY" -c "
import torch
from glyd_gpu import kernels as g, _lib
g.lib()
print('C API', _lib._lib.glyd_gpu_api_version(), _lib.API_VERSION, '| GPU code', _lib.gpu())" 2>&1 | tee "$R/log/load.txt"

SAMPLES=()
if [ "${F8_SAMPLES:-1}" = 1 ]; then
  step "wait for the samples"
  wait "$FETCH"; tail -8 "$R/log/fetch.txt"
  [ -f "$W/samples/manifest.json" ] && SAMPLES=(--samples "$W/samples") || echo "NO SAMPLES (the real-sample case will say SKIPPED)"
fi

step "check_f8.py"
cd "$W/src"
ONLY=(); [ -n "${F8_ONLY:-}" ] && ONLY=(--only "$F8_ONLY")
"$PY" gpu/fp8/check_f8.py "${SAMPLES[@]}" "${ONLY[@]}" > "$R/check_f8.log" 2>&1; echo "check_f8 exit $?" | tee -a "$R/check_f8.log"
tail -25 "$R/check_f8.log"
grep "^bandwidth" "$R/check_f8.log" > "$R/bandwidth.txt"

if [ "${F8_SAN:-0}" = 1 ]; then
  SANI=$(command -v compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer 2> /dev/null | head -1)
  if [ -n "$SANI" ]; then
    step "compute-sanitizer: racecheck and memcheck on small cases"
    for tool in racecheck memcheck; do
      for only in case_strides case_experts; do
        timeout 600 "$SANI" --tool $tool --error-exitcode 9 "$PY" gpu/fp8/check_f8.py --only $only --no-bw > "$R/san-$tool-$only.log" 2>&1; echo "$tool $only exit $?" | tee -a "$R/san.txt"
      done
    done
  else echo "no compute-sanitizer here" | tee "$R/san.txt"; fi
fi
step "done"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
