#!/usr/bin/env bash
# One immutable scale-decoder candidate, exact checks and a bounded same-GPU comparison; no whole-model speed claim.
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s); END=1800
exec > >(tee -a "$R/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$R/EXIT_STATUS"; touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > "$R/machine.txt"
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GLYD_SCALES GLYD_CODES GLYD_KV GLYD_EXACT GLYD_FP8 GLYD_FP8_VALUES GLYD_SC_ROUTED
export UV_HTTP_TIMEOUT=180 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=${CC/./}
[ "$CC" = "8.9" ] || { note "expected Ada GPU"; exit 1; }
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/routed_src.tar" || exit 1
cp "$W/src/COMMIT" "$R/COMMIT"
cp "$HOME/plan.txt" "$R/plan.txt"
cp "$HOME/bench_routed.py" "$R/bench_routed.py"
cp "$HOME/job.sh" "$R/job-source.sh"
sha256sum "$HOME/bench_routed.py" "$HOME/routed_src.tar" > "$R/inputs.sha256"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/download/0.12.21/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
note "vLLM 0.30.0, nvcc, the library (sm_$ARCH)"
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "torch==2.13.0" "transformers==5.19.0" "flashinfer-python==0.6.18.post1" "pillow==12.3.0"; } > "$R/log/env-install.txt" 2>&1 || { note "vLLM did not install"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64"
for f in "$CU"/lib/lib*.so.*; do b=${f##*/}; [ -e "$CU/lib64/${b%%.so*}.so" ] || ln -sf "../lib/$b" "$CU/lib64/${b%%.so*}.so"; done 2> /dev/null
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)
[ -n "$NVRTC" ] && for d in "$CU/lib" "$CU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib  # (as the RTX 4090 run of this checkpoint that served it, research/report-quantized-2026-10-06's qbm)
NVCC=$CU/bin/nvcc
MAJOR=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so; mkdir -p "$W/lib"
{ time { "$NVCC" "${F[@]}" -Xptxas -v -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && "$NVCC" "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1 || { note "the library did not build (log/build.txt)"; tail -5 "$R/log/build.txt"; exit 1; }
grep -A2 -E "entry function .*sc_unpack_kernel" "$R/log/build.txt" | grep -E "registers|spill" > "$R/ptxas-sc-unpack.txt"
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt"); library $(grep real "$R/log/build.txt"); scale unpack $(tr '\n' ' ' < "$R/ptxas-sc-unpack.txt")"
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache GLYD_I4_SAMPLES=$W/samples

note "scale plugin checks"
(cd "$W/src/glyd-gpu-py/tests" && GLYD_REQUIRE_VLLM=1 "$PY" test_vllm_scales.py) > "$R/tests.txt" 2>&1 || { note "plugin checks failed"; exit 1; }
note "full scale GPU byte/16-bit/reference/graph checks"
(cd "$W/src" && timeout -k 15 $(( $(left) - 120 )) "$PY" gpu/scales/check_sc.py) > "$R/check_sc.txt" 2>&1 || { note "scale GPU checks failed"; exit 1; }
grep -q 'check_sc: all passed' "$R/check_sc.txt" || exit 1
note "ABBA timing of original and direct selection in captured graphs"
timeout -k 15 $(( $(left) - 60 )) "$PY" "$HOME/bench_routed.py" "$R/timing" > "$R/timing.txt" 2>&1 || { note "timing/selection comparison failed"; exit 1; }
grep -q 'ROUTED EXACT PASS' "$R/timing.txt" || exit 1
note "end: all exact checks passed; timing.json contains synthetic operation results only"
