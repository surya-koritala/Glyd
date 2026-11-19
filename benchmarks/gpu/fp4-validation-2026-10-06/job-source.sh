#!/usr/bin/env bash
# fp4-codes: FP4 checkpoints' codes (NVFP4's and MXFP4's E2M1 values) under the option codes (vllm_codes.py) on one Runpod pod, unattended (rp_run.sh uploads
# this, f4_src.tar and f4env, pulls ~/results, deletes the pod). f4_src.tar: git archive of the branch (glyd-gpu-py/, gpu/glyd_gpu.cu and .h, gpu/vllm/,
# gpu/codes/, gpu/scales/, gpu/fp8/), COMMIT. f4env: F4_MODELS ("repo|quant_parts.py flags|FI=off" a line, the last two optional; no flags: fp4_bits.py only),
# END (seconds the job may take).
#   0. the checkpoints downloaded one after another in the background (huggingface_hub, a venv of its own; public repos, no token)
#   1. vLLM 0.30.0 (uv), nvcc from pip (its libraries linked by their plain names: FlashInfer builds its kernels with them on first use), the library built
#      from glyd_gpu.cu for this GPU alone; while it builds, test_int4ref.py --study (the code's bits on the INT4, MXFP4 and NVFP4 samples read by HTTP range,
#      CPU) fetches the samples check_i4.py then reads
#   2. the plugin's tests against the real vLLM (test_vllm_quant.py, test_vllm_scales.py, test_vllm_codes.py: they must pass); gpu/codes/check_i4.py on the
#      GPU: kernels.i4_pack and the i4 kernel against int4ref.py bit for bit (every value pattern, sizes, plain patches, the list's unpack, the real samples:
#      INT4, MXFP4 and NVFP4 tensors by HTTP range, CUDA graphs, the unpack's speed); nothing else runs unless it passes
#   3. each checkpoint once downloaded: gpu/codes/fp4_bits.py (every FP4 code tensor of it packed on the GPU and unpacked by the kernel, bit for bit, its bits
#      a code), then, where flags are given, quant_parts.py (stock and Glyd: weights, KV cache, 1 and 32 users' tokens a second, exact eager against stock
#      eager or Glyd's without the options, every pack verified at load); FI=off: FlashInfer's FP8 and NVFP4 GEMMs off for stock and Glyd alike
#      (VLLM_DISABLED_KERNELS), and a run that fails as FlashInfer builds a GEMM is made again so; its directory deleted after it
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp"
[ -f "$HOME/f4env" ] && . "$HOME/f4env"
T0=$(date +%s); END=${END:-6000}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GLYD_SCALES GLYD_CODES GLYD_KV GLYD_EXACT GLYD_I4_FORCE VLLM_DISABLED_KERNELS  # (public models: no token read, none sent)
export UV_HTTP_TIMEOUT=180 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/f4_src.tar" || { note "no f4_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
slug() { echo "$1" | tr '/' '_'; }
mapfile -t MODELS < <(printf '%s\n' "${F4_MODELS:-}" | sed '/^\s*$/d')

note "downloads (background): ${#MODELS[@]} checkpoints"
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet > "$R/log/dl-env.txt" 2>&1
  for line in "${MODELS[@]}"; do
    repo=${line%%|*}; s=$(slug "$repo"); t=$(date +%s)
    "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken']))" "$repo" > "$R/log/dl-$s.txt" 2>&1
    echo "exit $? after $(( $(date +%s) - t )) s; $(du -sh "$W/hf/hub/models--${repo/\//--}" 2> /dev/null | cut -f1)" >> "$R/log/dl-$s.txt"
  done ) &

note "vLLM 0.30.0, nvcc, the library (sm_$ARCH)"
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" pillow; } > "$R/log/env-install.txt" 2>&1 || { note "vLLM did not install"; exit 1; }
PY=$W/venv/bin/python
export GLYD_I4_SAMPLES=$W/i4-samples
( cd "$W/src/glyd-gpu-py/tests" && timeout 1500 "$PY" test_int4ref.py --study ) > "$R/study.txt" 2>&1 &
STUDY=$!
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64"
for f in "$CU"/lib/lib*.so.*; do b=${f##*/}; [ -e "$CU/lib64/${b%%.so*}.so" ] || ln -sf "../lib/$b" "$CU/lib64/${b%%.so*}.so"; done 2> /dev/null  # (-lcudart, -lcublasLt, ... for FlashInfer's builds)
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)
[ -n "$NVRTC" ] && for d in "$CU/lib" "$CU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so; mkdir -p "$W/lib"
{ time { nvcc "${F[@]}" -Xptxas -v -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1 || { note "the library did not build (log/build.txt)"; tail -5 "$R/log/build.txt"; exit 1; }
grep -A2 -E "entry function '_ZN3i4c" "$R/log/build.txt" | grep -E "registers|spill" > "$R/ptxas-i4c.txt"
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| flashinfer', m.version('flashinfer-python'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt"); library $(grep real "$R/log/build.txt"); i4 kernel $(tr '\n' ' ' < "$R/ptxas-i4c.txt")"
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache

( cd "$W/src/glyd-gpu-py/tests" && for t in test_vllm_quant.py test_vllm_scales.py test_vllm_codes.py; do GLYD_REQUIRE_VLLM=1 "$PY" "$t" || echo "FAIL $t exit $?"; done ) > "$R/tests.txt" 2>&1
note "plugin tests: $(grep -c ' ok' "$R/tests.txt") ok; $(grep -c '^FAIL' "$R/tests.txt") failed; $(grep -c Traceback "$R/tests.txt") tracebacks"
grep -q "^FAIL" "$R/tests.txt" && { note "the plugin's tests failed: no GPU run"; exit 1; }

wait "$STUDY"; note "test_int4ref.py --study: exit $? ($(grep -c '|' "$R/study.txt") lines)"
note "check_i4.py on the GPU"
( cd "$W/src" && timeout 1500 "$PY" gpu/codes/check_i4.py ) > "$R/check_i4.txt" 2>&1
note "check_i4: exit $? | $(grep -E 'speed|graphs|passed|Error|fp4\)|samples' "$R/check_i4.txt" | tail -40 | tr '\n' '|')"
grep -q "check_i4: all passed" "$R/check_i4.txt" || { note "check_i4 failed: no vLLM run (the kernel or the pack is not bit for bit)"; exit 1; }

FI_OFF=FlashInferFP8ScaledMMLinearKernel,FlashInferCuteDslNvFp4LinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel
for line in "${MODELS[@]}"; do
  IFS='|' read -r repo pflags fi <<< "$line"
  s=$(slug "$repo"); f="$R/log/dl-$s.txt"
  until grep -q "^exit" "$f" 2> /dev/null; do [ "$(left)" -lt 420 ] && break; sleep 5; done
  if ! grep -q "^exit 0" "$f" 2> /dev/null; then note "$repo: not downloaded ($(tail -2 "$f" 2> /dev/null | tr '\n' ' '))"; continue; fi
  dir=$(grep "^/.*/snapshots/" "$f" | tail -1)
  note "$repo: fp4_bits.py ($(tail -1 "$f"))"
  timeout 900 "$PY" "$W/src/gpu/codes/fp4_bits.py" "$dir" --name "$repo" --out "$R/fp4bits-$s.json" > "$R/fp4bits-$s.txt" 2>&1
  note "$repo: fp4_bits exit $? | $(tr '\n' '|' < "$R/fp4bits-$s.txt" | cut -c1-1500)"
  if [ -n "$pflags" ]; then
    l=$(left)
    if [ "$l" -lt 900 ]; then note "$repo: quant_parts.py not run, $l s left"
    else
      if [ "$fi" = "FI=off" ]; then export VLLM_DISABLED_KERNELS=$FI_OFF; else unset VLLM_DISABLED_KERNELS; fi
      for try in 1 2; do
        note "$repo: quant_parts.py $pflags (at most $((l - 120)) s)${VLLM_DISABLED_KERNELS:+; VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS}"
        timeout -k 30 $((l - 120)) "$PY" "$W/src/gpu/vllm/quant_parts.py" "$repo" --out "$R/$s" $pflags > "$R/$s.txt" 2>&1
        note "$repo: exit $? | $(grep -E '^(EXACT|PACKED|SCALES|CODES|STOCK|GLYD|stock|glyd)' "$R/$s.txt" | cut -c1-700 | tr '\n' '|')"
        if [ "$try" = 1 ] && [ -z "${VLLM_DISABLED_KERNELS:-}" ] && grep -q "Ninja build failed" "$R/$s"/*.log "$R/$s"/*.json 2> /dev/null; then
          note "$repo: FlashInfer could not build a GEMM here (Ninja build failed): its runs again with VLLM_DISABLED_KERNELS=$FI_OFF"
          mkdir -p "$R/$s-flashinfer-failed" && mv "$R/$s"/* "$R/$s.txt" "$R/$s-flashinfer-failed/" 2> /dev/null
          export VLLM_DISABLED_KERNELS=$FI_OFF; l=$(left); [ "$l" -lt 900 ] && { note "$repo: not run again, $l s left"; break; }; continue
        fi
        break
      done
      unset VLLM_DISABLED_KERNELS
    fi
  fi
  rm -rf "$W/hf/hub/models--${repo/\//--}"
done
nvidia-smi --query-gpu=memory.used --format=csv > "$R/smi-end.txt" 2>&1
note "end"
