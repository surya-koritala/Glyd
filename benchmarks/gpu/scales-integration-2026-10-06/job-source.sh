#!/usr/bin/env bash
# quant-scales: the scales of quantized 2026 checkpoints held packed (the option scales, vllm_scales.py) on one Runpod pod, unattended (rp_run.sh
# uploads this, qs_src.tar and qsenv, pulls ~/results, deletes the pod). qs_src.tar: glyd-gpu-py/, gpu/glyd_gpu.cu and .h, gpu/vllm/quant_parts.py and
# quant_layers.py, gpu/scales/, gpu/fp8/emu*.{py,cpp}, COMMIT. qsenv: QS_MODELS ("repo|quant_parts.py flags|quant_layers.py flags" a line, the last two
# optional; no quant_layers flags: not run), END (seconds the job may take).
#   0. the checkpoints downloaded one after another in the background (huggingface_hub, a venv of its own; public repos, no token)
#   1. vLLM 0.30.0 (uv), nvcc from pip (its libraries linked by their plain names: FlashInfer's kernels are built with them on first use), the library
#      built from glyd_gpu.cu for this GPU alone
#   2. the plugin's tests against the real vLLM (test_vllm_quant.py, test_vllm_scales.py: they must pass), quant_layers.py --selftest, then
#      gpu/scales/check_sc.py on the GPU: the scales kernels against scaleref.py bit for bit (every byte and 16-bit pattern, the real samples by HTTP range,
#      the list's unpack, CUDA graphs, the unpack's speed); nothing else runs unless it passes
#   3. each checkpoint once downloaded: quant_parts.py --scales (stock and Glyd: weights, KV cache, 1 and 32 users' tokens a second, the scales' packs
#      verified at load), then, where flags are given, quant_layers.py with GLYD_SCALES=1 (the moe-exact method: Glyd's 16-bit layers on stock's own inputs,
#      every tensor against stock's with each scales' pack decoded, end to end in a mode where stock repeats itself); its directory deleted after it.
#      Where a run of a checkpoint fails as FlashInfer builds a GEMM on first use, its runs are made again with FlashInfer's FP8 and NVFP4 GEMMs off
#      (VLLM_DISABLED_KERNELS, stock's and Glyd's alike; vLLM's own CUTLASS kernels instead), and the record says so. FI=off (qsenv): off from the start.
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp" "$W/cap"
[ -f "$HOME/qsenv" ] && . "$HOME/qsenv"
T0=$(date +%s); END=${END:-8400}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GLYD_SCALES GLYD_KV GLYD_EXACT VLLM_DISABLED_KERNELS  # (public models: no token read, none sent)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/qs_src.tar" || { note "no qs_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
slug() { echo "$1" | tr '/' '_'; }
mapfile -t MODELS < <(printf '%s\n' "${QS_MODELS:-}" | sed '/^\s*$/d')

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
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64"
for f in "$CU"/lib/lib*.so.*; do b=${f##*/}; [ -e "$CU/lib64/${b%%.so*}.so" ] || ln -sf "../lib/$b" "$CU/lib64/${b%%.so*}.so"; done 2> /dev/null  # (-lcudart, -lcublasLt, ... for FlashInfer's builds)
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)
[ -n "$NVRTC" ] && for d in "$CU/lib" "$CU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib
ls "$CU/include" | grep -E "^cublas" > "$R/log/toolkit-headers.txt" 2>&1
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so; mkdir -p "$W/lib"
{ time { nvcc "${F[@]}" -Xptxas -v -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1 || { note "the library did not build (log/build.txt)"; tail -5 "$R/log/build.txt"; exit 1; }
grep -A2 -E "entry function '_ZN3scc" "$R/log/build.txt" | grep -E "registers|spill" | sort | uniq -c > "$R/ptxas-scc.txt"
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| flashinfer', m.version('flashinfer-python'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt"); library $(grep real "$R/log/build.txt")"
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache GLYD_SCALE_SAMPLES=$W/scale-samples

( cd "$W/src/glyd-gpu-py/tests" && for t in test_vllm_quant.py test_vllm_scales.py; do GLYD_REQUIRE_VLLM=1 "$PY" "$t" || echo "FAIL $t exit $?"; done ) > "$R/tests.txt" 2>&1
note "plugin tests: $(grep -c ' ok' "$R/tests.txt") ok; $(grep -c '^FAIL' "$R/tests.txt") failed; $(grep -c Traceback "$R/tests.txt") tracebacks"
grep -q "^FAIL" "$R/tests.txt" && { note "the plugin's tests failed: no GPU run"; exit 1; }
( cd "$W/src" && "$PY" gpu/vllm/quant_layers.py --selftest ) > "$R/selftest.txt" 2>&1; LOK=$?
note "quant_layers.py --selftest: exit $LOK ($(tail -1 "$R/selftest.txt"))"

note "check_sc.py on the GPU"
( cd "$W/src" && timeout 1800 "$PY" gpu/scales/check_sc.py ) > "$R/check_sc.txt" 2>&1
note "check_sc: exit $? | $(grep -E ' ok|passed|Error|unpack|sc_pack' "$R/check_sc.txt" | tail -14 | tr '\n' '|')"
grep -q "check_sc: all passed" "$R/check_sc.txt" || { note "check_sc failed: no vLLM run (the kernels are not bit for bit)"; exit 1; }

FI_OFF=FlashInferFP8ScaledMMLinearKernel,FlashInferCuteDslNvFp4LinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel
[ "${FI:-}" = off ] && note "FlashInfer's FP8 and NVFP4 GEMMs off for every run, stock's and Glyd's alike: VLLM_DISABLED_KERNELS=$FI_OFF"
for line in "${MODELS[@]}"; do
  repo=${line%%|*}; rest=; case $line in *"|"*) rest=${line#*|};; esac; pflags=${rest%%|*}; lflags=; case $rest in *"|"*) lflags=${rest#*|};; esac
  s=$(slug "$repo"); f="$R/log/dl-$s.txt"
  until grep -q "^exit" "$f" 2> /dev/null; do [ "$(left)" -lt 420 ] && break; sleep 5; done
  if ! grep -q "^exit 0" "$f" 2> /dev/null; then note "$repo: not downloaded ($(tail -2 "$f" 2> /dev/null | tr '\n' ' '))"; continue; fi
  l=$(left); [ "$l" -lt 900 ] && { note "$repo: not run, $l s left"; continue; }
  if [ "${FI:-}" = off ]; then export VLLM_DISABLED_KERNELS=$FI_OFF; else unset VLLM_DISABLED_KERNELS; fi
  for try in 1 2; do
    note "$repo: quant_parts.py $pflags (at most $((l - 120)) s; $(tail -1 "$f"))${VLLM_DISABLED_KERNELS:+; VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS}"
    timeout -k 30 $((l - 120)) "$PY" "$W/src/gpu/vllm/quant_parts.py" "$repo" --out "$R/$s" $pflags > "$R/$s.txt" 2>&1
    note "$repo: exit $? | $(grep -E '^(EXACT|PACKED|SCALES|STOCK|GLYD|stock|glyd)' "$R/$s.txt" | cut -c1-500 | tr '\n' '|')"
    if [ "$try" = 1 ] && [ -z "${VLLM_DISABLED_KERNELS:-}" ] && grep -q "Ninja build failed" "$R/$s"/*.log "$R/$s"/*.json 2> /dev/null; then
      note "$repo: FlashInfer could not build a GEMM here (Ninja build failed): its runs again with VLLM_DISABLED_KERNELS=$FI_OFF"
      mkdir -p "$R/$s-flashinfer-failed" && mv "$R/$s"/* "$R/$s.txt" "$R/$s-flashinfer-failed/" 2> /dev/null
      export VLLM_DISABLED_KERNELS=$FI_OFF; l=$(left); continue
    fi
    break
  done
  if [ -n "$lflags" ] && [ "$LOK" = 0 ] && [ "$(left)" -gt 900 ]; then
    note "$repo: quant_layers.py $lflags (GLYD_SCALES=1)"
    GLYD_SCALES=1 GLYD_SC_VERIFY=1 timeout -k 30 $(( $(left) - 120 )) "$PY" "$W/src/gpu/vllm/quant_layers.py" "$repo" --out "$R/$s/layers" --cap "$W/cap/$s.pt" $lflags > "$R/$s/layers.txt" 2>&1
    note "$repo: quant_layers exit $? | $(grep -E '^(LAYERS|WEIGHTS|MOE|E2E|stock|glyd)' "$R/$s/layers.txt" | cut -c1-500 | tr '\n' '|')"
  elif [ -n "$lflags" ]; then note "$repo: quant_layers.py not run (selftest exit $LOK, $(left) s left)"; fi
  rm -rf "$W/hf/hub/models--${repo/\//--}" "$W/cap/$s.pt"
done
nvidia-smi --query-gpu=memory.used --format=csv > "$R/smi-end.txt" 2>&1
note "end"
