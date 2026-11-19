#!/usr/bin/env bash
# INT8 allocator candidate: an INT8 W8A8 checkpoint's codes held packed (the option codes, vllm_codes.py: signed bytes in the FP8 layout) on one Runpod pod, unattended
# (rp_run.sh uploads this and i8_src.tar, pulls ~/results, deletes the pod). i8_src.tar: git archive of the branch (glyd-gpu-py/, gpu/), COMMIT.
#   0. the checkpoint downloaded in the background (huggingface_hub, a venv of its own; a public repo, no token)
#   1. vLLM 0.30.0 (uv), nvcc from pip, the library built from glyd_gpu.cu for this GPU alone
#   2. the plugin's tests against the real vLLM (test_vllm_quant.py, test_vllm_codes.py) and test_fp8ref.py: they must pass; then gpu/codes/check_i8.py on the
#      GPU (kernels.i8_pack and the f8 unpack against fp8ref.py bit for bit, the real samples by HTTP range, a CUDA graph, the unpack's speed): nothing else
#      runs unless it passes
#   3. quant_parts.py --codes: stock eager; Glyd exact eager with the codes (every pack verified at load; tokens, logprobs, prompt logprobs and an image chat
#      against stock eager's); stock; Glyd without the codes; Glyd with them (weights, KV cache, 1 and 32 users' output tokens a second)
set -u
MODEL=RedHatAI/Qwen3.5-9B-quantized.w8a8
MODEL_REVISION=4257ccf0e5c6f87246fd96ea2e4fb16817101558
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s); END=${END:-4200}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GLYD_SCALES GLYD_CODES GLYD_KV GLYD_EXACT GLYD_FP8 GLYD_FP8_VALUES GLYD_F8V_FUSED_MAX GLYD_I4_FORCE VLLM_DISABLED_KERNELS  # (a public model: no token read, none sent)
export UV_HTTP_TIMEOUT=180 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/i8_src.tar" || { note "no i8_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT"
mkdir -p "$W/baseline" && tar -C "$W/baseline" -xf "$HOME/i8_baseline.tar" || { note "no i8_baseline.tar"; exit 1; }
cp "$W/baseline/COMMIT" "$R/BASE_COMMIT"
for f in gpu/glyd_gpu.cu gpu/glyd_gpu.h gpu/route_tables.h; do
  cmp "$W/src/$f" "$W/baseline/$f" || { note "GPU source differs: $f"; exit 1; }
done
cp "$HOME/model-meta.json" "$R/model-meta.json"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/download/0.12.21/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
S=$(echo "$MODEL" | tr '/' '_')

note "download (background): $MODEL"
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet > "$R/log/dl-env.txt" 2>&1
  t=$(date +%s)
  "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; p=snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken']); open(sys.argv[3], 'w').write(p); print(p)" "$MODEL" "$MODEL_REVISION" "$R/model-path.txt" > "$R/log/dl.txt" 2>&1
  echo "exit $? after $(( $(date +%s) - t )) s; $(du -sh "$W/hf/hub/models--${MODEL/\//--}" 2> /dev/null | cut -f1)" >> "$R/log/dl.txt" ) &

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
grep -A2 -E "entry function '_ZN3f8c16f8_unpack_kernel" "$R/log/build.txt" | grep -E "registers|spill" > "$R/ptxas-f8-unpack.txt"
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt"); library $(grep real "$R/log/build.txt"); f8 unpack $(tr '\n' ' ' < "$R/ptxas-f8-unpack.txt")"
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache GLYD_I4_SAMPLES=$W/samples

( cd "$W/src/glyd-gpu-py/tests" && for t in test_vllm_quant.py test_vllm_codes.py test_fp8ref.py; do GLYD_REQUIRE_VLLM=1 "$PY" "$t" || echo "FAIL $t exit $?"; done ) > "$R/tests.txt" 2>&1
note "tests: $(grep -c ' ok' "$R/tests.txt") ok; $(grep -c '^FAIL' "$R/tests.txt") failed; $(grep -c Traceback "$R/tests.txt") tracebacks"
grep -q "^FAIL" "$R/tests.txt" && { note "the tests failed: no GPU run"; exit 1; }

note "check_i8.py on the GPU"
( cd "$W/src" && timeout 1800 "$PY" gpu/codes/check_i8.py ) > "$R/check_i8.txt" 2>&1
note "check_i8: exit $? | $(grep -E 'samples:|speed|graph|passed|Error' "$R/check_i8.txt" | tail -12 | tr '\n' '|')"
grep -q "check_i8: all passed" "$R/check_i8.txt" || { note "check_i8 failed: no vLLM run (the kernel or the pack is not bit for bit)"; exit 1; }

f="$R/log/dl.txt"
until grep -q "^exit" "$f" 2> /dev/null; do [ "$(left)" -lt 600 ] && break; sleep 5; done
grep -q "^exit 0" "$f" 2> /dev/null || { note "$MODEL: not downloaded ($(tail -2 "$f" 2> /dev/null | tr '\n' ' '))"; exit 1; }
MODEL_PATH=$(cat "$R/model-path.txt")
for variant in baseline candidate; do
  l=$(left)
  [ "$l" -ge 1500 ] || { note "not enough time for $variant; no partial validation accepted"; exit 1; }
  export PYTHONPATH="$W/$([ "$variant" = baseline ] && echo baseline || echo src)/glyd-gpu-py"
  note "$variant: full INT8 comparison with fixed 64-sequence limit (at most $((l - 120)) s)"
  timeout -k 30 $((l - 120)) "$PY" "$W/src/gpu/vllm/quant_parts.py" "$MODEL_PATH" --out "$R/$variant" --codes --max-num-seqs 64 --mml 8192 --steps stock-eager,glyd-exact,stock,glyd,glyd-b,stock-b,glyd-16 > "$R/$variant.txt" 2>&1
  rc=$?
  note "$variant: exit $rc | $(grep -E '^(EXACT|PACKED|CODES|STOCK|GLYD|stock|glyd)' "$R/$variant.txt" | cut -c1-700 | tr '\n' '|')"
  [ "$rc" = 0 ] || exit 1
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("exact",{}).get("equal") is True, "exact comparison failed"; assert d.get("exact",{}).get("codes",{}).get("verified")==80, "expected all 80 verified INT8 packs"; assert d.get("exact",{}).get("verified")==159, "expected all 159 verified BF16 packs"' "$R/$variant/summary.json" || exit 1
done
nvidia-smi --query-gpu=memory.used --format=csv > "$R/smi-end.txt" 2>&1
note "end"
