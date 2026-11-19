#!/usr/bin/env bash
# quant-16bit-parts: gpu/vllm/quant_parts.py on each quantized 2026 checkpoint of QP_MODELS, on one Runpod pod, unattended (rp_run.sh uploads this,
# qp_src.tar and qpenv, pulls ~/results, deletes the pod). qp_src.tar: glyd-gpu-py/ (the branch's glyd_gpu), gpu/glyd_gpu.cu and .h, gpu/vllm/quant_parts.py, COMMIT.
# qpenv: QP_MODELS ("repo[@tag][|quant_parts.py flags]" a line; @tag: the same checkpoint again with other flags, its results in repo@tag's directory), END (seconds the job may take).
#   0. the checkpoints downloaded one after another in the background (huggingface_hub, a venv of its own; public repos, no token)
#   1. vLLM 0.30.0 (uv), nvcc from pip, the library built from glyd_gpu.cu for this GPU alone
#   2. glyd_gpu's plugin tests against the real vLLM (no GPU needed): test_vllm_quant.py, test_vllm_fp8.py, test_vllm.py
#   3. each checkpoint once downloaded: quant_parts.py (stock eager and Glyd exact eager bit for bit; stock and Glyd compiled: weights, KV
#      cache, 1 and 32 users' tokens a second; what Glyd packed), its directory deleted after it (disk)
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp"
[ -f "$HOME/qpenv" ] && . "$HOME/qpenv"
T0=$(date +%s); END=${END:-8100}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN  # (public models: no token read, none sent)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/qp_src.tar" || { note "no qp_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
slug() { echo "$1" | tr '/' '_'; }
mapfile -t MODELS < <(printf '%s\n' "${QP_MODELS:-}" | sed '/^\s*$/d')

note "downloads (background): ${#MODELS[@]} checkpoints"
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet > "$R/log/dl-env.txt" 2>&1
  for line in "${MODELS[@]}"; do
    spec=${line%%|*}; repo=${spec%%@*}; s=$(slug "$spec"); t=$(date +%s)
    "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken']))" "$repo" > "$R/log/dl-$s.txt" 2>&1
    echo "exit $? after $(( $(date +%s) - t )) s; $(du -sh "$W/hf/hub/models--${repo/\//--}" 2> /dev/null | cut -f1)" >> "$R/log/dl-$s.txt"
  done ) &

note "vLLM 0.30.0, nvcc, the library (sm_$ARCH)"
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-install.txt" 2>&1 || { note "vLLM did not install"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so; mkdir -p "$W/lib"
{ nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1 || { note "the library did not build (log/build.txt)"; tail -5 "$R/log/build.txt"; exit 1; }
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt")"
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache

( cd "$W/src/glyd-gpu-py/tests" && for t in test_vllm_quant.py test_vllm_fp8.py test_vllm.py; do GLYD_REQUIRE_VLLM=1 "$PY" "$t" || echo "FAIL $t exit $?"; done ) > "$R/tests.txt" 2>&1
note "plugin tests: $(grep -c ' ok$' "$R/tests.txt") ok; $(grep -c '^FAIL' "$R/tests.txt") failed; $(grep -c Traceback "$R/tests.txt") tracebacks"
grep -q "^FAIL test_vllm_quant" "$R/tests.txt" && { note "test_vllm_quant.py failed: no GPU run (the pod's time is not spent on a plugin that fails its own checks)"; exit 1; }

for i in "${!MODELS[@]}"; do
  line=${MODELS[$i]}; spec=${line%%|*}; repo=${spec%%@*}; flags=""; [ "$line" != "$spec" ] && flags=${line#*|}; s=$(slug "$spec"); f="$R/log/dl-$s.txt"
  until grep -q "^exit" "$f" 2> /dev/null; do [ "$(left)" -lt 420 ] && break; sleep 5; done
  if ! grep -q "^exit 0" "$f" 2> /dev/null; then note "$repo: not downloaded ($(tail -2 "$f" 2> /dev/null | tr '\n' ' '))"; continue; fi
  l=$(left); [ "$l" -lt 600 ] && { note "$repo: not run, $l s left"; continue; }
  note "$repo: quant_parts.py $flags (at most $((l - 120)) s; $(tail -1 "$f"))"
  timeout -k 30 $((l - 120)) "$PY" "$W/src/gpu/vllm/quant_parts.py" "$repo" --out "$R/$s" $flags > "$R/$s.txt" 2>&1
  note "$repo: exit $? | $(grep -E '^(EXACT|PACKED|STOCK|stock|glyd)' "$R/$s.txt" | cut -c1-400 | tr '\n' '|')"
  printf '%s\n' "${MODELS[@]:$((i + 1))}" | sed 's/[@|].*//' | grep -qxF "$repo" || rm -rf "$W/hf/hub/models--${repo/\//--}"  # (kept for a later line of it)
  if grep -q '"error"' "$R/$s/glyd.json" 2> /dev/null && grep -q '"error"' "$R/$s/glyd-exact.json" 2> /dev/null; then
    note "$repo: neither of Glyd's runs started: the rest is not run ($(grep -o '"error": "[^"]*' "$R/$s/glyd.json" | cut -c1-300))"; break
  fi
done
nvidia-smi --query-gpu=memory.used --format=csv > "$R/smi-end.txt" 2>&1
note "end"
