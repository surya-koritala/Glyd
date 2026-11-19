#!/usr/bin/env bash
# moe-experts, one L40S (Ada, 48 GB; the launcher refuses data-center GPUs), unattended (rp_run.sh uploads this and moex_src.tar, pulls ~/results, deletes the pod).
# moex_src.tar: pr1/glyd-gpu-py (the clamp fix, PR #27), pr2/{glyd-gpu-py,gpu} (the activations, PR #29), COMMIT. The library: main's glyd_gpu.cu (both
# branches leave it as it is), built here for this GPU alone.
#   0. slices in the background (make_slice.py: real weights by HTTP range, the model's first layers or the layers picked, renumbered):
#      inclusionAI/Ling-3.0-flash layer 35 (MLA, its routed experts SiLU with swiglu_limit 4), nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 layers 0-6
#      (3 MoE layers, relu2 without a gate), google/gemma-4-26B-A4B-it layers 0-5 (GELU tanh), stepfun-ai/Step-3.5-Flash layers 0 and 43 (43: swiglustep),
#      MiniMaxAI/MiniMax-M3 layers 0-3 (3: swigluoai)
#   1. the plugin's tests (no GPU needed) with each tree, against the real vLLM 0.30.0
#   2. PR1 on Ling: check_vllm.py --brief (fused: the clamped experts kept bf16; exact eager: bf16 eager's bits)
#   3. PR2 on each (GLYD_MOE_ACTS=all: a development build packs the activations not yet measured): check_vllm.py --brief (weights, greedy agreement, every pack and product checked, exact eager bit for bit) and moe_acts.py --no-tps
#      (a MoE layer's routes against bf16's by tokens a step)
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s); END=${END:-5700}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }
step() { local name=$1 secs=$2; shift 2; local l; l=$(left)
    [ "$l" -lt 240 ] && { note "skipped $name: $l s left"; return 99; }
    [ "$secs" -gt $((l - 60)) ] && secs=$((l - 60))
    note "$name (at most $secs s)"; local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$R/$name.out" 2> "$R/$name.err"; local rc=$?
    note "$name: exit $rc after $((SECONDS - t)) s; $(grep -c '^PASS' "$R/$name.out") PASS, $(grep -c '^FAIL' "$R/$name.out") FAIL; $(grep -E '^(FAIL|RULE)' "$R/$name.out" | cut -c1-200 | tr '\n' '|')"
    return $rc; }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN  # (public models: no token read, none sent)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/moex_src.tar" || { note "no moex_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT"
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }

note "slices (background)"
SL=$W/src/pr2/gpu/fp8/make_slice.py
sl() { ( python3 "$SL" "$2" "$W/$1" "${@:3}" > "$R/log/dl-$1.txt" 2>&1; echo "exit $?" >> "$R/log/dl-$1.txt" ) & }
sl ling inclusionAI/Ling-3.0-flash --pick 35 --max-gb 12
sl nemotron nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --layers 7 --max-gb 14
sl gemma google/gemma-4-26B-A4B-it --layers 6 --max-gb 14
sl step stepfun-ai/Step-3.5-Flash --pick 0,43 --max-gb 16
sl minimax MiniMaxAI/MiniMax-M3 ${MINIMAX_SLICE:---layers 4 --max-gb 25}  # (a 24 GB card: MINIMAX_SLICE="--pick 3 --max-gb 12", layer 3's swigluoai alone)

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
{ nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/pr2/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1 || { note "the library did not build (log/build.txt)"; tail -5 "$R/log/build.txt"; exit 1; }
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt")"
export GLYD_GPU_LIB=$LIB VLLM_CACHE_ROOT=$W/vllmcache

for t in pr1 pr2; do
  ( cd "$W/src/$t/glyd-gpu-py/tests" && PYTHONPATH=$W/src/$t/glyd-gpu-py GLYD_REQUIRE_VLLM=1 "$PY" test_vllm.py ) > "$R/tests-$t.txt" 2>&1
  note "tests $t: exit $? ($(grep -c ' ok$' "$R/tests-$t.txt") ok; $(grep -c Traceback "$R/tests-$t.txt") tracebacks)"
done

ready() {  # NAME: its slice made and checked; its directory on stdout
  local f="$R/log/dl-$1.txt"
  until grep -q "^exit" "$f" 2> /dev/null; do [ "$(left)" -lt 300 ] && return 1; sleep 5; done
  grep -q "^exit 0" "$f" && echo "$W/$1"
}
# check_vllm.py --brief (chk) and moe_acts.py --no-tps (acts) on one slice with one tree: NAME TREE DIR UTIL (LAYER, XSCALE: the caller's environment;
# UTIL_ALL, if set, takes the place of every UTIL: 0.85 on a 24 GB card)
kw() { python3 -c "import json,sys; print(json.dumps({'gpu_memory_utilization': float(sys.argv[1]), 'max_model_len': 2048, **({'language_model_only': True} if sys.argv[2] in ('gemma', 'minimax') else {})}))" "$1" "$2"; }
chk() { set -- "$1" "$2" "$3" "${UTIL_ALL:-$4}"; ( cd "$W/src/pr2" && export PYTHONPATH=$W/src/$2/glyd-gpu-py GLYD_MOE_ACTS=$([ "$2" = pr2 ] && echo all) CHECK_VLLM_LLM_KW=$(kw "$4" "${1%%-*}") && step "check-$1" "${CHECK_SECS:-1200}" "$PY" gpu/vllm/check_vllm.py --brief --out "$R/check-$1" "$3" ); }
acts() { set -- "$1" "$2" "$3" "${UTIL_ALL:-$4}"; ( cd "$W/src/pr2" && export PYTHONPATH=$W/src/$2/glyd-gpu-py GLYD_MOE_ACTS=$([ "$2" = pr2 ] && echo all) UTIL=$4 MOE_ACTS_LLM_KW=$(kw 0 "${1%%-*}" | python3 -c "import json,sys; d=json.load(sys.stdin); d.pop('gpu_memory_utilization'); print(json.dumps(d))") && step "acts-$1" "${ACTS_SECS:-900}" "$PY" gpu/vllm/moe_acts.py "$R/acts-$1" "$3" --no-tps ); }

# the clamp: Ling 3.0's layer 35 (limit 4). PR1 (fused keeps its experts bf16; exact: bf16's bits), then PR2 (packed, vLLM's clamp on the products' sums)
if dir=$(ready ling); then chk ling-pr1 pr1 "$dir" 0.35; chk ling-pr2 pr2 "$dir" 0.35; XSCALE=1 acts ling-pr2 pr2 "$dir" 0.35; else note "ling: no slice ($(tail -2 "$R/log/dl-ling.txt" | tr '\n' ' '))"; fi
# relu2 without a gate (Nemotron-H), GELU tanh (Gemma 4), swiglustep (Step 3.5's layer 43, the slice's last), swigluoai (MiniMax-M3)
if dir=$(ready nemotron); then chk nemotron pr2 "$dir" 0.4; acts nemotron pr2 "$dir" 0.4; else note "nemotron: no slice ($(tail -2 "$R/log/dl-nemotron.txt" | tr '\n' ' '))"; fi
if dir=$(ready gemma); then chk gemma pr2 "$dir" 0.4; acts gemma pr2 "$dir" 0.4; else note "gemma: no slice ($(tail -2 "$R/log/dl-gemma.txt" | tr '\n' ' '))"; fi
if dir=$(ready step); then chk step pr2 "$dir" 0.45; LAYER=-1 XSCALE=1 acts step pr2 "$dir" 0.45; else note "step: no slice ($(tail -2 "$R/log/dl-step.txt" | tr '\n' ' '))"; fi
if dir=$(ready minimax); then chk minimax pr2 "$dir" 0.6; XSCALE=1 acts minimax pr2 "$dir" 0.6; else note "minimax: no slice ($(tail -2 "$R/log/dl-minimax.txt" | tr '\n' ' '))"; fi
note "end"
