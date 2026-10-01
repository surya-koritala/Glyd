#!/usr/bin/env bash
# Glyd's `fraction` in vLLM on one GPU: Qwen3-32B served saturated by vLLM's bf16 and by --quantization glyd packing a
# fraction of the layers (GLYD_FRACTION 1, 0.5, 0.25, 0.75, 0), unattended, on an x86_64 or aarch64 host whose NVIDIA
# driver runs CUDA 13 (580 or newer), 30 minutes at most. In ~: this and budget_src.tar (the vllm-plugin branch's tree
# with its COMMIT: gpu/ for the library, bindings/python for the glyd package and its vLLM plugin, gpu/vllm/ for the
# bench). Its steps, in order, none started past VJ_BUDGET:
#   env    uv; a venv with vLLM (VJ_VLLM, from PyPI) and pip's nvcc for torch's CUDA; the glyd package from the tree
#          (editable, no dependencies). The model downloads meanwhile in a venv of its own, timed
#   build  the library for this GPU alone (build_lib.sh's flags)
#   sweep  gpu/vllm/bench_serve.sh a mode at a time, VJ_MODES in order (bf16 first, then the fractions by what they
#          tell most), each on an empty compile cache: one vllm serve (--max-model-len 4096, the same
#          --gpu-memory-utilization, VJ_SERVE_ARGS; a cold start's KV cache is VJ_COLD_GIB smaller than a warm one's,
#          so VJ_UTIL is raised by that share of the GPU's memory), then vllm bench serve at every request sent at once (VJ_PROMPTS
#          prompts of VJ_IN tokens in, VJ_OUT out, --ignore-eos): requests a second, the first token (TTFT) and each
#          token (TPOT), the server's weights and KV cache; a mode that cannot start in what is left of the budget is
#          skipped, and noted
# results/summary.txt rewritten after every step, its last table each mode against bf16's; results/DONE from the exit
# trap however the job ends. Every step's timeout ends by VJ_END.
#   bash ~/budget_job.sh
# Env: VJ_MODEL (Qwen/Qwen3-32B), VJ_MODES ("bf16 glyd@1 glyd@0.5 glyd@0.25 glyd@0.75 glyd@0"), VJ_PROMPTS (192), VJ_IN
# (1024), VJ_OUT (256), VJ_UTIL (0.9), VJ_COLD_GIB (1.54), VJ_SERVE_ARGS (--max-num-seqs 128 --compilation-config
# {"max_cudagraph_capture_size":128}: CUDA graphs captured to the most requests the bench has in flight, which takes
# less of the start), VJ_WARM (0: every server started once, on an empty compile cache; 1: twice, the second measured),
# VJ_EST_PLAIN (200 s) and VJ_EST_PACKED (290 s): what a mode takes, for bf16 and glyd@0 and for the rest, until
# one of each has run (then the longest seen), VJ_BUDGET (1560 s), VJ_END (1740 s), VJ_VLLM (vllm==0.30.0); FILES (~),
# R (~/results), W (~/budgetw), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/budgetw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W/tmp"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${VJ_BUDGET:-1560}
END=${VJ_END:-1740}
MODEL=${VJ_MODEL:-Qwen/Qwen3-32B}
MODES=${VJ_MODES:-bf16 glyd@1 glyd@0.5 glyd@0.25 glyd@0.75 glyd@0}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() {  # results/summary.txt from what is there
  { cat "$R/machine-short.txt" 2> /dev/null; cat "$R/steps.txt" 2> /dev/null; echo
    for f in "$R"/bench-*/summary.txt; do
      [ -f "$f" ] || continue
      echo "== $(basename "$(dirname "$f")")"; cat "$f"; echo
    done
  } > "$R/summary.tmp" 2>&1; mv "$R/summary.tmp" "$R/summary.txt"
}
done_() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
fail() { done_ "FAIL: $*"; exit 1; }

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; lscpu | grep -E "Model name|Architecture"; nproc
  free -g | head -2; df -h "$HOME" | tail -1; date -u; echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH  # the host's CUDA libraries never ahead of the venv's own
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
GB=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
NEED=$(case $MODEL in (*32B) echo 95 ;; (*14B) echo 55 ;; (*) echo 35 ;; esac)  # the model and the venv, GB
[ "${GB:-0}" -ge "$NEED" ] || echo "WARNING: $GB GB free in ~, $NEED wanted for $MODEL"
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/budget_src.tar" || fail "no budget_src.tar in $FILES"
echo "$NAME (compute $CC, sm_$ARCH, $MIB MiB), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); model $MODEL; modes $MODES; ${VJ_PROMPTS:-192} prompts, ${VJ_IN:-1024} tokens in, ${VJ_OUT:-256} out" | tee "$R/machine-short.txt"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ] && echo "WARNING: other processes on the GPU" | tee -a "$R/machine-short.txt"
done_ "machine"

step "uv, then the download (in the background, timed)"
UV=$(command -v uv || ls "$HOME/tools/uv/uv" "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then
  mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv
fi
[ -x "$UV" ] || fail "no uv"
export HF_HOME=${HF_HOME:-$HOME/hf} TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=${UV_CACHE_DIR:-$W/uvcache}
[ -x "$W/dl/bin/python" ] || ( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet ) > "$R/log/dl-env.txt" 2>&1 || fail "no download env (log/dl-env.txt)"
dl() {  # REPO: its snapshot, W/NAME.dir its directory; log/dl-NAME.txt its time and size
  local n t; n=$(basename "$1"); t=$(date +%s)
  timeout 1500 "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))" "$1" > "$W/$n.dir" 2> "$R/log/dl-$n.txt"
  local e=$?; echo "exit $e: $(du -sbL "$(tail -1 "$W/$n.dir")" 2> /dev/null | cut -f1) bytes in $(( $(date +%s) - t )) s (at +$(el) s)" >> "$R/log/dl-$n.txt"
}
dl "$MODEL" &
got() {  # REPO: 0 once its download is done and whole (1 if it failed, or past the budget)
  local n; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/dl-$n.txt" && [ -f "$(tail -1 "$W/$n.dir")/config.json" ]
}

step "env: vLLM from PyPI (${VJ_VLLM:-vllm==0.30.0}) and nvcc, then the glyd package"
t=$(date +%s)
if [ -x "$W/venv/bin/python" ] && "$W/venv/bin/python" -c "import vllm" 2> /dev/null; then
  echo "an earlier job's venv, $W/venv" > "$R/log/env-vllm.txt"
else
  { rm -rf "$W/venv" && "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "${VJ_VLLM:-vllm==0.30.0}"; } > "$R/log/env-vllm.txt" 2>&1 || fail "vLLM did not install (log/env-vllm.txt)"
fi
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || fail "nvcc did not install (log/env-nvcc.txt)"
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
cp "$W/src/LICENSE" "$W/src/COPYING" "$W/src/bindings/python/"; cp "$W/src/glyd-store/LICENSE" "$W/src/bindings/python/LICENSE-glyd-store"; cp "$W/src/gpu/LICENSE" "$W/src/bindings/python/LICENSE-glyd-gpu"
"$UV" pip install --python "$PY" --no-deps -e "$W/src/bindings/python" > "$R/log/env-glyd.txt" 2>&1 || fail "the glyd package did not install (log/env-glyd.txt)"
"$PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1 || fail "no vLLM on the GPU ($(tail -2 "$R/env.txt" | tr '\n' ' '))"
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"
cat "$R/env.txt"
done_ "env: vLLM and nvcc in $(( $(date +%s) - t )) s ($(head -1 "$R/env.txt"))"

step "build: the library for sm_$ARCH"
t=$(date +%s)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
mkdir -p "$W/lib"
if [ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] && [ "$W/lib/libglyd_gpu_cuda$MAJOR.so" -nt "$FILES/budget_src.tar" ]; then
  echo "an earlier job's build of this tarball" > "$R/log/build.txt"
else
  rm -f "$W/lib/libglyd_gpu_cuda$MAJOR.so"
  { timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/libglyd_gpu_cuda$MAJOR.so" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
fi
[ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] || fail "the library did not build (log/build.txt)"
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so PYTHONSAFEPATH=1
"$PY" -c "from glyd.gpu import _lib, kernels as g; g.lib(); print('library: C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" >> "$R/env.txt" 2>&1 || fail "the library does not load ($(tail -1 "$R/env.txt"))"
done_ "build: sm_$ARCH in $(( $(date +%s) - t )) s ($(tail -1 "$R/env.txt"))"

step "sweep: $MODEL, a mode at a time: $MODES (waiting for its download)"
got "$MODEL" || fail "no $MODEL: $(tail -2 "$R/log/dl-$(basename "$MODEL").txt" 2> /dev/null | tr '\n' ' ')"
B=$R/bench-$(basename "$MODEL")
# A cold start's compile takes 1.54 GiB from the KV cache vLLM sizes (M3, Qwen3-32B on a GH200: 17.94 GiB for bf16 against a
# warm start's 19.48, 30.73 for Glyd against 32.26): every mode's utilization is raised by that share of the GPU's memory, so
# that each cold server's KV cache is what a warm one gets at VJ_UTIL. Not with VJ_WARM=1, nor where VJ_COLD_GIB=0.
UTIL=$(awk -v u="${VJ_UTIL:-0.9}" -v c="${VJ_COLD_GIB:-1.54}" -v m="$MIB" -v w="${VJ_WARM:-0}" 'BEGIN { if (w == 1) c = 0; printf "%.4f", u + c / (m / 1024) }')
echo "gpu-memory-utilization $UTIL (VJ_UTIL ${VJ_UTIL:-0.9}, a cold start's ${VJ_COLD_GIB:-1.54} GiB of compile added; $MIB MiB)" | tee -a "$R/machine-short.txt"
CG='{"max_cudagraph_capture_size":128}'  # (no spaces: bench_serve.sh word-splits SERVE_ARGS)
SERVE=${VJ_SERVE_ARGS---max-num-seqs 128 --compilation-config $CG}
MTP=${VJ_EST_PLAIN:-200}; MTG=${VJ_EST_PACKED:-290}  # what a mode takes, seconds (bf16 and glyd@0: no packs), the longest so far
for mode in $MODES; do
  case $mode in (bf16|glyd@0) MT=$MTP ;; (*) MT=$MTG ;; esac
  left=$(( BUDGET - $(el) ))
  if [ "$left" -lt "$MT" ]; then done_ "sweep $mode: not run, $left s of the budget left and a mode takes $MT s"; continue; fi
  t=$(date +%s)
  rm -rf "$W/cache/$mode"  # (each mode's first start on an empty compile cache)
  ( cd "$R" && HF_HUB_OFFLINE=1 VLLM="$W/venv/bin/vllm" R="$B" VLLM_CACHE_ROOT="$W/cache/$mode" MODES="$mode" WARM="${VJ_WARM:-0}" \
      RATES=inf PROMPTS="${VJ_PROMPTS:-192}" IN="${VJ_IN:-1024}" OUT="${VJ_OUT:-256}" UTIL="$UTIL" BUSYWAIT=60 COOL=60 COOLWAIT=60 \
      SERVE_ARGS="$SERVE" \
      timeout "$(tmo 900)" bash "$W/src/gpu/vllm/bench_serve.sh" "$MODEL" ) >> "$R/bench-$(basename "$MODEL").txt" 2>&1
  e=$?; d=$(( $(date +%s) - t ))
  python3 "$W/src/gpu/vllm/bench_summary.py" "$B" > "$B/summary.txt" 2> "$R/log/summary-err.txt"  # (every mode so far, where a timeout cut the mode's own)
  case $mode in (bf16|glyd@0) [ "$d" -gt "$MTP" ] && MTP=$d ;; (*) [ "$d" -gt "$MTG" ] && MTG=$d ;; esac
  done_ "sweep $mode: exit $e in $d s"
done
for f in "$R"/log/dl-*.txt; do [ -f "$f" ] && echo "download $(basename "$f" .txt): $(tail -1 "$f")"; done | grep -v "dl-env" >> "$R/steps.txt"
step "done in $(el) s"
done_ "done"
