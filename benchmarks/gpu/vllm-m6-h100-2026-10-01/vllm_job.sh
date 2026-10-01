#!/usr/bin/env bash
# vllm serve --quantization glyd against vLLM's own bf16 (M3, M4): unattended, on an x86_64 or aarch64 host whose NVIDIA
# driver runs CUDA 13 (580 or newer), 30 minutes at most; over VJ_TP GPUs (tensor parallel), 1 by default. In ~: this and vllm_src.tar (the vllm-plugin branch's
# tree with its COMMIT: gpu/ for the library, bindings/python for the glyd package and its vLLM plugin, gpu/vllm/ for the
# checks and the bench). Its steps, in order, none started past VJ_BUDGET:
#   env    uv; a venv with vLLM (VJ_VLLM, from PyPI, timed) and pip's nvcc for torch's CUDA; the glyd package from the
#          tree (editable, no dependencies). The models download meanwhile in a venv of their own, one after another,
#          each timed: VJ_MODEL (for check, bench, profile), then the mixtures of experts (moe, moebench), then the big
#          model (big)
#   build  the library for this GPU alone (build_lib.sh's flags)
#   check  gpu/vllm/check_vllm.py --quick on VJ_MODEL: every pack decoded to its weights, every layer against F.linear,
#          tokens and logprobs against vLLM's bf16 (fused, both layouts, against bf16's own eager-against-graphs floor),
#          a compile cache each for bf16 and the two layouts; exact mode eager bit for bit, and compiled refused but with
#          inductor's deterministic mode, where compiled bf16's bit for bit; fused compiled in that mode the same bits
#          from one run to the next
#   moe    check_vllm.py on the mixtures of experts (VJ_MOE): granite-3.1-3b-a800m-instruct (--quick: every check),
#          then Qwen3-30B-A3B where its bf16 fits the GPUs (--brief: bf16, Glyd auto with its layers, exact eager)
#   bench  gpu/vllm/bench_serve.sh on VJ_MODEL: bf16, then Glyd (the layout best_layout picks for this GPU), each
#          server started first on an empty compile cache (cold: its KV cache noted) and then again on it (warm: its
#          graphs loaded, measured); vllm bench serve at VJ_RATES requests a second (VJ_PROMPTS prompts, 1,024 tokens in,
#          256 out); the same gpu_memory_utilization, VJ_UTIL
#   big    the same bench on the big model (VJ_BIG_PROMPTS): Qwen3-32B where the GPU has 70 GiB or more; Qwen3-14B
#          from 35 GiB (a 40 GB A100: Qwen3-32B does not fit there packed either, about 42 GiB of weights tiered and 47
#          12-bit); none below. A mode whose server does not start (bf16 that does not fit) is logged and skipped.
#   moebench  the same bench on the big mixture of experts (VJ_PROMPTS), Qwen3-30B-A3B where its bf16 fits the GPUs,
#          else granite-3.1-3b-a800m-instruct
#   moeroutes  gpu/vllm/moe_routes.py on the big mixture of experts (as moebench's), one GPU: a mixture of experts' layer
#          by tokens a step, the library's grouped products against the routed experts decoded for vLLM's Triton kernel,
#          and bf16's own layer (the threshold GLYD_MOE_DECODE_MIN routes by)
#   profile  gpu/vllm/profile_steps.py on VJ_MODEL, one GPU, bf16 then Glyd: decode steps of 1-256 sequences and prompt
#          steps of 512-8192 tokens under torch's profiler, the kernels' GPU time a step by kind (Glyd's, GEMMs,
#          attention, the rest): the linear layers' share by M
# results/summary.txt rewritten after every step; results/DONE from the exit trap however the job ends. Every step's
# timeout ends by VJ_END.
#   bash ~/vllm_job.sh                            (every step)
#   VJ_STEPS=check,bench bash ~/vllm_job.sh       (some of them)
# Env: VJ_STEPS (check,bench,big), VJ_TP (1), VJ_BUDGET (1560 s), VJ_END (1740 s), VJ_MODEL (Qwen/Qwen3-8B), VJ_BIG
# (auto, none or a model), VJ_MOE (auto, or the mixtures of experts to check), VJ_RATES ("1 4 inf"), VJ_PROMPTS
# ("64 128 256"), VJ_BIG_PROMPTS ("32 64 128"), VJ_UTIL (0.9), VJ_VLLM (vllm==0.30.0); FILES (~), R (~/results), W
# (~/vllmw), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/vllmw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W/tmp"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${VJ_BUDGET:-1560}
END=${VJ_END:-1740}
STEPS=${VJ_STEPS:-check,bench,big}
MODEL=${VJ_MODEL:-Qwen/Qwen3-8B}
TP=${VJ_TP:-1}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
want() { case ",$STEPS," in *",$1,"*) return 0 ;; esac; return 1; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() {  # results/summary.txt from what is there
  { cat "$R/machine-short.txt" 2> /dev/null; cat "$R/steps.txt" 2> /dev/null; echo
    for f in "$R"/check-*/report.txt "$R"/bench-*/summary.txt; do
      [ -f "$f" ] || continue
      echo "== $(basename "$(dirname "$f")")"; cat "$f"; echo
    done
    for f in "$R"/moeroutes/*.txt; do
      [ -f "$f" ] && { echo "== moeroutes $(basename "$f" .txt)"; grep "^{'T'" "$f"; echo; }
    done
    for f in "$R"/profile/*.txt; do
      [ -f "$f" ] && { echo "== profile $(basename "$f" .txt)"; grep "^|" "$f"; echo; }
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
NG=$(nvidia-smi -L | grep -c "^GPU")
[ "$NG" -ge "$TP" ] || fail "VJ_TP $TP, but $NG GPUs here"
# The mixtures of experts: granite's (small) everywhere; Qwen3-30B-A3B where its bf16 (57 GiB) and a KV cache fit the
# GPUs at 0.85 (75,000 MiB across them); the bench's the bigger of the two
MOE=${VJ_MOE:-auto}
[ "$MOE" = auto ] && MOE="ibm-granite/granite-3.1-3b-a800m-instruct$([ $(( MIB * TP )) -ge 75000 ] && echo " Qwen/Qwen3-30B-A3B")"
MOEBENCH=$(echo $MOE | awk '{print $NF}')
{ want moe || want moebench || want moeroutes; } || MOE=""
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
BIG=${VJ_BIG:-auto}
[ "$BIG" = auto ] && BIG=$([ "$MIB" -ge 71680 ] && echo Qwen/Qwen3-32B || { [ "$MIB" -ge 35840 ] && echo Qwen/Qwen3-14B || echo none; })
want big || BIG=none
GB=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
# the models, the venv and its cache, GB: Qwen3-8B's 16 and the venv's 14, granite's 7, Qwen3-30B-A3B's 61, the big one's
NEED=$(( 30 + $(case "$MOE" in (*30B*) echo 70 ;; ("") echo 0 ;; (*) echo 10 ;; esac) + $(case $BIG in (*32B) echo 80 ;; (*14B) echo 30 ;; (*) echo 0 ;; esac) ))
if [ "$GB" -lt "$NEED" ]; then
  echo "WARNING: $GB GB free in ~, $NEED wanted: the big model ($BIG) and Qwen3-30B-A3B skipped"
  BIG=none; MOE=$(echo $MOE | tr ' ' '\n' | grep -v 30B | tr '\n' ' '); MOEBENCH=$(echo $MOE | awk '{print $NF}')
fi
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/vllm_src.tar" || fail "no vllm_src.tar in $FILES"
echo "$NG x $NAME (compute $CC, sm_$ARCH, $MIB MiB), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); steps $STEPS; TP $TP; model $MODEL, big $BIG, experts ${MOE:-none}" | tee "$R/machine-short.txt"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ] && echo "WARNING: other processes on the GPU" | tee -a "$R/machine-short.txt"
done_ "machine"

step "uv, then the downloads (in the background, one after another, each timed)"
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
( { want check || want bench || want profile; } && dl "$MODEL"; for m in $MOE; do dl "$m"; done; [ "$BIG" != none ] && dl "$BIG" ) &
got() {  # REPO: 0 once its download is done and whole (1 if it failed, or past the budget)
  local n; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/dl-$n.txt" && [ -f "$(tail -1 "$W/$n.dir")/config.json" ]
}

step "env: vLLM from PyPI (${VJ_VLLM:-vllm==0.30.0}) and nvcc, then the glyd package"
t=$(date +%s)
if [ -x "$W/venv/bin/python" ] && "$W/venv/bin/python" -c "import vllm" 2> /dev/null; then
  echo "an earlier job's venv, $W/venv" > "$R/log/env-vllm.txt"  # (jobs one after another on one instance)
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
if [ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] && [ "$W/lib/libglyd_gpu_cuda$MAJOR.so" -nt "$FILES/vllm_src.tar" ]; then
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

if want check && left; then
  step "check: check_vllm.py on $MODEL (waiting for its download)"
  got "$MODEL" || fail "no $MODEL: $(tail -2 "$R/log/dl-$(basename "$MODEL").txt" 2> /dev/null | tr '\n' ' ')"
  t=$(date +%s)
  ( cd "$R" && HF_HUB_OFFLINE=1 timeout "$(tmo 1500)" "$PY" "$W/src/gpu/vllm/check_vllm.py" --quick --tp "$TP" --out "$R/check-$(basename "$MODEL")" "$MODEL" ) > "$R/check-$(basename "$MODEL").txt" 2>&1
  e=$?; done_ "check: exit $e in $(( $(date +%s) - t )) s ($(tail -1 "$R/check-$(basename "$MODEL").txt"))"
fi
if want moe; then
  for m in $MOE; do
    left || break
    step "moe: check_vllm.py on $m (waiting for its download)"
    got "$m" || { done_ "moe: no $m: $(tail -2 "$R/log/dl-$(basename "$m").txt" 2> /dev/null | tr '\n' ' ')"; continue; }
    t=$(date +%s)
    how=$(case $m in (*30B*|*32B*) echo --brief ;; (*) echo --quick ;; esac)  # (a big model: loaded 4 times, not 12)
    ( cd "$R" && HF_HUB_OFFLINE=1 timeout "$(tmo 1500)" "$PY" "$W/src/gpu/vllm/check_vllm.py" $how --tp "$TP" --out "$R/check-$(basename "$m")" "$m" ) > "$R/check-$(basename "$m").txt" 2>&1
    e=$?; done_ "moe $(basename "$m") $how: exit $e in $(( $(date +%s) - t )) s ($(tail -1 "$R/check-$(basename "$m").txt"))"
  done
fi
bench() {  # MODEL PROMPTS: bench_serve.sh, warm with the cold start noted
  local t=$(date +%s)
  rm -rf "$W/cache/bench-$(basename "$1")"  # (the first start cold, an earlier job's graphs or not)
  ( cd "$R" && HF_HUB_OFFLINE=1 VLLM="$W/venv/bin/vllm" R="$R/bench-$(basename "$1")" VLLM_CACHE_ROOT="$W/cache/bench-$(basename "$1")" WARM=1 \
      RATES="${VJ_RATES:-1 4 inf}" PROMPTS="$2" UTIL="${VJ_UTIL:-0.9}" TP="$TP" BUSYWAIT=60 COOL=60 COOLWAIT=60 \
      timeout "$(tmo 1700)" bash "$W/src/gpu/vllm/bench_serve.sh" "$1" ) > "$R/bench-$(basename "$1").txt" 2>&1
  local e=$?; done_ "bench $(basename "$1"): exit $e in $(( $(date +%s) - t )) s"
}
if want bench && left; then
  step "bench: $MODEL, bf16 then Glyd, cold then warm (waiting for its download)"
  got "$MODEL" && bench "$MODEL" "${VJ_PROMPTS:-64 128 256}" || done_ "bench: no $MODEL"
fi
if want moebench && [ -n "$MOEBENCH" ] && left; then
  step "moebench: $MOEBENCH, bf16 then Glyd, cold then warm (waiting for its download)"
  got "$MOEBENCH" && bench "$MOEBENCH" "${VJ_PROMPTS:-64 128 256}" || done_ "moebench: no $MOEBENCH"
fi
if want moeroutes && [ -n "$MOEBENCH" ] && left; then
  step "moeroutes: moe_routes.py on $MOEBENCH, one GPU, Glyd then bf16 (waiting for its download)"
  if got "$MOEBENCH"; then
    mkdir -p "$R/moeroutes"
    for m in glyd bf16; do
      t=$(date +%s)
      ( cd "$R/moeroutes" && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_MOE_DECODE_MIN=1 timeout "$(tmo 900)" "$PY" "$W/src/gpu/vllm/moe_routes.py" "$R/moeroutes/$m.json" "$MOEBENCH" $([ $m = bf16 ] && echo --bf16) ) > "$R/moeroutes/$m.txt" 2>&1
      e=$?; done_ "moeroutes $m: exit $e in $(( $(date +%s) - t )) s"
    done
  else
    done_ "moeroutes: no $MOEBENCH"
  fi
fi
if want profile && left; then
  step "profile: profile_steps.py on $MODEL, one GPU, bf16 then Glyd (waiting for its download)"
  if got "$MODEL"; then
    mkdir -p "$R/profile"
    for m in bf16 glyd; do
      t=$(date +%s)
      ( cd "$R/profile" && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT="$W/cache/profile" timeout "$(tmo 900)" "$PY" "$W/src/gpu/vllm/profile_steps.py" $m "$R/profile/$m.json" "$MODEL" ) > "$R/profile/$m.txt" 2>&1
      e=$?; done_ "profile $m: exit $e in $(( $(date +%s) - t )) s"
    done
  else
    done_ "profile: no $MODEL"
  fi
fi
if [ "$BIG" != none ] && left; then
  step "big: $BIG, bf16 then Glyd, cold then warm (waiting for its download)"
  got "$BIG" && bench "$BIG" "${VJ_BIG_PROMPTS:-32 64 128}" || done_ "big: no $BIG ($(tail -1 "$R/log/dl-$(basename "$BIG").txt" 2> /dev/null))"
fi
for f in "$R"/log/dl-*.txt; do [ -f "$f" ] && echo "download $(basename "$f" .txt): $(tail -1 "$f")"; done | grep -v "dl-env" >> "$R/steps.txt"
step "done in $(el) s"
done_ "done"
