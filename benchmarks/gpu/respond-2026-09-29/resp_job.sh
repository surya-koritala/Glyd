#!/usr/bin/env bash
# How fast a model responds on this GPU: gpu/respond.py (time to first token and tokens a second through generate(),
# bf16 eager and compiled against Glyd's default and Glyd exact=True, each mode a process of its own), unattended, on
# an x86_64 or aarch64 host, riding along with another job's session (the GPU to itself while it runs). In ~: this and
# resp_src.tar (the tree: gpu/, bindings/python, this directory; its COMMIT). In order: the library for this GPU alone
# (build_lib.sh's flags), the models downloading meanwhile (whole, in the runs' order); then the runs, most wanted
# first, each model's four modes:
#   Qwen3-8B Glyd, bf16 compiled, bf16 eager; the big model's the same; then each one's exact.
# The big model and each run's expected seconds are the GPU's plan (RESP_PLAN, by the GPU):
#   hopper  (a GH200, an H100): Qwen3-32B. Expected from a GH200's run (tree cad1d8c, results-hopall/resp): 8B 195,
#           213, 175 and 139 s; 32B 326, 354, 268 and 236 s; the library 35 s. 32.5 minutes in all, and exact's
#           several sequences 3.2 more.
#   a100    (an A100): Qwen3-32B with 60 GB or more, else Qwen3-14B. Scaled from the GH200's (eager as host-bound
#           there, 24 tokens a second for Qwen3-8B on a Lambda A100) and the A100's bandwidth for the compiled calls.
#   a10     (an A10, and any other GPU): Qwen3-14B, whose bf16 does not fit 24 GB (recorded so) where Glyd does.
#           Scaled from an A10G's rates (Qwen3-8B eager 23-26 tokens a second, compiled 27 and 37).
# Each run's deadline leaves the later runs their expected time (and it at least its own): a run past its time takes
# it from the last ones. respond.py cuts repeats first (at least one, more while they fit --rep-budget), then a run's
# last configurations (its several-sequence rates last: exact's, whose expected time leaves them out, go first).
# results/summary.txt (resp_summary.py: each model's table) is rewritten after every run, results/respond.json holds
# every result, results/DONE is written by the exit trap however the job ends. No run starts with under a minute
# left, and none runs past RESP_END, counted from the job's start: 35 minutes, with ~/gpuenv (PyTorch with CUDA 13,
# nvcc 13, transformers); where there is none, one is made here with uv first, inside the same time.
#   bash ~/resp_job.sh
# Env: RESP_PLAN (hopper, a100, a10), RESP_RUNS (MODEL:MODE[:SECONDS] ..., by the plan), RESP_END (2100 s),
# RESP_ARGS (respond.py's options; "--reps-ttft 3 --rep-budget 12"), RESP_COOL (20: seconds at most a run waits for the
# GPU to cool), FILES (~), R (~/results), W (~/respw), HF_HOME (~/hf); HF_HUB_OFFLINE=1: the models from the cache.
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/respw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
END=${RESP_END:-2100}
MARGIN=60  # a configuration started before its deadline may end past it by its warm-up and one repeat
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date +%T) (+$(el) s) $*"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() { [ -n "${PY:-}" ] && [ -f "${D:-}/resp_summary.py" ] && "$PY" "$D/resp_summary.py" "$R" --json "$R/respond.json" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
done_() { echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
fail() { echo "FAIL: $*" | tee "$R/summary.txt"; exit 1; }

step "machine"
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE
  lscpu | grep -E "Model name|Architecture"; nproc; free -g | head -2; df -h "$HOME" | tail -1; gcc --version | head -1; date -u
  echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH  # the host's CUDA libraries never ahead of the environment's own (a Deep Learning AMI's cuDNN)
OTHERS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$OTHERS" -gt 0 ] && echo "WARNING: $OTHERS other processes on the GPU (the timings share it)" | tee "$R/others.txt"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
PLAN=${RESP_PLAN:-}
[ -n "$PLAN" ] || case "$CC" in 9.0) PLAN=hopper ;; 8.0) PLAN=a100 ;; *) PLAN=a10 ;; esac
BIG=$([ "$PLAN" != a10 ] && [ "$MIB" -ge 60000 ] && echo Qwen/Qwen3-32B || echo Qwen/Qwen3-14B)
M8=Qwen/Qwen3-8B
RUNS=${RESP_RUNS:-"$M8:glyd $M8:bf16c $M8:bf16 $BIG:glyd $BIG:bf16c $BIG:bf16 $M8:exact $BIG:exact"}
# each run's expected seconds (size in billions:mode:seconds; exact's without its 8 and 32 sequences)
case $PLAN in
  hopper) EST="8:glyd:195 8:bf16c:213 8:bf16:175 8:exact:139 32:glyd:326 32:bf16c:354 32:bf16:268 32:exact:236" ;;
  a100) EST="8:glyd:200 8:bf16c:220 8:bf16:180 8:exact:145 32:glyd:355 32:bf16c:375 32:bf16:275 32:exact:285 14:glyd:235 14:bf16c:255 14:bf16:205 14:exact:210" ;;
  *) EST="8:glyd:230 8:bf16c:250 8:bf16:200 8:exact:240 14:glyd:365 14:bf16c:40 14:bf16:40 14:exact:355" ;;
esac
secs() {  # RUN: its seconds, given (MODEL:MODE:SECONDS) or the plan's (else 240)
  local r=$1 m s e; [ "${r//[^:]/}" = "::" ] && { echo "${r##*:}"; return; }
  m=${r%%:*}; s=$(basename "$m" | sed -E 's/^Qwen3-([0-9.]+)B.*/\1/')
  for e in $EST; do [ "${e%:*}" = "$s:${r#*:}" ] && { echo "${e##*:}"; return; }; done
  echo 240
}
MODELS=$(for r in $RUNS; do echo "${r%%:*}"; done | awk '!s[$0]++' | tr '\n' ' ')

step "environment"
envok() {  # PYTHON: the packages and a GPU; nvcc
  "$1" -c "import torch, transformers, accelerate, safetensors, huggingface_hub; assert torch.cuda.is_available(), 'no GPU for PyTorch'" && command -v nvcc
}
mkenv() {  # $E: uv, Python 3.12, PyTorch 2.14.0 (CUDA 13), transformers 5.17.0 and the rest, nvcc of PyTorch's CUDA; E/cuda.sh
  set -e
  local a uv=$W/uv/uv py=$E/bin/python cu mm
  a=$(uname -m); mkdir -p "$W/uv"
  [ -x "$uv" ] || curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$a-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"
  "$uv" venv -q --python 3.12 "$E"
  "$uv" pip install -q --python "$py" "torch==2.14.0" "transformers==5.17.0" accelerate safetensors numpy huggingface_hub hf_transfer hf_xet ninja
  mm=$("$py" -c "import torch; print(torch.version.cuda)")
  "$uv" pip install -q --python "$py" "nvidia-cuda-nvcc==$mm.*" "nvidia-cuda-cccl==$mm.*" "nvidia-cuda-crt==$mm.*" "nvidia-nvvm==$mm.*" "nvidia-cuda-runtime==$mm.*"
  cu=$("$py" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$mm'.split('.')[0]))")
  mkdir -p "$cu/lib64"
  ln -sf "../lib/$(ls "$cu/lib" | grep -m1 '^libcudart.so')" "$cu/lib64/libcudart.so"
  printf 'export CUDA_HOME=%s\nexport PATH=%s/bin:%s/bin:$PATH\n' "$cu" "$E" "$cu" > "$E/cuda.sh"
}
PY=""
if [ -f "$HOME/gpuenv/cuda.sh" ] && ( source "$HOME/gpuenv/cuda.sh" && envok python ) > "$R/log/env-gpuenv.txt" 2>&1; then
  source "$HOME/gpuenv/cuda.sh"; PY=python; ENVN="~/gpuenv"
elif E=$([ -e "$HOME/gpuenv" ] && echo "$W/env" || echo "$HOME/gpuenv") && ( timeout "$(tmo 600)" bash -c "$(declare -f mkenv); W='$W' E='$E' mkenv" ) > "$R/log/env-make.txt" 2>&1 &&
     source "$E/cuda.sh" && envok python >> "$R/log/env-make.txt" 2>&1; then
  PY=python; ENVN="made here in $E ($(uname -m))"
else
  fail "no environment: $(tail -3 "$R/log/env-make.txt" 2> /dev/null | tr '\n' ' ')"
fi
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1; }
IDLE=$(( $(temp) + 5 ))
cool() {  # each run from about the GPU's idle temperature (at most RESP_COOL s' wait): at its power cap a GPU slows as
  local i  # it heats (an L4, the same prompt: 739 ms at 66 C, 794 ms at 82 C), and the runs go one after another
  for i in $(seq 0 $(( ${RESP_COOL:-20} / 2 ))); do [ "$(temp)" -le "$IDLE" ] && break; sleep 2; done
  echo "the GPU at $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | head -1) after $((i * 2)) s (idle $IDLE C)"
}
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/resp_src.tar" || fail "no resp_src.tar in $FILES"
D=$W/src/benchmarks/gpu/respond-2026-09-29
echo "$NAME ($CC, sm_$ARCH, $MIB MiB), $(uname -m) host, $(nproc) CPUs; the tree $(cat "$W/src/COMMIT"); plan $PLAN; runs (expected s): $(for r in $RUNS; do printf '%s ' "${r%%:*}:$(echo "${r#*:}" | cut -d: -f1):$(secs "$r")"; done)" | tee "$R/machine-short.txt"
done_ "machine and environment ($ENVN)"

step "in the background, whole, in turn: $MODELS"
dl() {  # REPO: W/NAME.dir, its snapshot's directory
  local n; n=$(basename "$1")
  timeout "$(tmo 1100)" "$PY" - "$1" > "$W/$n.dir" 2> "$R/log/dl-$n.txt" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
PY
  echo "exit $? at +$(el) s" >> "$R/log/dl-$n.txt"
}
( for m in $MODELS; do dl "$m"; done ) &
got() {  # REPO: its directory once its download is done (nothing if it failed, or past the budget)
  local n d; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge $(( END - MARGIN - 60 )) ] && return 1; sleep 3; done
  d=$(tail -1 "$W/$n.dir" 2> /dev/null); [ -f "$d/config.json" ] && echo "$d"
}

step "the library for sm_$ARCH (build_lib.sh's flags)"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
mkdir -p "$W/src/lib"
{ timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$W/src/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
  timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/src/lib/libglyd_gpu_cuda$MAJOR.so" "$W/src/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
echo "exit $?" >> "$R/log/build.txt"
[ -f "$W/src/lib/libglyd_gpu_cuda$MAJOR.so" ] || fail "the library did not build: $(grep -m3 -i error "$R/log/build.txt" | tr '\n' ' ')"
export GLYD_GPU_LIB=$W/src/lib/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src/bindings/python
done_ "library built"

set -- $RUNS
while [ $# -gt 0 ]; do
  r=$1; shift; m=${r%%:*} mode=$(echo "${r#*:}" | cut -d: -f1) n=$(basename "${r%%:*}")
  step "$n, $mode (waiting for its download)"
  d=$(got "$m") || { done_ "$n $mode: no model ($(tail -2 "$R/log/dl-$n.txt" 2> /dev/null | tr '\n' ' '))"; continue; }
  cool
  later=0; for x in "$@"; do later=$(( later + $(secs "$x") )); done
  left=$(( END - MARGIN - $(el) )) own=$(secs "$r")
  win=$(( left - later )); [ "$win" -lt "$own" ] && win=$own; [ "$win" -gt "$left" ] && win=$left  # the later runs' time kept, at least its own
  [ "$win" -ge 60 ] || { done_ "$n $mode: under a minute left, not run"; continue; }
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms 1000 > "$R/smi-$n-$mode.csv" 2> /dev/null & S=$!
  ( cd "$W/src/gpu" && timeout "$(tmo $(( win + MARGIN )))" "$PY" -u respond.py "$d" --mode "$mode" --out "$R/$n-$mode.json" --deadline $(( $(date +%s) + win )) ${RESP_ARGS:---reps-ttft 3 --rep-budget 12} ) > "$R/log/$n-$mode.txt" 2>&1
  e=$?; kill $S 2> /dev/null
  echo "$n $mode: exit $e"; tail -n 12 "$R/log/$n-$mode.txt" | grep -v "^\s*$" | tail -n 10
  done_ "$n $mode (exit $e, its time $win s, expected $own s)"
done
step "done in $(el) s"
done_ "done"
