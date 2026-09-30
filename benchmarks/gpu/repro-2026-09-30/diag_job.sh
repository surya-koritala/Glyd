#!/usr/bin/env bash
# Whether generate()'s numbers repeat on this GPU, bf16 eager against Glyd exact (diag.py), unattended, 15 minutes at
# most, on an x86_64 or aarch64 host, riding along with another job's session (the GPU to itself while it runs). In ~:
# this and diag_src.tar (the tree: gpu/, bindings/python, benchmarks/gpu/repro-2026-09-30; its COMMIT). In order: the
# library for this GPU alone (build_lib.sh's flags), Qwen3-8B downloading meanwhile (whole); then pairs of processes,
# bf16 eager and exact, most wanted first, each pair followed by diag.py --compare (bf16's run A against exact's):
#   b32       32 sequences, 128 + 32 tokens, as the GH200's "rate 32" (whose tokens differed)
#   b1        1 sequence (where the GH200's tokens were the same)
#   b32-det   32, torch.use_deterministic_algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8
#   b32-math  32, scaled_dot_product_attention held to its math backend
#   b8        8 sequences
# then bf16 alone at 32 with each other attention backend held (efficient, flash, cudnn), and each BLAS library
# (cublas, cublaslt), where time is left.
# Each process runs generate() twice (A, B) and says whether B repeats A call by call (diag.py). results/summary.txt is
# rewritten after every step, results/DONE written by the exit trap however the job ends; no step starts with under a
# minute left of DIAG_END (900 s), counted from the job's start.
#   bash ~/diag_job.sh
# Env: DIAG_END (900), DIAG_MODEL (Qwen/Qwen3-8B), DIAG_NEW (32), FILES (~), R (~/results), W (~/diagw), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/diagw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
END=${DIAG_END:-900}
MODEL=${DIAG_MODEL:-Qwen/Qwen3-8B}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date +%T) (+$(el) s) $*"; echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }
left() { [ $(( END - $(el) )) -ge 60 ] || { echo "under a minute left: skipped"; false; }; }
summ() { { cat "$R/machine-short.txt" "$R/env.txt" 2> /dev/null; echo; for f in "$R"/run-*.txt "$R"/compare-*.txt; do [ -f "$f" ] && { echo "== $(basename "$f" .txt)"; grep -v "^\s*$" "$f" | grep -v -i "loading weights\|warn\|^ *warnings\.\|UserWarning" | tail -n 8; }; done; echo; cat "$R/steps.txt"; } > "$R/summary.txt" 2> /dev/null; }
fail() { echo "FAIL: $*" | tee "$R/summary.txt"; exit 1; }

step "machine"
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  lscpu | grep -E "Model name|Architecture"; nproc; free -g | head -2; date -u; echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH
OTHERS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$OTHERS" -gt 0 ] && echo "WARNING: $OTHERS other processes on the GPU" | tee "$R/others.txt"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")

step "environment"
envok() { "$1" -c "import torch, transformers, accelerate, safetensors, huggingface_hub; assert torch.cuda.is_available(), 'no GPU for PyTorch'" && command -v nvcc; }
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
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'cuDNN', torch.backends.cudnn.version(), '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false GLYD_COMPILE=0
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/diag_src.tar" || fail "no diag_src.tar in $FILES"
D=$W/src/benchmarks/gpu/repro-2026-09-30
echo "$NAME ($CC, sm_$ARCH), $(uname -m) host, $(nproc) CPUs; the tree $(cat "$W/src/COMMIT"); $MODEL" | tee "$R/machine-short.txt"

step "in the background: $MODEL, whole"
( timeout "$(tmo 800)" "$PY" - "$MODEL" > "$W/model.dir" 2> "$R/log/dl.txt" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
PY
  echo "exit $? at +$(el) s" >> "$R/log/dl.txt" ) &

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
until grep -q "^exit" "$R/log/dl.txt" 2> /dev/null; do [ "$(el)" -ge $(( END - 120 )) ] && fail "no $MODEL by +$(el) s"; sleep 3; done
M=$(tail -1 "$W/model.dir"); [ -f "$M/config.json" ] || fail "no $MODEL: $(tail -2 "$R/log/dl.txt" | tr '\n' ' ')"
summ

run() {  # NAME MODE ARGS...: diag.py in a process of its own, RESULT run-NAME-MODE.json / .txt
  local n=$1 m=$2; shift 2
  left || return 1
  step "run $n, $m: $*"
  ( cd "$D" && timeout "$(tmo 300)" "$PY" -u diag.py "$M" --mode "$m" --new "${DIAG_NEW:-32}" --out "$R/run-$n-$m.json" "$@" ) > "$R/run-$n-$m.txt" 2>&1
  echo "exit $?" >> "$R/run-$n-$m.txt"; tail -2 "$R/run-$n-$m.txt"; summ
}
pair() {  # NAME ARGS...: bf16 and exact, then bf16's run A against exact's
  local n=$1; shift
  run "$n" bf16 "$@" && run "$n" exact "$@" || return 0
  [ -f "$R/run-$n-bf16.json" ] && [ -f "$R/run-$n-exact.json" ] && ( cd "$D" && "$PY" diag.py --compare "$R/run-$n-bf16.json" "$R/run-$n-exact.json" ) > "$R/compare-$n.txt" 2>&1
  summ
}
pair b32 --batch 32
pair b1 --batch 1
pair b32-det --batch 32 --det
pair b32-math --batch 32 --attn math
pair b8 --batch 8
for a in efficient flash cudnn; do run "b32-$a" bf16 --batch 32 --attn "$a"; done
for b in cublas cublaslt; do run "b32-$b" bf16 --batch 32 --blas "$b"; done
step "done in $(el) s"
summ
