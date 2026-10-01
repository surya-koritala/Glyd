#!/usr/bin/env bash
# Option 2 built (gpu-option2: the route SPLIT, C API 5, COMMIT in o2_src.tar) on one GPU of a class, CLASS = a100 or
# hopper (o2_a100.sh and o2_hopper.sh set it and the lists). In order, most needed first, so that a cap leaves the
# first answers:
#   builds    the library for this GPU alone (build_lib.sh's flags) and the JIT build, and (with o2_old_src.tar) the
#             tree before the fix's library, at once; the models download meanwhile (MODELS, whole: e2e.py loads them);
#   (a) check_capi: the JIT build against the library, the route SPLIT's pins, its decode bit for bit, the ring's
#       products (within 1e-2 of fp32, the same bits run to run, off the queue, refused in a capture), GLinear by it
#       (as on an A100), exact bit for bit, today's route where it cannot run;
#   (b) test_gpu.py's split tests (test_split_route, test_split_order, test_c_header); then split_stress.py: every
#       Qwen3 layer's matrices (0.6B-32B) through the ring at 769-4096 tokens, rings of 3 to 16 slots, the order whole
#       or a few ahead, 36 passes each, then GLinear's recording pass and 6 after: every product the same bits across
#       layers, passes and slot counts, within 1e-2 of fp32; and the same stress, quick (14B, 32B at 769 and 1024),
#       on the tree before the fix (o2_old_src.tar), which it fails (the recording pass's row chunks not the plan's);
#   (c) layer.py (benchmarks/gpu/option2-2026-09-29): layer 10 of each model, a pass of 8 layers' products through
#       GLinear, the route SPLIT against today's route (v0.25.0's) and bf16, at LAYER_MS;
#   (d) e2e.py: bf16, then Glyd's 12-bit layout (fused, q k v and gate up merged) --prefill E2E_MS with the route SPLIT,
#       then --without-split (today's route) in the same process: a forward pass and the first token; then
#       --breakdown BREAK_MS: the host's time to issue a pass, and a profile of one pass with SPLIT and without (the
#       GPU's idle time; GEMMs, the decode beside GEMMs, beside the rest and alone, attention, the rest);
#   (e) the first model again, e2e.py without bf16: GLYD_SPLIT_SLOTS=3 (the decode's scheduling as the last session
#       ran it: each decode as a slot came free, beside the norms and attention too) at 1024 with its breakdown; and
#       SWEEP_SMS (the decode's SMs set) at SWEEP_MS;
#   then test_gpu.py whole as time allows.
# nvidia-smi logs the SM and memory clocks, power, temperature and clock event reasons every 250 ms throughout
# (smi.csv); the summary gives each step's means. results/summary.txt is rewritten after every step (first line CHECKS
# PASS or FAIL, then what the numbers decide), results/DONE written by the exit trap however the job ends; no step
# starts past BUDGET (1500 s from the job's start), each one's timeout ends by END (1680 s): 28 minutes at most.
#   bash ~/o2_<class>.sh     (in ~: it, o2_job.sh, o2_src.tar, o2_summary.py; x86_64 or aarch64; the GPU to itself)
# The environment: ~/gpuenv/cuda.sh where it has PyTorch with CUDA 13, nvcc 13, transformers, accelerate, safetensors
# and ninja; else one made here with uv (PyTorch 2.14.0, transformers 5.17.0; nvcc from NVIDIA's wheels), in ~/gpuenv
# where there is none, else in W/env; the host's LD_LIBRARY_PATH unset first (logged in machine.txt).
# Env: BUDGET, END, MODELS, LAYER_MS, E2E_MS, BREAK_MS, SWEEP_SMS, SWEEP_MS, SKIP (steps left out: capi,split_tests,
# stress,stress_old,layer,e2e,slots3,sweep,test_gpu), FILES
# (~), R (~/results), W (~/o2w), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/o2w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
SMI=""
trap '[ -n "$SMI" ] && kill $SMI 2> /dev/null; touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
CLASS=${CLASS:?set by o2_a100.sh or o2_hopper.sh}
MODELS=${MODELS:?} LAYER_MS=${LAYER_MS:?} E2E_MS=${E2E_MS:?} BREAK_MS=${BREAK_MS:?} SWEEP_SMS=${SWEEP_MS:+${SWEEP_SMS:?}} SWEEP_MS=${SWEEP_MS:-}
T0=$(date +%s)
BUDGET=${BUDGET:-1500}
END=${END:-1680}
el() { echo $(( $(date +%s) - T0 )); }
now() { date "+%Y/%m/%d %H:%M:%S"; }  # (nvidia-smi's timestamps: local time)
step() { echo "== $(date +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
skip() { case ",${SKIP:-}," in *",$1,"*) echo "$1: left out (SKIP)"; return 0 ;; esac; return 1; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() { "${PY:-python3}" "$FILES/o2_summary.py" "$R" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
done_() { echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
# run NAME TIMEOUT CMD...: CMD's output in R/NAME.txt, its exit code the file's last line; its window in windows.txt
run() {
  local n=$1 t=$2 a; shift 2; a=$(now)
  timeout "$(tmo "$t")" "$@" > "$R/$n.txt" 2>&1; local e=$?
  echo "exit $e" >> "$R/$n.txt"; printf '%s\t%s\t%s\n' "$n" "$a" "$(now)" >> "$R/windows.txt"
  echo "$n: exit $e"; done_ "$n (exit $e)"
}

step "machine"
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,power.default_limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; gcc --version | head -1
  lscpu | grep -E "Model name|Architecture"; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
case $CLASS in a100) WANT=8.0 ;; hopper) WANT=9.0 ;; *) WANT=? ;; esac
[ "$CC" = "$WANT" ] || echo "NOTE: CLASS=$CLASS expects compute capability $WANT; this GPU is $CC ($NAME): run as it is" | tee -a "$R/machine.txt"
OTHERS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$OTHERS" -gt 0 ] && echo "WARNING: $OTHERS other processes on the GPU (the timings share it)" | tee "$R/others.txt"
echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}" >> "$R/machine.txt"
unset LD_LIBRARY_PATH
Q=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu
nvidia-smi --query-gpu=$Q,clocks_event_reasons.active --format=csv,noheader,nounits > /dev/null 2>&1 && Q=$Q,clocks_event_reasons.active
echo "$Q" > "$R/smi-fields.txt"
nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -lms 250 > "$R/smi.csv" 2> "$R/log/smi.err" &
SMI=$!

step "environment"
envok() {  # PYTHON: the packages and a GPU; nvcc; ninja (the JIT build)
  "$1" -c "import torch, transformers, accelerate, safetensors, huggingface_hub; assert torch.cuda.is_available(), 'no GPU for PyTorch'" && command -v nvcc && "$1" -c "import ninja"
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
elif E=$([ -e "$HOME/gpuenv" ] && echo "$W/env" || echo "$HOME/gpuenv") && ( timeout 900 bash -c "$(declare -f mkenv); W='$W' E='$E' mkenv" ) > "$R/log/env-make.txt" 2>&1 &&
     source "$E/cuda.sh" && envok python >> "$R/log/env-make.txt" 2>&1; then
  PY=python; ENVN="made here in $E ($(uname -m))"
else
  echo "CHECKS FAIL: no environment: $(tail -3 "$R/log/env-make.txt" 2> /dev/null | tr '\n' ' ')" | tee "$R/summary.txt"; exit 1
fi
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), CUDA $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TORCH_EXTENSIONS_DIR=$W/torch_ext TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/o2_src.tar" || { echo "CHECKS FAIL: no o2_src.tar in $FILES" | tee "$R/summary.txt"; exit 1; }
B=$W/src/benchmarks/gpu/option2-2026-09-29
echo "$NAME ($CC, sm_$ARCH, ${MIB} MiB), $(uname -m) host, $(nproc) CPUs; CLASS=$CLASS; tree $(cat "$W/src/COMMIT")" | tee "$R/machine-short.txt"
done_ "machine and environment ($ENVN)"

step "the models in the background, in turn: $MODELS"
dl() {  # REPO: into the cache, whole; W/NAME.dir: its directory
  local n; n=$(basename "$1")
  timeout 1200 "$PY" - "$1" > "$W/$n.dir" 2> "$R/log/dl-$n.txt" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "original/*"]))
PY
  echo "exit $? at +$(el) s" >> "$R/log/dl-$n.txt"
}
( for m in $MODELS; do dl "$m"; done ) &
got() {  # REPO: waits for its download; its directory where it holds config.json (nothing past the budget)
  local n; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 5; done
  local d; d=$(tail -1 "$W/$n.dir" 2> /dev/null); [ -f "$d/config.json" ] && echo "$d"
}

step "builds for sm_$ARCH: the library (build_lib.sh's flags) and the JIT build"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
LIB=$W/src/lib/libglyd_gpu_cuda$MAJOR.so
mkdir -p "$W/src/lib"
{ timeout "$(tmo 1200)" nvcc "${F[@]}" -Xptxas -v -c -o "$W/src/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
  timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/src/lib/glyd_gpu.o"; } > "$R/log/build-lib.txt" 2>&1 &
P1=$!
( cd "$W/src/gpu" && env -u GLYD_GPU_LIB MAX_JOBS=4 timeout "$(tmo 1200)" "$PY" -c "import glyd_gpu; print('JIT build:', glyd_gpu._ext)" > "$R/log/build-jit.txt" 2>&1; echo "exit $?" >> "$R/log/build-jit.txt" ) &
P2=$!
OLD=$W/old OLIB=""
if [ -f "$FILES/o2_old_src.tar" ] && ! skip stress_old; then
  rm -rf "$OLD" && mkdir -p "$OLD/lib" && tar -C "$OLD" -xf "$FILES/o2_old_src.tar" && OLIB=$OLD/lib/libglyd_gpu_cuda$MAJOR.so
  { timeout "$(tmo 1200)" nvcc "${F[@]}" -c -o "$OLD/lib/glyd_gpu.o" "$OLD/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$OLIB" "$OLD/lib/glyd_gpu.o"; echo "exit $?"; } > "$R/log/build-old.txt" 2>&1 &
  P3=$!
fi
wait $P1; echo "exit $?" >> "$R/log/build-lib.txt"
wait $P2
[ -n "$OLIB" ] && wait $P3
grep -A2 "mma12_split_kernel" "$R/log/build-lib.txt" | grep -E "registers|spill" | head -2 > "$R/split-kernel-ptxas.txt"
for b in lib jit old; do [ -f "$R/log/build-$b.txt" ] && echo "build $b: $(tail -1 "$R/log/build-$b.txt")"; done
[ -f "$LIB" ] || { echo "CHECKS FAIL: the library did not build: $(grep -m3 -i error "$R/log/build-lib.txt" | tr '\n' ' ')" | tee -a "$R/steps.txt"; summ; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
done_ "builds (lib $(tail -1 "$R/log/build-lib.txt"), jit $(tail -1 "$R/log/build-jit.txt"))"

skip capi || { left && step "(a) check_capi" && ( cd "$W/src/gpu" && run check_capi 600 "$PY" -u check_capi.py "$LIB" ); }
skip split_tests || { left && step "(b) test_gpu.py's split tests" && ( cd "$W/src/bindings/python" && run test_split 300 "$PY" -u -c "
import test_gpu as t
for n in ('test_c_header', 'test_split_order', 'test_split_route'):
    getattr(t, n)(); print(n, 'ok', flush=True)" ); }
skip stress || { left && step "(b) split_stress.py: every model's matrices through the ring, 769-4096 tokens, 3-16 slots, many passes" &&
  ( cd "$W/src/gpu" && run split_stress 600 "$PY" -u split_stress.py "$LIB" ); }
if [ -n "$OLIB" ] && [ -f "$OLIB" ] && ! skip stress_old; then  # (its failures expected: the check against the tree before the fix)
  left && step "(b) split_stress.py on the tree before the fix ($(cat "$OLD/COMMIT")), quick: 14B and 32B at 769 and 1024" &&
    cp "$W/src/gpu/split_stress.py" "$OLD/gpu/" && ( cd "$OLD/gpu" && run split_stress_old 300 env PYTHONPATH="$OLD/bindings/python" GLYD_GPU_LIB="$OLIB" "$PY" -u split_stress.py "$OLIB" --models 14B,32B --ms 769,1024 --quick )
fi

FIRST=$(echo $MODELS | cut -d' ' -f1)
layer() {  # MODEL [NAME [env...]]: layer.py on its layer 10
  local m=$1 n=${2:-layer-$(basename "$1")} d; shift 2 2> /dev/null || shift $#
  d=$(got "$m") || { echo "$n: no $m"; return; }
  ( cd "$B" && run "$n" 420 env "$@" "$PY" -u layer.py "$d" --M "$LAYER_MS" )
}
e2e() {  # MODEL [NAME [ENV... --] [e2e.py options]]: bf16, then Glyd with the route SPLIT, then without it, then the breakdown
  local m=$1 n=${2:-e2e-$(basename "$1")} d e=(); shift 2 2> /dev/null || shift $#
  while [ $# -gt 0 ] && [ "$1" != -- ]; do e+=("$1"); shift; done; [ "${1:-}" = -- ] && shift
  d=$(got "$m") || { echo "$n: no $m"; return; }
  [ $# -gt 0 ] || set -- --baseline --prefill "$E2E_MS" --without-split --breakdown "$BREAK_MS"
  ( cd "$W/src/gpu" && run "$n" 900 env ${e[@]+"${e[@]}"} "$PY" -u e2e.py "$d" --format mma12 --fused --merge --tokens 32 "$@" )
}
export HF_HUB_OFFLINE=0
for m in $MODELS; do
  skip layer || { left && step "(c) layer.py, $m" && layer "$m"; }
  skip e2e || { left && step "(d) e2e.py, $m: bf16, the route SPLIT, today's route" && e2e "$m"; }
done
skip slots3 || { left && step "(e) e2e.py, $FIRST, GLYD_SPLIT_SLOTS=3: the last session's scheduling, at 1024 with its breakdown" &&
  e2e "$FIRST" "e2e-slots3-$(basename "$FIRST")" GLYD_SPLIT_SLOTS=3 -- --prefill 1024 --breakdown 1024; }
if [ -n "$SWEEP_MS" ] && ! skip sweep; then
  for sms in $SWEEP_SMS; do
    left && step "(e) e2e.py, $FIRST, GLYD_SPLIT_SMS=$sms at $SWEEP_MS" && e2e "$FIRST" "e2e-sms$sms-$(basename "$FIRST")" GLYD_SPLIT_SMS=$sms -- --prefill "$SWEEP_MS"
  done
fi
if ! skip test_gpu && [ $(( BUDGET - $(el) )) -gt 240 ]; then
  step "test_gpu.py whole" && ( cd "$W/src/bindings/python" && run test_gpu 600 "$PY" -u test_gpu.py )
else
  echo "test_gpu.py whole: skipped ($(( BUDGET - $(el) )) s left of the budget)"
fi
step "done in $(el) s"
done_ "done"
