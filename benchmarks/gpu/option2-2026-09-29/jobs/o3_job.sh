#!/usr/bin/env bash
# The route SPLIT against v0.25.1's routes on one Hopper GPU, to settle it (gpu-route-split, SPLIT opt-in, C API 6;
# COMMIT in o3_src.tar). On a GH200 or an H100 SXM it measures the shipping route (2048-8192 tokens, O and K at least 5120). On any
# other GPU (an H200, an H100 NVL, an H100 PCIe) SPLIT is forced on by GLYD_SPLIT_MIN=FORCE_MIN (2048: the GH200's
# range, its SMs, every matrix of these models' merged Linears O and K at least 4096): a measurement, not a route. In
# order, most needed first, so that a cap leaves the first answers:
#   builds    the library for this GPU alone (build_lib.sh's flags) and the JIT build, at once; the models download
#             meanwhile (MODELS, whole: e2e.py loads them), the first first;
#   (a) checks, on the shipping rules (no knob): check_capi, test_gpu.py's split tests, split_stress.py (every Qwen3
#       layer's matrices through the ring, many passes and slot counts, the same bits every time);
#   (b) per model: e2e.py, merged Linears, the 12-bit layout fused: bf16, then --prefill E2E_MS by SPLIT and by
#       v0.25.1's routes (--without-split: model.Split off, the routes without GLYD_GPU_WITH_SPLIT) in the same process,
#       ROUNDS times each way in turn (SPLIT, v0.25.1; v0.25.1, SPLIT; ...): a forward pass and the first token, each
#       phase's window logged; then layer.py (layer 10: SPLIT, v0.25.1's route and bf16 in turn, pass by pass) at
#       LAYER_MS if the budget allows;
#   (c) split_stress.py with the split skewed both ways (--sms=-2,1, quick: 0.6B, 8B and 14B at 769 and 2048), as the budget allows.
# nvidia-smi logs the SM and memory clocks, power, temperature and clock event reasons every 250 ms throughout
# (smi.csv); the summary gives each step's and each phase's (SPLIT, v0.25.1) means. results/summary.txt is rewritten
# after every step: first line CHECKS PASS or FAIL, last lines DECIDES, one per model and length (SPLIT's forward pass
# over v0.25.1's, the median of the rounds' ratios; SPLIT stays where it beats v0.25.1 by at least 2%, a ratio of
# 0.980 or less, and is dropped where it does not). results/DONE is written by the exit trap however the job ends; no
# step starts past BUDGET (1200 s from the job's start), each one's timeout ends by END (1440 s): 24 minutes at most.
#   bash ~/o3_job.sh     (in ~: it, o3_src.tar, o3_summary.py; x86_64 or aarch64; the GPU to itself)
#   MODELS=Qwen/Qwen3-14B bash ~/o3_job.sh     (another model's settle; SKIP=capi,split_tests,stress,stress_skewed after
#                                              a run of the same tree on this GPU)
# The environment: ~/gpuenv/cuda.sh where it has PyTorch with CUDA 13, nvcc 13, transformers, accelerate, safetensors
# and ninja; else one made here with uv (PyTorch 2.14.0, transformers 5.17.0; nvcc from NVIDIA's wheels), in ~/gpuenv
# where there is none, else in W/env; the host's LD_LIBRARY_PATH unset first (logged in machine.txt).
# Env: BUDGET, END, MODELS, E2E_MS, LAYER_MS, ROUNDS, FORCE_MIN, SKIP (steps left out: capi,split_tests,stress,e2e,
# layer,stress_skewed), FILES (~), R (~/results), W (~/o3w), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/o3w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
SMI="" DLP=""
trap '[ -n "$SMI" ] && kill $SMI 2> /dev/null; [ -n "$DLP" ] && { pkill -P $DLP; kill $DLP; } 2> /dev/null; touch "$R/DONE"' EXIT  # (a download still running stopped)
exec > >(tee -a "$R/job.log") 2>&1
MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-32B"} E2E_MS=${E2E_MS:-2048,4096,8192} LAYER_MS=${LAYER_MS:-2048,4096,8192} ROUNDS=${ROUNDS:-3} FORCE_MIN=${FORCE_MIN:-2048}
T0=$(date +%s)
BUDGET=${BUDGET:-1200}
END=${END:-1440}
el() { echo $(( $(date +%s) - T0 )); }
now() { date "+%Y/%m/%d %H:%M:%S"; }  # (nvidia-smi's timestamps: local time)
step() { echo "== $(date +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
skip() { case ",${SKIP:-}," in *",$1,"*) echo "$1: left out (SKIP)"; return 0 ;; esac; return 1; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() { "${PY:-python3}" "$FILES/o3_summary.py" "$R" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
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
PLIM=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
[ "$CC" = "9.0" ] || echo "NOTE: not Hopper (compute capability $CC, $NAME): run as it is" | tee -a "$R/machine.txt"
KNOB=()
if echo "$NAME" | grep -qw GH200 || { echo "$NAME" | grep -qw H100 && ! echo "$NAME" | grep -qiE 'pcie|nvl'; }; then
  echo "SPLIT: the shipping route (a GH200 or an H100 SXM: 2048-8192 tokens, O and K at least 5120)" > "$R/route.txt"
else
  KNOB=(GLYD_SPLIT_MIN="$FORCE_MIN")
  echo "SPLIT forced on by GLYD_SPLIT_MIN=$FORCE_MIN on an $NAME: a measurement, not a route" | tee "$R/route.txt" | sed 's/^/* /' > "$R/forced.txt"
fi
OTHERS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$OTHERS" -gt 0 ] && echo "WARNING: $OTHERS other processes on the GPU (the timings share it)" | tee "$R/others.txt"
echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}" >> "$R/machine.txt"
unset LD_LIBRARY_PATH GLYD_SPLIT_MIN GLYD_SPLIT_MAX GLYD_SPLIT_SMS GLYD_SPLIT_SLOTS
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
elif E=$([ -e "$HOME/gpuenv" ] && echo "$W/env" || echo "$HOME/gpuenv") && ( timeout 600 bash -c "$(declare -f mkenv); W='$W' E='$E' mkenv" ) > "$R/log/env-make.txt" 2>&1 &&
     source "$E/cuda.sh" && envok python >> "$R/log/env-make.txt" 2>&1; then
  PY=python; ENVN="made here in $E ($(uname -m))"
else
  echo "CHECKS FAIL: no environment: $(tail -3 "$R/log/env-make.txt" 2> /dev/null | tr '\n' ' ')" | tee "$R/summary.txt"; exit 1
fi
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), CUDA $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TORCH_EXTENSIONS_DIR=$W/torch_ext TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/o3_src.tar" || { echo "CHECKS FAIL: no o3_src.tar in $FILES" | tee "$R/summary.txt"; exit 1; }
B=$W/src/benchmarks/gpu/option2-2026-09-29
echo "$NAME ($CC, sm_$ARCH, ${MIB} MiB, power limit $PLIM), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); $(cat "$R/route.txt")" | tee "$R/machine-short.txt"
{ for m in $MODELS; do printf '%s ' "$(basename "$m")"; done; echo; echo "$E2E_MS"; } > "$R/expect.txt"  # (the summary's DECIDES lines: a model and length each)
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
DLP=$!
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
{ timeout "$(tmo 900)" nvcc "${F[@]}" -Xptxas -v -c -o "$W/src/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
  timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/src/lib/glyd_gpu.o"; } > "$R/log/build-lib.txt" 2>&1 &
P1=$!
P2=""
if ! skip capi; then
  ( cd "$W/src/gpu" && env -u GLYD_GPU_LIB MAX_JOBS=4 timeout "$(tmo 900)" "$PY" -c "import glyd_gpu; print('JIT build:', glyd_gpu._ext)" > "$R/log/build-jit.txt" 2>&1; echo "exit $?" >> "$R/log/build-jit.txt" ) &
  P2=$!
fi
wait $P1; echo "exit $?" >> "$R/log/build-lib.txt"
[ -n "$P2" ] && wait $P2
grep -A2 "mma12_split_kernel" "$R/log/build-lib.txt" | grep -E "registers|spill" | head -2 > "$R/split-kernel-ptxas.txt"
for b in lib jit; do [ -f "$R/log/build-$b.txt" ] && echo "build $b: $(tail -1 "$R/log/build-$b.txt")"; done
[ -f "$LIB" ] || { echo "CHECKS FAIL: the library did not build: $(grep -m3 -i error "$R/log/build-lib.txt" | tr '\n' ' ')" | tee -a "$R/steps.txt"; summ; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
done_ "builds (lib $(tail -1 "$R/log/build-lib.txt")$([ -f "$R/log/build-jit.txt" ] && echo ", jit $(tail -1 "$R/log/build-jit.txt")"))"

skip capi || { left && step "(a) check_capi" && ( cd "$W/src/gpu" && run check_capi 300 "$PY" -u check_capi.py "$LIB" ); }
skip split_tests || { left && step "(a) test_gpu.py's split tests" && ( cd "$W/src/bindings/python" && run test_split 180 "$PY" -u -c "
import test_gpu as t
for n in ('test_c_header', 'test_split_order', 'test_split_route'):
    getattr(t, n)(); print(n, 'ok', flush=True)" ); }
skip stress || { left && step "(a) split_stress.py: every model's matrices through the ring, many passes and slot counts" &&
  ( cd "$W/src/gpu" && run split_stress 300 "$PY" -u split_stress.py "$LIB" ); }

e2e() {  # MODEL: bf16, then Glyd's 12-bit layout by SPLIT and by v0.25.1's routes, ROUNDS each way in turn
  local m=$1 n=e2e-$(basename "$1") d
  d=$(got "$m") || { echo "$n: no $m"; return; }
  ( cd "$W/src/gpu" && run "$n" 900 env ${KNOB[@]+"${KNOB[@]}"} "$PY" -u e2e.py "$d" --format mma12 --fused --merge --tokens 32 --baseline --prefill "$E2E_MS" --without-split --rounds "$ROUNDS" )
}
layer() {  # MODEL: layer.py on its layer 10
  local m=$1 n=layer-$(basename "$1") d
  d=$(got "$m") || { echo "$n: no $m"; return; }
  ( cd "$B" && run "$n" 300 env ${KNOB[@]+"${KNOB[@]}"} "$PY" -u layer.py "$d" --M "$LAYER_MS" )
}
for m in $MODELS; do
  skip e2e || { left && step "(b) e2e.py, $m: bf16, then SPLIT and v0.25.1's routes, $ROUNDS rounds each way in turn" && e2e "$m"; }
  skip layer || { left && step "(b) layer.py, $m: layer 10" && layer "$m"; }
done
skip stress_skewed || { left && step "(c) split_stress.py, the split skewed both ways (the products on the fewest SMs, the decode on the fewest)" &&
  ( cd "$W/src/gpu" && run split_stress_skewed 300 "$PY" -u split_stress.py "$LIB" --sms=-2,1 --quick --models 0.6B,8B,14B --ms 769,2048,8192 ); }
step "done in $(el) s"
done_ "done"
