#!/usr/bin/env bash
# The decode fix as merged (v0.25.1: the 12-bit whole-matrix and experts' decodes in order 3's loads, the decode ahead
# in order 0's, no GLYD_DEC_ORDER) on an L4 (sm_89; the AWS dev machine), beside main's decode, v0.25.0's and 73b9560's
# four orders, and the release's checks at its head. In ~/dfx: this and dfx_src.tar (main/, rel/ and fix/ as
# hdec_src.tar's; final/: git archive of this tree's gpu/, bindings/, glyd-gpu/ and the two benchmarks' directories),
# each with its COMMIT; the models already in ~/hf (offline: nothing downloaded). Run with the GPU to itself:
#   flock ~/.glyd-box.lock bash ~/dfx/l4_dec.sh
# In order (v0.25.1's library as released, build_lib.sh; the others for this GPU alone, with its flags):
#   bits     v0.25.1's self-test (glyd_gpu.py) and xcheck.py's synthetic part (against main's library)
#   checks   check_capi.py (and with GLYD_DEC_MIN 1000 and 3000), test_gpu.py, the crate's tests (cargo test)
#   kernel   dec_time.py on layer 10 of Qwen3-8B and Qwen3-4B-Instruct-2507: main's, v0.25.0's, orders 0-3 and v0.25.1's
#            decodes in one process (whole: a warp a step; ahead: 2 warps an SM)
#   e2e      dec_e2e.py, exact mode, Qwen3-8B (the GPU the bound here): steps and 1024- and 4096-token prompts; 73b9560's
#            orders in one process, then main's, v0.25.0's and v0.25.1's libraries each in its own
#   kernel2  dec_time.py again
#   host     dec_e2e.py, exact steps, Qwen3-0.6B (the host the bound): the same processes, then each again (host/)
# Each e2e process starts from about the GPU's idle temperature (the L4's clock falls as it heats at its 72 W cap);
# smi.csv: its temperature, SM clock and power each second. results/summary.txt and host/summary.txt by dec_summary.py.
# L4DEC_TREE: the v0.25.1 tree's directory in ~/dfx/w (final), L4DEC_SRC its tar's name (dfx_src.tar), L4DEC_RESULTS
# (~/dfx/results), L4DEC_STEPS (builds,bits,checks,kernel,e2e,kernel2,host); L4DEC_REUSE=1: ~/dfx/w and the results so
# far kept, and a library already built there not built again (a run stopped after its builds, or another tree).
set -u
H=${H:-$HOME/dfx}
T=${L4DEC_TREE:-final} R=${L4DEC_RESULTS:-$H/results} W=$H/w
REUSE=${L4DEC_REUSE:-0}
STEPS=${L4DEC_STEPS:-builds,bits,checks,kernel,e2e,kernel2,host}
want() { case ",$STEPS," in *",$1,"*) return 0 ;; esac; return 1; }
[ "$REUSE" = 1 ] || rm -rf "$R" "$W"
mkdir -p "$R/log" "$R/host" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date +%T) (+$(el) s) $*"; echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; }
run() { local n=$1 t=$2; shift 2; timeout "$t" "$@" > "$n.txt" 2>&1; local e=$?; echo "exit $e" >> "$n.txt"; echo "$(basename "$n"): exit $e"; }
source "$HOME/gpuenv/cuda.sh"
unset LD_LIBRARY_PATH
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=python
[ -d "$W/$T" ] || tar -C "$W" -xf "$H/${L4DEC_SRC:-dfx_src.tar}"
D=$W/$T/benchmarks/gpu/decode-fix-2026-09-29 B=$W/$T/benchmarks/gpu/splitbyte-2026-09-29
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH=${CC/./}
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; lscpu | grep -E "Model name"; nproc; } > "$R/machine.txt" 2>&1
echo "$NAME ($CC, sm_$ARCH), $(uname -m) host, $(nproc) CPUs; v0.25.1 $(cat "$W/$T/COMMIT"), orders $(cat "$W/fix/COMMIT"), v0.25.0 $(cat "$W/rel/COMMIT"), main $(cat "$W/main/COMMIT")" | tee "$R/machine-short.txt"
{ echo "environment: ~/gpuenv; $($PY -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; } | tee "$R/env.txt"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv -l 1 >> "$R/smi.csv" 2>&1 &
SMI=$!
trap 'kill $SMI 2> /dev/null; touch "$R/DONE"' EXIT
IDLE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)
cool() {  # to within 3 C of the temperature at the start, 120 s at most
  local t0=$(date +%s) t
  while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1); [ "$t" -gt $(( IDLE + 3 )) ] && [ $(( $(date +%s) - t0 )) -lt 120 ]; do sleep 5; done
  echo "$t C after $(( $(date +%s) - t0 )) s (start $IDLE C)"
}

CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
L=libglyd_gpu_cuda$MAJOR.so
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
build() {
  mkdir -p "$W/$1/lib"
  { nvcc "${F[@]}" -c -o "$W/$1/lib/glyd_gpu.o" "$W/$1/gpu/glyd_gpu.cu" &&
    nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/$1/lib/libglyd_gpu_cuda$MAJOR.so" "$W/$1/lib/glyd_gpu.o"; } > "$R/log/build-$1.txt" 2>&1
  echo "exit $?" >> "$R/log/build-$1.txt"
}
if want builds; then
  step "builds: v0.25.1 ($T) by build_lib.sh (every architecture), 73b9560, v0.25.0 and main for sm_$ARCH (its flags), at once, each not yet built"
  P=()
  [ -f "$W/$T/lib/$L" ] || { ( cd "$W/$T/gpu" && timeout 1800 bash build_lib.sh "$W/$T/lib" > "$R/log/build-$T.txt" 2>&1; echo "exit $?" >> "$R/log/build-$T.txt" ) & P+=($!); }
  for t in fix rel main; do [ -f "$W/$t/lib/$L" ] || { build $t & P+=($!); }; done
  [ ${#P[@]} = 0 ] || wait "${P[@]}"  # (a bare wait would wait for nvidia-smi and the tee as well)
  for t in $T fix rel main; do [ -f "$R/log/build-$t.txt" ] && echo "build $t: $(tail -1 "$R/log/build-$t.txt")"; done
fi
NLIB=$W/$T/lib/$L FLIB=$W/fix/lib/$L RLIB=$W/rel/lib/$L MLIB=$W/main/lib/$L
for f in "$NLIB" "$FLIB" "$RLIB" "$MLIB"; do [ -f "$f" ] || { echo "FAIL: no $f"; exit 1; }; done
for t in main rel fix $T; do echo "== $t"; cuobjdump -res-usage "$W/$t/lib/$L" 2>&1 | grep -A1 "mma_unpack_kernelI3Nib" | grep -v "^--"; done > "$R/registers.txt"

if want bits; then
  step "bits: v0.25.1's self-test, then xcheck.py's synthetic part (against main's library)"
  ( cd "$W/$T/gpu" && run "$R/selftest-v0.25.1" 900 env GLYD_GPU_LIB="$NLIB" PYTHONPATH="$W/$T/bindings/python" $PY -u glyd_gpu.py )
  ( cd "$B" && run "$R/xcheck-v0.25.1" 900 env GLYD_GPU_LIB="$NLIB" PYTHONPATH="$W/$T/bindings/python" $PY -u xcheck.py "$W/main" "$MLIB" )
fi

if want checks; then
  step "checks at v0.25.1's head: check_capi.py (then with GLYD_DEC_MIN 1000 and 3000), test_gpu.py, cargo test"
  ( cd "$W/$T/gpu" && run "$R/check_capi" 1500 env GLYD_GPU_LIB="$NLIB" PYTHONPATH="$W/$T/bindings/python" MAX_JOBS=8 $PY -u check_capi.py "$NLIB" )
  ( cd "$W/$T/bindings/python" && run "$R/test_gpu" 1500 env GLYD_GPU_LIB="$NLIB" PYTHONPATH="$W/$T/bindings/python" $PY -u test_gpu.py )
  ( cd "$W/$T/glyd-gpu" && source "$HOME/.cargo/env" 2> /dev/null; run "$R/cargo_test" 1200 env GLYD_GPU_LIB="$NLIB" cargo test --release -- --test-threads 1 )
  for m in 1000 3000; do
    ( cd "$W/$T/gpu" && run "$R/check_capi-dec_min$m" 1500 env GLYD_DEC_MIN=$m GLYD_GPU_LIB="$NLIB" PYTHONPATH="$W/$T/bindings/python" MAX_JOBS=8 $PY -u check_capi.py "$NLIB" )
  done
fi

kernel() {
  for m in Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507; do
    step "kernel: dec_time.py, $m, run $1"
    ( cd "$D" && run "$R/time-$(basename $m)-run$1" 900 env GLYD_GPU_LIB="$FLIB" PYTHONPATH="$W/fix/bindings/python" $PY -u dec_time.py "$W/main" "$MLIB" "$RLIB" "$m" --final "$NLIB" )
  done
}
want kernel && kernel 1

e2e() {  # OUT TREE LIB ORDERS MODEL ARGS...: dec_e2e.py with TREE's package and LIB, from about the idle temperature
  local o=$1 t=$2 l=$3 ord=$4 m=$5; shift 5
  echo "$(basename "$o"): $(cool)" | tee -a "$R/temps.txt"
  ( cd "$D" && run "$o" 1200 env GLYD_COMPILE=0 GLYD_GPU_LIB="$l" PYTHONPATH="$W/$t/bindings/python" $PY -u dec_e2e.py "$m" --orders "$ord" --modes exact "$@" )
}
if want e2e; then
  step "e2e: dec_e2e.py, Qwen3-8B, exact: 73b9560's orders 0-3, then main's, v0.25.0's and v0.25.1's"
  A8=(--tokens 32 --reps 3 --prompts 1024,4096)
  e2e "$R/e2e-fix" fix "$FLIB" 0,1,2,3 Qwen/Qwen3-8B "${A8[@]}"
  e2e "$R/e2e-main" main "$MLIB" "" Qwen/Qwen3-8B "${A8[@]}"
  e2e "$R/e2e-v0.25.0" rel "$RLIB" "" Qwen/Qwen3-8B "${A8[@]}"
  e2e "$R/e2e-v0.25.1" $T "$NLIB" "" Qwen/Qwen3-8B "${A8[@]}"
fi

want kernel2 && kernel 2

if want host; then
  step "host: dec_e2e.py, Qwen3-0.6B, exact steps: the same processes, then each again"
  A06=(--tokens 64 --reps 5 --prompts "")
  for pass in 1 2; do
    s=$([ $pass = 2 ] && echo "-2")
    e2e "$R/host/e2e-fix$s" fix "$FLIB" 0,1,2,3 Qwen/Qwen3-0.6B "${A06[@]}"
    e2e "$R/host/e2e-main$([ $pass = 2 ] && echo 2)" main "$MLIB" "" Qwen/Qwen3-0.6B "${A06[@]}"
    e2e "$R/host/e2e-v0.25.0$s" rel "$RLIB" "" Qwen/Qwen3-0.6B "${A06[@]}"
    e2e "$R/host/e2e-v0.25.1$s" $T "$NLIB" "" Qwen/Qwen3-0.6B "${A06[@]}"
  done
  cp "$R/machine-short.txt" "$R/env.txt" "$R/host/"
  $PY "$D/dec_summary.py" "$R/host" > "$R/host/summary.txt" 2>> "$R/log/summary.err"
fi
$PY "$D/dec_summary.py" "$R" > "$R/summary.txt" 2>> "$R/log/summary.err"
step "done in $(el) s"
