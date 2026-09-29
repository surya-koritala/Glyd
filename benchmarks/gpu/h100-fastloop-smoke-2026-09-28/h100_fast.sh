#!/usr/bin/env bash
# One short unattended H100 smoke job for gpu-fastloop's head (fast_src.tar: gpu/ and bindings/python, its commit in
# HEAD): generate() compiled by default, with main's Hopper kernels merged in (mma12_wgp_kernel past 128 tokens,
# GLYD_WG_MAX 1024). In order: (a) the library from the head (build_lib.sh's compile line, sm_90a alone: ARCHS);
# (b) the self-test (glyd_gpu.py); (c) check_capi; (d) test_gpu; (e) check_api dense (Qwen3-0.6B and 1.7B); (f) plain
# generate() tokens/s on Qwen3-1.7B, the compiled default against compile=False, 1 sequence of a prompt of P1 tokens
# (160) and 8 of P8 (128), N generated (64), each in a process of its own (fast_gen.py): the prompt's products of 160
# and 1024 tokens, past 128 and to GLYD_WG_MAX, in mma_gemm_wg's mma12_wgp_kernel; 8 x (128 + 64) positions within
# the compile cap (2048 on an H100). torch's, transformers' and CUDA's versions in results/versions.txt; a line a step
# in results/summary.txt; results/DONE from the exit trap, however it ends. No step starts past BUDGET (620 s), and
# each one's timeout ends by DEADLINE (700 s): 12 minutes at most, with the downloads in the background.
#   bash ~/h100_fast.sh      (in ~: this script, fast_src.tar, fast_gen.py)
# Env, for a smoke test elsewhere: ENV_SH (~/gpuenv/cuda.sh), ARCHS (90a), FILES (~), R (~/results), W (~/fastw),
# HF_HOME (~/hf), BUDGET, DEADLINE, P1, P8, N.
set -u
T0=$(date +%s)
R=${R:-$HOME/results}
W=${W:-$HOME/fastw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
trap 'echo "$(date +%T) (+$(( $(date +%s) - T0 )) s) end" >> "$R/steps.txt"; touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
BUDGET=${BUDGET:-620}
DEADLINE=${DEADLINE:-700}
ARCHS=${ARCHS:-90a}
P1=${P1:-160}
P8=${P8:-128}
N=${N:-64}
step() { echo "== $(date +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
note() { echo "$*" >> "$R/summary.txt"; echo "$(date +%T) (+$(( $(date +%s) - T0 )) s) $*" >> "$R/steps.txt"; }
tmo() {  # CAP: a step's timeout, CAP or what is left to DEADLINE; 0 past BUDGET (the step skipped)
  local e=$(( $(date +%s) - T0 )); [ $e -lt "$BUDGET" ] || { echo 0; return; }
  local r=$(( DEADLINE - e )); [ "$1" -lt $r ] && echo "$1" || echo $r
}
run() {  # NAME CAP DIR CMD...: CMD in DIR under tmo CAP, its output in results/NAME.txt; its exit status (124: timed out)
  local n=$1 t; t=$(tmo "$2"); local d=$3; shift 3
  [ "$t" -gt 0 ] || { echo "$n: skipped, over the budget" > "$R/$n.txt"; return 125; }
  step "$n (timeout $t s)"
  ( cd "$d" && timeout "$t" "$@" ) > "$R/$n.txt" 2>&1
  local rc=$?; echo "$n exit $rc" >> "$R/$n.txt"; return $rc
}
source "${ENV_SH:-$HOME/gpuenv/cuda.sh}"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TORCH_EXTENSIONS_DIR=$W/torch_ext
python -c "import hf_transfer" 2>/dev/null || unset HF_HUB_ENABLE_HF_TRANSFER

step "machine and versions"
{ nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,power.limit --format=csv
  python - <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0])
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("transformers", transformers.__version__)
try:
    import accelerate
    print("accelerate", accelerate.__version__)
except ImportError as e:
    print("accelerate: none", e)
print("GPU", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")
PY
  echo "nvcc: $(nvcc --version | tail -2 | tr '\n' ' ')"; lscpu | grep "Model name"; echo "$(nproc) CPUs"; date -u; } > "$R/versions.txt" 2>&1
note "versions: $(grep -h '^torch\|^transformers\|^GPU' "$R/versions.txt" | tr '\n' ';') nvcc $(nvcc --version | grep -o 'release [0-9.]*')"

step "models, in the background: Qwen3-0.6B, Qwen3-1.7B"
( for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
    timeout 300 python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$m" >> "$R/log/dl.txt" 2>&1
    echo "$m exit $?" >> "$R/log/dl.txt"
  done ) & DL=$!

step "sources: the head"
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/fast_src.tar" || { note "sources: fast_src.tar did not unpack"; exit 1; }
note "sources: $(cat "$W/src/HEAD" 2>/dev/null || echo 'no HEAD file')"
export PYTHONPATH=$W/src/bindings/python

# (a) build_lib.sh's compile line, for ARCHS alone
CUDA=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
GEN=(); for a in $ARCHS; do GEN+=(-gencode "arch=compute_$a,code=sm_$a"); done
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CUDA/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ "${GEN[@]}" -Xcompiler -fPIC,-fvisibility=hidden)
mkdir -p "$W/lib"
run build 300 "$W/src/gpu" bash -c 'nvcc "$@" --threads 0 -c -o "'"$W"'/glyd_gpu.o" glyd_gpu.cu && nvcc "$@" -shared -Xlinker --exclude-libs,ALL -cudart static -L"'"$CUDA"'/lib" -L"'"$CUDA"'/lib64" -o "'"$LIB"'" "'"$W"'/glyd_gpu.o"' _ "${F[@]}"
rc=$?; note "(a) build, sm_$ARCHS: exit $rc"
[ $rc = 0 ] && [ -f "$LIB" ] || exit 1
export GLYD_GPU_LIB=$LIB

run selftest 150 "$W/src/gpu" python glyd_gpu.py; rc=$?
note "(b) self-test: exit $rc ($(grep -c 'within 1e-2' "$R/selftest.txt") lines within 1e-2; $(grep -o '^mma_gemm_wg [0-9x]*' "$R/selftest.txt" | wc -l) of them mma_gemm_wg's, 1-2100 tokens)"

run check_capi 240 "$W/src/gpu" python check_capi.py "$LIB"; rc=$?
note "(c) check_capi: exit $rc ($(grep -m1 -o '[0-9]* calls compared bit for bit, all identical' "$R/check_capi.txt"))"

run test_gpu 240 "$W/src/bindings/python" python test_gpu.py; rc=$?
note "(d) test_gpu: exit $rc ($(grep -c ' ok$' "$R/test_gpu.txt") ok; $(grep -c 'skipped' "$R/test_gpu.txt") skipped)"

wait $DL
run check_api 420 "$W/src/gpu" python check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; rc=$?
note "(e) check_api dense: exit $rc ($(grep -c 'compiled by default' "$R/check_api.txt") models compiled by default; $(grep -o "tokens as eager's [0-9]* of 32" "$R/check_api.txt" | tr '\n' ';') $(tail -2 "$R/check_api.txt" | head -1 | cut -c1-60))"

for B in 1 8; do
  P=$P1; [ $B = 8 ] && P=$P8
  for mode in default eager; do
    run "gen-$mode-$B" 120 "$W/src/gpu" python "$FILES/fast_gen.py" Qwen/Qwen3-1.7B $mode $B $P $N; rc=$?
    note "(f) exit $rc: $(grep -m1 'tokens/s' "$R/gen-$mode-$B.txt" || tail -1 "$R/gen-$mode-$B.txt")"
  done
done
step "done in $(( $(date +%s) - T0 )) s"
