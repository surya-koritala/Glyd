#!/usr/bin/env bash
# Round 1 for v0.25.0 on one GPU of a class, CLASS = hopper, a100, a10 or ada (r1_hopper.sh, r1_a100.sh, r1_a10.sh and
# r1_ada.sh set it and its lists): split byte's checks (sb_job.sh's, its steps kept) and the release's
# own, on the release's tree (rust-splitbyte: gpu-splitbyte and the Rust packer in split byte, COMMIT in r1_src.tar),
# main's 12-bit layout (sb_main.tar: origin/main db8e7b0) beside it. In order, most needed first, so that a cap leaves
# the first answers:
#   builds    the branch's library and main's for this GPU alone, the branch's JIT build, and the glyd-gpu crate (its
#             binary, examples and tests; Rust from rustup where there is none), with ATTR=1 main-once too, at once;
#             the models download meanwhile
#   (a) the self-test (the branch's: its products against fp32, every bf16 bit pattern decoded in both layouts);
#   (b) xcheck.py, synthetic: every 12-bit entry point this GPU takes, split byte's outputs the same bits as main's
#       12-bit layout's in the same kernels (or refused by both), and against the weights or fp32;
#   (c) check_capi: the JIT build against the library bit for bit, the routes against 0.24's rule (on Hopper its real
#       routes), and r1_routes.py: this device's own code and routes (an A10's class, 2086, on an A10);
#   (d) Rust: the crate's tests with the library and the GPU; Qwen3-0.6B packed by glyd pack and by python -m glyd.gpu
#       pack in both layouts, every file's sha256 compared; the Rust saves verified on the CPU, on the GPU and by
#       Python; the crate's examples (unpack, linear) on the tiered save;
#   (e) test_gpu.py;
#   (f) layer.py, run 1: layer 10 of TIME_MODELS, main's 12-bit layout against split byte, by route and by kernel;
#       with ATTR=1 (the Ada and A10 jobs), sb_job.sh's attribution: layer.py at 1, 256 and 1024 tokens on TIME_MODELS'
#       first, main against main-once (main's kernels with the exception loop an entry a pass, as the branch had it
#       before aa7a6fc: --old-both), then main-once against the branch (split byte, its loop main's again);
#   (g) e2e12.py: E2E_MODELS' logits and greedy tokens, main's package and library, the branch's, main's again;
#   (h) check_api dense (Qwen3-0.6B) and MoE (granite-3.1-3b-a800m-instruct);
#   (i) xcheck.py on MODELS as their downloads end (every Linear decoded bit for bit in both layouts, 5 layers'
#       products); (j) layer.py, run 2; (k) GeForce Ada: the grid prompt kernels (sb_job.sh's step h).
# Then, with DEC=1 (the Hopper and A100 jobs), the decoder test's follow-up: R=results/dec bash ~/dec_more.sh (dec_job.sh's
# check, a and c), the GPU to itself, its own budget (6-8 minutes, 14.5 at most); with DEC_QUICK=1 (the A10 job), the
# decoder test's smoke run (DEC_QUICK=1 DEC_MODELS=Qwen3-8B, about 5 minutes, 10 at most) once the environment is here,
# before the checks.
# results/summary.txt is rewritten after every step (first line CHECKS PASS or FAIL, then the decoder test's own line),
# results/DONE written by the exit trap however the job ends; no step of (a)-(k) starts past BUDGET (26 minutes from the
# checks' start), each one's timeout ends by END (29 minutes). A session: the environment (none where ~/gpuenv works,
# else about 5 minutes, 15 at most), the checks (29 at most), the decoder test (Hopper, A100: 21 at most; A10's smoke
# run: 10 at most).
#   bash ~/r1_<class>.sh      (in ~: it, r1_job.sh, r1_src.tar, sb_main.tar, r1_summary.py, r1_routes.py, r1_rust.sh,
#                              and for the decoder test dec_more.sh, dec_job.sh and dec_src.tar; x86_64 or aarch64)
# The environment: ~/gpuenv/cuda.sh where it has PyTorch with CUDA 13, nvcc 13, transformers, accelerate, safetensors
# and ninja; else one made here with uv (PyTorch 2.14.0, transformers 5.17.0, cuda-bindings for the decoder test; nvcc
# from NVIDIA's wheels), in ~/gpuenv where there is none, else in W/env; the host's LD_LIBRARY_PATH unset first (logged
# in machine.txt). Rust: cargo on PATH or in ~/.cargo, else rustup's stable, minimal. Downloads: the models, and those
# two where missing.
# Env: BUDGET (1560 s), END (1740 s), MODELS (xcheck's, downloaded whole; set empty: none), LAYER_MODELS (layer 10
# alone; a model in neither list is not waited for), E2E_MODELS, TIME_MODELS, MS (layer.py's token counts), SKIP (steps
# left out: selftest,xcheck,capi,routes,rust,test_gpu,layer,attr,e2e,api,xmodels,layer2,grid,dec,decquick), ONLY (those
# steps alone: the builds and downloads they need, no others), ATTR, DEC, DEC_QUICK, FILES (~), R (~/results), W
# (~/r1w), HF_HOME (~/hf). A step again, alone, in a few minutes (the environment, the branch's library, the step):
# ONLY=routes R=~/results-routes bash ~/r1_<class>.sh; the release candidate's runs: r1_<class>_rc.sh.
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/r1w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
CLASS=${CLASS:?set by r1_hopper.sh, r1_a100.sh, r1_a10.sh or r1_ada.sh}
MODELS=${MODELS-"Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ibm-granite/granite-3.1-3b-a800m-instruct Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B"}
E2E_MODELS=${E2E_MODELS:-"Qwen/Qwen3-1.7B ibm-granite/granite-3.1-3b-a800m-instruct"}
TIME_MODELS=${TIME_MODELS:?} MS=${MS:?}
T0=$(date +%s)  # (reset after the decoder test's smoke run: the checks' own 29 minutes count from there)
T00=$T0
el0() { echo $(( $(date +%s) - T00 )); }
BUDGET=${BUDGET:-1560}
END=${END:-1740}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
skip() {  # STEP: true (and says so) where it is left out: in SKIP, or ONLY is set and it is not there
  case ",${SKIP:-}," in *",$1,"*) echo "$1: left out (SKIP)"; return 0 ;; esac
  [ -z "${ONLY:-}" ] && return 1
  case ",$ONLY," in *",$1,"*) return 1 ;; esac
  echo "$1: left out (not in ONLY)"; return 0
}
need() { local x; for x in "$@"; do skip "$x" > /dev/null || return 0; done; return 1; }  # any of these steps to run
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() { "${PY:-python3}" "$FILES/r1_summary.py" "$R" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
done_() { echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
# run NAME TIMEOUT CMD...: CMD's output in R/NAME.txt, its exit code the file's last line
run() { local n=$1 t=$2; shift 2; timeout "$(tmo "$t")" "$@" > "$R/$n.txt" 2>&1; local e=$?; echo "exit $e" >> "$R/$n.txt"; echo "$n: exit $e"; done_ "$n (exit $e)"; }

step "machine"
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; gcc --version | head -1
  lscpu | grep -E "Model name|Architecture"; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
case $CLASS in ada) WANT=8.9 ;; a10) WANT=8.6 ;; a100) WANT=8.0 ;; hopper) WANT=9.0 ;; *) WANT=? ;; esac
[ "$CC" = "$WANT" ] || echo "NOTE: CLASS=$CLASS expects compute capability $WANT; this GPU is $CC ($NAME): run as it is" | tee -a "$R/machine.txt"
# The host's CUDA libraries never ahead of the environment's own: a Deep Learning AMI's LD_LIBRARY_PATH put another
# cuDNN's engine library ahead of the wheel's (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED on a g6's L4, 2026-09-29)
echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}" >> "$R/machine.txt"
ldconfig -p 2> /dev/null | grep -E "libcudnn(_graph)?\.so|libnvrtc\.so" | head >> "$R/machine.txt"
unset LD_LIBRARY_PATH

step "environment"
envok() {  # PYTHON: the packages, a GPU, and cuDNN's engines loading (a grouped Conv1d, as test_gpu's conv layers); nvcc
  "$1" -c "import torch, transformers, accelerate, safetensors, huggingface_hub; assert torch.cuda.is_available(), 'no GPU for PyTorch'
torch.nn.Conv1d(192, 192, 2, groups=6).cuda()(torch.randn(1, 192, 16, device='cuda')); torch.cuda.synchronize()" && command -v nvcc && "$1" -c "import ninja" ; }
mkenv() {  # $E: uv, Python 3.12, PyTorch 2.14.0 (CUDA 13), transformers 5.17.0 and the rest (cuda-bindings: the decoder
           # test's, which then takes it as ~/gpuenv), nvcc of PyTorch's CUDA; E/cuda.sh
  set -e
  local a uv=$W/uv/uv py=$E/bin/python cu mm
  a=$(uname -m); mkdir -p "$W/uv"
  [ -x "$uv" ] || curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$a-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"
  "$uv" venv -q --python 3.12 "$E"
  "$uv" pip install -q --python "$py" "torch==2.14.0" "transformers==5.17.0" accelerate safetensors numpy huggingface_hub hf_transfer hf_xet ninja cuda-bindings
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
  echo "no environment: $(tail -3 "$R/log/env-make.txt" 2> /dev/null | tr '\n' ' ')" | tee "$R/summary.txt"; exit 1
fi
export PATH="$HOME/.cargo/bin:$PATH"
if need rust && ! command -v cargo > /dev/null; then  # Rust, for the crate's tests and glyd pack: rustup's stable, minimal
  ( curl -sSf --retry 3 https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain stable ) > "$R/log/rustup.txt" 2>&1
  echo "rustup: exit $?" >> "$R/log/rustup.txt"
fi
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; echo "rust: $(cargo --version 2>&1 | head -1)"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TORCH_EXTENSIONS_DIR=$W/torch_ext TOKENIZERS_PARALLELISM=false
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/src" "$W/main" "$W/main-once" && mkdir -p "$W/src" "$W/main" && tar -C "$W/src" -xf "$FILES/r1_src.tar" && tar -C "$W/main" -xf "$FILES/sb_main.tar" || { echo "no r1_src.tar or sb_main.tar in $FILES" | tee "$R/summary.txt"; exit 1; }
B=$W/src/benchmarks/gpu/splitbyte-2026-09-29
echo "$NAME ($CC, sm_$ARCH, ${MIB} MiB), $(uname -m) host, $(nproc) CPUs; CLASS=$CLASS; release tree $(cat "$W/src/COMMIT"), main $(cat "$W/main/COMMIT")" | tee "$R/machine-short.txt"
done_ "machine and environment ($ENVN)"

# The decoder test's 5-minute smoke run (the A10 job), once the environment is here (it takes ~/gpuenv where that has
# cuda-bindings) and before anything else runs: the GPU to itself. Its own budget, 10 minutes at most.
if [ "${DEC_QUICK:-0}" = 1 ] && ! skip decquick; then
  if [ -f "$FILES/dec_job.sh" ] && [ -f "$FILES/dec_src.tar" ]; then
    echo "== $(date +%T) the decoder test's smoke run (DEC_QUICK=1 DEC_MODELS=Qwen3-8B), results/dec"
    R="$R/dec" W="$HOME/decw" FILES="$FILES" DEC_QUICK=1 DEC_MODELS=Qwen3-8B timeout "${DEC_QUICK_TIMEOUT:-600}" bash "$FILES/dec_job.sh" > "$R/log/dec-quick.txt" 2>&1
    echo "decoder smoke run: exit $? at +$(el0) s: $(head -1 "$R/dec/summary.txt" 2> /dev/null)"
    done_ "decoder smoke run ($(head -1 "$R/dec/summary.txt" 2> /dev/null | cut -c1-120))"
  else
    echo "no dec_job.sh or dec_src.tar in $FILES: the decoder test's smoke run skipped"
  fi
fi

T0=$(date +%s)  # (the checks' budget from here)

step "the models in the background, in turn: $MODELS; layer 10 alone: ${LAYER_MODELS:-none}"
dl() {  # REPO [layer]: into the cache (whole, or the index, config.json and layer 10's shards); W/NAME.dir: its directory
  local n; n=$(basename "$1")
  timeout 1200 "$PY" - "$1" "${2:-}" > "$W/$n.dir" 2> "$R/log/dl-$n.txt" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download, snapshot_download
if sys.argv[2] != "layer":
    print(snapshot_download(sys.argv[1], ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "original/*"]))
else:
    idx = hf_hub_download(sys.argv[1], "model.safetensors.index.json")
    hf_hub_download(sys.argv[1], "config.json")
    for f in sorted({f for t, f in json.load(open(idx))["weight_map"].items() if t.startswith("model.layers.10.")}):
        hf_hub_download(sys.argv[1], f)
    print(os.path.dirname(idx))
PY
  echo "exit $? at +$(el) s" >> "$R/log/dl-$n.txt"
}
need rust layer attr e2e api xmodels layer2 && ( for m in $MODELS; do dl "$m"; done; for m in ${LAYER_MODELS:-}; do dl "$m" layer; done ) &
got() {  # REPO: waits for its download; true where its directory holds config.json (false past the budget, or where
          # it is in neither MODELS nor LAYER_MODELS)
  local n; n=$(basename "$1")
  case " $MODELS ${LAYER_MODELS:-} " in *" $1 "*) ;; *) echo "no $1: in neither MODELS nor LAYER_MODELS"; return 1 ;; esac
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 5; done
  [ -f "$(tail -1 "$W/$n.dir" 2> /dev/null)/config.json" ] || { echo "no $1"; false; }
}

step "builds for sm_$ARCH: the branch's library and main's (build_lib.sh's flags), the branch's JIT build, the glyd-gpu crate"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
build() {  # TREE NAME: its gpu/glyd_gpu.cu as TREE/lib/libglyd_gpu_cudaN.so
  mkdir -p "$1/lib"
  { timeout "$(tmo 1200)" nvcc "${F[@]}" -c -o "$1/lib/glyd_gpu.o" "$1/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$1/lib/libglyd_gpu_cuda$MAJOR.so" "$1/lib/glyd_gpu.o"; } > "$R/log/build-$2.txt" 2>&1
  echo "exit $?" >> "$R/log/build-$2.txt"
}
LIB=$W/src/lib/libglyd_gpu_cuda$MAJOR.so MLIB=$W/main/lib/libglyd_gpu_cuda$MAJOR.so OLIB=$W/main-once/lib/libglyd_gpu_cuda$MAJOR.so
export CARGO_TARGET_DIR=$W/target
P=()  # the builds the steps to run need, at once
build "$W/src" src & P+=($!)
need xcheck e2e layer attr xmodels layer2 grid && { build "$W/main" main & P+=($!); }
if [ "${ATTR:-0}" = 1 ] && need attr; then  # main-once (sb_job.sh's): main's kernels, patch()'s loop over the exceptions not unrolled
  mkdir -p "$W/main-once/gpu" && cp "$W/main/gpu/glyd_gpu.h" "$W/main-once/gpu/"
  sed '/static __device__ __forceinline__ void patch(At at, int e0, int e1, int lane, uint32_t ew\[8\]) {/a #pragma unroll 1' "$W/main/gpu/glyd_gpu.cu" > "$W/main-once/gpu/glyd_gpu.cu"
  build "$W/main-once" main-once & P+=($!)
fi
need capi && { ( cd "$W/src/gpu" && env -u GLYD_GPU_LIB MAX_JOBS=4 timeout "$(tmo 1200)" "$PY" -c "import glyd_gpu; print('JIT build:', glyd_gpu._ext)" > "$R/log/build-jit.txt" 2>&1; echo "exit $?" >> "$R/log/build-jit.txt" ) & P+=($!); }
need rust && { ( cd "$W/src/glyd-gpu" && timeout "$(tmo 900)" cargo build --release --bins --examples > "$R/log/build-rust.txt" 2>&1 && timeout "$(tmo 600)" cargo test --release --no-run >> "$R/log/build-rust.txt" 2>&1; echo "exit $?" >> "$R/log/build-rust.txt" ) & P+=($!); }
wait "${P[@]}"
for b in src main main-once jit rust; do [ -f "$R/log/build-$b.txt" ] && echo "build $b: $(tail -1 "$R/log/build-$b.txt")"; done
[ -f "$LIB" ] && { [ -f "$MLIB" ] || ! need xcheck e2e layer xmodels layer2 grid; } || { echo "a library did not build: see log/build-*.txt" | tee -a "$R/steps.txt"; summ; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
done_ "builds ($(for b in src main main-once jit rust; do [ -f "$R/log/build-$b.txt" ] && echo -n "$b $(tail -1 "$R/log/build-$b.txt"), "; done))"

skip selftest || { left && step "(a) the self-test" && ( cd "$W/src/gpu" && run selftest 900 "$PY" -u glyd_gpu.py ); }
skip xcheck || { left && step "(b) xcheck.py, synthetic" && ( cd "$B" && run xcheck 1200 "$PY" -u xcheck.py "$W/main" "$MLIB" ); }
skip capi || { left && step "(c) check_capi" && ( cd "$W/src/gpu" && run check_capi 1200 "$PY" -u check_capi.py "$LIB" ); }
skip routes || { left && step "(c) r1_routes.py: this device's code and routes" && run routes 120 "$PY" -u "$FILES/r1_routes.py"; }

if ! skip rust && left; then
  step "(d) Rust: the crate's tests with the library and the GPU; glyd pack and python -m glyd.gpu pack of Qwen3-0.6B; verify"
  BIN=$W/target/release/glyd-gpu
  ( cd "$W/src/glyd-gpu" && run rust-tests 300 cargo test --release -- --test-threads 1 )
  got Qwen/Qwen3-0.6B && run rust-pack 600 env PY="$PY" BIN="$BIN" CRATE="$W/src/glyd-gpu" SNAP="$(tail -1 "$W/Qwen3-0.6B.dir")" O="$W/q06" bash "$FILES/r1_rust.sh"
fi
skip test_gpu || { left && step "(e) test_gpu.py" && ( cd "$W/src/bindings/python" && run test_gpu 900 "$PY" -u test_gpu.py ); }
if ! skip layer; then
  for m in $TIME_MODELS; do
    left || break
    step "(f) layer.py, $m, run 1"
    got "$m" || continue
    ( cd "$B" && run "layer-$(basename "$m")-run1" 600 "$PY" -u layer.py "$W/main" "$MLIB" "$m" --M "$MS" )
  done
fi
if [ "${ATTR:-0}" = 1 ] && ! skip attr && left; then
  step "(f) the attribution: layer.py at 1, 256 and 1024 tokens, main against main-once (--old-both), then main-once against the branch"
  n=$(( $(grep -c '^#pragma unroll 1$' "$W/main-once/gpu/glyd_gpu.cu") - $(grep -c '^#pragma unroll 1$' "$W/main/gpu/glyd_gpu.cu") ))
  if [ "$n" != 1 ] || [ ! -f "$OLIB" ]; then  # (main-once the same as main would read as a loop that costs nothing)
    run attr 10 sh -c 'echo "$1"; exit 1' - "no attribution: main-once has $n pragma(s) added, 1 expected; its library $([ -f "$OLIB" ] || echo "not ")built (log/build-main-once.txt)"
  else
    for m in $TIME_MODELS; do
      got "$m" || continue
      ( cd "$B" && run "attr-loop-$(basename "$m")" 600 env GLYD_GPU_LIB="$OLIB" "$PY" -u layer.py "$W/main" "$MLIB" "$m" --M 1,256,1024 --old-both )
      ( cd "$B" && run "attr-splitbyte-$(basename "$m")" 600 "$PY" -u layer.py "$W/main" "$OLIB" "$m" --M 1,256,1024 )
      break
    done
  fi
fi
if ! skip e2e && left; then
  step "(g) e2e12.py: $E2E_MODELS, main's then the branch's then main's again"
  for m in $E2E_MODELS; do got "$m" || true; done
  for side in main src main2; do
    left || break
    t=$W/${side%2}
    ( cd "$B" && run "e2e12-$side" 900 env GLYD_COMPILE=0 GLYD_GPU_LIB="$t/lib/libglyd_gpu_cuda$MAJOR.so" PYTHONPATH="$t/bindings/python" "$PY" -u e2e12.py $E2E_MODELS )
  done
  { cmp -s <(grep -E "logits|tokens:" "$R/e2e12-main.txt") <(grep -E "logits|tokens:" "$R/e2e12-main2.txt") && echo "main twice: the same lines" || echo "main twice: DIFFER"
    cmp -s <(grep -E "logits|tokens:" "$R/e2e12-main.txt") <(grep -E "logits|tokens:" "$R/e2e12-src.txt") &&
      echo "main and split byte: the same lines ($(grep -c logits "$R/e2e12-src.txt") logits, $(grep -c 'tokens: \[' "$R/e2e12-src.txt") generations)" || echo "main and split byte: DIFFER"; } > "$R/e2e12-compare.txt" 2>&1
  cat "$R/e2e12-compare.txt"; done_ "e2e12 compared"
fi
if ! skip api; then
  left && step "(h) check_api dense" && got Qwen/Qwen3-0.6B && ( cd "$W/src/gpu" && run check_api-dense 900 "$PY" -u check_api.py Qwen/Qwen3-0.6B )
  left && step "(h) check_api MoE" && got ibm-granite/granite-3.1-3b-a800m-instruct && ( cd "$W/src/gpu" && run check_api-moe 900 "$PY" -u check_api.py ibm-granite/granite-3.1-3b-a800m-instruct )
fi
export HF_HUB_OFFLINE=1  # the models from the cache from here on
if ! skip xmodels; then
  for m in $MODELS; do
    left || break
    step "(i) xcheck.py, $m (waiting for its download)"
    got "$m" || continue
    ( cd "$B" && run "xcheck-$(basename "$m")" 900 env SYNTHETIC=0 "$PY" -u xcheck.py "$W/main" "$MLIB" "$m" )
  done
fi
if ! skip layer2; then
  for m in $TIME_MODELS; do
    left || break
    step "(j) layer.py, $m, run 2"
    got "$m" || continue
    ( cd "$B" && run "layer-$(basename "$m")-run2" 600 "$PY" -u layer.py "$W/main" "$MLIB" "$m" --M "$MS" )
  done
fi
if [ "$CLASS" = ada ] && [[ $NAME == *GeForce* ]] && ! skip grid && left; then
  step "(k) the grid prompt kernels: both trees built with GeForce's test never matching, xcheck.py's synthetic part"
  for side in src main; do
    mkdir -p "$W/grid/$side/gpu" && sed 's/"GeForce")/"GeForce-not")/' "$W/$side/gpu/glyd_gpu.cu" > "$W/grid/$side/gpu/glyd_gpu.cu" && cp "$W/$side/gpu/glyd_gpu.h" "$W/grid/$side/gpu/"
    echo "grid $side: $(grep -c '"GeForce-not")' "$W/grid/$side/gpu/glyd_gpu.cu") tests patched"
    build "$W/grid/$side" "grid-$side"
  done
  ( cd "$B" && run xcheck-grid 900 env GLYD_GPU_LIB="$W/grid/src/lib/libglyd_gpu_cuda$MAJOR.so" "$PY" -u xcheck.py "$W/main" "$W/grid/main/lib/libglyd_gpu_cuda$MAJOR.so" )
fi
step "the checks done in $(el) s"
done_ "checks done"

# The decoder test's follow-up to option 2's test (the Hopper and A100 jobs: dec_more.sh, dec_job.sh's steps check, a
# and c), after everything here: the GPU to itself, its own budget (DEC_BUDGET 780 s, DEC_END 870 s: 6-8 minutes, 14.5
# at most), its results in results/dec (its DONE there; this job's at the end).
if [ "${DEC:-0}" = 1 ] && ! skip dec; then
  if [ -f "$FILES/dec_more.sh" ] && [ -f "$FILES/dec_job.sh" ] && [ -f "$FILES/dec_src.tar" ]; then
    step "the decoder test's follow-up: R=results/dec bash dec_more.sh"
    R="$R/dec" W="$HOME/decw" FILES="$FILES" timeout 900 env -u HF_HUB_OFFLINE bash "$FILES/dec_more.sh" > "$R/log/dec.txt" 2>&1  # (its models: the Hub)
    echo "decoder test: exit $?: $(head -1 "$R/dec/summary.txt" 2> /dev/null)"
    done_ "decoder test ($(head -1 "$R/dec/summary.txt" 2> /dev/null | cut -c1-120))"
  else
    echo "no dec_more.sh, dec_job.sh or dec_src.tar in $FILES: the decoder test skipped"; done_ "decoder test: no files, skipped"
  fi
fi
step "done in $(el0) s"
done_ "done"
