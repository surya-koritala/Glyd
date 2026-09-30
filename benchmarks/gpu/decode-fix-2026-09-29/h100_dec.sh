#!/usr/bin/env bash
# The 12-bit decode kernel's load orders (branch decode-fix, for v0.25.1) against main's decode (the 12-bit layout
# before split byte) and v0.25.0's, on one Hopper (an H100 SXM or PCIe, a GH200), or on an A10 (HDEC_MODE=a10, the
# default on any GPU but Hopper). Unattended, on an x86_64 or aarch64 host; in ~: this and hdec_src.tar (fix/:
# decode-fix, rel/: v0.25.0's tree, main/: origin/main, each with its COMMIT). In order, so that a cap leaves the first
# answers:
#   builds   the three libraries for this GPU alone (build_lib.sh's flags), at once; the models download meanwhile (layer
#            10 of TIME_MODELS, then E2E_MODEL whole), and where there is no ncu, NVIDIA's for this CUDA
#   bits     the self-test and xcheck.py's synthetic part with GLYD_DEC_ORDER 0, 1, 2 and 3: each order's decodes (a
#            warp a step, a few warps, a mixture of experts'; every bf16 bit pattern) the weights' bits and main's
#   kernel   dec_time.py, run 1: layer 10 of each of TIME_MODELS, main's decode, v0.25.0's and orders 0-3 in one
#            process, each checked bit for bit, then each call timed alone after an L2 flush, the median of 21 (whole:
#            a warp a step, for cuBLAS and exact mode; ahead: 2 warps an SM, the A10's decode ahead)
#   ncu      dec_prof.py under Nsight Compute: the first TIME_MODEL's layer-10 gate,up decoded once by each variant
#            (A10: whole and ahead); DRAM throughput and bytes, occupancy, stall reasons. The counters need this user
#            allowed to read them (root, or NVreg_RestrictProfilingToAdminUsers=0): else ERR_NVGPUCTRPERM, logged, and
#            the job goes on
#   e2e      dec_e2e.py on E2E_MODEL: exact mode's steps (TOKENS greedy tokens) and prompts of PROMPTS tokens, exact
#            then fused (Hopper: past wgmma's 1024, decoded for cuBLAS; A10: from 640, decoded ahead); this tree's orders
#            in one process, then main's library and v0.25.0's in their own; every output's sha256 compared across all
#   kernel2  dec_time.py, run 2; then main's end to end again (e2e-main2), where time is left
# results/summary.txt (dec_summary.py; first line BITS PASS or FAIL) is rewritten after every step, results/DONE written
# by the exit trap however the job ends. No step starts past HDEC_BUDGET, and each one's timeout ends by HDEC_END,
# counted from the environment: with ~/gpuenv (PyTorch with CUDA 13, nvcc 13, transformers) about 10 minutes on an
# H100 and 12 on an A10, 14.5 at most; where there is none, one is made here with uv first (about 5 minutes more).
#   bash ~/h100_dec.sh       (an A10: the same; the GPU to itself)
# Env: HDEC_MODE (hopper or a10: by the GPU), HDEC_BUDGET (720 s), HDEC_END (840 s), HDEC_STEPS (bits,kernel,ncu,e2e,
# kernel2), HDEC_NCU (0: no profile), HDEC_TIME_MODELS, HDEC_E2E_MODEL, HDEC_PROMPTS, HDEC_TOKENS, HDEC_REPS; FILES
# (~), R (~/results), W (~/hdecw), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/hdecw}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${HDEC_BUDGET:-720}
END=${HDEC_END:-840}
STEPS=${HDEC_STEPS:-bits,kernel,ncu,e2e,kernel2}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date +%T) (+$(el) s) $*"; }
left() { [ "$(el)" -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
want() { case ",$STEPS," in *",$1,"*) return 0 ;; esac; echo "$1: not in HDEC_STEPS"; return 1; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
summ() { [ -n "${PY:-}" ] && [ -f "${D:-}/dec_summary.py" ] && "$PY" "$D/dec_summary.py" "$R" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
done_() { echo "$(date +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
fail() { echo "FAIL: $*" | tee "$R/summary.txt"; exit 1; }
# run NAME TIMEOUT CMD...: CMD's output in R/NAME.txt, its exit code the file's last line
run() { local n=$1 t=$2; shift 2; timeout "$(tmo "$t")" "$@" > "$R/$n.txt" 2>&1; local e=$?; echo "exit $e" >> "$R/$n.txt"; echo "$n: exit $e"; done_ "$n (exit $e)"; return $e; }

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
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
MODE=${HDEC_MODE:-$([ "$CC" = "9.0" ] && echo hopper || echo a10)}
if [ "$MODE" = hopper ]; then
  TIME_MODELS=${HDEC_TIME_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B"} PROMPTS=${HDEC_PROMPTS:-1280,2048,4096}
  TOKENS=${HDEC_TOKENS:-64} REPS=${HDEC_REPS:-5} NCU_KINDS=whole
else
  TIME_MODELS=${HDEC_TIME_MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507"} PROMPTS=${HDEC_PROMPTS:-1024,2048}
  TOKENS=${HDEC_TOKENS:-32} REPS=${HDEC_REPS:-4} NCU_KINDS=whole,ahead
fi
E2E_MODEL=${HDEC_E2E_MODEL:-Qwen/Qwen3-8B}

step "environment"
envok() {  # PYTHON: the packages and a GPU; nvcc
  "$1" -c "import torch, transformers, safetensors, huggingface_hub; assert torch.cuda.is_available(), 'no GPU for PyTorch'" && command -v nvcc
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
  fail "no environment: $(tail -3 "$R/log/env-make.txt" 2> /dev/null | tr '\n' ' ')"
fi
T0=$(date +%s)  # (the budget from here)
{ echo "environment: $ENVN; $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__)")"
  echo "nvcc: $(nvcc --version | tail -1)"; } | tee "$R/env.txt"
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
"$PY" -c "import hf_transfer" 2> /dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
rm -rf "$W/fix" "$W/rel" "$W/main" && tar -C "$W" -xf "$FILES/hdec_src.tar" || fail "no hdec_src.tar in $FILES"
D=$W/fix/benchmarks/gpu/decode-fix-2026-09-29 B=$W/fix/benchmarks/gpu/splitbyte-2026-09-29
echo "$NAME ($CC, sm_$ARCH), $(uname -m) host, $(nproc) CPUs; mode $MODE; decode-fix $(cat "$W/fix/COMMIT"), v0.25.0 $(cat "$W/rel/COMMIT"), main $(cat "$W/main/COMMIT")" | tee "$R/machine-short.txt"
done_ "machine and environment ($ENVN)"

step "in the background: layer 10 of $TIME_MODELS, then $E2E_MODEL whole"
dl() {  # REPO KIND (layer: the index, config.json and layer 10's shards; whole): W/NAME-KIND.dir, its directory
  local n; n=$(basename "$1")-$2
  timeout 1200 "$PY" - "$1" "$2" > "$W/$n.dir" 2> "$R/log/dl-$n.txt" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download, snapshot_download
if sys.argv[2] == "whole":
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
( for m in $TIME_MODELS; do dl "$m" layer; done; dl "$E2E_MODEL" whole ) &
got() {  # REPO KIND: its directory once its download is done (nothing if it failed, or past the budget)
  local n d; n=$(basename "$1")-$2
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 3; done
  d=$(tail -1 "$W/$n.dir" 2> /dev/null); [ -f "$d/config.json" ] && echo "$d"
}
findncu() { { command -v ncu; ls -d /usr/local/cuda/bin/ncu /usr/local/cuda*/bin/ncu /usr/local/cuda*/nsight-compute*/ncu /opt/nvidia/nsight-compute/*/ncu; find "$W/ncu" -maxdepth 4 -type f -name ncu; } 2> /dev/null | head -1; }
getncu() {  # NVIDIA's redistributable Nsight Compute for this CUDA (major.minor, its latest update), sha256-checked, in W/ncu
  set -e
  local a plat v u rel sha B=https://developer.download.nvidia.com/compute/cuda/redist
  a=$(uname -m); plat=$([ "$a" = aarch64 ] && echo linux-sbsa || echo linux-x86_64)
  v=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  rm -rf "$W/ncu" && mkdir -p "$W/ncu"
  for u in 5 4 3 2 1 0; do curl -sSfL --retry 2 -o "$W/ncu/manifest.json" "$B/redistrib_$v.$u.json" && break; done
  read -r rel sha < <("$PY" -c "import json; x = json.load(open('$W/ncu/manifest.json'))['nsight_compute']['$plat']; print(x['relative_path'], x['sha256'])")
  echo "$v.$u: $rel"
  curl -sSfL --retry 3 -o "$W/ncu/nc.tar.xz" "$B/$rel"
  echo "$sha  $W/ncu/nc.tar.xz" | sha256sum -c --quiet -
  tar -xJf "$W/ncu/nc.tar.xz" -C "$W/ncu" --strip-components=1 && rm -f "$W/ncu/nc.tar.xz"
}
NCU=""
if [ "${HDEC_NCU:-1}" != 0 ] && want ncu > /dev/null; then
  NCU=$(findncu)
  [ -n "$NCU" ] || ( timeout 600 bash -c "$(declare -f getncu); W='$W' PY='$PY' getncu"; echo "exit $?" ) > "$R/log/ncu-fetch.txt" 2>&1 &
fi

step "builds for sm_$ARCH: decode-fix, v0.25.0 and main (build_lib.sh's flags), at once"
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
build() {  # TREE NAME: its gpu/glyd_gpu.cu as TREE/lib/libglyd_gpu_cudaN.so
  mkdir -p "$1/lib"
  { timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$1/lib/glyd_gpu.o" "$1/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$1/lib/libglyd_gpu_cuda$MAJOR.so" "$1/lib/glyd_gpu.o"; } > "$R/log/build-$2.txt" 2>&1
  echo "exit $?" >> "$R/log/build-$2.txt"
}
build "$W/fix" fix & P1=$!
build "$W/rel" rel & P2=$!
build "$W/main" main & P3=$!
wait $P1 $P2 $P3
FLIB=$W/fix/lib/libglyd_gpu_cuda$MAJOR.so RLIB=$W/rel/lib/libglyd_gpu_cuda$MAJOR.so MLIB=$W/main/lib/libglyd_gpu_cuda$MAJOR.so
for b in fix rel main; do echo "build $b: $(tail -1 "$R/log/build-$b.txt")"; done
[ -f "$FLIB" ] && [ -f "$RLIB" ] && [ -f "$MLIB" ] || { done_ "builds: a library did not build (log/build-*.txt)"; exit 1; }
OBJ=$(command -v cuobjdump || ls "$CU/bin/cuobjdump" /usr/local/cuda/bin/cuobjdump 2> /dev/null | head -1)
[ -n "$OBJ" ] && for b in main rel fix; do echo "== $b"; "$OBJ" -res-usage "$W/$b/lib/libglyd_gpu_cuda$MAJOR.so" 2>&1 | grep -A1 "mma_unpack_kernelI3Nib" | grep -v "^--"; done > "$R/registers.txt"
export GLYD_GPU_LIB=$FLIB PYTHONPATH=$W/fix/bindings/python
done_ "builds (fix $(tail -1 "$R/log/build-fix.txt"), rel $(tail -1 "$R/log/build-rel.txt"), main $(tail -1 "$R/log/build-main.txt"))"

if want bits; then
  for o in 0 1 2 3; do
    left || break
    step "bits, GLYD_DEC_ORDER=$o: the self-test, then xcheck.py's synthetic part (against main's library)"
    ( cd "$W/fix/gpu" && run "selftest-o$o" 300 env GLYD_DEC_ORDER=$o "$PY" -u glyd_gpu.py )
    ( cd "$B" && run "xcheck-o$o" 600 env GLYD_DEC_ORDER=$o "$PY" -u xcheck.py "$W/main" "$MLIB" )
  done
fi
kernel() {  # RUN: dec_time.py on each of TIME_MODELS
  local m d
  for m in $TIME_MODELS; do
    left || break
    step "kernel: dec_time.py, $m, run $1 (waiting for its download)"
    d=$(got "$m" layer) || { echo "no $m: $(tail -2 "$R/log/dl-$(basename "$m")-layer.txt" 2> /dev/null | tr '\n' ' ')"; continue; }
    ( cd "$D" && run "time-$(basename "$m")-run$1" 600 "$PY" -u dec_time.py "$W/main" "$MLIB" "$RLIB" "$d" )
  done
}
want kernel && kernel 1
if want ncu && [ "${HDEC_NCU:-1}" != 0 ] && left; then
  step "ncu: dec_prof.py, the layer-10 gate,up of $(echo $TIME_MODELS | cut -d' ' -f1), each variant once ($NCU_KINDS)"
  if [ -z "$NCU" ]; then  # NVIDIA's, fetched meanwhile: its end awaited (to 4 minutes before END)
    until grep -q "^exit" "$R/log/ncu-fetch.txt" 2> /dev/null || [ "$(el)" -ge $(( END - 240 )) ]; do sleep 3; done
    NCU=$(findncu)
  fi
  d=$(got "$(echo $TIME_MODELS | cut -d' ' -f1)" layer)
  if [ -z "$NCU" ]; then
    done_ "ncu: none here, and none fetched ($(tail -2 "$R/log/ncu-fetch.txt" 2> /dev/null | tr '\n' ' '))"
  elif [ -z "$d" ]; then
    done_ "ncu: no model"
  else
    mkdir -p "$R/ncu"
    echo "ncu: $NCU ($("$NCU" --version 2>&1 | tail -1))" | tee "$R/ncu/version.txt"
    ( cd "$D" && run prof 300 "$NCU" --target-processes all -k regex:mma_unpack_kernel --cache-control all --clock-control none \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SchedulerStats \
        --section LaunchStats --metrics dram__bytes_read.sum,dram__bytes_write.sum -f -o "$R/ncu/dec" \
        "$PY" -u dec_prof.py "$W/main" "$MLIB" "$RLIB" "$d" --kinds "$NCU_KINDS" )
    if [ -f "$R/ncu/dec.ncu-rep" ]; then
      "$NCU" -i "$R/ncu/dec.ncu-rep" --page raw --csv --print-units base > "$R/ncu/raw.csv" 2> "$R/log/ncu-raw.txt"
      "$NCU" -i "$R/ncu/dec.ncu-rep" --page details > "$R/ncu/details.txt" 2>&1
    fi
    grep -m1 -E "ERR_NVGPUCTRPERM|permission" "$R/prof.txt" && echo "ncu: this user may not read the GPU's counters (root, or NVreg_RestrictProfilingToAdminUsers=0)"
    done_ "ncu ($(ls "$R/ncu" | tr '\n' ' '))"
  fi
fi
e2e() {  # NAME TREE LIB ORDERS: dec_e2e.py on E2E_MODEL with TREE's package and LIB
  ( cd "$D" && run "e2e-$1" 900 env GLYD_COMPILE=0 GLYD_GPU_LIB="$3" PYTHONPATH="$2/bindings/python" "$PY" -u dec_e2e.py "$DE" --orders "$4" --tokens "$TOKENS" --reps "$REPS" --prompts "$PROMPTS" )
}
DE=""
if want e2e && left; then
  step "e2e: dec_e2e.py, $E2E_MODEL (waiting for its download): this tree's orders 0-3, then main's, then v0.25.0's"
  DE=$(got "$E2E_MODEL" whole) || echo "no $E2E_MODEL: $(tail -2 "$R/log/dl-$(basename "$E2E_MODEL")-whole.txt" 2> /dev/null | tr '\n' ' ')"
  if [ -n "$DE" ]; then
    e2e fix "$W/fix" "$FLIB" 0,1,2,3
    left && e2e main "$W/main" "$MLIB" ""
    left && e2e v0.25.0 "$W/rel" "$RLIB" ""
  fi
fi
want kernel2 && kernel 2
[ -n "$DE" ] && want e2e > /dev/null && left && { step "e2e: main's again"; e2e main2 "$W/main" "$MLIB" ""; }
step "done in $(el) s"
done_ "done"
