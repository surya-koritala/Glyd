#!/usr/bin/env bash
# One unattended H100 job for gpu-hopper2 (3add84a): (a) CUDA 12 against CUDA 13, with and without the accumulator
# zeroed in both Hopper kernels (CUDA 12.8's ptxas serializes every wgmma where it is left unset, its C7515); (b) N1's
# whole-tiles rule in mma12_wgp_run; (c) one e2e check with the winning library. In that order: past the budget the
# steps left are skipped, and results/summary.txt is written after every step, so a cap still leaves partial answers.
#   bash ~/h100_cu12.sh      (in ~: this script, cu12_src.tar, cu12_fix.cu, cu12_fixn1.cu, cu12_fetch.sh,
#                             cu12_check.py, cu12_layer.py, cu12_summary.py; ~/gpuenv/cuda.sh: PyTorch and nvcc 13)
# Libraries (sm_90a alone, build_lib.sh's compile line with -Xptxas -v): cu12-base, cu12-fix, cu13-base, cu13-fix,
# cu13-fixn1 (base: 3add84a; fix: d zeroed in mma12_tma_kernel and mma12_wgp_kernel; fixn1: that and the rule).
# CUDA 12: the instance's own where it is 12.8 (the CUDA 12 the release builds with; any other CUDA 12 found is recorded
# in toolchains.txt), else NVIDIA's 12.8.2 redistributables fetched by cu12_fetch.sh, sha256-checked.
# Models: through the Hugging Face cache (HF_HOME, ~/hf), layer 10's shards of each and Qwen3-8B whole for the e2e check.
# Env, for a smoke test elsewhere: ARCHS (90a), KERNEL (wg on Hopper, else mid), N8 N14 N32 NE (the repos in Qwen/:
# Qwen3-8B, Qwen3-14B, Qwen3-32B, and the e2e check's, Qwen3-8B), C12 (a CUDA 12 dir), CCBIN12 (its host compiler),
# BUDGET (1560 s: no step starts past it) and E2E_BY (1260 s: nor the e2e check), MS_A, MS_B, CHECK_MAX, E2E_PREFILL,
# FILES (~: the files above), R (~/results), W (~/cu12w).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/cu12w}
mkdir -p "$R/log" "$W"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${BUDGET:-1560}  # s: past it no step starts (a step's own timeouts keep the job within its 30 minutes)
E2E_BY=${E2E_BY:-1260}  # s: past it the e2e check (at most 9 minutes) does not start
FILES=${FILES:-$HOME}
ARCHS=${ARCHS:-90a}
MS_A=${MS_A:-17,64,128,129,256,512,1024}
MS_B=${MS_B:-129,160,256,384,512,1024}
E2E_PREFILL=${E2E_PREFILL:-256,1024}
LIBS="cu12-base cu12-fix cu13-base cu13-fix cu13-fixn1"
step() { echo "== $(date +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
left() { [ $(( $(date +%s) - T0 )) -lt "${1:-$BUDGET}" ] || { echo "over the budget: skipped"; false; }; }
done_() { echo "$(date +%T) (+$(( $(date +%s) - T0 )) s) $*" >> "$R/steps.txt"; summ; }
summ() { python "$FILES/cu12_summary.py" "$R" > "$R/summary.tmp" 2> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
source ~/gpuenv/cuda.sh
C13=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
python -c "import hf_transfer" 2>/dev/null || unset HF_HUB_ENABLE_HF_TRANSFER

step "machine"
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"; lscpu | grep "Model name"; nproc; free -g | head -2
  gcc --version | head -1; ls -d /usr/local/cuda* /usr/lib/cuda 2>&1; which -a nvcc; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
KERNEL=${KERNEL:-$([ "$CC" = "9.0" ] && echo wg || echo mid)}
echo "$(nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,power.limit --format=csv,noheader | head -1); kernels timed: $KERNEL" > "$R/machine-short.txt"
done_ "machine"

step "models, in the background: layer 10 of Qwen3-8B, 14B and 32B, and Qwen3-8B whole (the e2e check)"
N8=${N8:-Qwen3-8B} N14=${N14:-Qwen3-14B} N32=${N32:-Qwen3-32B} NE=${NE:-Qwen3-8B}
getlayer() {  # REPO: the index, config.json and the shards holding layer 10, into the cache; W/REPO.dir: its directory
  timeout 900 python - "Qwen/$1" > "$W/$1.dir" 2> "$R/log/dl-$1.txt" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download
idx = hf_hub_download(sys.argv[1], "model.safetensors.index.json")
hf_hub_download(sys.argv[1], "config.json")
for f in sorted({f for t, f in json.load(open(idx))["weight_map"].items() if t.startswith("model.layers.10.")}):
    hf_hub_download(sys.argv[1], f)
print(os.path.dirname(idx))
PY
  echo "exit $?" >> "$R/log/dl-$1.txt"
}
getlayer "$N8" & DL8=$!
getlayer "$N14" & DL14=$!
getlayer "$N32" & DL32=$!
# (one cache: a shard two of these want is fetched once, the second waiting on the first's lock)
( timeout 1200 python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "Qwen/$NE" > "$W/e2e.dir" 2> "$R/log/dl-e2e-$NE.txt"
  echo "exit $?" >> "$R/log/dl-e2e-$NE.txt" ) & DLE=$!

step "sources: 3add84a, and the two variants"
for v in base fix fixn1; do rm -rf "$W/$v" && mkdir -p "$W/$v" && tar -C "$W/$v" -xf "$FILES/cu12_src.tar"; done
cp "$FILES/cu12_fix.cu" "$W/fix/gpu/glyd_gpu.cu" && cp "$FILES/cu12_fixn1.cu" "$W/fixn1/gpu/glyd_gpu.cu"
{ diff -u "$W/base/gpu/glyd_gpu.cu" "$W/fix/gpu/glyd_gpu.cu"; diff -u "$W/fix/gpu/glyd_gpu.cu" "$W/fixn1/gpu/glyd_gpu.cu"; } > "$R/variants.diff"
done_ "sources ($(grep -c '^[-+][^-+]' "$R/variants.diff") changed lines in variants.diff)"

step "CUDA 12: the instance's own where it is 12.8, else NVIDIA's 12.8.2 redistributables"
C12=${C12:-}
if [ -z "$C12" ]; then
  for n in /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc /usr/lib/cuda/bin/nvcc /usr/bin/nvcc; do
    v=$([ -x "$n" ] && "$n" --version 2>/dev/null | grep -o "release 12\.[0-9]*")
    [ -n "$v" ] && echo "the instance's CUDA 12: $n, $v" >> "$R/log/cuda12-found.txt"
    [ "$v" = "release 12.8" ] && [ -z "$C12" ] && C12=$(cd "$(dirname "$n")/.." && pwd)
  done
  [ -n "$C12" ] && echo "CUDA 12: the instance's, $C12" || { echo "no CUDA 12.8 on the instance: fetching"; timeout 600 bash "$FILES/cu12_fetch.sh" "$W/cu12" > "$R/log/cu12_fetch.txt" 2>&1 && C12=$W/cu12; }
fi
CCBIN12=${CCBIN12:-}
if [ -n "$C12" ] && [ -z "$CCBIN12" ] && [ "$(gcc -dumpversion | cut -d. -f1)" -gt 14 ]; then  # (CUDA 12.8 takes GCC to 14)
  for g in g++-14 g++-13 g++-12; do command -v $g > /dev/null && { CCBIN12=$(command -v $g); break; }; done
fi
{ echo "CUDA 13: $C13: $("$C13/bin/nvcc" --version | tail -2 | tr '\n' ' ')"
  [ -n "$C12" ] && echo "CUDA 12: $C12: $("$C12/bin/nvcc" --version | tail -2 | tr '\n' ' ')${CCBIN12:+ (host compiler $CCBIN12)}" || echo "CUDA 12: NONE (the cu12 libraries are skipped)"
  cat "$R/log/cuda12-found.txt" 2>/dev/null || echo "the instance's CUDA 12: none"
  echo "gcc: $(gcc --version | head -1)"; } | tee "$R/toolchains.txt"
done_ "CUDA 12 ${C12:-missing}"

step "the five libraries, sm_90a alone ($ARCHS), in parallel"
build() {  # NAME SRC CUDA_DIR [HOST_COMPILER]: build_lib.sh's compile line, -Xptxas -v kept in results/build-NAME.txt
  local n=$1 src=$2 cu=$3 cc=${4:-} gen="" a
  for a in $ARCHS; do gen="$gen -gencode arch=compute_$a,code=sm_$a"; done
  local F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$cu/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ $gen -Xcompiler -fPIC,-fvisibility=hidden)
  [ -n "$cc" ] && F+=(-ccbin "$cc")
  { "$cu/bin/nvcc" --version | tail -2
    timeout 900 "$cu/bin/nvcc" "${F[@]}" -Xptxas -v -c -o "$W/$n.o" "$src/gpu/glyd_gpu.cu" &&
    timeout 300 "$cu/bin/nvcc" "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$cu/lib" -L"$cu/lib64" -o "$W/$n.so" "$W/$n.o" &&
    touch "$R/lib-$n.ok"; } > "$R/build-$n.txt" 2>&1
  echo "build $n exit $?"
}
BJ=""
[ -n "$C12" ] && { build cu12-base "$W/base" "$C12" "$CCBIN12" & BJ="$BJ $!"; build cu12-fix "$W/fix" "$C12" "$CCBIN12" & BJ="$BJ $!"; }
build cu13-base "$W/base" "$C13" & BJ="$BJ $!"
build cu13-fix "$W/fix" "$C13" & BJ="$BJ $!"
build cu13-fixn1 "$W/fixn1" "$C13" & BJ="$BJ $!"
wait $BJ
# The SASS: of each mma12_wgp_kernel<256> and mma12_tma_kernel<64> HGMMA, whether a wait for every product in flight
# (WARPGROUP.DEPBAR.LE gsb0, 0x0) comes before the next: ptxas's serialization, seen in the code it made.
CO=$(ls "$C13/bin/cuobjdump" ~/gpuenv/lib/python3*/site-packages/triton/backends/nvidia/bin/cuobjdump /usr/local/cuda*/bin/cuobjdump 2>/dev/null | head -1)
for n in $LIBS; do
  [ -f "$R/lib-$n.ok" ] && [ -n "$CO" ] || continue
  "$CO" -sass -arch sm_90a "$W/$n.so" > "$W/$n.sass" 2>&1
  python - "$W/$n.sass" "$R/log/hgmma-$n.txt" > "$R/sass-$n.txt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
out = []
for label, name in (("wgp256", "_Z16mma12_wgp_kernelILi256ELi2EE"), ("tma64", "_Z16mma12_tma_kernelILi64ELi2EE")):
    m = re.search(r"Function : (" + name + r"\S*)(.*?)(?=\n\s*Function : |\Z)", text, re.S)
    if not m:
        out.append(f"{label} not found")
        continue
    ops = [f"{a} {o.strip()}" for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+([^;]*);", m.group(2))]  # (address, instruction)
    idx = [i for i, o in enumerate(ops) if "HGMMA" in o]
    waited = sum(any("DEPBAR.LE gsb0, 0x0" in o for o in ops[i + 1:(idx[j + 1] if j + 1 < len(idx) else len(ops))]) for j, i in enumerate(idx))
    out.append(f"{label} {waited}/{len(idx)} HGMMAs waited out")
    with open(sys.argv[2], "a") as f:  # (the two kernels' HGMMAs and waits, for the record)
        f.write(f"== {m.group(1)}\n" + "\n".join(o for o in ops if "HGMMA" in o or "DEPBAR" in o) + "\n")
print("SASS: " + ", ".join(out))
PY
done
done_ "builds: $(ls "$R"/lib-*.ok 2>/dev/null | wc -l) of $([ -n "$C12" ] && echo 5 || echo 3)"

step "(a) each library's Hopper products: within 1e-2 of fp32, the same every run, bit for bit against the others"
for n in $LIBS; do
  [ -f "$R/lib-$n.ok" ] && left || continue
  GLYD_GPU_DIR="$W/base/gpu" GLYD_GPU_LIB="$W/$n.so" timeout 600 python "$FILES/cu12_check.py" "$R/check-$n.json" > "$R/log/check-$n.txt" 2>&1
  echo "check $n exit $?"; done_ "check $n"
done

step "(a) layers against cuBLAS, each library, two rounds: Qwen3-8B's four products, 32B's o and gate_up"
wait $DL8 $DL32
MODEL8=$(tail -1 "$W/$N8.dir" 2>/dev/null) MODEL32=$(tail -1 "$W/$N32.dir" 2>/dev/null)
for r in 1 2; do
  for n in cu13-base cu12-base cu13-fix cu12-fix; do
    [ -f "$R/lib-$n.ok" ] && left || continue
    for spec in "$N8 $MODEL8 qkv,o,gate_up,down" "$N32 $MODEL32 o,gate_up"; do
      set -- $spec
      [ -f "$2/model.safetensors.index.json" ] || { echo "no model at $2"; continue; }
      GLYD_GPU_DIR="$W/base/gpu" GLYD_GPU_LIB="$W/$n.so" timeout 600 python "$FILES/cu12_layer.py" "$2" --name "$1" --tag "$n-r$r" --only "$3" --M "$MS_A" --kernel "$KERNEL" >> "$R/layer-a.txt" 2>> "$R/log/layer-a.err"
    done
    done_ "layers $n round $r"
  done
done

step "(b) the whole-tiles rule (cu13-fixn1) against library 4 (cu13-fix): 8B's and 14B's o and qkv, 32B's o"
wait $DL14
MODEL14=$(tail -1 "$W/$N14.dir" 2>/dev/null)
for r in 1 2; do
  for n in cu13-fix cu13-fixn1; do
    [ -f "$R/lib-$n.ok" ] && left || continue
    for spec in "$N8 $MODEL8 o,qkv" "$N14 $MODEL14 o,qkv" "$N32 $MODEL32 o"; do
      set -- $spec
      [ -f "$2/model.safetensors.index.json" ] || { echo "no model at $2"; continue; }
      GLYD_GPU_DIR="$W/base/gpu" GLYD_GPU_LIB="$W/$n.so" timeout 600 python "$FILES/cu12_layer.py" "$2" --name "$1" --tag "$n-r$r" --only "$3" --M "$MS_B" --kernel "$KERNEL" >> "$R/layer-b.txt" 2>> "$R/log/layer-b.err"
    done
    done_ "rule $n round $r"
  done
done

step "(c) e2e with the winning library: prompts of $E2E_PREFILL tokens and generated tokens, then --exact"
wait $DLE
E2E_MODEL=$(tail -1 "$W/e2e.dir" 2>/dev/null)
WIN=$(python "$FILES/cu12_summary.py" "$R" --winner)
echo "$WIN" > "$R/winner.txt"
if [ -f "$R/lib-$WIN.ok" ] && [ -f "$E2E_MODEL/config.json" ] && left "$E2E_BY"; then
  cd "$W/base/gpu"
  GLYD_GPU_LIB="$W/$WIN.so" timeout 300 python e2e.py "$E2E_MODEL" --format auto --fused --merge --baseline --prefill "$E2E_PREFILL" --tokens 8 --batch 1 > "$R/e2e.txt" 2>&1
  echo "e2e exit $?"; done_ "e2e $NE $WIN"
  GLYD_GPU_LIB="$W/$WIN.so" timeout 240 python e2e.py "$E2E_MODEL" --format auto --exact --baseline --tokens 8 --prefill "$E2E_PREFILL" > "$R/e2e-exact.txt" 2>&1
  echo "e2e exact exit $?"; done_ "e2e exact $NE $WIN"
  cd - > /dev/null
fi

step "done in $(( $(date +%s) - T0 )) s"
done_ "done"
touch "$R/DONE"
