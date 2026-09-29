#!/usr/bin/env bash
# One short unattended H100 job: gpu-hopper2's committed state (val_src.tar), after re-review 2's cap. In order:
# (a) its library for CUDA 12 and 13, sm_90a alone, -Xptxas -v kept (C7515 a kernel) and the SASS's waits; (b) the full
# self-test (glyd_gpu.py, the (8960, 128) matrix included) through each, and each library's mma_gemm_wg outputs on the
# self-test's matrices by sha256 against each other and against the h100f run's (prev-check-*.json: base's or fixn1's,
# whichever split the committed rule takes); (c) Qwen3-8B's q, k, v, o, gate_up and down and 14B's o and q, k, v at 17,
# 128, 129, 256, 512 and 1024 tokens against cuBLAS, each library, two rounds (the summary marks where the committed
# split differs from fixn1's); (d) e2e.py on Qwen3-8B, prompts of 256 and 1024 tokens and 8 generated, fused and
# --exact, the CUDA 13 library, then the CUDA 12 one; (e) 479139a's library (no cap) on (8960, 128): the K = 128 split
# tile left unwritten. results/summary.txt is rewritten after every step, and results/DONE written however it ends.
#   bash ~/h100_val.sh      (in ~: this script, val_src.tar, val_479139a.cu, cu12_fetch.sh, val_check.py,
#                            cu12_layer.py, val_summary.py, prev-check-cu13-base.json, prev-check-cu13-fixn1.json)
# Env, for a smoke test elsewhere: ARCHS (90a), KERNEL (wg on Hopper, else mid), N8 N14 (Qwen3-8B, Qwen3-14B: repos in
# Qwen/; N8 whole, for the e2e check too), C12 (a CUDA 12 dir), CCBIN12 (its host compiler), BUDGET (660 s: no step
# starts past it), E2E_BY (480 s: nor the first e2e check), MS, E2E_PREFILL, FILES (~), R (~/results), W (~/valw).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/valw}
mkdir -p "$R/log" "$W"
trap 'touch "$R/DONE"' EXIT
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${BUDGET:-660}
E2E_BY=${E2E_BY:-480}
FILES=${FILES:-$HOME}
ARCHS=${ARCHS:-90a}
MS=${MS:-17,128,129,256,512,1024}
E2E_PREFILL=${E2E_PREFILL:-256,1024}
LIBS="cu13 cu12"
step() { echo "== $(date +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
left() { [ $(( $(date +%s) - T0 )) -lt "${1:-$BUDGET}" ] || { echo "over the budget: skipped"; false; }; }
done_() { echo "$(date +%T) (+$(( $(date +%s) - T0 )) s) $*" >> "$R/steps.txt"; summ; }
summ() { python "$FILES/val_summary.py" "$R" > "$R/summary.tmp" 2> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"; }
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
SMS=$(python -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)")
echo "$(nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,power.limit --format=csv,noheader | head -1), $SMS SMs; kernels timed: $KERNEL" > "$R/machine-short.txt"
echo "$SMS" > "$R/sms.txt"
done_ "machine"

step "models, in the background: Qwen3-8B whole (layers and e2e), 14B's layer 10"
N8=${N8:-Qwen3-8B} N14=${N14:-Qwen3-14B}
( timeout 600 python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "Qwen/$N8" > "$W/$N8.dir" 2> "$R/log/dl-$N8.txt"
  echo "exit $?" >> "$R/log/dl-$N8.txt" ) & DL8=$!
( timeout 600 python - "Qwen/$N14" > "$W/$N14.dir" 2> "$R/log/dl-$N14.txt" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download
idx = hf_hub_download(sys.argv[1], "model.safetensors.index.json")
hf_hub_download(sys.argv[1], "config.json")
for f in sorted({f for t, f in json.load(open(idx))["weight_map"].items() if t.startswith("model.layers.10.")}):
    hf_hub_download(sys.argv[1], f)
print(os.path.dirname(idx))
PY
  echo "exit $?" >> "$R/log/dl-$N14.txt" ) & DL14=$!

step "sources: the committed state, and 479139a's glyd_gpu.cu for (e)"
for v in val old; do rm -rf "$W/$v" && mkdir -p "$W/$v" && tar -C "$W/$v" -xf "$FILES/val_src.tar"; done
cp "$FILES/val_479139a.cu" "$W/old/gpu/glyd_gpu.cu"
diff -u "$W/old/gpu/glyd_gpu.cu" "$W/val/gpu/glyd_gpu.cu" > "$R/cap.diff"
done_ "sources ($(grep -c '^[-+][^-+]' "$R/cap.diff") changed lines in cap.diff)"

step "CUDA 12: the instance's own where it is 12.8, else NVIDIA's 12.8.2 redistributables"
C12=${C12:-}
if [ -z "$C12" ]; then
  for n in /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc /usr/lib/cuda/bin/nvcc /usr/bin/nvcc; do
    v=$([ -x "$n" ] && "$n" --version 2>/dev/null | grep -o "release 12\.[0-9]*")
    [ -n "$v" ] && echo "the instance's CUDA 12: $n, $v" >> "$R/log/cuda12-found.txt"
    [ "$v" = "release 12.8" ] && [ -z "$C12" ] && C12=$(cd "$(dirname "$n")/.." && pwd)
  done
  [ -n "$C12" ] && echo "CUDA 12: the instance's, $C12" || { echo "no CUDA 12.8 on the instance: fetching"; timeout 300 bash "$FILES/cu12_fetch.sh" "$W/cu12" > "$R/log/cu12_fetch.txt" 2>&1 && C12=$W/cu12; }
fi
CCBIN12=${CCBIN12:-}
if [ -n "$C12" ] && [ -z "$CCBIN12" ] && [ "$(gcc -dumpversion | cut -d. -f1)" -gt 14 ]; then  # (CUDA 12.8 takes GCC to 14)
  for g in g++-14 g++-13 g++-12; do command -v $g > /dev/null && { CCBIN12=$(command -v $g); break; }; done
fi
{ echo "CUDA 13: $C13: $("$C13/bin/nvcc" --version | tail -2 | tr '\n' ' ')"
  [ -n "$C12" ] && echo "CUDA 12: $C12: $("$C12/bin/nvcc" --version | tail -2 | tr '\n' ' ')${CCBIN12:+ (host compiler $CCBIN12)}" || echo "CUDA 12: NONE (the cu12 library is skipped)"
  cat "$R/log/cuda12-found.txt" 2>/dev/null || echo "the instance's CUDA 12: none"
  echo "gcc: $(gcc --version | head -1)"; } | tee "$R/toolchains.txt"
done_ "CUDA 12 ${C12:-missing}"

step "(a) the libraries, sm_90a alone ($ARCHS), in parallel: cu13 and cu12 of the committed state, and 479139a's cu13 (old)"
build() {  # NAME SRC CUDA_DIR [HOST_COMPILER]: build_lib.sh's compile line, -Xptxas -v kept in results/build-NAME.txt
  local n=$1 src=$2 cu=$3 cc=${4:-} gen="" a
  for a in $ARCHS; do gen="$gen -gencode arch=compute_$a,code=sm_$a"; done
  local F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$cu/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ $gen -Xcompiler -fPIC,-fvisibility=hidden)
  [ -n "$cc" ] && F+=(-ccbin "$cc")
  { "$cu/bin/nvcc" --version | tail -2
    timeout 600 "$cu/bin/nvcc" "${F[@]}" -Xptxas -v -c -o "$W/$n.o" "$src/gpu/glyd_gpu.cu" &&
    timeout 300 "$cu/bin/nvcc" "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$cu/lib" -L"$cu/lib64" -o "$W/$n.so" "$W/$n.o" &&
    touch "$R/lib-$n.ok"; } > "$R/build-$n.txt" 2>&1
  echo "build $n exit $?"
}
BJ=""
build cu13 "$W/val" "$C13" & BJ="$BJ $!"
[ -n "$C12" ] && { build cu12 "$W/val" "$C12" "$CCBIN12" & BJ="$BJ $!"; }
build old "$W/old" "$C13" & BJ="$BJ $!"
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
done_ "builds: $(ls "$R"/lib-*.ok 2>/dev/null | wc -l) of $([ -n "$C12" ] && echo 3 || echo 2)"

step "(b) the self-test through each library, then its mma_gemm_wg outputs by sha256"
for n in $LIBS; do
  [ -f "$R/lib-$n.ok" ] && left || continue
  ( cd "$W/val/gpu" && GLYD_GPU_LIB="$W/$n.so" timeout 480 python glyd_gpu.py > "$R/selftest-$n.txt" 2>&1; echo "exit $?" >> "$R/selftest-$n.txt" )
  GLYD_GPU_DIR="$W/val/gpu" GLYD_GPU_LIB="$W/$n.so" timeout 300 python "$FILES/val_check.py" "$R/check-$n.json" > "$R/log/check-$n.txt" 2>&1
  echo "self-test $n: $(tail -1 "$R/selftest-$n.txt"); check exit $?"; done_ "self-test and check $n"
done

step "(c) layers against cuBLAS, each library, two rounds: Qwen3-8B's four products, 14B's o and q, k, v"
wait $DL8 $DL14
MODEL8=$(tail -1 "$W/$N8.dir" 2>/dev/null) MODEL14=$(tail -1 "$W/$N14.dir" 2>/dev/null)
for r in 1 2; do
  for n in $LIBS; do
    [ -f "$R/lib-$n.ok" ] && left || continue
    for spec in "$N8 $MODEL8 qkv,o,gate_up,down" "$N14 $MODEL14 o,qkv"; do
      set -- $spec
      [ -f "$2/model.safetensors.index.json" ] || { echo "no model at $2"; continue; }
      GLYD_GPU_DIR="$W/val/gpu" GLYD_GPU_LIB="$W/$n.so" timeout 300 python "$FILES/cu12_layer.py" "$2" --name "$1" --tag "$n-r$r" --only "$3" --M "$MS" --kernel "$KERNEL" >> "$R/layer.txt" 2>> "$R/log/layer.err"
    done
    done_ "layers $n round $r"
  done
done

step "(d) e2e, $N8: prompts of $E2E_PREFILL tokens and 8 generated, fused, then --exact; the CUDA 13 library, then CUDA 12's"
for n in $LIBS; do
  by=$E2E_BY; [ "$n" = cu12 ] && by=$(( BUDGET - 60 ))
  [ -f "$R/lib-$n.ok" ] && [ -f "$MODEL8/config.json" ] && left "$by" || continue
  ( cd "$W/val/gpu"
    GLYD_GPU_LIB="$W/$n.so" timeout 240 python e2e.py "$MODEL8" --format auto --fused --merge --baseline --prefill "$E2E_PREFILL" --tokens 8 --batch 1 > "$R/e2e-$n.txt" 2>&1
    echo "exit $?" >> "$R/e2e-$n.txt"
    GLYD_GPU_LIB="$W/$n.so" timeout 180 python e2e.py "$MODEL8" --format auto --exact --baseline --tokens 8 --prefill "$E2E_PREFILL" > "$R/e2e-exact-$n.txt" 2>&1
    echo "exit $?" >> "$R/e2e-exact-$n.txt" )
  done_ "e2e $n"
done

step "(e) 479139a's library (no cap) on (8960, 128), 129-256 tokens: a split tile of two stages with an empty cluster"
if [ -f "$R/lib-old.ok" ] && left; then
  GLYD_GPU_DIR="$W/old/gpu" GLYD_GPU_LIB="$W/old.so" KERNEL="$KERNEL" timeout 120 python - > "$R/e-479139a.txt" 2>&1 <<'PY'
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.environ["GLYD_GPU_DIR"])
import glyd_gpu as g
prod = g.mma_gemm_wg if os.environ["KERNEL"] == "wg" else g.mma_gemm_mid
torch.manual_seed(0)
w = (torch.randn(8960, 128, device="cuda") * 0.02).to(torch.bfloat16)
q = g.pack_mma12(w)
for M in (129, 160, 256):
    x = torch.randn(M, 128, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x.float(), w.float())
    y = prod(q, x)
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    rows = ((y.float() - ref).abs().amax(0) / ref.abs().max() > 1e-2).nonzero().flatten().tolist()
    print(f"{os.environ['KERNEL']} 8960x128 M={M}: error {err:.3g}; columns off: {len(rows)}" + (f" ({rows[0]}-{rows[-1]})" if rows else ""))
PY
  echo "(e) exit $?"; done_ "e 479139a"
fi

step "done in $(( $(date +%s) - T0 )) s"
done_ "done"
