#!/usr/bin/env bash
# gpu-format's confirming run (format-report-1.md section 4), unattended, on one GPU; h100_format.sh and a100_format.sh
# run it. The split-byte 12-bit layout against the 12-bit layout in the same kernels, both against cuBLAS bf16, per
# layer (layer 10, q,k,v and gate,up merged), and check.py's bit-for-bit check. In order of importance:
#   1. sb12.py's self-check;  2. Qwen3-8B's layer at MS tokens, twice;  3. check.py on Qwen3-8B (every tensor bit for
#   bit, the products the 12-bit layout's bits);  4. Qwen3-14B's layer twice, then Qwen3-32B's.
# In ~: this script, the launcher, format_src.tar (gpu-format's gpu/glyd_gpu.cu and .h, gpu/experimental/format,
# bindings/python/glyd), format_summary.py; ~/gpuenv/cuda.sh (PyTorch and nvcc). Models through the Hugging Face cache
# (HF_HOME, ~/hf: another job's downloads are reused): Qwen3-8B whole, layer 10's shards of Qwen3-14B and 32B.
# Results in R (~/results/format) alone: machine.txt, steps.txt, selfcheck.txt, layer-MODEL-runN.txt, check-MODEL.txt,
# build-*.txt, log/, summary.txt (rewritten after every step: a cap still leaves partial answers), END last (never
# results/DONE). Every step has its own timeout, none past DEADLINE; no step starts past BUDGET.
# Env: GPU (the compute capability expected, "9.0" or "8.0"; empty: any), MS (the tokens), N8 N14 N32 (repos in Qwen/;
# NC: check.py's), R, W (~/formatw), FILES (~), HF_HOME, BUDGET (1260 s), DEADLINE (1440 s), OFFLINE (1: the cache
# only, no downloads), ROWS (0 or 1: the mid or TMA kernel's build; default: not on an A100, where it is not used).
set -u
GPU=${GPU:-}
MS=${MS:-1,8,16,32,64,128,256,512}
R=${R:-$HOME/results/format}
W=${W:-$HOME/formatw}
FILES=${FILES:-$HOME}
BUDGET=${BUDGET:-1260}
DEADLINE=${DEADLINE:-1440}
N8=${N8:-Qwen3-8B} N14=${N14:-Qwen3-14B} N32=${N32:-Qwen3-32B} NC=${NC:-Qwen3-8B}
mkdir -p "$R/log" "$W/tmp"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
step() { echo "== $(date +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
left() { [ $(( $(date +%s) - T0 )) -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
tmo() { local r=$(( T0 + DEADLINE - $(date +%s) )); r=$(( r < $1 ? r : $1 )); echo $(( r < 5 ? 5 : r )); }  # at most $1 s, and to the deadline
done_() {
  echo "$(date +%T) (+$(( $(date +%s) - T0 )) s) $*" >> "$R/steps.txt"
  python "$FILES/format_summary.py" "$R" > "$R/summary.tmp" 2>> "$R/log/summary.err" && mv "$R/summary.tmp" "$R/summary.txt"
}
source ~/gpuenv/cuda.sh
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
python -c "import hf_transfer" 2>/dev/null || unset HF_HUB_ENABLE_HF_TRANSFER
[ "${OFFLINE:-0}" = 1 ] && export HF_HUB_OFFLINE=1
export TMPDIR="$W/tmp" TORCH_EXTENSIONS_DIR="$W/torch_ext" MAX_JOBS=${MAX_JOBS:-8} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
has_ninja() { python -c "import torch.utils.cpp_extension as c; c.verify_ninja_availability()" 2>/dev/null; }  # PyTorch's JIT builds need it
has_ninja || { [ -x ~/tools/uv/uv ] && timeout 120 ~/tools/uv/uv pip install -q --python "$(command -v python)" ninja; has_ninja || timeout 120 python -m pip install -q ninja; }

step "machine"
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvcc --version | tail -2; python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"; echo "ninja $(command ninja --version 2>&1)"
  python -c "import sys, tarfile; print('sources: gpu-format', tarfile.open(sys.argv[1]).pax_headers.get('comment', '?'))" "$FILES/format_src.tar"
  nvidia-smi -q -d CLOCK; lscpu | grep "Model name"; nproc; free -g | head -2; df -h ~ | tail -1; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ -n "$GPU" ] && [ "$CC" != "$GPU" ]; then
  echo "compute capability $CC, not $GPU: nothing run"; done_ "stopped: compute capability $CC, not $GPU"; touch "$R/END"; exit 1
fi
ROWS=${ROWS:-$([ "$CC" = "8.0" ] && echo 0 || echo 1)}
done_ "machine: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"

step "models, in the background: $N8's layer 10, then the rest of it; layer 10 of $N14 and of $N32"
getlayer() {  # REPO: its index, config.json and the shards holding layer 10, into the cache; W/REPO.dir: the snapshot's directory
  timeout "$(tmo 1200)" python - "Qwen/$1" > "$W/$1.dir" 2> "$R/log/dl-$1.txt" <<'PY'
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
rm -f "$W/$N8.ok"
( getlayer "$N8"; touch "$W/$N8.ok"  # (then the rest of the check's model: a shard both want is fetched once)
  timeout "$(tmo 1200)" python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors']))" "Qwen/$NC" > "$W/check.dir" 2> "$R/log/dl-check-$NC.txt"
  echo "exit $?" >> "$R/log/dl-check-$NC.txt" ) & DLC=$!
getlayer "$N14" & DL14=$!
getlayer "$N32" & DL32=$!

step "sources, and the builds (the step and prompt kernels; the mid or TMA kernel$([ "$ROWS" = 1 ] || echo ": not on this GPU")), side by side"
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/format_src.tar"
X="$W/src/gpu/experimental/format"
( cd "$X" && timeout "$(tmo 900)" python -c "import sb12; sb12.ext(); print('built')" > "$R/build-ext.txt" 2>&1; echo "exit $?" >> "$R/build-ext.txt" ) & B1=$!
B2=""
[ "$ROWS" = 1 ] && { ( cd "$X" && timeout "$(tmo 900)" python -c "import sb12; sb12.ext_rows(); print('built')" > "$R/build-rows.txt" 2>&1; echo "exit $?" >> "$R/build-rows.txt" ) & B2=$!; }
wait $B1 $B2
done_ "builds: $(tail -1 "$R/build-ext.txt")$([ -f "$R/build-rows.txt" ] && echo ", rows $(tail -1 "$R/build-rows.txt")")"

step "1. the self-check"
( cd "$X" && timeout "$(tmo 600)" python sb12.py ) > "$R/selfcheck.txt" 2>&1; echo "exit $?" >> "$R/selfcheck.txt"
done_ "self-check: $(tail -1 "$R/selfcheck.txt")"

layer() {  # MODEL DIR RUN
  [ -f "$2/model.safetensors.index.json" ] || [ -f "$2/model.safetensors" ] || { done_ "layer $1 run $3: no model ($2)"; return; }
  left || { done_ "layer $1 run $3: over the budget"; return; }
  ( cd "$X" && timeout "$(tmo 600)" python layer.py "$2" --M "$MS" --reps 15 ) > "$R/layer-$1-run$3.txt" 2>> "$R/log/layer-$1.err"
  done_ "layer $1 run $3: exit $?"
}

step "2. $N8's layer, twice"
until [ -f "$W/$N8.ok" ] || [ $(( $(date +%s) - T0 )) -ge "$BUDGET" ]; do sleep 2; done
M8=$(tail -1 "$W/$N8.dir" 2>/dev/null)
layer "$N8" "$M8" 1
layer "$N8" "$M8" 2

step "3. check.py on $NC"
wait $DLC
MC=$(tail -1 "$W/check.dir" 2>/dev/null)
if [ -f "$MC/config.json" ] && left; then
  ( cd "$X" && timeout "$(tmo 900)" python check.py "$MC" ) > "$R/check-$NC.txt" 2>&1; done_ "check $NC: exit $?"
else
  done_ "check $NC: not run ($([ -f "$MC/config.json" ] && echo "over the budget" || echo "no model: $MC"))"
fi

step "4. $N14's layer twice, then $N32's"
wait $DL14
M14=$(tail -1 "$W/$N14.dir" 2>/dev/null)
layer "$N14" "$M14" 1
layer "$N14" "$M14" 2
wait $DL32
M32=$(tail -1 "$W/$N32.dir" 2>/dev/null)
layer "$N32" "$M32" 1
layer "$N32" "$M32" 2

step "done in $(( $(date +%s) - T0 )) s"
done_ "end"
touch "$R/END"
