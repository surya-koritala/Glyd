#!/bin/bash
# GLY-235 on the owner's PC: `glyd-engine rangecheck` of Qwen3.5-9B at penguin, linnet and kestrel (the glyd/Qwen3.5-9B-* snapshots), a prompt of 1,024 tokens and 256 greedy
# tokens: the whole model, then two stages (layers 0..16 and 16..32) chained in one process, then the same two as worker processes through the glyd-range/1 protocol, every step's
# logits and token compared bit for bit; linnet once more cut in three (0..11, 11..23, 23..32). Around them `glyd-engine bench` of the linnet snapshot with the binary built from
# main (85529126, "before") and with this one ("after"), three times each, interleaved. nvidia-smi before, every second during, and after each run. The GPU is shared: run it
# under the lock, with the build's commit in $OUT/../src-commit:
#   flock ~/glyd-scratch/gpu.lock ./record.sh
set -u
OUT=${OUT:-$HOME/glyd-scratch/gly235/rec}
ENGINE=${ENGINE:-$HOME/glyd-scratch/gly235/target/release/glyd-engine}
BASE=${BASE:-$HOME/glyd-scratch/gly235/glyd-engine-base}
HUB=$HOME/glyd-scratch/rel032/hub
# (the library built from main's kernels, set here whatever the shell had: qwen-build's env.sh exports an older one)
export GLYD_GPU_LIB=${LIB:-$HOME/glyd-scratch/reg166/lib2/libglyd_gpu_cuda13.so} HF_HUB_CACHE=$HUB HF_HOME=$HOME/.config/huggingface HF_HUB_OFFLINE=1
LINNET=$HUB/models--glyd--Qwen3.5-9B-linnet/snapshots/073a2132abeaa41c119678b9687092871521eb4d
PENGUIN=$HUB/models--glyd--Qwen3.5-9B-penguin/snapshots/5985564d007e847509609c23d7647a8688867889
KESTREL=$HUB/models--glyd--Qwen3.5-9B-kestrel/snapshots/4d7e583a7cce212734140927824f91ad9ad11cf7
mkdir -p "$OUT"
cd "$OUT" || exit 1

now() { date +%s.%N | cut -c1-14; }
log() { echo "[$(date -Is)] $*" | tee -a steps.txt; }
smi() {
  { echo "== $1 ($(now))"; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> nvidia-smi.txt
}
# the GPU's used memory and each process's, every second while $1 runs, into $2
watch() {
  while kill -0 "$1" 2> /dev/null; do
    echo "$(now) used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) procs=[$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')]"
    sleep 1
  done >> "$2"
}
# rangecheck NAME DIR ARGS...: its output, its stderr (the engines' own lines, the workers' too), nvidia-smi around and during it
run() {
  local name=$1 dir=$2
  shift 2
  smi "before rangecheck $name"
  log "start: glyd-engine rangecheck $dir --prompt 1024 --new 256 $*"
  local t0
  t0=$(now)
  "$ENGINE" rangecheck "$dir" --prompt 1024 --new 256 "$@" > "rangecheck-$name.txt" 2> "rangecheck-$name.err" &
  local pid=$!
  watch $pid "smi-during-$name.txt"
  wait $pid
  local st=$?
  log "rangecheck $name: exit $st after $(awk "BEGIN {printf \"%.1f\", $(now) - $t0}") s: $(tail -1 "rangecheck-$name.txt")"
  smi "after rangecheck $name"
}
bench() { # bench NAME BINARY: the linnet snapshot, 1,024 prompt tokens, 256 new
  log "bench $1"
  "$2" bench "$LINNET" --prompt 1024 --new 256 >> "bench-$1.txt" 2>> "bench-$1.err"
}
gen() { # gen NAME BINARY: the whole-model path's tokens (Engine35, as bench runs it), 64 greedy tokens of one prompt, to compare before and after
  log "generate $1"
  "$2" generate "$LINNET" "The lighthouse keeper climbed the stairs again." --new 64 --ignore-eos > "generate-$1.txt" 2> "generate-$1.err"
}

{ echo "commit: $(cat "$OUT/../src-commit" 2> /dev/null)"
  echo "engine: $(sha256sum "$ENGINE")"
  echo "engine from main (bench before): $(sha256sum "$BASE")"
  echo "library: $(sha256sum "$GLYD_GPU_LIB")"
  "$ENGINE" --version
  ( . "$HOME/glyd-scratch/qwen-build/env.sh" 2> /dev/null; rustc --version; nvcc --version | tail -1 )
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  lscpu | grep 'Model name'
  uname -r
  echo "snapshots: penguin $PENGUIN, linnet $LINNET, kestrel $KESTREL"; } > build.txt
nvidia-smi > nvidia-smi-full-before.txt
log "start"

bench before "$BASE"
bench after "$ENGINE"
gen before "$BASE"
gen after "$ENGINE"
if cmp -s generate-before.txt generate-after.txt; then log "generate: the same text before and after"; else log "generate: DIFFERENT text before and after"; fi
run penguin "$PENGUIN" --workers
bench before "$BASE"
bench after "$ENGINE"
run linnet "$LINNET" --workers
run kestrel "$KESTREL" --workers
run linnet-3 "$LINNET" --cuts 11,23 --workers
bench before "$BASE"
bench after "$ENGINE"
smi "at the end"
nvidia-smi > nvidia-smi-full-after.txt
log "done"
