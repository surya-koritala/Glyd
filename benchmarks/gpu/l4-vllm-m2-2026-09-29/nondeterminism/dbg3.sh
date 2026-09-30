#!/bin/bash
# Inductor's deterministic mode (no on-device benchmarking that moves numerics): Qwen3-1.7B compiled (CUDA graphs off)
# three times on one cache, Glyd tiered, then bf16; each trio's first compiles, the others load.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/dbg3; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_LAYOUT=mma
LONG=$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json
D='{"inductor_compile_config": {"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}}'
one() { echo "== $(date -u +%T) $1"; (cd $O && env VLLM_CACHE_ROOT=$3 DBG_Q=$2 timeout 900 $PY $W/dbg_hash.py compiled $O/$1.json $LONG "$D") > $O/$1.log 2>&1; echo "$1 exit $?"; grep "products hashed" $O/$1.log; }
trio() { local c; c=$(mktemp -d -p $TMPDIR); for i in 1 2 3; do one $1$i "$2" $c; done; rm -rf $c; }
[ -f $O/d3.json ] || trio d glyd
trio f ""
$PY $W/dbg_cmp.py $O d1:d2 d2:d3 d1:d3 f1:f2 f2:f3 f1:f3 | tee $O/summary.txt
