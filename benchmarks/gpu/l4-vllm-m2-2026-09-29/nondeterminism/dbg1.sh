#!/bin/bash
# dbg_hash.py: Qwen3-1.7B tiered compiled (CUDA graphs off) three times on one compile cache (the first compiles, the
# others load), and eager twice; then dbg_cmp.py. Under the box's lock, holding ~/.glyd-busy.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/dbg1; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_LAYOUT=mma
LONG=$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json
C=$(mktemp -d -p $TMPDIR)
one() { echo "== $(date -u +%T) $1"; (cd $O && env VLLM_CACHE_ROOT=${3:-$C} timeout 900 $PY $W/dbg_hash.py $2 $O/$1.json $LONG) > $O/$1.log 2>&1; echo "$1 exit $?"; grep "products hashed" $O/$1.log; }
one c1 compiled
one c2 compiled
one c3 compiled
one e1 eager $(mktemp -d -p $TMPDIR)
one e2 eager $(mktemp -d -p $TMPDIR)
$PY $W/dbg_cmp.py $O c1:c2 c2:c3 c1:c3 e1:e2 | tee $O/summary.txt
rm -rf $C
