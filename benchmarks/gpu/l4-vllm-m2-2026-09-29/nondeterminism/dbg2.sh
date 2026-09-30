#!/bin/bash
# dbg_hash.py with FlashAttention hooked: Qwen3-1.7B tiered compiled (CUDA graphs off) three times on one cache, as vLLM
# sets inductor (combo kernels, benchmarked); then with combo kernels off; then combo kernels without their benchmark.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/dbg2; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_LAYOUT=mma
LONG=$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json
one() { echo "== $(date -u +%T) $1"; (cd $O && env VLLM_CACHE_ROOT=$3 timeout 900 $PY $W/dbg_hash.py compiled $O/$1.json $LONG "$2") > $O/$1.log 2>&1; echo "$1 exit $?"; grep "products hashed" $O/$1.log; }
trio() { local c; c=$(mktemp -d -p $TMPDIR); for i in 1 2 3; do one $1$i "$2" $c; done; rm -rf $c; }
trio v '{}'
trio n '{"inductor_compile_config": {"enable_auto_functionalized_v2": false, "combo_kernels": false, "benchmark_combo_kernel": false}}'
trio b '{"inductor_compile_config": {"enable_auto_functionalized_v2": false, "combo_kernels": true, "benchmark_combo_kernel": false}}'
$PY $W/dbg_cmp.py $O v1:v2 v2:v3 v1:v3 n1:n2 n2:n3 n1:n3 b1:b2 b2:b3 b1:b3 | tee $O/summary.txt
