#!/bin/bash
# detcost.py: Qwen3-8B, bf16 and Glyd (the L4's layout), inductor as vLLM sets it and deterministic; each on an empty
# compile cache, from about the GPU's idle temperature, under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m3prep/detcost; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0
for r in "bf16 default" "bf16 deterministic" "glyd default" "glyd deterministic"; do
  set -- $r
  [ -f $O/$1-$2.json ] && continue
  for i in $(seq 1 24); do [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -le 50 ] && break; sleep 5; done
  echo "== $(date -u +%T) $1 $2 (the GPU at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) C)"
  (cd $O && VLLM_CACHE_ROOT=$(mktemp -d -p $TMPDIR) flock ~/.glyd-box.lock timeout 900 $PY $W/detcost.py $1 $2 $O/$1-$2.json) > $O/$1-$2.log 2>&1; echo "exit $?"
  tail -1 $O/$1-$2.log | cut -c1-250
done
