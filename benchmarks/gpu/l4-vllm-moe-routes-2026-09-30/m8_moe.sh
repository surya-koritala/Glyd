#!/bin/bash
# A mixture of experts' two routes on the L4, granite-3.1-3b-a800m-instruct, each run under the box's lock: moe_routes.py
# (a layer's GPU time a call by tokens: grouped, decoded, and bf16's own), then profile_steps.py (a step's GPU time by
# kind) for bf16, Glyd grouped throughout (GLYD_MOE_DECODE_MIN=-1) and Glyd decoded throughout (=1).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
W=~/vllm-work; V=$W/venv/bin; O=$W/m8; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m8
M=ibm-granite/granite-3.1-3b-a800m-instruct
run() {  # NAME ENV... -- CMD...
  local n=$1; shift
  [ -s $O/$n.json ] && { echo "$n: done before"; return; }
  touch ~/.glyd-busy
  echo "== $(date -u +%T) $n"
  (cd $O && flock ~/.glyd-box.lock timeout 1800 env "$@") > $O/$n.txt 2>&1; echo "exit $?"
  grep "^{'T'\|^| \|Error" $O/$n.txt | cut -c1-200 | tail -24
}
run routes-glyd GLYD_MOE_DECODE_MIN=1 $V/python $W/glyd/gpu/vllm/moe_routes.py $O/routes-glyd.json $M
run routes-bf16 $V/python $W/glyd/gpu/vllm/moe_routes.py $O/routes-bf16.json $M --bf16
export BATCHES="1 8 32 64 128 256" PROMPTS="512 2048 4096"
run steps-bf16 $V/python $W/glyd/gpu/vllm/profile_steps.py bf16 $O/steps-bf16.json $M
run steps-grouped GLYD_MOE_DECODE_MIN=-1 $V/python $W/glyd/gpu/vllm/profile_steps.py glyd $O/steps-grouped.json $M
run steps-decoded GLYD_MOE_DECODE_MIN=1 $V/python $W/glyd/gpu/vllm/profile_steps.py glyd $O/steps-decoded.json $M
echo "== $(date -u +%T) done"
