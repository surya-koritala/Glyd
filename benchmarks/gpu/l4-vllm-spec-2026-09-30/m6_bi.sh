#!/bin/bash
# After m6_spec.sh: the same runs under VLLM_BATCH_INVARIANT=1 (every product's bits independent of the batch), eager:
# bf16 without and with n-gram speculation, and Glyd exact with it, each under the box's lock. Results in ~/vllm-work/m6/.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m6
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m6 VLLM_BATCH_INVARIANT=1
while pgrep -f m6_spec.sh > /dev/null; do sleep 30; done
run() {  # NAME JSON [ENV ...]
  local n=$1 cfg=$2; shift 2
  [ -s $O/$n.json ] && { echo "$n: done before"; return; }
  touch ~/.glyd-busy
  echo "== $(date -u +%T) $n $cfg $* (VLLM_BATCH_INVARIANT=1)"
  (cd $O && env "$@" flock ~/.glyd-box.lock timeout 2400 $V/python $W/glyd/gpu/vllm/spec_decode.py $O/$n.json "$cfg") > $O/$n.log 2>&1; echo "exit $?"
  grep "^edit:\|^chat:\|Error" $O/$n.log | grep -v "^INFO\|^WARNING" | tail -4 | cut -c1-240
}
run bi-bf16-eager '{"mode": "bf16", "eager": true}'
run bi-bf16-eager-ngram '{"mode": "bf16", "spec": "ngram", "eager": true}'
run bi-glyd-exact-eager-ngram '{"mode": "glyd", "spec": "ngram", "eager": true}' GLYD_EXACT=1
echo "== $(date -u +%T) done"
