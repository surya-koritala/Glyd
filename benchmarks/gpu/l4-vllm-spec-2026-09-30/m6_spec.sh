#!/bin/bash
# Speculative decoding with Glyd on the L4: gpu/vllm/spec_decode.py on Qwen3-8B, a vLLM each, each under the box's
# lock (a run done before skipped), on one compile cache. Results in ~/vllm-work/m6/.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m6; mkdir -p $O $W/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m6
run() {  # NAME JSON [ENV ...]
  local n=$1 cfg=$2; shift 2
  [ -s $O/$n.json ] && { echo "$n: done before"; return; }
  touch ~/.glyd-busy
  echo "== $(date -u +%T) $n $cfg $*"
  (cd $O && env "$@" flock ~/.glyd-box.lock timeout 2400 $V/python $W/glyd/gpu/vllm/spec_decode.py $O/$n.json "$cfg") > $O/$n.log 2>&1; echo "exit $?"
  grep "^edit:\|^chat:\|Error\|error" $O/$n.log | grep -v "^INFO\|^WARNING" | tail -4 | cut -c1-240
}
run bf16 '{"mode": "bf16"}'
run bf16-ngram '{"mode": "bf16", "spec": "ngram"}'
run bf16-eagle3 '{"mode": "bf16", "spec": "eagle3"}'
run glyd '{"mode": "glyd"}'
run glyd-ngram '{"mode": "glyd", "spec": "ngram"}'
run glyd-eagle3 '{"mode": "glyd", "spec": "eagle3"}'
run glyd-eagle3-packed '{"mode": "glyd", "spec": "eagle3", "draft_glyd": true}'
run glyd-eagle3-packed-again '{"mode": "glyd", "spec": "eagle3", "draft_glyd": true}'
run bf16-eager-ngram '{"mode": "bf16", "spec": "ngram", "eager": true}'
run glyd-exact-eager-ngram '{"mode": "glyd", "spec": "ngram", "eager": true}' GLYD_EXACT=1
run bf16-eager-eagle3 '{"mode": "bf16", "spec": "eagle3", "eager": true}'
run glyd-exact-eager-eagle3 '{"mode": "glyd", "spec": "eagle3", "eager": true}' GLYD_EXACT=1
run glyd-exact-eager-eagle3-packed '{"mode": "glyd", "spec": "eagle3", "eager": true, "draft_glyd": true}' GLYD_EXACT=1
echo "== $(date -u +%T) done"
