#!/bin/bash
# After m6_bi.sh: bf16 with the EAGLE-3 draft did not fit the L4 at 0.9 (no memory left for the KV cache). The same runs
# with room made for it (--gpu-memory-utilization 0.95, --max-num-batched-tokens 2048), bf16 and Glyd alike, and bf16
# without speculation in those settings, each under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m6
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m6
while pgrep -f "m6_spec.sh|m6_bi.sh" > /dev/null; do sleep 30; done
run() {  # NAME JSON [ENV ...]
  local n=$1 cfg=$2; shift 2
  [ -s $O/$n.json ] && { echo "$n: done before"; return; }
  touch ~/.glyd-busy
  echo "== $(date -u +%T) $n $cfg $*"
  (cd $O && env "$@" flock ~/.glyd-box.lock timeout 2400 $V/python $W/glyd/gpu/vllm/spec_decode.py $O/$n.json "$cfg") > $O/$n.log 2>&1; echo "exit $?"
  grep "^edit:\|^chat:\|Error" $O/$n.log | grep -v "^INFO\|^WARNING" | tail -4 | cut -c1-240
}
run bf16-tight '{"mode": "bf16", "util": 0.95, "batched": 2048}'
run bf16-eagle3-tight '{"mode": "bf16", "spec": "eagle3", "util": 0.95, "batched": 2048}'
run glyd-eagle3-tight '{"mode": "glyd", "spec": "eagle3", "util": 0.95, "batched": 2048}'
echo "== $(date -u +%T) done"
