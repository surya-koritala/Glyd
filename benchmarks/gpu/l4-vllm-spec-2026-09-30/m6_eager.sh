#!/bin/bash
# bf16 eager without speculation (eager is the same from one process to the next): against bf16-eager-ngram and
# bf16-eager-eagle3, speculation's own effect on the tokens, without compiled runs' variation. Under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
W=~/vllm-work; V=$W/venv/bin; O=$W/m6
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m6
echo "== $(date -u +%T) bf16-eager"
(cd $O && flock ~/.glyd-box.lock timeout 2400 $V/python $W/glyd/gpu/vllm/spec_decode.py $O/bf16-eager.json '{"mode": "bf16", "eager": true}') > $O/bf16-eager.log 2>&1; echo "exit $?"
grep "^edit:\|^chat:" $O/bf16-eager.log
echo "== $(date -u +%T) done"
