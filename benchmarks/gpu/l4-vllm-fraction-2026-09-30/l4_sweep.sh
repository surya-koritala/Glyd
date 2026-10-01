#!/bin/bash
# The L4 sanity sweep: Qwen3-8B, bf16 and Glyd at fractions 0, 0.5 and 1 back to back in one session under the box's lock,
# as M5 ran bf16 and Glyd (bench_serve.sh, WARM=1: each server started cold on an empty compile cache, then again on it,
# the second measured), rate 1 (64 prompts) and every request at once (256), 1,024 tokens in and 256 out, util 0.9.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
# the GPU free of others' processes before anything starts (up to 10 minutes)
for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 5; done
echo "gpu used ${used} MiB at $(date -u +%T)"
set -u
B=~/budget; S=$B/src; O=$B/sweep; mkdir -p $O $B/tmp
V=~/mmoedry/w/venv/bin
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$HOME/mmoedry/w/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$S/bindings/python TMPDIR=$B/tmp
rm -rf $B/cache/sweep $O/bench-Qwen3-8B
cd $O
echo "== $(date -u +%T) sweep"
PATH=$V:$PATH R=$O/bench-Qwen3-8B VLLM_CACHE_ROOT=$B/cache/sweep WARM=1 RATES="1 inf" PROMPTS="64 256" MODES="bf16 glyd@0 glyd@0.5 glyd@1" \
  UTIL=0.9 BUSYWAIT=600 COOL=50 COOLWAIT=180 timeout 7200 bash $S/gpu/vllm/bench_serve.sh Qwen/Qwen3-8B > $O/bench-Qwen3-8B.txt 2>&1
echo "exit $? at $(date -u +%T)"; cat $O/bench-Qwen3-8B/summary.txt
