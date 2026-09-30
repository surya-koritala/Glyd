#!/bin/bash
# M5: Qwen3-8B on the L4 in one session, bf16 then Glyd back to back, under the box's lock: M3's bench (bench_serve.sh,
# warm servers with the cold start noted, on an empty compile cache) at 0.25, 1, 4 and inf requests a second (the L4
# saturates below 1), v0.25.1's library, the plugin as synced (61825a1). Results in ~/vllm-work/m5/bench-Qwen3-8B.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m5; mkdir -p $O $W/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
rm -rf $W/cache/m5-l4 $O/bench-Qwen3-8B
echo "== $(date -u +%T) waiting for the box's lock"
(cd $O && PATH=$V:$PATH R=$O/bench-Qwen3-8B VLLM_CACHE_ROOT=$W/cache/m5-l4 WARM=1 RATES="0.25 1 4 inf" PROMPTS="32 64 128 256" \
  UTIL=0.9 BUSYWAIT=600 COOL=50 COOLWAIT=180 flock ~/.glyd-box.lock timeout 5400 bash $W/glyd/gpu/vllm/bench_serve.sh Qwen/Qwen3-8B) \
  > $O/bench-Qwen3-8B.txt 2>&1
echo "== $(date -u +%T) exit $?"
