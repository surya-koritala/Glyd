#!/bin/bash
# The MoE route at 1,152 tokens (the plugin's default now) on the L4, granite-3.1-3b-a800m-instruct, under the box's lock:
# profile_steps.py with the default route; then vllm bench serve bf16 against Glyd (the default route), and Glyd with the
# grouped products throughout (GLYD_MOE_DECODE_MIN=-1), warm with the cold start noted, at 1, 4 and inf.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
W=~/vllm-work; V=$W/venv/bin; O=$W/m8; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
M=ibm-granite/granite-3.1-3b-a800m-instruct
cd $W/glyd/bindings/python && $V/python test_vllm.py 2>&1 | grep -v "^INFO\|^WARNING" | tail -8 > $O/test_vllm.txt; cat $O/test_vllm.txt
echo "== $(date -u +%T) steps-routed"
(cd $O && VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/m8 BATCHES="1 8 32 64 128 256" PROMPTS="512 1024 2048 4096" flock ~/.glyd-box.lock timeout 1800 $V/python $W/glyd/gpu/vllm/profile_steps.py glyd $O/steps-routed.json $M) > $O/steps-routed.txt 2>&1; echo "exit $?"
grep "^| " $O/steps-routed.txt
echo "== $(date -u +%T) bench: bf16 and Glyd (routed)"
(cd $O && PATH=$V:$PATH R=$O/bench-routed VLLM_CACHE_ROOT=$W/cache/m8-bench WARM=1 RATES="1 4 inf" PROMPTS="64 128 256" UTIL=0.9 BUSYWAIT=600 COOL=50 COOLWAIT=180 flock ~/.glyd-box.lock timeout 3600 bash $W/glyd/gpu/vllm/bench_serve.sh $M) > $O/bench-routed.txt 2>&1; echo "exit $?"
cat $O/bench-routed/summary.txt
echo "== $(date -u +%T) bench: Glyd grouped throughout"
(cd $O && PATH=$V:$PATH R=$O/bench-grouped MODES=glyd GLYD_MOE_DECODE_MIN=-1 VLLM_CACHE_ROOT=$W/cache/m8-bench-grouped WARM=1 RATES="1 4 inf" PROMPTS="64 128 256" UTIL=0.9 BUSYWAIT=600 COOL=50 COOLWAIT=180 flock ~/.glyd-box.lock timeout 3600 bash $W/glyd/gpu/vllm/bench_serve.sh $M) > $O/bench-grouped.txt 2>&1; echo "exit $?"
cat $O/bench-grouped/summary.txt
echo "== $(date -u +%T) done"
