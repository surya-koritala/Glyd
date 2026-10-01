#!/bin/bash
# check_vllm.py --quick --fraction F on the dev L4 under the box's lock (F, the model: arguments; Qwen/Qwen3-8B, 0.5): the
# plugin as synced to ~/budget/src, v0.25.1's library built for sm_89 (the moe job's: gpu/ has not changed since).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
# the GPU free of others' processes before anything starts (up to 10 minutes)
for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 5; done
echo "gpu used ${used} MiB at $(date -u +%T)"
F=${1:-0.5}; M=${2:-Qwen/Qwen3-8B}; H=${3:---quick}
B=~/budget; S=$B/src; O=$B/check-$(basename $M)-f$F; mkdir -p $O $B/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$HOME/mmoedry/w/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$S/bindings/python TMPDIR=$B/tmp
PY=~/mmoedry/w/venv/bin/python
cd $B   # (not a directory holding gpu/: it would shadow glyd.gpu)
echo "== $(date -u +%T) check_vllm $H --fraction $F $M"
$PY $S/gpu/vllm/check_vllm.py $H --fraction $F --out $O $M > $O.txt 2>&1
echo "exit $? at $(date -u +%T)"; grep -E "^(PASS|FAIL)|^check_vllm" $O.txt | cut -c1-330
