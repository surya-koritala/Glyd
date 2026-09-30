#!/bin/bash
# M4 on the L4, each step under the box's lock (a step done before skipped): check_vllm.py --quick on
# granite-3.1-3b-a800m-instruct (a mixture of experts), then STEPS' others.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m4; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
for m in ${MODELS:-ibm-granite/granite-3.1-3b-a800m-instruct}; do
  t=$(basename $m)
  [ -f $O/check-$t/report.txt ] && { echo "check $t: done before"; continue; }
  echo "== $(date -u +%T) check_vllm --quick $m"
  (cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py --quick --out $O/check-$t $m) > $O/check-$t.txt 2>&1; echo "exit $?"
  grep "PASS\|FAIL\|check_vllm:" $O/check-$t.txt | cut -c1-240
done
echo "== $(date -u +%T) done"
