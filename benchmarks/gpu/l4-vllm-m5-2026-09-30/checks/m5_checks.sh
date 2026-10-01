#!/bin/bash
# M5, the review's fixes on the L4, each step under the box's lock (a step done before skipped): the missing-weight
# refusal (b2check.py), then check_vllm.py --quick on Qwen2.5-1.5B-Instruct (Linears with biases), granite (a mixture of
# experts) and Qwen3-8B.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m5/checks; mkdir -p $O $W/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
if [ ! -s $O/b2check.txt ]; then
  echo "== $(date -u +%T) b2check"
  (cd $O && VLLM_ENABLE_V1_MULTIPROCESSING=0 flock ~/.glyd-box.lock timeout 900 $V/python $W/b2check.py) > $O/b2check.txt 2>&1; echo "exit $?"
  grep "refused\|loaded" $O/b2check.txt | cut -c1-300
fi
for m in ${MODELS:-Qwen/Qwen2.5-1.5B-Instruct ibm-granite/granite-3.1-3b-a800m-instruct Qwen/Qwen3-8B}; do
  t=$(basename $m)
  [ -f $O/check-$t/report.txt ] && { echo "check $t: done before"; continue; }
  echo "== $(date -u +%T) check_vllm --quick $m"
  (cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py --quick --out $O/check-$t $m) > $O/check-$t.txt 2>&1; echo "exit $?"
  grep "PASS\|FAIL\|check_vllm:" $O/check-$t.txt | cut -c1-240
  if [ "$t" = Qwen2.5-1.5B-Instruct ] && grep -q "FAIL .*exact, eager" $O/check-$t.txt; then echo "exact eager not bit for bit on $t: stopped"; exit 1; fi
done
echo "== $(date -u +%T) done"
