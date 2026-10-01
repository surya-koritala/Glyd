#!/bin/bash
# M5, part 3, after parts 1 and 2 (the tree unchanged since part 2 started): the missing-weight refusal again and
# granite's check again (its first run met a tree synced between its runs), each under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m5/checks
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
while pgrep -f "m5_checks.sh|m5_checks2.sh" > /dev/null; do sleep 20; done
echo "== $(date -u +%T) b2check"
(cd $O && VLLM_ENABLE_V1_MULTIPROCESSING=0 flock ~/.glyd-box.lock timeout 900 $V/python $W/b2check.py) > $O/b2check.txt 2>&1; echo "exit $?"
grep "refused\|loaded" $O/b2check.txt | cut -c1-300
m=ibm-granite/granite-3.1-3b-a800m-instruct; t=$(basename $m)
rm -rf $O/check-$t $O/check-$t.txt
echo "== $(date -u +%T) check_vllm --quick $m"
(cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py --quick --out $O/check-$t $m) > $O/check-$t.txt 2>&1; echo "exit $?"
grep "PASS\|FAIL\|check_vllm:" $O/check-$t.txt | cut -c1-240
echo "== $(date -u +%T) done"
