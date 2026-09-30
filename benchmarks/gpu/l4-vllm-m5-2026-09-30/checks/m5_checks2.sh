#!/bin/bash
# M5, the review's fixes on the L4, part 2, each step under the box's lock: check_vllm.py --quick on
# Qwen2.5-1.5B-Instruct again (the bias fix), then --quick --saves on Yi-1.5-6B-Chat (a Llama save with its own LM head).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m5/checks; mkdir -p $O $W/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export GLYD_SAVE_PYTHON=$HOME/gpuenv/bin/python PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
run() {  # MODEL FLAGS
  t=$(basename $1)
  [ -f $O/check-$t/report.txt ] && { echo "check $t: done before"; return; }
  rm -rf $O/check-$t
  echo "== $(date -u +%T) check_vllm $2 $1"
  (cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py $2 --out $O/check-$t $1) > $O/check-$t.txt 2>&1; echo "exit $?"
  grep "PASS\|FAIL\|check_vllm:\|Error" $O/check-$t.txt | cut -c1-240
}
run Qwen/Qwen2.5-1.5B-Instruct --quick
if grep -q "FAIL .*exact, eager" $O/check-Qwen2.5-1.5B-Instruct.txt; then echo "exact eager not bit for bit on Qwen2.5-1.5B-Instruct: stopped"; exit 1; fi
run 01-ai/Yi-1.5-6B-Chat "--quick --saves"
echo "== $(date -u +%T) done"
