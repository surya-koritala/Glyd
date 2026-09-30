#!/bin/bash
# M2 on the L4: STEPS in order, each "check:MODEL[:saves]" (check_vllm.py) or "bench:MODEL"
# (bench_serve.sh, bf16 against glyd), each under the box's lock on its own (other helpers' jobs run between); a step done before (its report.txt or summary.txt) skipped. Results in ~/vllm-work/m2/.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m2; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export GLYD_SAVE_PYTHON=$HOME/gpuenv/bin/python PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
mkdir -p $TMPDIR
step() { echo "== $(date -u +%T) $*"; }
for s in ${STEPS:-check:Qwen/Qwen3-1.7B:saves bench:Qwen/Qwen3-8B check:Qwen/Qwen3-8B check:Qwen/Qwen3-4B-Instruct-2507:saves}; do
  IFS=: read -r kind m opt <<< "$s"
  t=$(basename $m)
  if [ $kind = check ]; then
    [ -f $O/check-$t/report.txt ] && { echo "check $t: done before"; continue; }
    step "check_vllm $m ${opt:+--$opt}"
    (cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py ${opt:+--$opt} --out $O/check-$t $m) > $O/check-$t.txt 2>&1; echo "exit $?"
    grep "PASS\|FAIL\|check_vllm:" $O/check-$t.txt | cut -c1-240
  else
    [ -f $O/bench-$t/summary.txt ] && { echo "bench $t: done before"; continue; }
    step "bench_serve $m"
    (cd $O && PATH=$V:$PATH R=$O/bench-$t VLLM_CACHE_ROOT=$W/cache/bench flock ~/.glyd-box.lock timeout 5400 bash $W/glyd/gpu/vllm/bench_serve.sh $m) > $O/bench-$t.txt 2>&1; echo "exit $?"
    cat $O/bench-$t/summary.txt 2>/dev/null
  fi
done
step done
