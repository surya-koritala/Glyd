#!/bin/bash
# After m2_extra3.sh: opcheck.py (the library loaded first) under compute-sanitizer, initcheck and memcheck, under the
# box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m2
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
while pgrep -f "^bash /home/ubuntu/vllm-work/m2_extra3.sh" > /dev/null; do sleep 10; done
for tool in initcheck memcheck; do
  echo "== $(date -u +%T) opcheck.py under compute-sanitizer --tool $tool"
  (cd $O/diag && PYTORCH_NO_CUDA_MEMORY_CACHING=1 flock ~/.glyd-box.lock timeout 1500 /usr/local/cuda-13.0/bin/compute-sanitizer --tool $tool --print-limit 20 $V/python $W/opcheck.py) > $O/diag/opcheck-$tool.txt 2>&1; echo "exit $?"
  tail -4 $O/diag/opcheck-$tool.txt
done
echo "== $(date -u +%T) done"
