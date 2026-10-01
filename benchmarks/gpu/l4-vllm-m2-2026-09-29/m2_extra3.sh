#!/bin/bash
# After m2_extra2.sh's warm bench, on the L4, each step under the box's lock (a step done before skipped):
# 1. repeat.py: glyd and bf16 compiled, the same prompts three times in one process (prefix caching off);
# 2. opcheck.py under compute-sanitizer, initcheck and memcheck: the library's routed linear on random packs;
# 3. m2_extra2.sh's l4-routes steps.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m2; L=$W/lib-l4routes
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
step() { echo "== $(date -u +%T) $*"; }
while [ ! -f $O/bench-Qwen3-8B-warm/summary.txt ] && pgrep -f "bench_serve.sh Qwen/Qwen3-8B" > /dev/null; do sleep 10; done
mkdir -p $O/diag
for r in "glyd compiled" "bf16 compiled"; do
  set -- $r
  [ -f $O/diag/repeat-$1-$2.json ] && continue
  step "repeat.py $1 $2"
  (cd $O/diag && VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$W/cache/diag flock ~/.glyd-box.lock timeout 900 $V/python $W/repeat.py $1 $2 $O/diag/repeat-$1-$2.json $W/glyd/gpu/vllm) > $O/diag/repeat-$1-$2.log 2>&1; echo "exit $?"
  grep "one process" $O/diag/repeat-$1-$2.log
done
for tool in initcheck memcheck; do
  [ -f $O/diag/opcheck-$tool.txt ] && continue
  step "opcheck.py under compute-sanitizer --tool $tool"
  (cd $O/diag && PYTORCH_NO_CUDA_MEMORY_CACHING=1 flock ~/.glyd-box.lock timeout 1200 /usr/local/cuda-13.0/bin/compute-sanitizer --tool $tool --print-limit 20 $V/python $W/opcheck.py) > $O/diag/opcheck-$tool.txt 2>&1; echo "exit $?"
  tail -4 $O/diag/opcheck-$tool.txt
done
if [ ! -f $L/libglyd_gpu_cuda13.so ]; then
  step "the l4-routes library ($(cat $W/l4routes/COMMIT))"
  (cd $W/l4routes && flock ~/.glyd-box.lock bash gpu/build_lib.sh $L) > $O/l4routes-build.txt 2>&1; echo "exit $?"
fi
export GLYD_GPU_LIB=$L/libglyd_gpu_cuda13.so
mkdir -p $O/l4routes
if [ ! -f $O/l4routes/Qwen3-8B-glyd-mma.json ]; then
  step "Qwen3-8B glyd tiered on the l4-routes library: layers, tokens, continuation"
  spec=$($V/python -c "import json; d = json.load(open('$O/check-Qwen3-8B/Qwen3-8B-bf16.json')); print(json.dumps(dict(model='Qwen/Qwen3-8B', quantization='glyd', layers=True, long_ids=d['long_ids'])))")
  (cd $O/l4routes && VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_LAYOUT=mma GLYD_VERIFY=1 VLLM_CACHE_ROOT=$W/cache/l4routes flock ~/.glyd-box.lock timeout 1800 $V/python $W/glyd/gpu/vllm/check_vllm.py --child "$spec" $O/l4routes/Qwen3-8B-glyd-mma.json) > $O/l4routes/Qwen3-8B-glyd-mma.log 2>&1; echo "exit $?"
  $V/python - $W/glyd/gpu/vllm $O <<'PY' | tee $O/l4routes/compare.txt
import json, sys
sys.path.insert(0, sys.argv[1])
from check_vllm import compare
O = sys.argv[2]
L = lambda p: json.load(open(p))
bf16, eager = L(f"{O}/check-Qwen3-8B/Qwen3-8B-bf16.json"), L(f"{O}/check-Qwen3-8B/Qwen3-8B-bf16-eager.json")
new, old = L(f"{O}/l4routes/Qwen3-8B-glyd-mma.json"), L(f"{O}/check-Qwen3-8B/Qwen3-8B-glyd-mma.json")
print("l4-routes library, tiered, against bf16:", compare(new, bf16))
print("layers:", new["layers"])
print("v0.25.0's library (the 8B check), tiered, against bf16:", compare(old, bf16))
print("bf16 eager against its graphs (the floor):", compare(eager, bf16))
PY
fi
if [ ! -f $O/bench-Qwen3-8B-l4routes/summary.txt ]; then
  step "bench_serve Qwen/Qwen3-8B, glyd on the l4-routes library"
  (cd $O && PATH=$V:$PATH R=$O/bench-Qwen3-8B-l4routes VLLM_CACHE_ROOT=$W/cache/bench MODES=glyd RATES="0.25 1 inf" PROMPTS="32 64 256" flock ~/.glyd-box.lock timeout 3600 bash $W/glyd/gpu/vllm/bench_serve.sh Qwen/Qwen3-8B) > $O/bench-Qwen3-8B-l4routes.txt 2>&1; echo "exit $?"
  cat $O/bench-Qwen3-8B-l4routes/summary.txt
fi
step done
