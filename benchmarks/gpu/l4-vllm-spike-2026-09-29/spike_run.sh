#!/bin/bash
# M1 spike on the L4, run under the box's lock: the library built from the worktree (main v0.25.0 + the plugin), glyd
# installed editable into the vLLM venv (its vllm.general_plugins entry point), then spike.py on Qwen3-1.7B (bf16;
# glyd tiered and 12-bit with CUDA graphs; glyd tiered eager) and Qwen3-8B (bf16, glyd tiered), and vllm serve
# --quantization glyd answering one request. Results in ~/vllm-work/results.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; R=$W/results; mkdir -p $R
source ~/gpuenv/cuda.sh  # nvcc (CUDA 13), for the library
# PYTHONSAFEPATH: the script's directory not on sys.path (its glyd/ copy of the repo would shadow the package)
export HF_HOME=~/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$W/cache VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1
T0=$(date +%s); step() { echo "== $(date -u +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
step "versions"
$PY -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'CUDA', torch.version.cuda, 'transformers', transformers.__version__)" | tee $R/versions.txt
step "library"
[ -f $W/lib/libglyd_gpu_cuda13.so ] || { (cd $W/glyd && bash gpu/build_lib.sh $W/lib) > $R/build.txt 2>&1; echo "build exit $?"; }
export GLYD_GPU_LIB=$(ls $W/lib/libglyd_gpu_cuda*.so | head -1); echo "GLYD_GPU_LIB=$GLYD_GPU_LIB"
step "glyd into the venv"
cp $W/glyd/LICENSE $W/glyd/COPYING $W/glyd/bindings/python/ && cp $W/glyd/glyd-store/LICENSE $W/glyd/bindings/python/LICENSE-glyd-store && cp $W/glyd/gpu/LICENSE $W/glyd/bindings/python/LICENSE-glyd-gpu
~/tools/uv/uv pip install -q --python $PY -e $W/glyd/bindings/python > $R/pip-glyd.txt 2>&1; echo "pip exit $?"
$PY -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.general_plugins')))"
run() {  # NAME ARGS...: spike.py, its output in results/NAME.txt and NAME.json (a run done already kept)
  local n=$1; shift
  [ -f $R/$n.json ] && { echo "$n: kept"; return; }
  step "$n"
  (cd $R && VLLM_CACHE_ROOT=${CACHE:-$W/cache/$n} timeout 900 $PY $W/spike.py "$@" $R/$n.json) > $R/$n.txt 2>&1; echo "$n exit $?"
  grep -h "num_gpu_blocks\|tokens_per_s\|load_s\|KV cache size\|Maximum concurrency\|Model loading took\|Capturing CUDA graphs\|Graph capturing finished\|Traceback\|Error" $R/$n.txt | head -20
}
run q17-bf16 Qwen/Qwen3-1.7B bf16
run q17-glyd-mma Qwen/Qwen3-1.7B glyd --layout mma
run q17-glyd-mma12 Qwen/Qwen3-1.7B glyd --layout mma12
run q17-glyd-mma-eager Qwen/Qwen3-1.7B glyd --layout mma --eager
GLYD_EXACT=1 run q17-glyd-exact Qwen/Qwen3-1.7B glyd --layout mma
run q17-bf16-eager Qwen/Qwen3-1.7B bf16 --eager
GLYD_EXACT=1 run q17-glyd-exact-eager Qwen/Qwen3-1.7B glyd --layout mma --eager
run q8-bf16 Qwen/Qwen3-8B bf16
run q8-glyd-mma Qwen/Qwen3-8B glyd --layout mma
# vLLM's compile cache is keyed by the quantization's name, not its options: the 12-bit layout from the tiered run's cache
[ -f $R/q17-glyd-mma12-sharedcache.txt ] || CACHE=$W/cache/q17-glyd-mma run q17-glyd-mma12-sharedcache Qwen/Qwen3-1.7B glyd --layout mma12
step "vllm serve --quantization glyd"
(cd $R && timeout 600 $W/venv/bin/vllm serve Qwen/Qwen3-1.7B --quantization glyd --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8011 > $R/serve.txt 2>&1) &
SV=$!
for i in $(seq 1 120); do curl -s localhost:8011/v1/models > /dev/null 2>&1 && break; sleep 5; done
curl -s localhost:8011/v1/completions -H 'Content-Type: application/json' -d '{"model": "Qwen/Qwen3-1.7B", "prompt": "The history of data compression began", "max_tokens": 32, "temperature": 0}' > $R/serve-reply.json; echo "request exit $?"
cat $R/serve-reply.json | head -c 600; echo
kill $SV; sleep 5; pkill -f "vllm serve Qwen/Qwen3-1.7B" 2>/dev/null
step "compare"
$PY $W/compare.py $R | tee $R/compare.txt
step "done"
