#!/bin/bash
# The GPU memory peak while Qwen3-8B loads (vLLM's debug line "Peak GPU memory after loading weights"), bf16 and glyd.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
W=~/vllm-work; PY=$W/venv/bin/python; R=$W/results
export HF_HOME=~/hf HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 VLLM_LOGGING_LEVEL=DEBUG
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so
for q in bf16 glyd; do
  (cd $R && VLLM_CACHE_ROOT=$W/cache/peak-$q timeout 600 $PY -c "
import sys
from vllm import LLM
LLM(model='Qwen/Qwen3-8B', quantization=None if sys.argv[1] == 'bf16' else 'glyd', dtype='bfloat16', gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True)
" $q) > $R/peak-$q.txt 2>&1
  echo "$q exit $?: $(grep -h "Peak GPU memory after loading weights\|Model loading took" $R/peak-$q.txt | sed 's/.*\] //' | tr '\n' ' ')"
done
