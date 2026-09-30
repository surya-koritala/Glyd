#!/bin/bash
# A fraction vLLM's server is asked for and Glyd refuses: outside 0 to 1 and not a number, as --additional-config's option and
# as GLYD_FRACTION, on Qwen3-0.6B (the last lines of each server's output, and its exit), under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 5; done
echo "gpu used ${used} MiB at $(date -u +%T)"
B=~/budget; S=$B/src; O=$B/bad; mkdir -p $O $B/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$HOME/mmoedry/w/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$S/bindings/python TMPDIR=$B/tmp VLLM_CACHE_ROOT=$B/cache/bad
V=~/mmoedry/w/venv/bin
cd $B
try() {  # NAME, then the environment and arguments
  local name=$1; shift
  echo "== $(date -u +%T) $name"
  timeout 240 "$@" > $O/$name.txt 2>&1; echo "exit $?"
  grep -E "glyd:|Error" $O/$name.txt | tail -3 | cut -c1-330
}
try json-1.5 $V/vllm serve Qwen/Qwen3-0.6B --quantization glyd --additional-config '{"glyd": {"fraction": 1.5}}' --port 8041
try json-minus $V/vllm serve Qwen/Qwen3-0.6B --quantization glyd --additional-config '{"glyd": {"fraction": -0.25}}' --port 8041
try json-half-word $V/vllm serve Qwen/Qwen3-0.6B --quantization glyd --additional-config '{"glyd": {"fraction": "half"}}' --port 8041
try env-abc env GLYD_FRACTION=abc $V/vllm serve Qwen/Qwen3-0.6B --quantization glyd --port 8041
try env-2 env GLYD_FRACTION=2 $V/vllm serve Qwen/Qwen3-0.6B --quantization glyd --port 8041
