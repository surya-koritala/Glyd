#!/bin/bash
# The refusals over several processes, on the one L4: check_vllm.py --quick --mp (vLLM's workers in processes of their
# own, as over several GPUs) on Qwen3-1.7B, under the box's lock; then a refused run's first lines as a user sees them.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
W=~/vllm-work; V=$W/venv/bin; O=$W/m7; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp
echo "== $(date -u +%T) check_vllm --quick --mp Qwen/Qwen3-1.7B"
(cd $O && flock ~/.glyd-box.lock timeout 5400 $V/python $W/glyd/gpu/vllm/check_vllm.py --quick --mp --out $O/check-Qwen3-1.7B Qwen/Qwen3-1.7B) > $O/check-Qwen3-1.7B.txt 2>&1; echo "exit $?"
grep "PASS\|FAIL\|check_vllm:\|Error" $O/check-Qwen3-1.7B.txt | cut -c1-230
echo "== $(date -u +%T) vllm serve, exact and compiled, workers in processes of their own: the output's last lines"
(cd $O && GLYD_EXACT=1 flock ~/.glyd-box.lock timeout 600 $V/vllm serve Qwen/Qwen3-1.7B --quantization glyd --distributed-executor-backend mp --max-model-len 4096 --port 8031) > $O/serve-refused.txt 2>&1; echo "exit $?"
tail -4 $O/serve-refused.txt | cut -c1-300
echo "== $(date -u +%T) done"
