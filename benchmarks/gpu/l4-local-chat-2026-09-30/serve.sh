#!/bin/bash
# vllm serve in the quickstart's venv, in the background, logged to ~/quick/serve-NAME.log; waits until its log says
# it is up or it has exited (15 minutes at most), then prints the log's memory and KV cache lines.
#   bash serve.sh NAME vllm-serve-args...        (the server stays up: bash stop.sh)
Q=~/quick; NAME=$1; shift
export HF_HOME=~/hf HF_HUB_OFFLINE=1 PYTHONPATH=$Q/geforce
unset LD_LIBRARY_PATH
L=$Q/serve-$NAME.log
echo "$ vllm serve $*" > $L
cd $Q && nohup $Q/venv/bin/vllm serve "$@" >> $L 2>&1 < /dev/null &
PID=$!; disown $PID; echo $PID > $Q/serve.pid
t0=$(date +%s)
for i in $(seq 1 300); do
  grep -q "Application startup complete" $L && { echo "up after $(( $(date +%s) - t0 )) s"; break; }
  kill -0 $PID 2>/dev/null || { echo "exited after $(( $(date +%s) - t0 )) s"; break; }
  sleep 3
done
grep -E "glyd|Model loading took|Available KV|GPU KV cache size|Maximum concurrency|maximum model length|Total non KV|Error:|CUDA out of memory" $L | grep -v "^\$ vllm\|DEBUG.*cache_usage" | cut -c1-330 | tail -12
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
