#!/bin/bash
# The server serve.sh started, stopped (and its engine process gone).
Q=~/quick
P=$(cat $Q/serve.pid 2>/dev/null); [ -n "$P" ] && kill $P 2>/dev/null
for i in $(seq 1 30); do pgrep -f "quick/venv/bin/vllm serve" > /dev/null || pgrep -f "VLLM::EngineCore" > /dev/null || break; sleep 2; done
pkill -9 -f "quick/venv/bin/vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
