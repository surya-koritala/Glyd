#!/usr/bin/env bash
# One engine on the GPU, the load, the GPU's memory, the engine stopped.
#   run.sh engine|vllm OUTDIR      (glyd-pc: takes ~/glyd-scratch/gpu.lock, refuses while a CI job's container is up, and says if one appears)
# engine: glyd-engine serve (BIN, default ~/glyd-scratch/bench-bin/glyd-engine) with the context and users below.
# vllm:   what `glyd serve` starts for the frozen path: vLLM 0.30.0 with --quantization glyd (glyd-gpu 0.32.1, VENV) and Glyd's weights; glyd-pc has no nvcc, which vLLM's
#         FlashInfer sampler compiles with when it first runs, so VLLM_USE_FLASHINFER_SAMPLER=0 (vLLM's own sampler, a little slower per step, the same tokens).
set -uo pipefail
WHICH=$1; OUT=$2; mkdir -p "$OUT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=${HF_HOME:-$HOME/glyd-demo/hf} HF_HUB_OFFLINE=1
MODEL=${MODEL:-Qwen/Qwen3-8B} CTX=${CTX:-8192} USERS=${USERS:-8} PORT=${PORT:-8100}
BIN=${BIN:-$HOME/glyd-scratch/bench-bin/glyd-engine} VENV=${VENV:-$HOME/glyd-scratch/vllm-env}
exec 9> "$HOME/glyd-scratch/gpu.lock"; flock 9
ci() { docker ps --format '{{.Image}} {{.Names}}' | grep -E 'nvidia/cuda' ; }
if ci; then echo "a CI container is up; not measuring"; exit 3; fi
nvidia-smi --query-gpu=name,driver_version,memory.used,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.txt"
case $WHICH in
  engine) LD_LIBRARY_PATH=$(dirname "$BIN") GLYD_GPU_LIB=$(dirname "$BIN")/libglyd_gpu_cuda13.so "$BIN" serve "$MODEL" --port "$PORT" --context "$CTX" --users "$USERS" > "$OUT/server.log" 2>&1 & ;;
  vllm)   VLLM_USE_FLASHINFER_SAMPLER=0 "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --quantization glyd --port "$PORT" --max-model-len "$CTX" --max-num-seqs "$USERS" > "$OUT/server.log" 2>&1 & ;;
esac
SERVER=$!
# (a CI job that starts at any moment of the run, short or not, spoils it: the container's name is written to DISCARD, once a second looked for)
( while :; do c=$(ci); [ -n "$c" ] && echo "$(date +%T) $c" >> "$OUT/DISCARD"; sleep 1; done ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null; kill $SERVER 2>/dev/null; sleep 5; pkill -P $SERVER 2>/dev/null' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break; kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -5 "$OUT/server.log"; exit 1; }; sleep 2; done
sleep 5
python3 "$HERE/load.py" --url "http://127.0.0.1:$PORT" --label "$WHICH" --out "$OUT/load.json" | tee "$OUT/load.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/memory.txt"
[ -e "$OUT/DISCARD" ] && echo "A CI CONTAINER WAS UP DURING THE RUN: discard $OUT"
