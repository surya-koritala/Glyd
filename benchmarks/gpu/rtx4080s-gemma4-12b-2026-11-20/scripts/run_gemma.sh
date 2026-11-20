#!/usr/bin/env bash
# One program on the GPU under the load of ../../rtx4080s-engine-vs-vllm-2026-11-20/scripts/load.py (copy it beside this), the GPU's memory, the program stopped.
#   run_gemma.sh engine MODEL[:LEVEL] OUTDIR      glyd-engine serve (BIN, default ~/glyd-scratch/ttft/out/glyd-engine-new): Gemma 4 runs one request at a time
#   run_gemma.sh vllm  MODEL         OUTDIR      vLLM 0.30.0 in VENV, the checkpoint as it is (a compressed-tensors one: no --quantization)
# Takes ~/glyd-scratch/gpu.lock and refuses while a CI container is up, as the earlier record's run.sh does; DISCARD is written if one appears.
set -uo pipefail
WHICH=$1; MODEL=$2; OUT=$3; mkdir -p "$OUT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=${HF_HOME:-$HOME/glyd-scratch/hf2} HF_HUB_OFFLINE=1
CTX=${CTX:-8192} PORT=${PORT:-8104} BIN=${BIN:-$HOME/glyd-scratch/ttft/out/glyd-engine-new} VENV=${VENV:-$HOME/glyd-scratch/vllm-env}
exec 9> "$HOME/glyd-scratch/gpu.lock"; flock 9
ci() { docker ps --format '{{.Image}} {{.Names}}' | grep -E 'nvidia/cuda'; }
while ci > /dev/null; do sleep 5; done
case $WHICH in
  engine) LD_LIBRARY_PATH=$(dirname "$BIN") GLYD_GPU_LIB=$(dirname "$BIN")/libglyd_gpu_cuda13.so "$BIN" serve "$MODEL" --port "$PORT" --context "$CTX" > "$OUT/server.log" 2>&1 & ;;
  vllm)   VLLM_USE_FLASHINFER_SAMPLER=0 "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --port "$PORT" --max-model-len "$CTX" --max-num-seqs 8 > "$OUT/server.log" 2>&1 & ;;
esac
SERVER=$!
( while :; do c=$(ci); if [ -n "$c" ]; then echo "$(date +%T) $c" >> "$OUT/DISCARD"; kill $SERVER 2>/dev/null; pkill -f "load.py --url http://127.0.0.1:$PORT" 2>/dev/null; fi; sleep 1; done ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null; kill $SERVER 2>/dev/null; sleep 5; pkill -P $SERVER 2>/dev/null' EXIT
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break; kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -8 "$OUT/server.log"; exit 1; }; sleep 2; done
sleep 5
python3 "$HERE/load.py" --url "http://127.0.0.1:$PORT" --label "$WHICH" --out "$OUT/load.json" --max-tokens 128 ${ROUNDS:+--rounds $ROUNDS} | tee "$OUT/load.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$OUT/memory.txt"
[ -e "$OUT/DISCARD" ] && echo "A CI CONTAINER WAS UP DURING THE RUN: discard $OUT"
