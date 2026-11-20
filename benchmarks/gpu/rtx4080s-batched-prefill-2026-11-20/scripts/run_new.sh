#!/usr/bin/env bash
# a fresh server of ENGINE, the token check in MODE (alone or together), stopped.  run_new.sh ENGINE OUT.json MODE
set -uo pipefail
BIN=$1; OUT=$2; MODE=$3; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=$HOME/glyd-demo/hf HF_HUB_OFFLINE=1
exec 9> "$HOME/glyd-scratch/gpu.lock"; flock 9
while docker ps --format '{{.Image}}' | grep -q nvidia/cuda; do sleep 5; done
LD_LIBRARY_PATH=$(dirname "$BIN") GLYD_GPU_LIB=$(dirname "$BIN")/libglyd_gpu_cuda13.so "$BIN" serve Qwen/Qwen3-8B --port 8102 --context 8192 --users 8 > "$OUT.log" 2>&1 &
S=$!; trap 'kill $S; sleep 3' EXIT
until curl -sf http://127.0.0.1:8102/v1/models > /dev/null; do sleep 2; kill -0 $S 2>/dev/null || { tail -5 "$OUT.log"; exit 1; }; done
python3 "$HERE/tokens_check.py" --url http://127.0.0.1:8102 --mode "$MODE" --out "$OUT"
