#!/usr/bin/env bash
# vllm bench serve, bf16 against Glyd (--quantization glyd), on this GPU: one server a mode, the same
# --gpu-memory-utilization, the random dataset (IN tokens in, OUT out, --ignore-eos) at request rates 0.25, 1, 4, 16 and inf,
# a result JSON a (mode, rate); the server's KV cache (its log's "GPU KV cache size" and "Maximum concurrency") and
# nvidia-smi's memory; each mode from about the GPU's idle temperature (COOL, 50 C: at most 2 minutes' wait), since at
# its power cap an L4 slows as it heats, and nvidia-smi's temperature, SM clock and power each second of a rate
# (smi-MODE-rateR.csv); then bench_summary.py's table.
#   bash bench_serve.sh [MODEL]       (default Qwen/Qwen3-8B)
# Env: VLLM (the vllm command), R (results; default ./bench-MODEL), UTIL (0.9), IN (1024), OUT (256), RATES
# ("0.25 1 4 16 inf"), PROMPTS (per rate: "32 64 128 256 256"), MODES ("bf16 glyd"), GLYD_* (the plugin's options).
set -u
MODEL=${1:-Qwen/Qwen3-8B}
VLLM=${VLLM:-vllm}
R=${R:-$PWD/bench-$(basename "$MODEL")}
UTIL=${UTIL:-0.9}; IN=${IN:-1024}; OUT=${OUT:-256}
RATES=(${RATES:-0.25 1 4 16 inf}); PROMPTS=(${PROMPTS:-32 64 128 256 256})
PORT=${PORT:-8012}
mkdir -p "$R"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SV= SMI=
trap '[ -n "$SV" ] && kill $SV 2> /dev/null; [ -n "$SMI" ] && kill $SMI 2> /dev/null; wait 2> /dev/null' EXIT
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
for mode in ${MODES:-bf16 glyd}; do
  q=(); [ "$mode" = bf16 ] || q=(--quantization glyd)
  for i in $(seq 1 24); do [ "$(temp)" -le "${COOL:-50}" ] && break; sleep 5; done
  echo "== $(date -u +%T) $mode: vllm serve $MODEL ${q[*]} (the GPU at $(temp) C)"
  "$VLLM" serve "$MODEL" "${q[@]}" --max-model-len 4096 --gpu-memory-utilization "$UTIL" --port "$PORT" > "$R/serve-$mode.txt" 2>&1 &
  SV=$!
  for i in $(seq 1 180); do curl -sf "localhost:$PORT/v1/models" > /dev/null && break; kill -0 $SV 2> /dev/null || break; sleep 5; done
  if ! curl -sf "localhost:$PORT/v1/models" > /dev/null; then echo "$mode: the server did not start (serve-$mode.txt)"; kill $SV 2> /dev/null; SV=; continue; fi
  grep -h "GPU KV cache size\|Maximum concurrency\|Model loading took\|glyd:" "$R/serve-$mode.txt" | sed 's/.*\] //' | tee "$R/kv-$mode.txt"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$R/kv-$mode.txt"
  for i in "${!RATES[@]}"; do
    rate=${RATES[$i]}; n=${PROMPTS[$i]}
    echo "== $(date -u +%T) $mode: rate $rate, $n prompts"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -l 1 > "$R/smi-$mode-rate$rate.csv" &
    SMI=$!
    "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" --dataset-name random --random-input-len "$IN" \
      --random-output-len "$OUT" --ignore-eos --num-prompts "$n" --request-rate "$rate" --seed 0 --save-result \
      --result-dir "$R" --result-filename "$mode-rate$rate.json" --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,99 > "$R/bench-$mode-rate$rate.txt" 2>&1
    echo "   exit $?"
    kill $SMI; SMI=
  done
  kill $SV; wait $SV 2> /dev/null; SV=
  sleep 5
done
python3 "$here/bench_summary.py" "$R" | tee "$R/summary.txt"
