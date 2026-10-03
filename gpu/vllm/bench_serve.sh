#!/usr/bin/env bash
# vllm bench serve, bf16 against Glyd (--quantization glyd), on this GPU: one server a mode, the same
# --gpu-memory-utilization, the random dataset (IN tokens in, OUT out, --ignore-eos) at request rates 0.25, 1, 4, 16 and inf,
# a result JSON a (mode, rate); the server's KV cache (its log's "GPU KV cache size" and "Maximum concurrency") and
# nvidia-smi's memory; each mode on a GPU no other process is on (BUSYWAIT, at most 1800 s' wait, else not run) and
# from about its idle temperature (COOL, 50 C: at most COOLWAIT, 120 s), since at its power cap an L4 slows as it heats,
# and nvidia-smi's temperature, SM clock and power each second of a rate (smi-MODE-rateR.csv); then bench_summary.py's
# table. WARM=1: each mode's server started twice, the first on VLLM_CACHE_ROOT as it is (on an empty one, a cold
# start: its compile, and its KV cache in kv-MODE-cold.txt), the one measured after it (its graphs loaded).
# A mode glyd@F is Glyd packing the fraction F of the layers (GLYD_FRACTION=F for its server: glyd@0 is vLLM's own bf16
# method through the plugin, glyd@1 is glyd), its results in files of its own.
# The passes of a mode share one server, and vLLM's prefix cache is on (its default, as deployments run it) and skips the prefill
# of a prompt it still holds: so pass i (the i-th rate) draws its prompts with --seed SEED+i, none of them any earlier pass's, and
# each mode's pass i gets the same prompts. The summary gives each server's highest prefix cache hit rate and flags a run above 1%.
#   bash bench_serve.sh [MODEL]       (default Qwen/Qwen3-8B)
#   MODES="bf16 glyd@0 glyd@0.5 glyd@1" RATES="1 inf" bash bench_serve.sh [MODEL]
# Env: VLLM (the vllm command), R (results; default ./bench-MODEL), UTIL (0.9), IN (1024), OUT (256), RATES
# ("0.25 1 4 16 inf"), PROMPTS (per rate: "32 64 128 256 256"), SEED (0: the first pass's seed), MODES ("bf16 glyd"), WARM,
# BUSYWAIT, COOL, COOLWAIT, TP (1: the servers over that many GPUs, tensor parallel), SERVE_ARGS (more arguments for every
# vllm serve), GLYD_* (the plugin's options).
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
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
serve() {  # LOG: vllm serve for this mode (SV its pid) once it answers; else stopped, 1
  "$VLLM" serve "$MODEL" "${q[@]}" --max-model-len 4096 --gpu-memory-utilization "$UTIL" --tensor-parallel-size "${TP:-1}" --port "$PORT" ${SERVE_ARGS:-} > "$1" 2>&1 &
  SV=$!
  for i in $(seq 1 180); do curl -sf "localhost:$PORT/v1/models" > /dev/null && return 0; kill -0 $SV 2> /dev/null || break; sleep 5; done
  kill $SV 2> /dev/null; wait $SV 2> /dev/null; SV=
  return 1
}
kv() { grep -h "GPU KV cache size\|Maximum concurrency\|Model loading took\|glyd:" "$1" | sed 's/.*\] //'; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }
GF=${GLYD_FRACTION-}
for mode in ${MODES:-bf16 glyd}; do
  q=(); [ "$mode" = bf16 ] || q=(--quantization glyd)
  case $mode in glyd@*) export GLYD_FRACTION=${mode#glyd@} ;; *) export GLYD_FRACTION=$GF ;; esac
  for i in $(seq 1 $(( ${BUSYWAIT:-1800} / 10 ))); do [ "$(others)" -eq 0 ] && break; sleep 10; done
  if [ "$(others)" -ne 0 ]; then echo "$mode: another process is on the GPU (nvidia-smi): not run"; continue; fi
  if [ "${WARM:-0}" = 1 ]; then
    echo "== $(date -u +%T) $mode: vllm serve $MODEL ${q[*]}, first (cold where VLLM_CACHE_ROOT is empty)"
    if ! serve "$R/serve-$mode-cold.txt"; then echo "$mode: the server did not start (serve-$mode-cold.txt)"; continue; fi
    kv "$R/serve-$mode-cold.txt" | tee "$R/kv-$mode-cold.txt"
    kill $SV; wait $SV 2> /dev/null; SV=
    for i in $(seq 1 30); do [ "$(others)" -eq 0 ] && break; sleep 2; done  # (its engine's processes gone)
  fi
  for i in $(seq 1 $(( ${COOLWAIT:-120} / 5 ))); do [ "$(temp)" -le "${COOL:-50}" ] && break; sleep 5; done
  echo "== $(date -u +%T) $mode: vllm serve $MODEL ${q[*]} (the GPU at $(temp) C)"
  if ! serve "$R/serve-$mode.txt"; then echo "$mode: the server did not start (serve-$mode.txt)"; continue; fi
  kv "$R/serve-$mode.txt" | tee "$R/kv-$mode.txt"
  for i in "${!RATES[@]}"; do
    rate=${RATES[$i]}; n=${PROMPTS[$i]}
    echo "== $(date -u +%T) $mode: rate $rate, $n prompts"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -l 1 > "$R/smi-$mode-rate$rate.csv" &
    SMI=$!
    "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" --dataset-name random --random-input-len "$IN" \
      --random-output-len "$OUT" --ignore-eos --num-prompts "$n" --request-rate "$rate" --seed $(( ${SEED:-0} + i )) --save-result \
      --result-dir "$R" --result-filename "$mode-rate$rate.json" --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,99 > "$R/bench-$mode-rate$rate.txt" 2>&1
    echo "   exit $?"
    kill $SMI; SMI=
  done
  kill $SV; wait $SV 2> /dev/null; SV=
  sleep 5
done
python3 "$here/bench_summary.py" "$R" | tee "$R/summary.txt"
