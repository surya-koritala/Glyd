#!/usr/bin/env bash
# A few users at once on this GPU, the fixed bench's way (vllm bench serve, random prompts of IN tokens and OUT out with --ignore-eos, a new set of prompts for every pass: --seed SEED+i), 1, 4 and 8 users
# at a time (--max-concurrency), for one server: MODE engine (glyd-engine serve), kvglyd (vLLM with Glyd's weights and the lossless cache, as `bench_serve.sh` runs kv+glyd) or bf16 (vLLM as it comes). The prefix cache's hit rate is
# printed for every pass (the engine's /stats, the vLLM server's log): a run above 1% is flagged, its prompts were not new.
#   engine_conc_bench.sh MODE RESULT_DIR
# Env: GLYD_GPU_LIB, B (the glyd-engine binary), M (its model directory), T (its KV tables), VLLM, MODEL (Qwen/Qwen3-8B), CONC ("1 4 8"), NP (prompts a pass: "8 24 32"), SEED (0), IN (1024), OUT (256),
# COOL (50: C to start from), COOLWAIT (240 s), PORT, BENCH_ARGS (more arguments for the client: the sampling the requests ask for is the server's own default unless --temperature and the like are given).
set -u
MODE=${1:?MODE engine|kvglyd}; R=${2:?RESULT_DIR}
mkdir -p "$R"
CONC=(${CONC:-1 4 8}); NP=(${NP:-8 24 32}); SEED=${SEED:-0}; IN=${IN:-1024}; OUT=${OUT:-256}
PORT=${PORT:-8030}; MODEL=${MODEL:-Qwen/Qwen3-8B}; VLLM=${VLLM:-$HOME/v027/venv/bin/vllm}
unset PYTHONPATH LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-$HOME/hf} TOKENIZERS_PARALLELISM=false
SV= SMI=
trap '[ -n "$SV" ] && kill $SV 2> /dev/null; [ -n "$SMI" ] && kill $SMI 2> /dev/null; wait 2> /dev/null' EXIT
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
[ "$(others)" -eq 0 ] || { echo "another process is on the GPU: not run"; exit 1; }
for i in $(seq 1 $(( ${COOLWAIT:-240} / 5 ))); do [ "$(temp)" -le "${COOL:-50}" ] && break; sleep 5; done
echo "== $(date -u +%T) $MODE: the GPU at $(temp) C"
case $MODE in
  engine) GLYD_GPU_LIB=${GLYD_GPU_LIB:?} "${B:?}" serve "${M:?}" --tables "${T:?}" --port $PORT --ctx 4096 --users 8 > "$R/serve-$MODE.log" 2>&1 9>&- & ;;
  kvglyd) GLYD_FRACTION=1 GLYD_KV=lossless "$VLLM" serve "$MODEL" --quantization glyd --max-model-len 4096 --gpu-memory-utilization 0.9 --port $PORT > "$R/serve-$MODE.log" 2>&1 9>&- & ;;
  bf16) "$VLLM" serve "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.9 --port $PORT > "$R/serve-$MODE.log" 2>&1 9>&- & ;;
  *) echo "MODE engine, kvglyd or bf16"; exit 2 ;;
esac
SV=$!
for i in $(seq 1 180); do curl -sf "localhost:$PORT/v1/models" > /dev/null && break; kill -0 $SV 2> /dev/null || { echo "the server did not start ($R/serve-$MODE.log)"; exit 1; }; sleep 5; done
echo "== $(date -u +%T) $MODE: server up"
bench() {  # N C SEED NAME
  "$VLLM" bench serve --backend vllm --model "$MODEL" --port $PORT --dataset-name random --random-input-len $IN --random-output-len $OUT --ignore-eos --num-prompts "$1" --max-concurrency "$2" --request-rate inf \
    --seed "$3" --save-result --result-dir "$R" --result-filename "$4.json" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 ${BENCH_ARGS:-} > "$R/bench-$4.txt" 2>&1
}
hit() {  # the highest prefix cache hit rate (%) the server has reported so far
  if [ $MODE = engine ]; then curl -s localhost:$PORT/stats | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefix_hit_rate_percent"])'
  else grep -o "Prefix cache hit rate: [0-9.]*%" "$R/serve-$MODE.log" | grep -o "[0-9.]*" | sort -g | tail -1 | grep . || echo 0; fi
}
snap() {  # the engine's counts so far (a pass's own: the difference of two)
  [ $MODE = engine ] && curl -s localhost:$PORT/stats > "$R/stats-$MODE-$1.json"
  return 0
}
bench 4 4 99 warmup; echo "   warm-up exit $?"
snap warmup
for i in "${!CONC[@]}"; do
  c=${CONC[$i]}; n=${NP[$i]}
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -l 1 > "$R/smi-$MODE-c$c.csv" 9>&- &
  SMI=$!
  echo "== $(date -u +%T) $MODE: $c at a time, $n prompts (the GPU at $(temp) C)"
  bench "$n" "$c" $(( SEED + i )) "$MODE-c$c"; echo "   exit $?"
  kill $SMI; SMI=
  snap c$c
  h=$(hit); echo "   prefix cache hit rate so far: $h%"; echo "$h" > "$R/hit-$MODE-c$c.txt"
  awk -v h="$h" 'BEGIN { if (h + 0 > 1) print "   NOT COMPARABLE: the hit rate is above 1%" }'
done
[ $MODE = engine ] && curl -s localhost:$PORT/stats > "$R/stats-$MODE.json"
kill $SV; wait $SV 2> /dev/null; SV=
echo "== $(date -u +%T) $MODE: done"
