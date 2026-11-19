#!/bin/bash
# GLY-236 on the owner's PC (RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D, 60 GiB): the cost profiles of Qwen3.5-9B at swift, kestrel and penguin from their
# packed directories (glyd-engine costs), this machine's memory tiers (glyd-engine rates), the predictor against the engine's own bench (glyd-engine bench,
# the hand-written Qwen3.5 engine) and against the dynamic engine served (a streamed reply of 256 tokens after a 1,024-token prompt, timed from the client),
# the costs object of /v1/glyd/info with the allocation, and Nemotron 3 Super's profile from the Hub's safetensors headers (no download) with the predictor
# against the measured decode of the records in benchmarks/gpu/nh-*-2026-10-10 (their tiers in tiers-*.json beside this script). The GPU is shared: run it under the lock,
#   flock ~/glyd-scratch/gpu.lock ./record.sh
# A fresh API key is made here and kept in files of this directory only (removed at the end); nothing prints it.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/glyd-scratch/gly236/rec}
ENGINE=${ENGINE:-$HOME/glyd-scratch/gly236/target/release/glyd-engine}
HUB=${HUB:-$HOME/glyd-scratch/rel032/hub}
SWIFT=${SWIFT:-$HUB/models--glyd--Qwen3.5-9B-swift/snapshots/073a2132abeaa41c119678b9687092871521eb4d}
KESTREL=${KESTREL:-$(ls -d "$HUB"/models--glyd--Qwen3.5-9B-kestrel/snapshots/* 2> /dev/null | head -1)}
PENGUIN=${PENGUIN:-$(ls -d "$HUB"/models--glyd--Qwen3.5-9B-penguin/snapshots/* 2> /dev/null | head -1)}
NEMOTRON=${NEMOTRON:-glyd/nemotron-3-hummingbird}
export GLYD_GPU_LIB=${GLYD_GPU_LIB:-$HOME/glyd-scratch/reg166/lib2/libglyd_gpu_cuda13.so} HF_HUB_CACHE=$HUB HF_HOME=$HOME/.config/huggingface
PORT=8236
B=http://127.0.0.1:$PORT
CTX=4096
PROMPT=1024
NEW=256
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$OUT" || exit 1
(umask 077; head -c 48 /dev/urandom | base64 | tr -dc A-Za-z0-9 | head -c 40 > .key; printf 'Authorization: Bearer %s\n' "$(cat .key)" > .auth)
trap 'rm -f "$OUT/.key" "$OUT/.auth"' EXIT

now() { date +%s.%N | cut -c1-14; }
log() { echo "[$(date -Is)] $*" | tee -a steps.txt; }
smi() {
  { echo "== $1 ($(now))"; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> nvidia-smi.txt
}

# the build
{ echo "engine: $($ENGINE --version), commit ${COMMIT:-$(cd "$HERE/../../.." && git rev-parse HEAD 2> /dev/null || echo unknown)} (GLYD_SOURCE_COMMIT in the JSON), GLYD_ENGINE_ARCHS=89, cargo build --release -p glyd-engine (the development build, default features)"
  sha256sum "$ENGINE" "$GLYD_GPU_LIB"; rustc --version; nvcc --version | tail -2; uname -sr; grep -m1 'model name' /proc/cpuinfo; head -1 /proc/driver/nvidia/version; free -g | head -2; nproc; } > build.txt 2>&1

nvidia-smi > nvidia-smi-full-before.txt
smi "before"

# this machine's tiers: the GPU's memory (read, copy, and the engine's own products over the swift directory's first layers), the link (pinned, pageable), the computer's memory, the disk under the hub cache
log "tiers (512 MiB, 10 passes; the disk under $SWIFT)"
$ENGINE tiers --dir "$SWIFT" --mib 512 --reps 10 > tiers.json 2> tiers.err
log "tiers done: $(jq -c '.rates | map_values(.gbs)' tiers.json 2> /dev/null)"

# a 1,024-token prompt for the served measurement: the sentence repeated (counted by /v1/glyd/tokenize below)
P=$(printf 'The lighthouse keeper climbed the stairs again. %.0s' $(seq 1 114))

served_decode() {
  # $1: the level's name. A streamed plain completion of NEW tokens (the end token ignored) after the prompt: the time from the first token's event to the last's over NEW - 1 tokens.
  local level=$1 t0 tf tl n=0
  curl -s -H @.auth -H 'Content-Type: application/json' $B/v1/glyd/tokenize -d "{\"prompt\":\"$P\"}" > "tokenize-$level.txt"
  t0=$(now)
  tf=
  tl=
  while IFS= read -r line; do
    case "$line" in
      "data: {"*'"text"'*) n=$((n + 1)); [ -z "$tf" ] && tf=$(now); tl=$(now) ;;
    esac
  done < <(curl -sN -H @.auth -H 'Content-Type: application/json' $B/v1/completions -d "{\"prompt\":\"$P\",\"max_tokens\":$NEW,\"temperature\":0,\"ignore_eos\":true,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" | tee "serve-stream-$level.txt")
  local toks; toks=$(jq -r .prompt_tokens "tokenize-$level.txt")
  awk -v n="$n" -v t0="$t0" -v tf="$tf" -v tl="$tl" -v toks="$toks" -v level="$level" 'BEGIN {
    printf "{\"level\":\"%s\",\"prompt_tokens\":%s,\"tokens_streamed\":%d,\"first_token_s\":%.3f,\"decode_s\":%.3f,\"decode_tokens_per_s\":%.2f,\"prefill_tokens_per_s_with_http\":%.1f,\"how\":\"/v1/completions streamed, timed by the client between the first and the last token event; the first token includes the prompt and the HTTP round trip\"}\n",
      level, toks, n, tf - t0, tl - tf, (n - 1) / (tl - tf), toks / (tf - t0) }' > "serve-decode-$level.json"
  log "served $level: $(cat "serve-decode-$level.json")"
}

for level in swift kestrel penguin; do
  case $level in swift) DIR=$SWIFT ;; kestrel) DIR=$KESTREL ;; penguin) DIR=$PENGUIN ;; esac
  if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    log "$level: no snapshot in the cache, skipped"
    continue
  fi
  log "$level: $DIR"
  # the profile with the prediction from this machine's tiers; then the model loaded (the dynamic engine), its allocation beside the profile
  $ENGINE costs "$DIR" --predict --rates tiers.json --ctx $CTX --prompt $PROMPT --new $NEW > "costs-$level.json" 2> "costs-$level.err"
  log "costs $level: resident $(jq .totals.resident_bytes "costs-$level.json"), active $(jq .totals.active_bytes_per_token "costs-$level.json"), predicted $(jq -c '[.predictions[0].placement, .predictions[0].tok_s]' "costs-$level.json")"
  $ENGINE costs "$DIR" --load --ctx $CTX > "costs-$level-loaded.json" 2> "costs-$level-loaded.err"
  log "costs --load $level: $(jq -c .allocated "costs-$level-loaded.json" 2> /dev/null || tail -1 "costs-$level-loaded.err")"
  # the engine's own bench (the hand-written Qwen3.5 engine): the prompt, then NEW tokens, each step timed on the host
  $ENGINE bench "$DIR" --prompt $PROMPT --new $NEW --ctx $CTX --reps 5 > "bench-$level.txt" 2> "bench-$level.err"
  log "bench $level: $(grep '^{' "bench-$level.txt" | tail -1 || tail -2 "bench-$level.err" | tr '\n' ' ')"
  smi "after bench $level"
  # the dynamic engine, served: /v1/glyd/info's costs and memory, and a streamed reply timed
  t0=$(now)
  env GLYD_API_KEY="$(cat .key)" HF_HUB_OFFLINE=1 "$ENGINE" serve "$DIR" --ctx $CTX --port $PORT --status-file "$OUT/status-$level" > "server-$level.out" 2> "server-$level.err" &
  SRV=$!
  for _ in $(seq 1 1200); do
    [ "$(cat "status-$level" 2> /dev/null)" = ready ] && curl -s -o /dev/null $B/health && break
    kill -0 $SRV 2> /dev/null || break
    sleep 0.1
  done
  if kill -0 $SRV 2> /dev/null && [ "$(cat "status-$level" 2> /dev/null)" = ready ]; then
    log "serve $level ready after $(awk "BEGIN {printf \"%.1f\", $(now) - $t0}") s"
    curl -s -H @.auth $B/v1/glyd/info | jq . > "info-$level.json"
    jq '{costs_totals: .costs.totals, costs_allocated: .costs.allocated, memory}' "info-$level.json" > "info-memory-$level.json"
    smi "serving $level, idle"
    served_decode $level
    smi "serving $level, after the reply"
    kill -TERM $SRV
    wait $SRV
  else
    log "serve $level did not come up: $(tail -3 "server-$level.err" | tr '\n' ' ')"
    kill -TERM $SRV 2> /dev/null
    wait $SRV 2> /dev/null
  fi
done

# Nemotron 3 Super at hummingbird: the profile from the Hub's headers (nothing downloaded), the predictor against the records' measured decode with each pod's tiers
log "Nemotron: $NEMOTRON by the Hub's headers"
$ENGINE costs "$NEMOTRON" --ctx $CTX --prompt 42 --new 255 > costs-nemotron-hummingbird.json 2> costs-nemotron-hummingbird.err
log "costs nemotron: resident $(jq .totals.resident_bytes costs-nemotron-hummingbird.json 2> /dev/null), experts $(jq .totals.resident_expert_bytes costs-nemotron-hummingbird.json 2> /dev/null), active $(jq .totals.active_bytes_per_token costs-nemotron-hummingbird.json 2> /dev/null) ($(tail -1 costs-nemotron-hummingbird.err))"
# (pod, tiers file, GPU share of the experts, the measured hit rate, the memory's share of the experts, the placement): the records' rows
while read -r name tiers share hit ram placement; do
  [ -z "$name" ] && continue
  $ENGINE costs "$NEMOTRON" --predict --rates "$HERE/$tiers" --placement "$placement" --gpu-experts "$share" --hit "$hit" --ram-experts "$ram" --ctx $CTX --prompt 42 --new 255 > "predict-nemotron-$name.json" 2> "predict-nemotron-$name.err"
  log "predict nemotron $name: $(jq -c '.predictions[0] | [.placement, .tok_s, .tok_s_overlapped, .bytes_per_token]' "predict-nemotron-$name.json" 2> /dev/null || tail -1 "predict-nemotron-$name.err")"
done <<'EOF'
4090A-copy tiers-rtx4090-podA.json 0.371 0.779 1.0 experts_ram_copy
4090B-copy tiers-rtx4090-podB.json 0.371 0.779 1.0 experts_ram_copy
4090B-cpu tiers-rtx4090-podB.json 0.371 0.723 1.0 experts_ram_cpu
4090B-cpu-kernel tiers-rtx4090-podB-kernel.json 0.371 0.723 1.0 experts_ram_cpu
4090B-16g-copy tiers-rtx4090-podB.json 0.164 0.607 1.0 experts_ram_copy
4090B-16g-cpu tiers-rtx4090-podB.json 0.164 0.547 1.0 experts_ram_cpu
4090B-16g-cpu-kernel tiers-rtx4090-podB-kernel.json 0.164 0.547 1.0 experts_ram_cpu
4090C-ssd-cpu tiers-rtx4090-podC.json 0.371 0.723 0.510 experts_ssd_cpu
4090C-16g-ssd-cpu tiers-rtx4090-podC.json 0.164 0.547 0.510 experts_ssd_cpu
EOF
# the wren pack (the experts four-bit, the rest six-bit) from the base model's config alone: sizes estimated, for the wren records on the RTX 4090 and RTX 5090
$ENGINE costs nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:wren --predict --rates "$HERE/tiers-rtx4090-podE.json" --placement experts_ram_copy --gpu-experts 0.240 --hit 0.677 --ctx $CTX --prompt 42 --new 255 > predict-nemotron-wren-4090E-copy.json 2> predict-nemotron-wren-4090E-copy.err
$ENGINE costs nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:wren --predict --rates "$HERE/tiers-rtx5090-podD.json" --placement experts_ram_copy --gpu-experts 0.371 --hit 0.779 --ctx $CTX --prompt 42 --new 255 > predict-nemotron-wren-5090D-copy.json 2> predict-nemotron-wren-5090D-copy.err
log "predict nemotron wren: 4090E $(jq -c '[.basis.sizes, .totals.resident_bytes, .predictions[0].tok_s, .predictions[0].tok_s_overlapped]' predict-nemotron-wren-4090E-copy.json 2> /dev/null || tail -1 predict-nemotron-wren-4090E-copy.err); 5090D $(jq -c '[.predictions[0].tok_s, .predictions[0].tok_s_overlapped]' predict-nemotron-wren-5090D-copy.json 2> /dev/null || tail -1 predict-nemotron-wren-5090D-copy.err)"
smi "at the end"
log "done"
