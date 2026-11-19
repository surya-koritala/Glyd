#!/bin/bash
# GLY-238 on the owner's PC: glyd-engine serve of Qwen3.5-9B at swift (the glyd/Qwen3.5-9B-swift snapshot, --ctx 4096): /v1/glyd/info,
# the API's answers, and nvidia-smi with the engine's own counters before, during and after a cancelled streamed request and a client
# that goes away; then the same model served by its Hub name and level. The GPU is shared: run it under the lock,
#   flock ~/glyd-scratch/gpu.lock ./record.sh
# A fresh API key is made here and kept in files of this directory only (removed at the end); nothing prints it.
set -u
OUT=${OUT:-$HOME/glyd-scratch/gly238/rec}
ENGINE=${ENGINE:-$HOME/glyd-scratch/gly238/target/release/glyd-engine}
SNAP=${SNAP:-$HOME/glyd-scratch/rel032/hub/models--glyd--Qwen3.5-9B-swift/snapshots/073a2132abeaa41c119678b9687092871521eb4d}
export GLYD_GPU_LIB=${GLYD_GPU_LIB:-$HOME/glyd-scratch/reg166/lib2/libglyd_gpu_cuda13.so} HF_HUB_CACHE=$HOME/glyd-scratch/rel032/hub HF_HOME=$HOME/.config/huggingface
PORT=8238
B=http://127.0.0.1:$PORT
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
ours() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$1" '$1 == p {print $2}'; }
mem() {
  curl -s -H @.auth $B/v1/glyd/info | jq -c --arg l "$1" --arg smi "$(ours "$SRV")" '{label: $l, nvidia_smi_this_process_mib: ($smi | tonumber? // null), state, requests, memory}' >> memory.jsonl
}
status() { curl -s -H @.auth "$B/v1/glyd/requests/$1"; }
until_ended() {
  local s i
  for i in $(seq 1 1000); do
    s=$(status "$1")
    [ "$(echo "$s" | jq -r .state)" = ended ] && break
    sleep 0.01
  done
  echo "$s"
}
post() { curl -s -H @.auth -H 'Content-Type: application/json' -w '\nHTTP %{http_code}\n' "$B$1" -d "$2"; }
serve() {
  log "start: glyd-engine serve $1 --ctx 4096 --port $PORT ($2)"
  local t0
  t0=$(now)
  env GLYD_API_KEY="$(cat .key)" ${OFFLINE:+HF_HUB_OFFLINE=1} "$ENGINE" serve "$1" --ctx 4096 --port $PORT --status-file "$OUT/status-$2" > "server-$2.out" 2> "server-$2.err" &
  SRV=$!
  for _ in $(seq 1 1200); do
    [ "$(cat "status-$2" 2> /dev/null)" = ready ] && curl -s -o /dev/null $B/health && break
    kill -0 $SRV 2> /dev/null || { log "the server exited: $(tail -3 "server-$2.err")"; exit 1; }
    sleep 0.1
  done
  log "ready ($2) after $(awk "BEGIN {printf \"%.1f\", $(now) - $t0}") s, pid $SRV"
}
stop() {
  kill -TERM $SRV
  wait $SRV
  log "the server stopped (exit status $?)"
}

nvidia-smi > nvidia-smi-full-before.txt
smi "before the server"
OFFLINE=1 serve "$SNAP" dir  # (offline: the directory alone is read)

# what is open and what the key guards; the probe
{ echo "GET /health, no key: $(curl -s -w ' HTTP %{http_code}' $B/health)"
  echo "GET /v1/models, no key: $(curl -s -w ' HTTP %{http_code}' $B/v1/models)"
  echo "GET /v1/glyd/info, no key: $(curl -s -o /dev/null -w 'HTTP %{http_code}' $B/v1/glyd/info)"
  echo "GET /v1/models, the key: $(curl -s -H @.auth $B/v1/models)"; } > access.txt
curl -s -H @.auth $B/v1/glyd/info | jq . > info.json
smi "loaded, idle"
mem "loaded, idle"

# the API's answers: JSON (thinking off), SSE (thinking on, final usage), a length cut, the count of a prompt, refusals
Q='[{"role":"user","content":"Name three primary colors."}]'
post /v1/chat/completions "{\"messages\":$Q,\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > chat-json.txt
post /v1/glyd/tokenize "{\"messages\":$Q,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > tokenize.txt
post /v1/chat/completions "{\"messages\":$Q,\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > chat-length.txt
curl -sN -H @.auth -H 'Content-Type: application/json' $B/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":1024,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' > chat-sse.txt
{ echo "logprobs at this level:"; post /v1/chat/completions "{\"messages\":$Q,\"max_tokens\":8,\"logprobs\":true}"
  echo "max_tokens past the context:"; post /v1/chat/completions "{\"messages\":$Q,\"max_tokens\":5000}"
  echo "a request id no request has:"; curl -s -H @.auth -w '\nHTTP %{http_code}\n' $B/v1/glyd/requests/chatcmpl-none
  echo "a cancel by GET:"; curl -s -H @.auth -w '\nHTTP %{http_code}\n' $B/v1/glyd/requests/chatcmpl-none/cancel; } > errors.txt

# a streamed request cancelled by its id
LONG='[{"role":"user","content":"Write a long story about a lighthouse keeper and a winter storm."}]'
smi "before the streamed request"
mem "before the streamed request"
curl -sN -H @.auth -H 'Content-Type: application/json' $B/v1/chat/completions \
  -d "{\"messages\":$LONG,\"max_tokens\":3000,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true},\"request_id\":\"rec-1\"}" > cancel-stream.txt &
CURL=$!
sleep 3
smi "during the streamed request"
mem "during the streamed request"
status chatcmpl-rec-1 > status-1-during.json
TC=$(now)
curl -s -X POST -H @.auth $B/v1/glyd/requests/chatcmpl-rec-1/cancel > status-1-cancel.json
until_ended chatcmpl-rec-1 > status-1-ended.json
TE=$(now)
log "cancel asked at $TC; the request ended (the engine let go of it) at $TE, $(awk "BEGIN {printf \"%.3f\", $TE - $TC}") s later"
wait $CURL
smi "after the cancel"
mem "after the cancel"

# a client that goes away
curl -sN -H @.auth -H 'Content-Type: application/json' $B/v1/chat/completions \
  -d "{\"messages\":$LONG,\"max_tokens\":3000,\"temperature\":0,\"stream\":true,\"request_id\":\"rec-2\"}" > disconnect-stream.txt &
CURL=$!
sleep 3
smi "during the second streamed request"
mem "during the second streamed request"
kill $CURL
TK=$(now)
until_ended chatcmpl-rec-2 > status-2-ended.json
TE=$(now)
log "the client went away at $TK; the request ended at $TE, $(awk "BEGIN {printf \"%.3f\", $TE - $TK}") s later"
smi "after the client went away"
mem "after the client went away"

# a cancel while a long prompt is read (this path reads the whole prompt before it looks at a cancel)
P=$(printf 'The lighthouse keeper climbed the stairs again. %.0s' $(seq 1 420))
post /v1/glyd/tokenize "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}]}" > tokenize-long.txt
curl -sN -H @.auth -H 'Content-Type: application/json' $B/v1/chat/completions \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":64,\"temperature\":0,\"stream\":true,\"request_id\":\"rec-3\"}" > prompt-cancel-stream.txt &
CURL=$!
sleep 0.2
TC=$(now)
curl -s -X POST -H @.auth $B/v1/glyd/requests/chatcmpl-rec-3/cancel > status-3-cancel.json
until_ended chatcmpl-rec-3 > status-3-ended.json
TE=$(now)
log "a cancel 0.2 s into a prompt of $(head -1 tokenize-long.txt | jq .prompt_tokens) tokens, asked at $TC: the request ended at $TE, $(awk "BEGIN {printf \"%.3f\", $TE - $TC}") s later"
wait $CURL
smi "after the cancel during a prompt"
mem "after the cancel during a prompt"
curl -s $B/stats > stats.json
stop
smi "after the server stopped"

# the same model by its Hub name and level (the base model's config.json is read from the Hub; the pre-packed repo is the one in the cache)
OFFLINE= serve "Qwen/Qwen3.5-9B:swift" hubname
curl -s -H @.auth $B/v1/glyd/info | jq . > info-hubname.json
curl -s -H @.auth $B/v1/models > models-hubname.json
smi "loaded by its Hub name"
stop
smi "at the end"
log "done"
