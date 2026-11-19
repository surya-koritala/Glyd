#!/usr/bin/env bash
# run-2026-families GPU check, second pod: the fixes of the first (the plugin's count and Gemma 4's config, MiniCPM5's reasoning parser, the sampler's peak) on the models they touch
# (one RTX 4090 pod): glyd run --prompt, glyd serve and the API (a tool call and the turn after it, thinking apart,
# a prompt of ~90% of the chosen context) for Qwen3.5-9B, gemma-4-12B-it, granite-4.2-8b and NVIDIA-Nemotron-3-Nano-4B (MiniCPM5-2B in its
# place if it does not start), with glyd_gpu from branch run-2026-families (a pure-Python wheel of it, the v0.29.0 release's CUDA libraries)
# installed by v0.29.0's install.sh. Uploaded beside it: install.sh, glyd_gpu-0.29.0+runfam-py3-none-any.whl, api_check.py. Results in ~/results; DONE at the end.
set -uo pipefail
R="$HOME/results"; mkdir -p "$R"; T0=$(date +%s); MAX_MIN="${MAX_MIN:-100}"; PASS=(); FAIL=()
log() { echo "$(date -u +%T) ($(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
trap 'log "ending"; touch "$R/DONE"' EXIT
check() { if [ "$2" = 0 ]; then PASS+=("$1"); log "ok   $1"; else FAIL+=("$1"); log "FAIL $1${3:+ -- $3}"; fi; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
step() { local name=$1 secs=$2; shift 2; local l; l=$(left)
    if [ "$l" -lt 60 ]; then log "skipped $name: $l s left"; RC=99; return 0; fi
    [ "$secs" -gt $((l - 20)) ] && secs=$((l - 20))
    log "$name (at most $secs s)"; local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$R/$name.out" 2> "$R/$name.err"; RC=$?
    log "$name: exit $RC after $((SECONDS - t)) s"; echo "$name $RC $((SECONDS - t))" >> "$R/steps.txt"; }
serve() { local name=$1 port=$2; shift 2; log "serve $name on $port"
    "$@" > "$R/$name.serve.out" 2> "$R/$name.serve.err" < /dev/null & SRV=$!; local t=$SECONDS
    for _ in $(seq 1 450); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > "$R/$name.models.json" 2> /dev/null && break
        kill -0 "$SRV" 2> /dev/null || { log "$name exited before it answered"; SRV=""; return 0; }; sleep 2; done
    curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 || { log "$name never answered"; kill -KILL "$SRV" 2> /dev/null; SRV=""; return 0; }
    echo "$name up in $((SECONDS - t)) s, GPU memory used $(gpu_used) MiB" | tee -a "$R/times.txt"; }
stop() { [ -n "${SRV:-}" ] || return 0; kill -TERM "$SRV" 2> /dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
    kill -KILL "$SRV" 2> /dev/null; pkill -f "vllm.entrypoints" 2> /dev/null; sleep 5; SRV=""; }
vlog() { grep -h -o -E "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|--max-model-len [0-9]+|--tool-call-parser [a-z0-9_]+|--reasoning-parser [a-z0-9_]+|--gpu-memory-utilization [0-9.]+|CUDA out of memory" "$@" 2> /dev/null | sort -u | tr '\n' ';'; }

{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"

# ---- install: v0.29.0's install.sh, glyd-gpu from this branch's wheel ---------------------------------------------------------------------
mkdir -p "$HOME/wheels"; mv "$HOME"/glyd_gpu-*.whl "$HOME/wheels/"; export UV_FIND_LINKS="$HOME/wheels"
sha256sum "$HOME"/wheels/*.whl > "$R/wheel.sha256"
step install 1500 sh "$HOME/install.sh"
check "install.sh (glyd 0.29.0, glyd-gpu from the branch's wheel): exit 0" "$RC" "see install.err"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir 2> /dev/null)/glyd"; PY="$TOOL/bin/python"
"$PY" -c 'import glyd_gpu, glyd_gpu.preflight as p, os; print(glyd_gpu.__file__, "PARSERS" in dir(p), "kv_fit" in dir(p))' > "$R/which_glyd_gpu.txt" 2>&1
check "the installed glyd_gpu is the branch's (PARSERS, kv_fit)" "$(grep -q 'True True' "$R/which_glyd_gpu.txt"; echo $?)" "$(cat "$R/which_glyd_gpu.txt")"
grep -q 'True True' "$R/which_glyd_gpu.txt" || { log "not the branch's glyd_gpu: stopping here"; exit 1; }
"$PY" -m pip list 2> /dev/null | grep -i -E "^(vllm|torch|glyd|glyd-gpu|transformers) " > "$R/versions.txt" || uv pip list --python "$PY" 2> /dev/null | grep -i -E "^(vllm|torch|glyd|glyd-gpu|transformers) " > "$R/versions.txt"
step doctor 300 glyd doctor
check "glyd doctor: exit 0" "$RC" "$(tail -3 "$R/doctor.out")"

# ---- the models ----------------------------------------------------------------------------------------------------------------------------
PORT=8100
one() { local repo=$1 tag=$2
    log "=== $repo"
    step "run_$tag" 1800 glyd run "$repo" --prompt "What is 17 times 23? Give the number."
    local glog; glog=$(ls -t "$HOME"/.local/state/glyd/logs/*.log 2> /dev/null | head -1); [ -n "$glog" ] && cp "$glog" "$R/run_$tag.vllm.log"
    check "glyd run $repo --prompt: exit 0, an answer, no thinking or tool marks in it" "$([ "$RC" = 0 ] && [ -s "$R/run_$tag.out" ] && ! grep -q -E '<think>|</think>|<\|channel|<tool_call>' "$R/run_$tag.out"; echo $?)" "$(tail -c 300 "$R/run_$tag.err")"
    check "glyd run $repo: one start (the chosen context fit vLLM's first try)" "$(grep -q 'starting again' "$R/run_$tag.err"; [ $? = 1 ]; echo $?)" "$(grep -h 'starting again' "$R/run_$tag.err")"
    echo "$repo glyd run: $(grep -h '^glyd run' "$R/run_$tag.err" | head -3 | tr '\n' ' ') | $(vlog "$R/run_$tag.vllm.log")" >> "$R/numbers.txt"
    PORT=$((PORT + 1))
    serve "serve_$tag" "$PORT" glyd serve "$repo" --port "$PORT"
    glog=$(ls -t "$HOME"/.local/state/glyd/logs/*.log 2> /dev/null | head -1); [ -n "$glog" ] && cp "$glog" "$R/serve_$tag.vllm.log"
    if [ -z "${SRV:-}" ]; then check "glyd serve $repo: answers /v1/models" 1 "$(tail -c 400 "$R/serve_$tag.serve.err")"; return 1; fi
    check "glyd serve $repo: answers /v1/models" 0
    step "api_$tag" 2400 python3 "$HOME/api_check.py" "http://127.0.0.1:$PORT" "$R/api_$tag.json"
    for k in tool_call tool_result_turn thinking long_prompt; do
        check "glyd serve $repo: $k" "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get(sys.argv[2], {}).get("ok") else 1)' "$R/api_$tag.json" "$k" 2> /dev/null; echo $?)" "$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get(sys.argv[2]))[:400])' "$R/api_$tag.json" "$k" 2> /dev/null)"
    done
    glog=$(ls -t "$HOME"/.local/state/glyd/logs/*.log 2> /dev/null | head -1); [ -n "$glog" ] && cp "$glog" "$R/serve_$tag.vllm.log"
    echo "$repo glyd serve: $(grep -h '^glyd serve' "$R/serve_$tag.serve.err" | head -3 | tr '\n' ' ') | GPU used $(gpu_used) MiB | $(vlog "$R/serve_$tag.vllm.log" "$R/serve_$tag.serve.err")" >> "$R/numbers.txt"
    stop
    rm -rf "$HOME/.cache/huggingface/hub/models--$(echo "$repo" | sed 's|/|--|')"  # (the disk for the next)
    return 0
}
one google/gemma-4-12B-it gemma4_12b
one nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 nemotron3_4b
one openbmb/MiniCPM5-2B minicpm5_2b
one Qwen/Qwen3.5-9B q35_9b

tb=$(grep -l "Traceback" "$R"/run_*.err "$R"/api_*.err 2> /dev/null | tr '\n' ' ')
check "no traceback in the user commands' output" "$([ -z "$tb" ]; echo $?)" "$tb"
{ echo "run-2026-families on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"
  for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; echo "--- versions"; cat "$R/versions.txt" 2> /dev/null; } > "$R/summary.txt"
cat "$R/summary.txt" | tee -a "$R/job.log"
