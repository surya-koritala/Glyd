#!/usr/bin/env bash
# v0.29.1 validation, third RTX 4090 pod (pod6; pod4 = job_pod4.sh, pod5 = job_pod5.sh): glyd_gpu built from private main + run-2026-families
# (a Mamba2 model served at glyd run's memory share) + coverage-2026 (release.yml's x86_64 wheel, versioned 0.29.0+r291 here so that the installer
# takes it), installed by v0.29.0's install.sh. run-2026-families' check for NVIDIA-Nemotron-3-Nano-4B-BF16 only (pods 4 and 5: glyd serve out of
# memory at 92% and 88%; the change touches the memory share of models with Mamba2 layers alone): glyd run --prompt, one start, glyd serve and
# api_check.py's four. In ~: install.sh, api_check.py, glyd_gpu-*.whl. Results in ~/results/runfam; DONE at the end.
set -uo pipefail
R0="$HOME/results"; R="$R0/runfam"; C="$R0/coverage"; W="$HOME/w"; mkdir -p "$R" "$C" "$W/tmp"
T0=$(date +%s); MAX_MIN="${MAX_MIN:-25}"; PASS=(); FAIL=()
log() { echo "$(date -u +%T) ($(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R0/job.log"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
trap 'log "ending"; pkill -f "vllm serve|EngineCore|vllm.entrypoints" 2> /dev/null; touch "$R0/DONE"' EXIT
check() { if [ "$2" = 0 ]; then PASS+=("$1"); log "ok   $1"; else FAIL+=("$1"); log "FAIL $1${3:+ -- $3}"; fi; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
step() { local name=$1 secs=$2; shift 2; local l; l=$(left)
    if [ "$l" -lt 60 ]; then log "skipped $name: $l s left"; RC=99; return 0; fi
    [ "$secs" -gt $((l - 20)) ] && secs=$((l - 20))
    log "$name (at most $secs s)"; local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"; RC=$?
    log "$name: exit $RC after $((SECONDS - t)) s"; echo "$name $RC $((SECONDS - t))" >> "$OUT/steps.txt"; }
serve() { local name=$1 port=$2; shift 2; log "serve $name on $port"
    "$@" > "$OUT/$name.serve.out" 2> "$OUT/$name.serve.err" < /dev/null & SRV=$!; local t=$SECONDS
    for _ in $(seq 1 450); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > "$OUT/$name.models.json" 2> /dev/null && break
        kill -0 "$SRV" 2> /dev/null || { log "$name exited before it answered"; SRV=""; return 0; }; sleep 2; done
    curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 || { log "$name never answered"; kill -KILL "$SRV" 2> /dev/null; SRV=""; return 0; }
    echo "$name up in $((SECONDS - t)) s, GPU memory used $(gpu_used) MiB" | tee -a "$OUT/times.txt"; }
stop() { [ -n "${SRV:-}" ] || return 0; kill -TERM "$SRV" 2> /dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
    kill -KILL "$SRV" 2> /dev/null; pkill -f "vllm.entrypoints|vllm serve|EngineCore" 2> /dev/null; sleep 5; SRV=""; }
vlog() { grep -h -o -E "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|--max-model-len [0-9]+|--tool-call-parser [a-z0-9_]+|--reasoning-parser [a-z0-9_]+|--gpu-memory-utilization [0-9.]+|CUDA out of memory" "$@" 2> /dev/null | sort -u | tr '\n' ';'; }

OUT=$R
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R0/machine.txt" 2>&1
cat "$R0/machine.txt" | tee -a "$R0/job.log"; cp "$R0/machine.txt" "$R/machine.txt"; cp "$R0/machine.txt" "$C/machine.txt"

# ---- the wheel, versioned 0.29.0+r291 (glyd 0.29.0 pins glyd-gpu==0.29.0, which 0.29.0+r291 meets; PyPI's 0.29.0 would win a tie) -------------
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
mkdir -p "$HOME/wheels" "$W/unp"; WH=$(ls "$HOME"/glyd_gpu-0.29.0-*.whl | head -1)
sha256sum "$WH" > "$R/wheel-built.sha256"
"$UV" tool run --from wheel==0.48.0 wheel unpack "$WH" -d "$W/unp" > "$R/reversion.txt" 2>&1
D="$W/unp/glyd_gpu-0.29.0"; mv "$D/glyd_gpu-0.29.0.dist-info" "$D/glyd_gpu-0.29.0+r291.dist-info" && sed -i 's/^Version: 0\.29\.0$/Version: 0.29.0+r291/' "$D/glyd_gpu-0.29.0+r291.dist-info/METADATA"
"$UV" tool run --from wheel==0.48.0 wheel pack "$D" -d "$HOME/wheels" >> "$R/reversion.txt" 2>&1
ls "$HOME/wheels" >> "$R/reversion.txt"; export UV_FIND_LINKS="$HOME/wheels"
sha256sum "$HOME"/wheels/*.whl > "$R/wheel.sha256"
WHL=$(ls "$HOME"/wheels/glyd_gpu-0.29.0+r291-*.whl 2> /dev/null | head -1)
[ -n "$WHL" ] || { log "the wheel was not re-versioned (reversion.txt)"; exit 1; }

# ==== A. run-2026-families ======================================================================================================================
step install 1500 sh "$HOME/install.sh"
check "install.sh (glyd 0.29.0, glyd-gpu from the branch's wheel): exit 0" "$RC" "see install.err"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir 2> /dev/null)/glyd"; PY="$TOOL/bin/python"
"$PY" -c 'import glyd_gpu, glyd_gpu.preflight as p, importlib.metadata as m; print(glyd_gpu.__file__, "PARSERS" in dir(p), "kv_fit" in dir(p), m.version("glyd-gpu"))' > "$R/which_glyd_gpu.txt" 2>&1
check "the installed glyd_gpu is the branch's (PARSERS, kv_fit, 0.29.0+r291)" "$(grep -q 'True True 0.29.0+r291' "$R/which_glyd_gpu.txt"; echo $?)" "$(cat "$R/which_glyd_gpu.txt")"
grep -q 'True True 0.29.0+r291' "$R/which_glyd_gpu.txt" || { log "not the branch's glyd_gpu: stopping here"; exit 1; }
"$UV" pip list --python "$PY" 2> /dev/null | grep -i -E "^(vllm|torch|glyd|glyd-gpu|transformers) " > "$R/versions.txt"; cp "$R/versions.txt" "$C/versions.txt"
step doctor 300 glyd doctor
check "glyd doctor: exit 0" "$RC" "$(tail -3 "$R/doctor.out")"

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
    return 0
}
one nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 nemotron3_4b

tb=$(grep -l "Traceback" "$R"/run_*.err "$R"/api_*.err 2> /dev/null | tr '\n' ' ')
check "no traceback in the user commands' output" "$([ -z "$tb" ]; echo $?)" "$tb"
{ echo "run-2026-families (with coverage-2026, on private main) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"
  for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; echo "--- versions"; cat "$R/versions.txt" 2> /dev/null; } > "$R/summary.txt"
cat "$R/summary.txt" | tee -a "$R0/job.log"

