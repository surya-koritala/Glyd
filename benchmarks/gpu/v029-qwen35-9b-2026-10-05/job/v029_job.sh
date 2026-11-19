#!/usr/bin/env bash
# v0.29 pre-release GPU run (one pod, RTX 4090 or RTX 5090): the user flow of the wheels CI built from engine-v029, the engine's own checks
# (its API features, Open WebUI's requests, the loopback guard) on Qwen3-8B (the only family the engine runs: pass/fail only, nothing published),
# one user's speed of the engine against vLLM with Glyd (v0.28's path) and plain vLLM bf16 on that model (to decide the engine's GPU list),
# and the 2026 models through the release's real path: Qwen3.5-9B (glyd run, glyd serve, its numbers against plain vLLM bf16), Qwen3.5-0.8B (the fallback line).
# Uploaded beside it: install.sh, glyd_gpu-*.whl (x86_64), api_features_check.py, speed1.py. Results in ~/results (summary.txt last); DONE at the end.
set -uo pipefail
R="$HOME/results"; mkdir -p "$R"; T0=$(date +%s); MAX_MIN="${MAX_MIN:-85}"; PASS=(); FAIL=()
log() { echo "$(date -u +%T) ($(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
trap 'log "ending"; touch "$R/DONE"' EXIT
check() { if [ "$2" = 0 ]; then PASS+=("$1"); log "ok   $1"; else FAIL+=("$1"); log "FAIL $1${3:+ -- $3}"; fi; }
has() { grep -q -- "$1" "$2" 2> /dev/null; echo $?; }
lines() { local n; n=$(grep -c -- "$1" "$2" 2> /dev/null); echo "${n:-0}"; }
once() { [ "$(lines "$1" "$2")" = 1 ]; echo $?; }
answered() { [ "$RC" = 0 ] && [ -s "$1" ]; echo $?; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
gpu_idle() { sleep 4; local u; u=$(gpu_used); [ "${u:-99999}" -lt 700 ]; echo $?; }
step() { local name=$1 secs=$2; shift 2; local l; l=$(left)
    if [ "$l" -lt 60 ]; then log "skipped $name: $l s left"; RC=99; SECS=0; return 0; fi
    [ "$secs" -gt $((l - 20)) ] && secs=$((l - 20))
    log "$name (at most $secs s)"; local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$R/$name.out" 2> "$R/$name.err"; RC=$?; SECS=$((SECONDS - t))
    log "$name: exit $RC after $SECS s"; echo "$name $RC $SECS" >> "$R/steps.txt"; }
# a server in the background: NAME PORT CMD...; waits for /v1/models (at most 600 s); SRV is its pid (empty if it never answered)
serve() { local name=$1 port=$2; shift 2; log "serve $name on $port"
    "$@" > "$R/$name.serve.out" 2> "$R/$name.serve.err" < /dev/null & SRV=$!; local t=$SECONDS
    for _ in $(seq 1 300); do curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > "$R/$name.models.json" 2> /dev/null && break
        kill -0 "$SRV" 2> /dev/null || { log "$name exited before it answered"; SRV=""; return 0; }; sleep 2; done
    curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 || { log "$name never answered"; kill -KILL "$SRV" 2> /dev/null; SRV=""; return 0; }
    echo "$name up in $((SECONDS - t)) s, GPU memory used $(gpu_used) MiB" | tee -a "$R/times.txt"; }
stop() { [ -n "${SRV:-}" ] || return 0; kill -TERM "$SRV" 2> /dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
    kill -KILL "$SRV" 2> /dev/null; pkill -f "vllm.entrypoints" 2> /dev/null; pkill -f "glyd-engine" 2> /dev/null; sleep 4; SRV=""; }
vlog() { grep -h -o -E "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$R/$1.serve.err" "$R/$1.serve.out" 2> /dev/null | sort -u | tr '\n' ';'; }

{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# ---- install, as a user does it -------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p "$HOME/wheels"; mv "$HOME"/glyd_gpu-*.whl "$HOME/wheels/"; WHEEL=$(ls "$HOME"/wheels/glyd_gpu-*.whl | head -1); export UV_FIND_LINKS="$HOME/wheels"
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$HOME" > "$R/http.log" 2>&1 & HTTP=$!; sleep 2
step install 1500 sh -c 'curl -LsSf http://127.0.0.1:8765/install.sh | sh'; kill "$HTTP" 2> /dev/null
check "install.sh as curl | sh: exit 0" "$RC" "see install.err"; echo "install: $SECS s" >> "$R/times.txt"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir 2> /dev/null)/glyd"; PY="$TOOL/bin/python"
ENG_DIR=$("$PY" -c 'import glyd_gpu, os; print(os.path.dirname(glyd_gpu.__file__))' 2> /dev/null); ENG="$ENG_DIR/glyd-engine"
want=$(python3 -c 'import hashlib, sys, zipfile; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read("glyd_gpu/glyd-engine")).hexdigest())' "$WHEEL" 2> /dev/null)
check "the engine in the tool is CI's" "$([ -n "$want" ] && [ "$want" = "$(sha256sum "$ENG" 2> /dev/null | cut -d' ' -f1)" ]; echo $?)"
{ echo "tool env: $(du -sh "$TOOL" | cut -f1)"; echo "engine program: $(du -h "$ENG" | cut -f1)"; echo "glyd_gpu package: $(du -sh "$ENG_DIR" | cut -f1)"; } | tee -a "$R/sizes.txt"
"$PY" -m pip --version > /dev/null 2>&1 || "$PY" -m ensurepip > /dev/null 2>&1; "$PY" -m pip install -q openai > "$R/pip-openai.txt" 2>&1 || uv pip install --python "$PY" openai > "$R/pip-openai.txt" 2>&1
PROMPT="Say hello in five words."

# ---- Qwen3-8B (the engine's family; pass/fail only) --------------------------------------------------------------------------------------------------------------------------
step run8b_default 1800 glyd run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
check "glyd run Qwen3-8B (engine GPU list empty): exit 0 and an answer" "$(answered "$R/run8b_default.out")" "see run8b_default.err"
check "glyd run Qwen3-8B: one line says vLLM runs it" "$(( $(once '^glyd run: ' "$R/run8b_default.err") + $(has 'vLLM' "$R/run8b_default.err") ))" "$(grep '^glyd run:' "$R/run8b_default.err" | head -3)"
# the engine itself, by its own program (the hand-off keeps it for the GPUs it is measured on: none yet)
step run8b_engine 1800 "$ENG" run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
check "glyd-engine run Qwen3-8B: exit 0 and an answer" "$(answered "$R/run8b_engine.out")" "see run8b_engine.err"
check "GPU let go after glyd run" "$(gpu_idle)"

serve engine8b 8000 "$ENG" serve Qwen/Qwen3-8B --port 8000
check "glyd-engine serve Qwen3-8B: answers /v1/models" "$([ -n "${SRV:-}" ]; echo $?)" "see engine8b.serve.err"
if [ -n "${SRV:-}" ]; then
    step api_check 900 "$PY" "$HOME/api_features_check.py" http://127.0.0.1:8000 --out "$R/api.json"
    check "the engine's API checks (tools, logprobs, n, logit_bias, Open WebUI, loopback): exit 0" "$RC" "$(grep -c FAIL "$R/api_check.out") FAIL lines"
    curl -s -o /dev/null -w '%{http_code}' -H 'Host: evil.example' http://127.0.0.1:8000/v1/models > "$R/guard_host.txt"
    check "a foreign Host is 421" "$([ "$(cat "$R/guard_host.txt")" = 421 ]; echo $?)" "$(cat "$R/guard_host.txt")"
    python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",8000)); s.sendall(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Big: " + b"a"*(200*1024) + b"\r\n\r\n"); print(s.recv(64).decode(errors="replace").split("\r\n")[0])' > "$R/guard_big.txt" 2>&1
    check "an oversized header block is 431" "$(has ' 431' "$R/guard_big.txt")" "$(cat "$R/guard_big.txt")"
    step speed_engine8b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8000 Qwen/Qwen3-8B "$R/speed-engine-qwen3-8b.json"
    echo "engine idle GPU memory: $(gpu_used) MiB" >> "$R/sizes.txt"
    stop
fi
check "GPU let go after the engine's server" "$(gpu_idle)"
serve glydvllm8b 8001 glyd serve Qwen/Qwen3-8B --port 8001
if [ -n "${SRV:-}" ]; then
    check "glyd serve (default): vLLM runs it" "$(has 'vLLM' "$R/glydvllm8b.serve.err")" "$(grep '^glyd serve:' "$R/glydvllm8b.serve.err" | head -2)"
    step speed_glydvllm8b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8001 Qwen/Qwen3-8B "$R/speed-glydvllm-qwen3-8b.json"; stop
fi
serve vllm8b 8002 "$TOOL/bin/vllm" serve Qwen/Qwen3-8B --port 8002 --max-model-len 8192
[ -n "${SRV:-}" ] && { step speed_vllm8b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8002 Qwen/Qwen3-8B "$R/speed-vllm-qwen3-8b.json"; stop; }

# ---- 2026 models, through the release's real path -------------------------------------------------------------------------------------------------------------------------
step run_q35_9b 1800 glyd run Qwen/Qwen3.5-9B --prompt "$PROMPT" --no-think
check "glyd run Qwen/Qwen3.5-9B (2026): exit 0 and an answer" "$(answered "$R/run_q35_9b.out")" "see run_q35_9b.err"
check "glyd run Qwen3.5-9B: one line says vLLM runs it" "$(( $(once '^glyd run: ' "$R/run_q35_9b.err") + $(has 'vLLM' "$R/run_q35_9b.err") ))" "$(grep '^glyd run:' "$R/run_q35_9b.err" | head -3)"
serve glyd_q35_9b 8003 glyd serve Qwen/Qwen3.5-9B --port 8003
if [ -n "${SRV:-}" ]; then
    curl -s -m 180 http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"What is 2 plus 3? Answer with one word."}],"max_tokens":300}' > "$R/q35_chat.json"
    check "glyd serve Qwen3.5-9B: a chat completion answers" "$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; sys.exit(0 if (m.get("content") or m.get("reasoning_content") or "").strip() else 1)' "$R/q35_chat.json" 2> /dev/null; echo $?)" "$(head -c 300 "$R/q35_chat.json")"
    step speed_glyd_q35_9b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8003 Qwen/Qwen3.5-9B "$R/speed-glyd-qwen3.5-9b.json"
    echo "glyd serve Qwen3.5-9B: $(vlog glyd_q35_9b)" >> "$R/numbers.txt"; stop
fi
serve vllm_q35_9b 8004 "$TOOL/bin/vllm" serve Qwen/Qwen3.5-9B --port 8004 --max-model-len 4096
[ -n "${SRV:-}" ] && { step speed_vllm_q35_9b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8004 Qwen/Qwen3.5-9B "$R/speed-vllm-qwen3.5-9b.json"; echo "vllm bf16 Qwen3.5-9B: $(vlog vllm_q35_9b)" >> "$R/numbers.txt"; stop; }
GLYD_ENGINE=0 step off_q35_08b 900 glyd run Qwen/Qwen3.5-0.8B --prompt "$PROMPT" --no-think
check "GLYD_ENGINE=0 glyd run Qwen3.5-0.8B (2026): exit 0, an answer, the GLYD_ENGINE=0 line" "$(( $(answered "$R/off_q35_08b.out") + $(has 'GLYD_ENGINE=0' "$R/off_q35_08b.err") ))" "$(grep '^glyd run:' "$R/off_q35_08b.err" | head -2)"

# ---- the whole ------------------------------------------------------------------------------------------------------------------------------------------------------------
tb=$(grep -l "Traceback" "$R"/*.err "$R"/*.out 2> /dev/null | grep -v -E "api_check|serve\.(out|err)" | tr '\n' ' ')
check "no traceback in the user commands' output" "$([ -z "$tb" ]; echo $?)" "$tb"
check "the GPU is idle at the end" "$(gpu_idle)"
python3 - "$R" > "$R/speed.txt" 2>&1 <<'EOF'
import json, os, sys
R = sys.argv[1]
for f in sorted(os.listdir(R)):
    if f.startswith("speed-") and f.endswith(".json"):
        d = json.load(open(os.path.join(R, f))); print(f"{f[6:-5]:28} TTFT {d['ttft_ms_median']:8.1f} ms  decode {d['decode_tok_s_median']:7.2f} tok/s  rounds {[r['decode_tok_s'] for r in d['rounds']]}")
EOF
{ echo "v0.29 pre-release run on $GPU"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"; for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- one user's speed (Qwen3-8B: engine vs vLLM+Glyd vs vLLM bf16; Qwen3.5-9B: glyd vs vLLM bf16)"; cat "$R/speed.txt"
  echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- sizes"; cat "$R/sizes.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; } > "$R/summary.txt"
cat "$R/summary.txt" | tee -a "$R/job.log"
