#!/usr/bin/env bash
# E1 (glyd-engine as what `glyd run` and `glyd serve` start): the user flow, on one RTX 4090 pod, of the glyd-gpu wheel that release.yml built from engine-v029.
#
# Uploaded beside this script (the pod's home): install.sh (the branch's scripts/install.sh), glyd_gpu-*.whl (x86_64, from the CI run), glyd (the Rust glyd, x86_64: its hand-off),
# serve_smoke.py (the engine's API check). Everything a user does is run as a user does it: `curl -LsSf http://127.0.0.1:PORT/install.sh | sh` (the script served from here, not from
# getglyd.com), then glyd run / glyd serve. UV_FIND_LINKS makes uv take the wheel from ~/wheels where it asks for glyd-gpu 0.28.0 (the same version PyPI has).
# Results in ~/results: summary.txt (made last), job.log, one .out/.err a step; DONE when it ends. MAX_MIN (default 44): a step that would not end by then is skipped.
set -uo pipefail
R="$HOME/results"
mkdir -p "$R"
T0=$(date +%s)
MAX_MIN="${MAX_MIN:-44}"
PASS=()
FAIL=()
log() { echo "$(date -u +%T) ($(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
trap 'log "ending"; touch "$R/DONE"' EXIT
check() { # NAME, STATUS (0: it holds), DETAIL
    if [ "$2" = 0 ]; then PASS+=("$1"); log "ok   $1"; else FAIL+=("$1"); log "FAIL $1${3:+ -- $3}"; fi
}
has() { grep -q -- "$1" "$2" 2> /dev/null; echo $?; }   # grep's status for the file: 0 where it says it
lines() { local n; n=$(grep -c -- "$1" "$2" 2> /dev/null); echo "${n:-0}"; }   # how many lines of the file say it
answered() { [ "$RC" = 0 ] && [ -s "$1" ]; echo $?; }   # the last step ended well and wrote to the file
once() { [ "$(lines "$1" "$2")" = 1 ]; echo $?; }   # exactly one line of the file says it
gpu_idle() { sleep 4; local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '); [ "${used:-99999}" -lt 700 ]; echo $?; }
# a step: NAME, seconds it may take, then the command (its stdout in NAME.out, its stderr in NAME.err); the exit status in $RC, the seconds it took in $SECS
step() {
    local name=$1 secs=$2
    shift 2
    local l
    l=$(left)
    if [ "$l" -lt 60 ]; then log "skipped $name: $l s left"; RC=99; SECS=0; return 0; fi
    [ "$secs" -gt $((l - 20)) ] && secs=$((l - 20))
    log "$name (at most $secs s)"
    local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$R/$name.out" 2> "$R/$name.err"
    RC=$?
    SECS=$((SECONDS - t))
    log "$name: exit $RC after $SECS s"
    echo "$name $RC $SECS" >> "$R/steps.txt"
}

{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nvidia-smi | sed -n 3,4p; uname -a; grep PRETTY /etc/os-release; python3 --version; gcc --version | head -1; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"
case "$(head -2 "$R/machine.txt" | tail -1)" in *4090*) ;; *) log "NOTE: this is not an RTX 4090" ;; esac

# ---- the install, as the README's two commands have it (the script from here) -------------------------------------------------------------------------------------------------
mkdir -p "$HOME/wheels"
mv "$HOME"/glyd_gpu-*.whl "$HOME/wheels/"
ls -l "$HOME/wheels" | tee -a "$R/job.log"
WHEEL=$(ls "$HOME"/wheels/glyd_gpu-*.whl | head -1)
export UV_FIND_LINKS="$HOME/wheels"
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$HOME" > "$R/http.log" 2>&1 &
HTTP=$!
sleep 2
step install 1500 sh -c 'curl -LsSf http://127.0.0.1:8765/install.sh | sh'
kill "$HTTP" 2> /dev/null
check "install.sh from the branch, as curl | sh: exit 0" "$RC" "see install.err"
echo "install: $SECS s" >> "$R/times.txt"
check "install.sh: no traceback in its output" "$(( $(lines Traceback "$R/install.err") + $(lines Traceback "$R/install.out") > 0 ))"
export PATH="$HOME/.local/bin:$PATH"
TOOL="$(uv tool dir 2> /dev/null)/glyd"
ENG_DIR=$("$TOOL/bin/python" -c 'import glyd_gpu, os; print(os.path.dirname(glyd_gpu.__file__))' 2> /dev/null)
ENG="$ENG_DIR/glyd-engine"
{ uv tool list; "$TOOL/bin/python" -c 'import importlib.metadata as m; print("glyd", m.version("glyd"), "glyd-gpu", m.version("glyd-gpu"))'; ls -l "$ENG_DIR" | head -40; } > "$R/tool.txt" 2>&1
check "the tool's glyd-gpu has the engine's program, executable" "$([ -x "$ENG" ]; echo $?)" "$ENG_DIR"
want=$(python3 -c 'import hashlib, sys, zipfile; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read("glyd_gpu/glyd-engine")).hexdigest())' "$WHEEL" 2> /dev/null)
got=$(sha256sum "$ENG" 2> /dev/null | cut -d' ' -f1)
check "the engine in the tool is the one in CI's wheel (sha256 $want)" "$([ -n "$want" ] && [ "$want" = "$got" ]; echo $?)" "$got"
step engine_version 20 "$ENG" --version
check "glyd-engine --version" "$(has 'glyd-engine 0.28.0' "$R/engine_version.out")"
step engine_doctor 120 "$ENG" doctor
step glyd_doctor 300 glyd doctor
check "glyd doctor lists the engine, found in the tool's package" "$(has "Glyd engine .*$ENG" "$R/glyd_doctor.out")" "$(grep 'Glyd engine' "$R/glyd_doctor.out" | head -2)"
export PATH="$HOME/.local/bin:$PATH"

# ---- glyd run Qwen/Qwen3-8B: the engine, from an empty machine (download, packing, the KV tables, the load) ------------------------------------------------------------------
PROMPT="Say hello in five words."
step run8b 1800 glyd run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
check "glyd run Qwen/Qwen3-8B --prompt: exit 0" "$RC" "see run8b.err"
echo "first run of Qwen3-8B from an empty machine, to the answer: $SECS s" >> "$R/times.txt"
check "glyd run: one line says glyd-engine runs it" "$(once '^glyd run: ' "$R/run8b.err")" "$(grep '^glyd run:' "$R/run8b.err" | head -3)"
check "glyd run: that line is glyd-engine's" "$(has '^glyd run: Qwen/Qwen3-8B on glyd-engine' "$R/run8b.err")"
check "glyd run: an answer on stdout" "$([ -s "$R/run8b.out" ]; echo $?)" "$(head -c 200 "$R/run8b.out")"
check "glyd run: the GPU is let go" "$(gpu_idle)"

# ---- glyd serve + a request -------------------------------------------------------------------------------------------------------------------------------------------------
if [ "$(left)" -gt 120 ]; then
    glyd serve Qwen/Qwen3-8B --port 8000 > "$R/serve.out" 2> "$R/serve.err" < /dev/null &
    SERVE=$!
    t=$SECONDS
    for _ in $(seq 1 240); do
        curl -sf -m 3 http://127.0.0.1:8000/v1/models > "$R/models.json" 2> /dev/null && break
        kill -0 "$SERVE" 2> /dev/null || break
        sleep 2
    done
    echo "glyd serve, to /v1/models answering: $((SECONDS - t)) s" >> "$R/times.txt"
    check "glyd serve: /v1/models answers" "$(has 'Qwen' "$R/models.json")" "$(head -c 200 "$R/models.json" 2> /dev/null)"
    check "glyd serve: one line says glyd-engine runs it" "$(once '^glyd serve: ' "$R/serve.err")" "$(grep '^glyd serve:' "$R/serve.err" | head -3)"
check "glyd serve: that line is glyd-engine's" "$(has '^glyd serve: Qwen/Qwen3-8B on glyd-engine' "$R/serve.err")"
    curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"What is 2 plus 3? Answer with one word."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}' > "$R/chat.json" 2> "$R/chat.err"
    check "a chat completion over the API has an answer" "$(python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["choices"][0]["message"]["content"].strip() else 1)' "$R/chat.json" 2> /dev/null; echo $?)" "$(head -c 300 "$R/chat.json")"
    curl -s -m 10 http://127.0.0.1:8000/ > "$R/page.html" 2> /dev/null
    check "the chat page is served at /" "$(has '<html' "$R/page.html")"
    step serve_smoke 600 python3 "$HOME/serve_smoke.py" http://127.0.0.1:8000
    check "the API's own check (serve_smoke.py): exit 0" "$RC" "$(grep -c FAIL "$R/serve_smoke.out") lines FAIL"
    kill -TERM "$SERVE" 2> /dev/null
    for _ in $(seq 1 40); do kill -0 "$SERVE" 2> /dev/null || break; sleep 1; done
    kill -0 "$SERVE" 2> /dev/null && { kill -KILL "$SERVE" 2> /dev/null; check "glyd serve stops on SIGTERM" 1; } || check "glyd serve stops on SIGTERM" 0
    check "glyd serve: the GPU is let go" "$(gpu_idle)"
fi

# ---- a model the engine does not run: vLLM, said in one line ----------------------------------------------------------------------------------------------------------------
FB="Qwen/Qwen2.5-1.5B-Instruct"
step fallback 900 glyd run "$FB" --prompt "$PROMPT"
check "glyd run of a model the engine does not run: exit 0" "$RC" "see fallback.err"
check "glyd run: one line says why vLLM runs it" "$(once '^glyd run: ' "$R/fallback.err")" "$(grep '^glyd run:' "$R/fallback.err" | head -3)"
check "glyd run: that line is the engine's reason" "$(has '^glyd run: Qwen2.5-1.5B-Instruct is a Qwen2ForCausalLM model, and this engine runs Qwen3 models so far; using vLLM\.$' "$R/fallback.err")"
check "glyd run via vLLM: an answer on stdout" "$([ -s "$R/fallback.out" ]; echo $?)" "$(head -c 200 "$R/fallback.out")"
check "glyd run via vLLM: the GPU is let go" "$(gpu_idle)"

# ---- the second start of the engine, warm ------------------------------------------------------------------------------------------------------------------------------------
step run8b_warm 300 glyd run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
check "glyd run Qwen/Qwen3-8B again (warm): exit 0 and an answer" "$(answered "$R/run8b_warm.out")"
echo "second start of Qwen3-8B, to the answer: $SECS s" >> "$R/times.txt"

# ---- the Rust glyd first on PATH (the compression program: it passes run and serve on, or starts the engine itself) -------------------------------------------------------------
mkdir -p "$HOME/rust"
if [ -f "$HOME/glyd" ]; then
    cp "$HOME/glyd" "$HOME/rust/glyd"
    chmod +x "$HOME/rust/glyd"
    export PATH="$HOME/rust:$PATH"
    step rust_version 20 glyd --version
    check "the Rust glyd runs here" "$RC" "$(head -2 "$R/rust_version.out")"
    # 1. no engine of its own to find: it passes the command to the Python tool, which starts the engine of its package
    step rust_via_tool 300 glyd run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
    check "Rust glyd -> Python tool -> engine: exit 0 and an answer" "$(answered "$R/rust_via_tool.out")"
    check "Rust glyd -> Python tool -> engine: one line, glyd-engine's" "$(( $(once '^glyd run: ' "$R/rust_via_tool.err") + $(has '^glyd run: Qwen/Qwen3-8B on glyd-engine' "$R/rust_via_tool.err") ))" "$(grep '^glyd run:' "$R/rust_via_tool.err" | head -3)"
    # 2. an engine it finds (GLYD_ENGINE=PATH): it asks the engine and starts it, with its own line
    export GLYD_ENGINE="$ENG"
    step rust_direct 300 glyd run Qwen/Qwen3-8B --prompt "$PROMPT" --no-think
    check "Rust glyd starts the engine itself (GLYD_ENGINE=program): exit 0 and an answer" "$(answered "$R/rust_direct.out")"
    check "Rust glyd starts the engine itself: one line, glyd-engine's" "$(( $(once '^glyd run: ' "$R/rust_direct.err") + $(has '^glyd run: Qwen/Qwen3-8B on glyd-engine' "$R/rust_direct.err") ))" "$(grep '^glyd run:' "$R/rust_direct.err" | head -3)"
    # 3. an engine it finds that does not run the model: its line, then the Python tool's vLLM path (told GLYD_ENGINE=0: no second line); stopped once it is seen to go on
    step rust_unsupported 45 glyd run "$FB" --prompt "$PROMPT"
    check "Rust glyd, an unsupported model: one line, then the Python tool goes on with vLLM" "$(( $(once '^glyd run: ' "$R/rust_unsupported.err") + $(has 'using vLLM' "$R/rust_unsupported.err") + $(has 'Qwen2ForCausalLM' "$R/rust_unsupported.err") ))" "$(grep '^glyd run:' "$R/rust_unsupported.err" | head -4)"
    unset GLYD_ENGINE
    # 4. GLYD_ENGINE=0
    GLYD_ENGINE=0 step rust_off 45 glyd run Qwen/Qwen3-0.6B --prompt "$PROMPT"
    check "Rust glyd, GLYD_ENGINE=0: one line, then vLLM" "$(( $(once '^glyd run: ' "$R/rust_off.err") + $(has 'GLYD_ENGINE=0; using vLLM' "$R/rust_off.err") ))" "$(grep '^glyd run:' "$R/rust_off.err" | head -3)"
    # and the Python tool's own GLYD_ENGINE=0 (the Rust glyd out of the way)
    export PATH="${PATH#"$HOME"/rust:}"
    GLYD_ENGINE=0 step py_off 45 glyd run Qwen/Qwen3-0.6B --prompt "$PROMPT"
    check "Python tool, GLYD_ENGINE=0: one line, then vLLM" "$(( $(once '^glyd run: ' "$R/py_off.err") + $(has 'GLYD_ENGINE=0; using vLLM' "$R/py_off.err") ))" "$(grep '^glyd run:' "$R/py_off.err" | head -3)"
    pkill -f "vllm.entrypoints" 2> /dev/null
    sleep 3
else
    log "no Rust glyd uploaded: its steps are skipped"
fi

# ---- the whole: no traceback in any log, nothing holds the GPU ------------------------------------------------------------------------------------------------------------------
tb=$(grep -l "Traceback" "$R"/*.err "$R"/*.out 2> /dev/null | tr '\n' ' ')
check "no traceback in any step's output" "$([ -z "$tb" ]; echo $?)" "$tb"
oom=$(grep -il "out of memory\|OutOfMemory" "$R"/*.err "$R"/*.out 2> /dev/null | tr '\n' ' ')
check "no out-of-memory in any step's output" "$([ -z "$oom" ]; echo $?)" "$oom"
check "the GPU is idle at the end" "$(gpu_idle)"
mkdir -p "$R/glyd-logs"
cp -r "$HOME/.local/state/glyd/logs/." "$R/glyd-logs/" 2> /dev/null
ls -la "$HOME/.cache/glyd" "$HOME/.cache/glyd/engine" > "$R/cache.txt" 2>&1
nvidia-smi > "$R/nvidia-smi-end.txt" 2>&1
{
    echo "E1 user flow on $(sed -n 2p "$R/machine.txt")"
    echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"
    for f in "${FAIL[@]}"; do echo "FAIL $f"; done
    for p in "${PASS[@]}"; do echo "ok   $p"; done
    echo "--- times"
    cat "$R/times.txt" 2> /dev/null
} > "$R/summary.txt"
cat "$R/summary.txt" | tee -a "$R/job.log"
