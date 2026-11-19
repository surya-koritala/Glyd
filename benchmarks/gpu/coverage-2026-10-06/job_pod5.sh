#!/usr/bin/env bash
# v0.29.1 validation, second RTX 4090 pod (pod5; pod4 = job_pod4.sh): glyd_gpu built from private main + run-2026-families (with the Mamba2
# sizing fix) + coverage-2026 (release.yml's x86_64 wheel, versioned 0.29.0+r291 here so that the installer takes it), installed by v0.29.0's install.sh.
#   A. run-2026-families' check, for NVIDIA-Nemotron-3-Nano-4B-BF16 only (pod4's one failure: glyd serve out of memory; the fix changes
#      the memory settings of models with Mamba2 layers alone): glyd run --prompt, one start, glyd serve and api_check.py's four. -> results/runfam
#   B as pod4's, with the environment glyd run and glyd serve give vLLM (VLLM_USE_FLASHINFER_SAMPLER=0: pod4's part B stopped on FlashInfer's
#      sampler compiling without ninja).
#   B. coverage-2026's check: test_vllm.py (GLYD_REQUIRE_VLLM=1; the installed wheel, then the sources on the wheel's library); check_vllm.py
#      --brief on Qwen3.5-0.8B, LFM2.5-350M, Nemotron-3-Nano-4B, AI21-Jamba2-3B, Qwen3.5-9B; Qwen3.5-9B's speed at 1 and 32 users against
#      glyd-gpu 0.29.0 (PyPI), this wheel / 0.29.0 / this wheel again; the towers on granite-vision-4.1-4b.              -> results/coverage
# In ~: install.sh, api_check.py, glyd_gpu-*.whl, r291_src.tar. Results in ~/results; DONE at the end.
set -uo pipefail
R0="$HOME/results"; R="$R0/runfam"; C="$R0/coverage"; W="$HOME/w"; mkdir -p "$R" "$C" "$W/tmp"
T0=$(date +%s); MAX_MIN="${MAX_MIN:-92}"; PASS=(); FAIL=()
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

# ==== B. coverage-2026 ==========================================================================================================================
OUT=$C; PASS=(); FAIL=()
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/r291_src.tar" && cp "$W/src/COMMIT" "$C/COMMIT"
LIB=$("$PY" -c "import glyd_gpu.kernels as k; print(k.library())" 2> "$C/lib.err"); echo "$LIB" > "$C/lib.txt"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1  # (as glyd run and glyd serve start vLLM)
VLLM="$TOOL/bin/vllm"

# 1. the plugin's tests: the installed (compiled) wheel, then the sources (a development build) on the wheel's library
cd "$W/src/glyd-gpu-py/tests"
step tests_wheel 900 env GLYD_REQUIRE_VLLM=1 "$PY" test_vllm.py
check "test_vllm.py, the installed wheel: exit 0" "$RC" "$(tail -c 400 "$C/tests_wheel.err")"
step tests_src 900 env GLYD_REQUIRE_VLLM=1 PYTHONPATH="$W/src/glyd-gpu-py" GLYD_GPU_LIB="$LIB" "$PY" test_vllm.py
check "test_vllm.py, the sources on the wheel's library: exit 0" "$RC" "$(tail -c 400 "$C/tests_src.err")"
grep -h -E "skipped" "$C/tests_wheel.out" "$C/tests_src.out" > "$C/tests_skipped.txt" 2> /dev/null
for t in test_moe_activations test_padded_rows test_after_load test_embedding_ops test_scratch_rule; do
    check "$t: its GPU part ran (sources run)" "$(grep -q -E "^$t(:| ).*skipped" "$C/tests_src.out"; [ $? = 1 ] && grep -q "^$t ok" "$C/tests_src.out"; echo $?)" "$(grep -h "^$t" "$C/tests_src.out" | head -2)"
done
cd "$HOME"

# 2. check_vllm --brief (the installed wheel)
step check_brief 2700 "$PY" "$W/src/gpu/vllm/check_vllm.py" --brief --out "$C/check_brief" Qwen/Qwen3.5-0.8B LiquidAI/LFM2.5-350M nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 ai21labs/AI21-Jamba2-3B Qwen/Qwen3.5-9B
check "check_vllm --brief: every check PASS" "$RC" "$(grep -h '^FAIL' "$C/check_brief.out" | head -5)"
grep -h -E "^(PASS|FAIL|COVERAGE)" "$C/check_brief.out" > "$C/check_brief_report.txt" 2> /dev/null

# 3. Qwen3.5-9B's speed at 1 and 32 users: this wheel, glyd-gpu 0.29.0 (PyPI), this wheel again
speed() { local name=$1 port=8300 d="$C/speed-$1"; mkdir -p "$d"
    [ "$(left)" -lt 500 ] && { log "skipped speed $name: $(left) s left"; return; }
    "$PY" -c 'import importlib.metadata as m, glyd_gpu; print(m.version("glyd-gpu"), glyd_gpu.__file__, "PARSERS" in dir(__import__("glyd_gpu.preflight").preflight))' > "$d/glyd_gpu_version.txt" 2>&1
    "$UV" pip list --python "$PY" 2> /dev/null | grep -i "^glyd-gpu " >> "$d/glyd_gpu_version.txt"
    OUT=$d serve vllm "$port" "$VLLM" serve Qwen/Qwen3.5-9B --quantization glyd --max-model-len 4096 --port "$port"
    [ -n "${SRV:-}" ] || { log "speed $name: the server did not start"; return; }
    vlog "$d/vllm.serve.out" "$d/vllm.serve.err" > "$d/vllm.txt"
    local i=0
    for u in warm 1 32; do local n=$([ "$u" = warm ] && echo 8 || ([ "$u" = 1 ] && echo 8 || echo 128)); local c=$([ "$u" = warm ] && echo 8 || echo "$u")
        timeout 900 "$VLLM" bench serve --backend vllm --model Qwen/Qwen3.5-9B --port "$port" --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
            --max-concurrency "$c" --num-prompts "$n" --seed $((100 + i)) --save-result --result-dir "$d" --result-filename "u$u.json" --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 > "$d/bench-$u.txt" 2>&1
        log "speed $name at $u: exit $?"; i=$((i + 1))
    done
    stop
}
speed r291-a
env -u UV_FIND_LINKS "$UV" pip install --python "$PY" --no-deps --reinstall "glyd-gpu===0.29.0" > "$C/swap1.txt" 2>&1; log "glyd-gpu 0.29.0 in: exit $?"
speed v029
"$UV" pip install --python "$PY" --no-deps --reinstall "$WHL" > "$C/swap2.txt" 2>&1; log "the branch's wheel back: exit $?"
speed r291-b
"$PY" - "$C" > "$C/speed.txt" 2>&1 <<'EOF'
import json, os, sys
C = sys.argv[1]
print("| glyd-gpu | users | output tok/s | TTFT median ms | TPOT median ms | weights (vLLM) |")
print("| :-- | --: | --: | --: | --: | --: |")
for name in ("r291-a", "v029", "r291-b"):
    d = os.path.join(C, f"speed-{name}")
    v = open(os.path.join(d, "glyd_gpu_version.txt")).read().split()[0] if os.path.exists(os.path.join(d, "glyd_gpu_version.txt")) else "?"
    w = open(os.path.join(d, "vllm.txt")).read().strip() if os.path.exists(os.path.join(d, "vllm.txt")) else ""
    for u in ("1", "32"):
        p = os.path.join(d, f"u{u}.json")
        if os.path.exists(p):
            j = json.load(open(p))
            print(f"| {name} ({v}) | {u} | {j['output_throughput']:.1f} | {j['median_ttft_ms']:.1f} | {j['median_tpot_ms']:.2f} | {w} |")
EOF
cat "$C/speed.txt" | tee -a "$R0/job.log"

# 4. the towers (decides their default only): the sources, GLYD_PACK_TOWERS=1 (a development build's variable)
step towers 1500 env PYTHONPATH="$W/src/glyd-gpu-py" GLYD_GPU_LIB="$LIB" "$PY" "$W/src/gpu/vllm/check_vllm.py" --towers --out "$C/towers" ibm-granite/granite-vision-4.1-4b
check "check_vllm --towers granite-vision-4.1-4b: every check PASS" "$RC" "$(grep -h '^FAIL' "$C/towers.out" | head -3)"
grep -h -E "^(PASS|FAIL|TOWERS)" "$C/towers.out" > "$C/towers_report.txt" 2> /dev/null

{ echo "coverage-2026 (with run-2026-families, on private main) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"
  for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- check_vllm --brief"; cat "$C/check_brief_report.txt" 2> /dev/null; echo "--- speed"; cat "$C/speed.txt" 2> /dev/null
  echo "--- towers"; cat "$C/towers_report.txt" 2> /dev/null; echo "--- versions"; cat "$C/versions.txt" 2> /dev/null; } > "$C/summary.txt"
cat "$C/summary.txt" | tee -a "$R0/job.log"
