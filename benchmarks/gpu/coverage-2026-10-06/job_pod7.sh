#!/usr/bin/env bash
# v0.29.1 validation, fourth RTX 4090 pod (pod7; pods 4-6 = job_pod4.sh .. job_pod6.sh): glyd_gpu built from r291-validate 5a4be1f0 (private main
# + run-2026-families 1dab49a3 + coverage-2026 1612a0c0; release.yml run 37461453970's x86_64 wheel, versioned 0.29.0+r291), installed by v0.29.0's install.sh.
# coverage-2026's checks pod5 left open:
#   1. check_vllm --brief Qwen/Qwen3.5-9B (pod5: bf16 ran out of memory capturing CUDA graphs at max_num_seqs 256, so no check ran): every run with
#      CHECK_VLLM_LLM_KW {"language_model_only": true, "max_num_seqs": 16} (bf16 and Glyd alike; the vision tower is covered by Qwen3.5-0.8B's run).
#   2. check_vllm --towers ibm-granite/granite-vision-4.1-4b (the sources, GLYD_PACK_TOWERS=1; decides the towers' default only).
#   3. Qwen3.5-0.8B's layer products (pod5: 1.02e-02 against the 1e-2 bound): the branch, then glyd-gpu 0.29.0 (PyPI) with v0.29.0's check_vllm.py,
#      each with every layer over 6e-3 printed (WORSTLAYER lines in the glyd-auto log; a copy of the script, not committed).
# In ~: install.sh, glyd_gpu-*.whl, r291_src7.tar. Results in ~/results; DONE at the end.
set -uo pipefail
R0="$HOME/results"; R="$R0/runfam"; C="$R0/coverage"; W="$HOME/w"; mkdir -p "$R" "$C" "$W/tmp"
T0=$(date +%s); MAX_MIN="${MAX_MIN:-75}"; PASS=(); FAIL=()
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

step install 1500 sh "$HOME/install.sh"
check "install.sh (glyd 0.29.0, glyd-gpu from the branch's wheel): exit 0" "$RC" "see install.err"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir 2> /dev/null)/glyd"; PY="$TOOL/bin/python"
"$PY" -c 'import glyd_gpu, glyd_gpu.preflight as p, importlib.metadata as m; print(glyd_gpu.__file__, "PARSERS" in dir(p), "kv_fit" in dir(p), m.version("glyd-gpu"))' > "$R/which_glyd_gpu.txt" 2>&1
check "the installed glyd_gpu is the branch's (PARSERS, kv_fit, 0.29.0+r291)" "$(grep -q 'True True 0.29.0+r291' "$R/which_glyd_gpu.txt"; echo $?)" "$(cat "$R/which_glyd_gpu.txt")"
grep -q 'True True 0.29.0+r291' "$R/which_glyd_gpu.txt" || { log "not the branch's glyd_gpu: stopping here"; exit 1; }
"$UV" pip list --python "$PY" 2> /dev/null | grep -i -E "^(vllm|torch|glyd|glyd-gpu|transformers) " > "$C/versions.txt"

OUT=$C
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/r291_src7.tar" && cp "$W/src/COMMIT" "$C/COMMIT"
LIB=$("$PY" -c "import glyd_gpu.kernels as k; print(k.library())" 2> "$C/lib.err"); echo "$LIB" > "$C/lib.txt"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1  # (as glyd run and glyd serve start vLLM)
# the copies that print every layer over 6e-3 (item 3 only)
for f in gpu/vllm/check_vllm.py check_vllm_v029.py; do "$PY" - "$W/src/$f" "$W/src/${f%.py}_dbg.py" <<'EOF'
import sys
s = open(sys.argv[1]).read()
old = "            worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())\n"
new = ("            _e = ((y.float() - ref).abs().max() / ref.abs().max()).item(); worst = max(worst, _e)\n"
       "            if _e > 6e-3: print('WORSTLAYER', type(m).__name__, tuple(m.weight.shape), O, K, M, m.bias is not None, f'{_e:.3e}', flush=True)\n")
assert s.count(old) == 1, sys.argv[1]
open(sys.argv[2], "w").write(s.replace(old, new))
EOF
done
ls "$W/src/gpu/vllm/check_vllm_dbg.py" "$W/src/check_vllm_v029_dbg.py" > "$C/dbg.txt" 2>&1

# 1. Qwen3.5-9B
step check_9b 2400 env CHECK_VLLM_LLM_KW='{"language_model_only": true, "max_num_seqs": 16}' "$PY" "$W/src/gpu/vllm/check_vllm.py" --brief --out "$C/check_9b" Qwen/Qwen3.5-9B
check "check_vllm --brief Qwen3.5-9B: every check PASS" "$RC" "$(grep -h '^FAIL' "$C/check_9b.out" | head -5; tail -c 300 "$C/check_9b.err")"
grep -h -E "^(PASS|FAIL|COVERAGE)" "$C/check_9b.out" > "$C/check_9b_report.txt" 2> /dev/null

# 2. the towers
step towers 1500 env PYTHONPATH="$W/src/glyd-gpu-py" GLYD_GPU_LIB="$LIB" "$PY" "$W/src/gpu/vllm/check_vllm.py" --towers --out "$C/towers" ibm-granite/granite-vision-4.1-4b
check "check_vllm --towers granite-vision-4.1-4b: every check PASS" "$RC" "$(grep -h '^FAIL' "$C/towers.out" | head -3; tail -c 300 "$C/towers.err")"
grep -h -E "^(PASS|FAIL|TOWERS)" "$C/towers.out" > "$C/towers_report.txt" 2> /dev/null

# 3. Qwen3.5-0.8B's layer products: the branch, then 0.29.0
step l08_r291 900 "$PY" "$W/src/gpu/vllm/check_vllm_dbg.py" --brief --out "$C/l08_r291" Qwen/Qwen3.5-0.8B
grep -h -E "^(PASS|FAIL|COVERAGE)" "$C/l08_r291.out" > "$C/l08_r291_report.txt" 2> /dev/null
grep -h WORSTLAYER "$C"/l08_r291/*glyd-auto.log > "$C/l08_r291_worst.txt" 2> /dev/null
env -u UV_FIND_LINKS "$UV" pip install --python "$PY" --no-deps --reinstall "glyd-gpu===0.29.0" > "$C/swap1.txt" 2>&1; log "glyd-gpu 0.29.0 in: exit $?"
"$UV" pip list --python "$PY" 2> /dev/null | grep -i "^glyd-gpu " >> "$C/swap1.txt"
step l08_v029 900 "$PY" "$W/src/check_vllm_v029_dbg.py" --brief --out "$C/l08_v029" Qwen/Qwen3.5-0.8B
grep -h -E "^(PASS|FAIL|COVERAGE)" "$C/l08_v029.out" > "$C/l08_v029_report.txt" 2> /dev/null
grep -h WORSTLAYER "$C"/l08_v029/*glyd-auto.log > "$C/l08_v029_worst.txt" 2> /dev/null

{ echo "coverage-2026 (with run-2026-families, on private main), pod7, on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"
  for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- check_vllm --brief Qwen3.5-9B"; cat "$C/check_9b_report.txt" 2> /dev/null
  echo "--- towers"; cat "$C/towers_report.txt" 2> /dev/null
  echo "--- Qwen3.5-0.8B, the branch"; cat "$C/l08_r291_report.txt" 2> /dev/null; echo "worst layers:"; cat "$C/l08_r291_worst.txt" 2> /dev/null
  echo "--- Qwen3.5-0.8B, glyd-gpu 0.29.0"; cat "$C/l08_v029_report.txt" 2> /dev/null; echo "worst layers:"; cat "$C/l08_v029_worst.txt" 2> /dev/null
  echo "--- versions"; cat "$C/versions.txt" 2> /dev/null; } > "$C/summary.txt"
cat "$C/summary.txt" | tee -a "$R0/job.log"
