#!/usr/bin/env bash
# The M1 gate (ROADMAP.md, "M1 Front door"), run on one clean machine with only the NVIDIA driver: install, then for each pinned model of m1-models.txt
# the planner's answer, the first token, the printed plan against what loaded, and JSON-schema output; the checks other sessions measured are read from --ext.
# One line a step: PASS, FAIL or SKIP, the step, what it saw. Exit 0 only where nothing FAILs and every required step has a result. No Python is used here
# (jq is), and python and pip are shimmed during the run: a call to either is a FAIL.
#
#   benchmarks/gates/m1-gate.sh [--tier 8gb|16gb|24gb] [--models FILE] [--glyd PATH] [--install-cmd CMD | --skip-install] [--out DIR] [--ext DIR] [--limit SECONDS]
#   benchmarks/gates/m1-gate.sh --plan-only --machine PRESET [--glyd PATH]      the planner's answer for each model on a described machine (glyd fit --machine), no GPU needed:
#                                                                                 what m1-models.txt's expectations were made from, and a check they still hold
# --ext DIR: files NAME.result, each a line "PASS|FAIL|SKIP<TAB>NAME<TAB>detail": P6's driver runs (driver-r535, driver-r550, driver-r570), P2a's no-OOM check (p2a-no-oom), R5a's
# schema conformance on real weights (r5a-schema-real-weights), R8's reference results (r8-reference-gpu-results) and P17's client matrix (p17-open-webui, p17-continue, p17-openai-sdk).
set -u

TIER=""; MODELS="$(dirname "$0")/m1-models.txt"; GLYD="${GLYD:-glyd}"; INSTALL_CMD='curl -LsSf https://getglyd.com/install.sh | sh'; SKIP_INSTALL=0
OUT=""; EXT=""; LIMIT=120; PLAN_ONLY=0; MACHINE=""; PORT=18734
REQUIRED_EXT="driver-r535 driver-r550 driver-r570 p2a-no-oom r5a-schema-real-weights r8-reference-gpu-results p17-open-webui p17-continue p17-openai-sdk"

while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;; --models) MODELS="$2"; shift 2 ;; --glyd) GLYD="$2"; shift 2 ;;
    --install-cmd) INSTALL_CMD="$2"; shift 2 ;; --skip-install) SKIP_INSTALL=1; shift ;; --out) OUT="$2"; shift 2 ;;
    --ext) EXT="$2"; shift 2 ;; --limit) LIMIT="$2"; shift 2 ;; --plan-only) PLAN_ONLY=1; shift ;; --machine) MACHINE="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "m1-gate.sh: $1: not an option (--help)"; exit 2 ;;
  esac
done
if [ -z "$OUT" ]; then OUT="$(mktemp -d "${TMPDIR:-/tmp}/m1-gate.XXXXXX")" || exit 2   # (a directory made new, 0700: not a name another user could have made first)
elif [ -e "$OUT" ] && [ ! -O "$OUT" ]; then echo "m1-gate.sh: $OUT exists and is not yours"; exit 2
else mkdir -p "$OUT" && chmod 700 "$OUT" || exit 2; fi
RESULTS="$OUT/results.tsv"; : > "$RESULTS"; FAILS=0
# (the tools the gate runs, by their absolute paths, before the shims are put on PATH)
SH="$(command -v sh)"; CURL="$(command -v curl || true)"; JQ="$(command -v jq || true)"

step() { # STATUS NAME DETAIL
  printf '%-5s %-34s %s\n' "$1" "$2" "$3"; printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RESULTS"
  [ "$1" = FAIL ] && FAILS=$((FAILS + 1)); return 0
}

# the pinned models: MODEL | released | 8gb | 16gb | 24gb | note, each tier "run" or "refuse"; # lines and blanks skipped
models() { grep -v '^[[:space:]]*#' "$MODELS" | grep -v '^[[:space:]]*$'; }
field() { echo "$1" | awk -F'|' -v n="$2" '{gsub(/^[ \t]+|[ \t]+$/, "", $n); print $n}'; }

# the planner's answer for MODEL: "run<TAB>plan line" or "refuse<TAB>first line of why"
plan_of() {
  local model="$1" out
  if [ -n "$MACHINE" ]; then out="$("$GLYD" fit "$model" --machine "$MACHINE" 2>&1)"; else out="$("$GLYD" fit "$model" 2>&1)"; fi
  local plan; plan="$(echo "$out" | grep -m1 '^Plan: ')"
  if [ -n "$plan" ] && ! echo "$plan" | grep -q 'none fits'; then printf 'run\t%s\n' "$plan"
  else printf 'refuse\t%s\n' "$(echo "$out" | grep -m1 -E '^glyd: |does not fit|none fits|does not run' | cut -c1-200)"; fi
}

if [ "$PLAN_ONLY" = 1 ]; then
  [ -n "$MACHINE" ] || { echo "--plan-only needs --machine PRESET (rtx4060-laptop, rtx4080s, rtx4090, ...)"; exit 2; }
  case "$MACHINE" in rtx4060-laptop) col=3 ;; rtx4080s) col=4 ;; rtx4090) col=5 ;; *) col=0 ;; esac
  while IFS= read -r line; do
    m="$(field "$line" 1)"; r="$(plan_of "$m")"; got="${r%%$'\t'*}"; why="${r#*$'\t'}"
    if [ "$col" != 0 ]; then want="$(field "$line" $col)"; if [ "$got" = "$want" ]; then s=PASS; else s=FAIL; fi
    else want="-"; s=INFO; fi
    step "$s" "plan:$m" "$got (expected $want) $why"
  done < <(models)
  [ "$FAILS" = 0 ] && exit 0 || exit 1
fi

# ---- the machine
if command -v nvidia-smi > /dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  MIB="$(echo "$GPU" | awk -F', ' '{print $2}' | tr -dc 0-9)"
  if [ -z "$TIER" ]; then if [ "${MIB:-0}" -lt 9000 ]; then TIER=8gb; elif [ "${MIB:-0}" -lt 20000 ]; then TIER=16gb; else TIER=24gb; fi; fi
  step PASS machine "$GPU; tier $TIER; $(free -g | awk '/^Mem:/{print $2}') GiB RAM; $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-$(uname -sr)}")"
else
  step INFO machine "no nvidia-smi: the one-line answer is what is checked (below)"
fi
case "$TIER" in 8gb) col=3 ;; 16gb) col=4 ;; 24gb) col=5 ;; "") col=0 ;; *) echo "--tier 8gb, 16gb or 24gb"; exit 2 ;; esac

# python and pip are shimmed for the whole run: a call is a FAIL
SHIM="$OUT/nopython"; mkdir "$SHIM" && chmod 700 "$SHIM" || exit 2; CALLED="$OUT/python-called"; : > "$CALLED"
for p in python python3 pip pip3; do printf '#!/bin/sh\necho "$0 $*" >> "%s"\nexit 127\n' "$CALLED" > "$SHIM/$p"; chmod +x "$SHIM/$p"; done
export PATH="$SHIM:$PATH"
[ -n "$JQ" ] || { step FAIL jq "jq is not installed (apt-get install jq): the gate reads JSON with it, not with Python"; exit 1; }

# ---- install
if [ "$SKIP_INSTALL" = 0 ]; then
  t0=$SECONDS; if "$SH" -c "$INSTALL_CMD" > "$OUT/install.log" 2>&1; then step PASS install "$((SECONDS - t0)) s: $INSTALL_CMD"; else step FAIL install "failed after $((SECONDS - t0)) s: see $OUT/install.log"; fi
fi
if ! "$GLYD" --version > "$OUT/version.txt" 2>&1; then step FAIL glyd-version "$GLYD --version failed"; fi

# ---- no NVIDIA: Apple or AMD get one line saying what to do
if ! command -v nvidia-smi > /dev/null 2>&1; then
  first="$(models | head -1)"; m="$(field "$first" 1)"
  out="$(timeout 60 "$GLYD" run "$m" --prompt hi 2>&1)"; code=$?; n="$(echo "$out" | grep -c .)"
  if [ "$code" -ne 0 ] && [ "$n" -le 2 ]; then step PASS vendor-one-line "exit $code: $(echo "$out" | head -1 | cut -c1-160)"; else step FAIL vendor-one-line "exit $code, $n lines: $(echo "$out" | head -1 | cut -c1-120)"; fi
fi

# ---- each pinned model
SCHEMA='{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"},"coastal":{"type":"boolean"}},"required":["city","population","coastal"],"additionalProperties":false}'
chat() { # BODY -> stdout
  "$CURL" -s -m 120 -H 'Content-Type: application/json' -d "$1" "http://127.0.0.1:$PORT/v1/chat/completions"
}
while IFS= read -r line; do
  m="$(field "$line" 1)"; want="$([ "$col" != 0 ] && field "$line" "$col" || echo "")"
  r="$(plan_of "$m")"; got="${r%%$'\t'*}"; why="${r#*$'\t'}"
  if [ -n "$want" ] && [ "$got" != "$want" ]; then step FAIL "plan:$m" "the planner says $got, m1-models.txt expects $want on $TIER: $why"; continue; fi
  step PASS "plan:$m" "$got: $why"
  if [ "$got" = refuse ]; then
    out="$(timeout 60 "$GLYD" run "$m" --prompt hi 2>&1)"; code=$?; n="$(echo "$out" | grep -c .)"
    if [ "$code" -ne 0 ] && [ "$n" -le 3 ]; then step PASS "refusal:$m" "exit $code, $n lines: $(echo "$out" | head -1 | cut -c1-140)"; else step FAIL "refusal:$m" "exit $code, $n lines (a refusal is a few lines and a non-zero exit)"; fi
    continue
  fi
  # first token: the server started, the model packed or downloaded where it is not here, and the first streamed token, all inside the limit
  t0=$SECONDS; "$GLYD" serve "$m" --host 127.0.0.1 --port "$PORT" --users 1 > "$OUT/serve-$(echo "$m" | tr '/:' '__').log" 2>&1 & SRV=$!
  up=0; while [ $((SECONDS - t0)) -lt 900 ]; do kill -0 $SRV 2> /dev/null || break; "$CURL" -sf -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null && { up=1; break; }; sleep 1; done
  if [ "$up" = 0 ]; then step FAIL "first-token:$m" "the server did not come up ($(tail -1 "$OUT/serve-$(echo "$m" | tr '/:' '__').log" | cut -c1-120))"; kill $SRV 2> /dev/null; wait $SRV 2> /dev/null; continue; fi
  ft="$("$CURL" -sN -m 120 -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":4,"stream":true}' "http://127.0.0.1:$PORT/v1/chat/completions" | grep -m1 -E '"content":"[^"]' )"
  total=$((SECONDS - t0))
  if [ -n "$ft" ] && [ "$total" -le "$LIMIT" ]; then step PASS "first-token:$m" "$total s (limit $LIMIT)"; else step FAIL "first-token:$m" "$total s (limit $LIMIT), token: ${ft:-none}"; fi
  # the printed plan is what loads: the level, and the weights' size within 10%
  info="$("$CURL" -s -m 10 "http://127.0.0.1:$PORT/v1/glyd/info")"
  lvl="$(echo "$info" | jq -r '.level.effective // empty')"; wb="$(echo "$info" | jq -r '.memory.weights_bytes // empty')"
  plan_lvl="$(echo "$why" | sed -n 's/^Plan: \([a-z]*\).*/\1/p')"; plan_gib="$("$GLYD" fit "$m" ${MACHINE:+--machine "$MACHINE"} 2>&1 | sed -n 's/.*: \([0-9.]*\) GiB of weights on the GPU.*/\1/p' | head -1)"
  if [ -n "$lvl" ] && [ -n "$plan_lvl" ] && [ "$lvl" = "$plan_lvl" ]; then
    if [ -n "$wb" ] && [ -n "$plan_gib" ] && awk -v w="$wb" -v p="$plan_gib" 'BEGIN{g=w/1073741824; d=(g>p?g-p:p-g)/p; exit !(d<=0.10)}'; then step PASS "plan-is-load:$m" "level $lvl, weights $(awk -v w="$wb" 'BEGIN{printf "%.2f", w/1073741824}') GiB against the plan's $plan_gib"
    else step FAIL "plan-is-load:$m" "weights $wb bytes against the plan's $plan_gib GiB"; fi
  else step FAIL "plan-is-load:$m" "level loaded '$lvl', planned '$plan_lvl'"; fi
  # JSON-schema output is valid
  body="$(jq -nc --argjson s "$SCHEMA" '{messages:[{role:"user",content:"Give a city, its population and whether it is coastal."}],max_tokens:200,temperature:0,response_format:{type:"json_schema",json_schema:{name:"city",strict:true,schema:$s}}}')"
  txt="$(chat "$body" | jq -r '.choices[0].message.content // empty')"
  if [ -n "$txt" ] && echo "$txt" | jq -e 'type=="object" and (keys|sort)==["city","coastal","population"] and (.city|type=="string") and (.population|type=="number" and floor==.) and (.coastal|type=="boolean")' > /dev/null 2>&1; then
    step PASS "json-schema:$m" "$(echo "$txt" | tr -d '\n' | cut -c1-100)"; else step FAIL "json-schema:$m" "not valid against the schema: $(echo "$txt" | tr -d '\n' | cut -c1-100)"; fi
  kill $SRV 2> /dev/null; wait $SRV 2> /dev/null; sleep 2
done < <(models)

# ---- python was never called
if [ -s "$CALLED" ]; then step FAIL no-python "called: $(head -3 "$CALLED" | tr '\n' ';')"; else step PASS no-python "no call to python or pip during the run"; fi

# ---- what the other checks measured
for name in $REQUIRED_EXT; do
  f="$EXT/$name.result"
  if [ -n "$EXT" ] && [ -f "$f" ]; then
    s="$(cut -f1 "$f" | head -1)"; d="$(cut -f3- "$f" | head -1)"; step "$s" "ext:$name" "$d"
  else step FAIL "ext:$name" "no result (--ext DIR with $name.result)"; fi
done

PASSES="$(grep -c '^PASS' "$RESULTS")"
echo "---"; echo "$PASSES passed, $FAILS failed; results in $RESULTS"
jq -Rn '[inputs|split("\t")|{status:.[0],step:.[1],detail:.[2]}]' < "$RESULTS" > "$OUT/m1-gate.json" 2> /dev/null || true
[ "$FAILS" = 0 ]
