#!/usr/bin/env bash
# The M2 gate (ROADMAP.md, M2; epic #121), run on one GPU machine (an RTX 4080 SUPER or an L4) with Glyd installed: for each row of m2-models.txt for this GPU, at each of the row's levels
# and contexts, the planner's answer for 64 users, then `serve --users 64` with 64 concurrent streaming requests through the one server, the server's memory against the printed plan, and
# the logits against the CPU Ref backend; a model the list says does not run (NONE) must be refused well. Checks other runs measured are read from --ext.
# One line a step: PASS, FAIL, SKIP or INFO, the step, what it saw. Exit 0 only where nothing FAILs, no step is SKIPped and every required step has a result. No Python is used (jq is).
#
#   benchmarks/gates/m2-gate.sh [--gpu rtx4080s|l4] [--models FILE] [--glyd PATH] [--engine PATH] [--out DIR] [--ext DIR] [--limit SECONDS] [--rows REGEX] [--machine PRESET]
#                               [--ref-dir DIR]
#   benchmarks/gates/m2-gate.sh --plan-only --machine PRESET [--gpu rtx4080s|l4] [--models FILE] [--glyd PATH] [--rows REGEX] [--out DIR]
#                                    the planner's answer for each row of the GPU on a described machine (glyd fit --machine PRESET), no GPU needed: a RUN row must be a plan for 64
#                                    users at each of its levels and contexts, a NONE row must be refused (exit 3; its first line is printed)
#
# The list (FILE, default m2-models.txt beside this script): rows `KIND | GPU | MODEL | family | released | users | levels | contexts | the plan | note`, KIND RUN, ALT or NONE. A level
# followed by * is one where the planner does not count 64 users (the line says so, and that is no FAIL); an ALT row (the owner's choice instead of its family's RUN row) is checked by
# --plan-only but run only where --rows selects it. --gpu: the rows of this GPU (default: the NVIDIA GPU here; in --plan-only, the machine's name). --glyd: the glyd command (default $GLYD, else
# glyd). --engine: the glyd-engine program of a development build: glyd is told to use it (GLYD_ENGINE), and the pack-compatibility suite is run with it (below); without it GLYD_ENGINE=1, so
# serve is the engine's. --limit: the longest the gate waits for one thing, in seconds (a fit, a server to come up with its download and packing, the 64 requests; default 1800). --rows: only the
# rows whose "KIND GPU MODEL family" match the extended regex; a subset is not the gate (its required results are read only with --ext). --machine: plan for that machine preset instead of this
# computer. --out: where the server logs, every request's stream, results.tsv and m2-gate.json go (default a new directory under $TMPDIR).
#
# The steps for each RUN row, each level and each context: plan:MODEL:LEVEL@CONTEXT (the planner's plan, 64 users); serve-64 (the server up at --users 64 and 64 streaming requests at once, each
# ending with its tokens: /v1/glyd/info says 64 at once, /stats' running peaks at 64 with no sequence sent back for room); plan-is-load (the level and the memory the server says against the
# printed plan: /v1/glyd/info); and, for each level, logits-vs-ref. NONE rows: plan:MODEL (refused, exit 3) and refusal:MODEL (the real command: a non-zero exit, a few lines).
#
# logits-vs-ref runs the development command of --engine's build, `glyd-engine usercheck --ref --dir MODEL --level LEVEL --users 64` on the model the cache holds (serve fetched it):
# 64 prompts asked at once through the one scheduler, 5 tokens each, and every row of logits a token was chosen from (the prompt's last position and 4 decode steps) held to the CPU
# Ref backend running the model's description on the same weights, row by row: at penguin within the kind's bound on the run's own choices of experts, at a lossy level within the
# level's threshold. Its line says how many rows were held and the worst row's KL. Without --engine it is a SKIP, which keeps the gate red: an installed glyd has no development
# commands. The reference holds the weights as stored (bf16: a model's parameters times 2 bytes, a tied head once, in the computer's memory); it is given four times --limit.
# --ref-dir DIR: the reference's rows recorded on another computer, where this one's memory does not hold the model (gemma-4-12B's ~22 GiB, Nemotron-3.5-Lightning's ~58 GiB): a file
# DIR/MODEL.LEVEL.ref.json.gz (MODEL's / written --) made on any CPU computer by `glyd-engine usercheck --ref --record FILE --dir MODEL --level LEVEL --users 64 --tokens 5
# --prompt-tokens 32` holds that row instead (`usercheck --ref-from FILE`: the prompts served with the tokens the file's reference chose, every row held to the file's); a file of other
# weights, level, prompts or reference FAILS in words. A row with no file runs the reference here, and where this computer's memory does not hold it, it FAILS in words: never a skip.
#
# --ext DIR: files NAME.result, each a line "PASS|FAIL|SKIP<TAB>NAME<TAB>detail": parity-kimi-k3, parity-deepseek-v4, parity-glm-5.3 (single-user parity on their reference hardware),
# bench-no-regression (`glyd bench --compare BASE.json CAND.json` against the previous record on the gate's GPU: exit 0) and packs-since-v0.29 (every published pack loads). Where --engine is
# given, packs-since-v0.29 is not read but run: `glyd-engine packcheck --gpu --engine` (the pack-compatibility suite of the tree it was built from), and the line says so.
set -u

GPU=""; MODELS="$(dirname "$0")/m2-models.txt"; GLYD="${GLYD:-glyd}"; ENGINE=""; OUT=""; EXT=""; LIMIT=1800; ROWS=""; MACHINE=""; PLAN_ONLY=0; REFDIR=""
PORT="${GLYD_GATE_PORT:-18735}"; USERS=64; REPLY=128
REQUIRED_EXT="parity-kimi-k3 parity-deepseek-v4 parity-glm-5.3 bench-no-regression packs-since-v0.29"
NO_REF="not run: the check is a development command (glyd-engine usercheck --ref), and no --engine names a development build that has it"

need() { [ "$1" -ge 2 ] || { echo "m2-gate.sh: $2 needs a value (--help)"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) need $# "$1"; GPU="$2"; shift 2 ;; --models) need $# "$1"; MODELS="$2"; shift 2 ;; --glyd) need $# "$1"; GLYD="$2"; shift 2 ;;
    --engine) need $# "$1"; ENGINE="$2"; shift 2 ;; --out) need $# "$1"; OUT="$2"; shift 2 ;; --ext) need $# "$1"; EXT="$2"; shift 2 ;;
    --limit) need $# "$1"; LIMIT="$2"; shift 2 ;; --rows) need $# "$1"; ROWS="$2"; shift 2 ;; --machine) need $# "$1"; MACHINE="$2"; shift 2 ;;
    --ref-dir) need $# "$1"; REFDIR="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "m2-gate.sh: $1: not an option (--help)"; exit 2 ;;
  esac
done
case "$LIMIT" in ''|*[!0-9]*) echo "m2-gate.sh: --limit takes a number of seconds"; exit 2 ;; esac
[ -f "$MODELS" ] || { echo "m2-gate.sh: $MODELS: no such file"; exit 2; }
if [ "$PLAN_ONLY" = 1 ] && [ -z "$MACHINE" ]; then echo "m2-gate.sh: --plan-only needs --machine PRESET (rtx4080s, l4)"; exit 2; fi
if [ -n "$ENGINE" ] && [ ! -x "$ENGINE" ]; then echo "m2-gate.sh: --engine $ENGINE: not an executable file"; exit 2; fi
if [ -n "$REFDIR" ] && [ ! -d "$REFDIR" ]; then echo "m2-gate.sh: --ref-dir $REFDIR: no such directory"; exit 2; fi
if [ -z "$OUT" ]; then OUT="$(mktemp -d "${TMPDIR:-/tmp}/m2-gate.XXXXXX")" || exit 2   # (a directory made new, 0700: not a name another user could have made first)
elif [ -e "$OUT" ] && [ ! -O "$OUT" ]; then echo "m2-gate.sh: $OUT exists and is not yours"; exit 2
else mkdir -p "$OUT" && chmod 700 "$OUT" || exit 2; fi
RESULTS="$OUT/results.tsv"; : > "$RESULTS"; FAILS=0; BLOCKED=0
CURL="$(command -v curl || true)"; JQ="$(command -v jq || true)"; TIMEOUT="$(command -v timeout || command -v gtimeout || true)"
SRV=""; POLL=""; PJ=""; PCODE=0; PWHY=""; PLAN=0; PLAN_J=""
export GLYD_ENGINE="${ENGINE:-1}"   # (a path, or 1: serve is the engine's, not another tool's)
unset GLYD_API_KEY                   # (the server is asked for /v1 with no key)

step() { # STATUS NAME DETAIL
  local d; d="$(printf '%s' "$3" | tr '\t\n' '  ')"
  printf '%-5s %-52s %s\n' "$1" "$2" "$d"; printf '%s\t%s\t%s\n' "$1" "$2" "$d" >> "$RESULTS"
  case "$1" in FAIL) FAILS=$((FAILS + 1)) ;; SKIP) BLOCKED=$((BLOCKED + 1)) ;; esac
  return 0
}

# killtree PID: the process and everything under it
killtree() {
  local c kids
  kids="$(pgrep -P "$1" 2> /dev/null)"; kill "$1" 2> /dev/null
  for c in $kids; do killtree "$c"; done
  return 0
}

stop_server() {
  local i=0
  [ -n "$SRV" ] || return 0
  killtree "$SRV"
  while kill -0 "$SRV" 2> /dev/null && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
  kill -9 "$SRV" 2> /dev/null; wait "$SRV" 2> /dev/null; SRV=""
  sleep 2   # (the GPU's memory is given back)
}
# shellcheck disable=SC2329
cleanup() { if [ -n "$POLL" ]; then kill "$POLL" 2> /dev/null; fi; stop_server; }
trap cleanup EXIT
trap 'exit 130' INT TERM

finish() { # the summary, the JSON, the exit status
  local p; p="$(grep -c '^PASS' "$RESULTS")"
  echo "---"
  if [ "$PLAN_ONLY" = 1 ]; then echo "plan only: the planner's answers for $GPU on '$MACHINE'; the run's steps and the --ext results were not asked"; fi
  if [ -n "$ROWS" ]; then echo "a subset (--rows $ROWS): this is not the gate"; fi
  echo "$p passed, $FAILS failed, $BLOCKED skipped (a skipped step keeps the gate red); results in $RESULTS"
  if [ -n "$JQ" ]; then
    jq -Rn --arg gpu "$GPU" --argjson fails "$FAILS" --argjson skips "$BLOCKED" --argjson passes "$p" --argjson plan "$([ "$PLAN_ONLY" = 1 ] && echo true || echo false)" \
      --argjson subset "$([ -n "$ROWS" ] && echo true || echo false)" \
      '{gate: "m2", gpu: $gpu, plan_only: $plan, subset: $subset, passed: $passes, failed: $fails, skipped: $skips, steps: [inputs | split("\t") | {status: .[0], step: .[1], detail: .[2]}]}' \
      < "$RESULTS" > "$OUT/m2-gate.json" 2> /dev/null || true
  fi
  [ "$FAILS" = 0 ] && [ "$BLOCKED" = 0 ]
  exit $?
}

# tmo SECONDS CMD...: CMD as it is, ended after SECONDS (status 124): timeout(1) where there is one, else a watchdog
tmo() {
  local s="$1" p w c; shift
  if [ -n "$TIMEOUT" ]; then "$TIMEOUT" "$s" "$@"; return $?; fi
  "$@" & p=$!
  ( sleep "$s"; killtree "$p" ) > /dev/null 2>&1 & w=$!
  wait "$p"; c=$?
  killtree "$w"; wait "$w" 2> /dev/null
  if [ "$c" -gt 128 ]; then c=124; fi
  return "$c"
}

# the computer's memory, where /proc says
ram_gib() { awk '/^MemTotal/ { printf "%d GiB RAM; ", $2 / 1048576 }' /proc/meminfo 2> /dev/null; return 0; }

# the OS, as its own file names it
os_name() {
  # shellcheck source=/dev/null
  ( . /etc/os-release 2> /dev/null; echo "${PRETTY_NAME:-$(uname -sr)}" )
}

# the rows: no # lines, no blank ones; field LINE N is the Nth, trimmed
rows() { grep -v '^[[:space:]]*#' "$MODELS" | grep -v '^[[:space:]]*$'; }
field() { printf '%s\n' "$1" | awk -F'|' -v n="$2" '{gsub(/^[ \t]+|[ \t]+$/, "", $n); print $n}'; }

# ask TARGET CONTEXT: `glyd fit TARGET --users 64 [--context N] [--machine M] --json`: PLAN=1 and PJ the plan's JSON where there is a plan, else PLAN=0, PCODE its exit
# status and PWHY the first line of the refusal
ask() {
  local target="$1" ctx="$2" err="$OUT/fit.err"
  set -- fit "$target" --users "$USERS"
  if [ -n "$ctx" ]; then set -- "$@" --context "$ctx"; fi
  if [ -n "$MACHINE" ]; then set -- "$@" --machine "$MACHINE"; fi
  PJ="$(tmo "$LIMIT" "$GLYD" "$@" --json 2> "$err")"; PCODE=$?
  PWHY="$( (grep -m1 '^glyd: ' "$err" || grep -m1 . "$err") | sed 's/^glyd: //' | cut -c1-260)"
  if [ -z "$PWHY" ]; then PWHY="$(printf '%s' "$PJ" | jq -r '.degrade.refusal // .notes[0] // empty' 2> /dev/null | head -1 | cut -c1-260)"; fi
  if [ "$PCODE" = 0 ] && printf '%s' "$PJ" | jq -e '.level != null' > /dev/null 2>&1; then PLAN=1; else PLAN=0; PJ=""; fi
}

# the plan in the list's own words: LEVEL: weights + cache GiB (total of free GiB), users x tokens
plan_words() {
  printf '%s' "$1" | jq -r '[.level, (.weights_on_gpu_bytes // .weights_bytes // 0), (.kv_bytes // 0), (.gpu_total_bytes // 0), (.free_bytes // 0), (.users // 0), (.context // 0), (.experts.ram_bytes // 0), (.experts.places_per_layer // 0), (.experts.experts_per_layer // 0), .exact] | @tsv' |
    awk -F'\t' '{ g = 1073741824; printf "%s: %.1f + %.1f GiB (%.1f of %.1f), %d users x %d tokens", $1, $2 / g, $3 / g, $4 / g, $5 / g, $6, $7
                  if ($8 > 0) printf ", experts in RAM %.1f GiB (%d places a layer of %d on the GPU)", $8 / g, $9, $10
                  if ($11 != "true") printf " (an estimate)"
                  printf "\n" }'
}

# the numbers of a text, one a line
numbers() { printf '%s' "$1" | grep -oE '[0-9]+(\.[0-9]+)?'; }

# plan_is_load NAME INFO: what /v1/glyd/info says loaded and holds, against the plan asked before the server started (PLAN_J, glyd-plan/1)
plan_is_load() {
  local name="$1" info="$2" v bad txt
  v="$(jq -nc --argjson p "$PLAN_J" --argjson i "$info" --argjson probe "$([ -z "$MACHINE" ] && echo true || echo false)" '
    def g: ((. / 1073741824 * 100 | round) / 100);
    ($p.gpu_total_bytes - $p.reserve_bytes) as $used | $i.memory as $m
    | { bad: [ (if $i.level.effective != $p.level then "level loaded \($i.level.effective), planned \($p.level)" else empty end),
               (if $m.weights_bytes == null then "the server does not say its weights"
                elif $p.weights_on_gpu_bytes > 0 and ((($m.weights_bytes - $p.weights_on_gpu_bytes) | fabs) / $p.weights_on_gpu_bytes) > 0.1
                  then "weights \($m.weights_bytes | g) GiB, the plan has \($p.weights_on_gpu_bytes | g) (over 10% apart)" else empty end),
               (if $m.engine_bytes == null then "the server does not say its memory"
                elif $m.engine_bytes > $used then "the engine holds \($m.engine_bytes | g) GiB, the plan has \($used | g)" else empty end),
               (if $m.device_free_bytes == null then "the server does not say the GPU free memory"
                elif $probe and ($p.free_bytes - $m.device_free_bytes) > $used then "the GPU lost \(($p.free_bytes - $m.device_free_bytes) | g) GiB since the plan, the plan has \($used | g)" else empty end) ],
        text: ("level \($i.level.effective); weights \($m.weights_bytes // 0 | g) GiB (plan \($p.weights_on_gpu_bytes | g)); engine \($m.engine_bytes // 0 | g) GiB (plan \($used | g))"
               + (if $probe then "; GPU lost \(($p.free_bytes - ($m.device_free_bytes // 0)) | g) GiB (plan \($used | g))" else "" end)) }' 2> /dev/null)"
  if [ -z "$v" ]; then step FAIL "plan-is-load:$name" "/v1/glyd/info or the plan could not be read"; return; fi
  bad="$(printf '%s' "$v" | jq -r '.bad | join("; ")')"; txt="$(printf '%s' "$v" | jq -r '.text')"
  if [ -z "$bad" ]; then step PASS "plan-is-load:$name" "$txt"; else step FAIL "plan-is-load:$name" "$bad ($txt)"; fi
}

# a request's stream: the completion tokens its last chunk counts, or - where it did not end with [DONE]
sse_tokens() {
  if ! grep -q '^data: \[DONE\]' "$1" 2> /dev/null; then echo -; return; fi
  sed -n 's/^data: //p' "$1" | grep -v '^\[DONE\]' | jq -r '.usage.completion_tokens? // empty' 2> /dev/null | tail -1
}
filler() { awk -v k="$1" 'BEGIN { for (i = 0; i < k; i++) printf "The quick brown fox jumps over the lazy dog. " }'; }
req_body() { # REQUEST SENTENCES: a chat request that makes REPLY tokens whatever the model would say, streamed with its usage
  jq -nc --arg t "Request $1. $(filler "$2")" --argjson n "$REPLY" \
    '{messages: [{role: "user", content: $t}], max_tokens: $n, min_tokens: $n, ignore_eos: true, temperature: 0, stream: true, stream_options: {include_usage: true}}'
}
prompt_tokens() { # SENTENCES: what the server counts of such a prompt
  "$CURL" -s -m 30 -H 'Content-Type: application/json' -d "$(jq -nc --arg t "Request 0. $(filler "$1")" '{messages: [{role: "user", content: $t}]}')" \
    "http://127.0.0.1:$PORT/v1/glyd/tokenize" | jq -r '.prompt_tokens // empty' 2> /dev/null
}

# serve_64 TARGET CONTEXT NAME: the server at --users 64 up, 64 streaming requests at once through it (each prompt about half the context, each reply REPLY tokens), what the
# server says of itself: serve-64 and plan-is-load
serve_64() {
  local target="$1" ctx="$2" name="$3" tag slog d t0 up_s up info i k p1 p11 want ptok pids ok first n peak wpeak info1 st conc steps nrows pre why took
  tag="$(printf '%s' "$name" | tr '/:@' '___')"; slog="$OUT/serve-$tag.log"; d="$OUT/streams-$tag"; mkdir -p "$d"
  if "$CURL" -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then step FAIL "serve-64:$name" "port $PORT is busy: a server answers there already (GLYD_GATE_PORT)"; return; fi
  "$GLYD" serve "$target" --users "$USERS" --context "$ctx" --host 127.0.0.1 --port "$PORT" > "$slog" 2>&1 & SRV=$!
  t0=$SECONDS; up=0
  while [ $((SECONDS - t0)) -lt "$LIMIT" ]; do
    kill -0 "$SRV" 2> /dev/null || break
    info="$("$CURL" -s -m 2 "http://127.0.0.1:$PORT/v1/glyd/info")"
    case "$(printf '%s' "$info" | jq -r '.state // empty' 2> /dev/null)" in ready) up=1; break ;; failed) break ;; esac
    sleep 1
  done
  up_s=$((SECONDS - t0))
  if [ "$up" = 0 ]; then step FAIL "serve-64:$name" "the server did not come up within $LIMIT s (ended or failed after $up_s s): $(tail -1 "$slog" | cut -c1-160)"; stop_server; return; fi
  # (the prompts: about half the context, counted by the server's own tokenizer from a one-sentence and an eleven-sentence prompt; each begins with its own words, so no two share a prefix)
  want=$((ctx / 2)); k=1; ptok=""
  p1="$(prompt_tokens 1)"; p11="$(prompt_tokens 11)"
  if [ -n "$p1" ] && [ -n "$p11" ]; then
    k="$(awk -v a="$p1" -v b="$p11" -v w="$want" 'BEGIN { per = (b - a) / 10; k = (per > 0) ? int((w - a) / per) + 1 : 1; print (k < 1) ? 1 : k }')"
    ptok="$(prompt_tokens "$k")"
  fi
  i=1; while [ "$i" -le "$USERS" ]; do req_body "$i" "$k" > "$d/$i.req"; i=$((i + 1)); done
  : > "$d/poll"
  ( while :; do "$CURL" -s -m 2 "http://127.0.0.1:$PORT/stats" | jq -r '"\(.running // 0) \(.waiting // 0)"' >> "$d/poll" 2> /dev/null; sleep 0.1; done ) > /dev/null 2>&1 & POLL=$!
  t0=$SECONDS; pids=""; i=1
  while [ "$i" -le "$USERS" ]; do
    ( "$CURL" -sN -m "$LIMIT" -H 'Content-Type: application/json' -d "@$d/$i.req" "http://127.0.0.1:$PORT/v1/chat/completions" > "$d/$i.sse" 2> /dev/null; echo $? > "$d/$i.rc" ) & pids="$pids $!"
    i=$((i + 1))
  done
  # shellcheck disable=SC2086
  wait $pids
  took=$((SECONDS - t0)); kill "$POLL" 2> /dev/null; wait "$POLL" 2> /dev/null; POLL=""
  ok=0; first=""; i=1
  while [ "$i" -le "$USERS" ]; do
    n="$(sse_tokens "$d/$i.sse")"
    if [ "$n" = "$REPLY" ]; then ok=$((ok + 1))
    elif [ -n "$first" ]; then :
    elif [ "$n" = - ]; then first="request $i did not end: $(head -c 160 "$d/$i.sse" | tr '\n' ' ')"
    else first="request $i: ${n:-no usage line} tokens, not $REPLY"; fi
    i=$((i + 1))
  done
  peak="$(awk '$1 > m { m = $1 } END { print m + 0 }' "$d/poll")"; wpeak="$(awk '$2 > m { m = $2 } END { print m + 0 }' "$d/poll")"
  info1="$("$CURL" -s -m 10 "http://127.0.0.1:$PORT/v1/glyd/info")"; st="$("$CURL" -s -m 10 "http://127.0.0.1:$PORT/stats")"
  printf '%s\n' "$info1" > "$d/info.json"; printf '%s\n' "$st" > "$d/stats.json"
  conc="$(printf '%s' "$info1" | jq -r '.limits.concurrency // empty' 2> /dev/null)"
  steps="$(printf '%s' "$st" | jq -r '.steps // empty' 2> /dev/null)"; nrows="$(printf '%s' "$st" | jq -r '.rows // empty' 2> /dev/null)"; pre="$(printf '%s' "$st" | jq -r '.preemptions // empty' 2> /dev/null)"
  why="$ok of $USERS requests ended with their $REPLY tokens in $took s (prompts ${ptok:-not measured} tokens, context $ctx); running peaked at $peak, waiting at $wpeak; limits.concurrency ${conc:--}; ${steps:--} steps, ${nrows:--} rows, ${pre:--} sent back for room; up in $up_s s"
  if [ "$ok" = "$USERS" ] && [ "${conc:-0}" = "$USERS" ] && [ "$peak" -ge "$USERS" ] && [ "${pre:-1}" = 0 ]; then step PASS "serve-64:$name" "$why"
  else step FAIL "serve-64:$name" "${first:+$first; }$why"; fi
  plan_is_load "$name" "$info1"
  stop_server
}

# a RUN row's levels and contexts: the plan at 64 users and, on a GPU, the server
run_row() { # MODEL LEVELS CONTEXTS
  local m="$1" levels="$2" ctxs="$3" l c star name words pl pu
  levels="$(printf '%s' "$levels" | tr -d ' ' | tr ',' ' ' | tr '*' '+')"; ctxs="$(printf '%s' "$ctxs" | tr -d ' ' | tr ',' ' ')"
  if [ -z "$levels" ] || [ -z "$ctxs" ] || printf '%s%s' "$levels" "$ctxs" | grep -q '[^a-z0-9+ ]'; then step FAIL row "$m: levels '$levels' or contexts '$ctxs' are not a list"; return; fi
  for l in $levels; do
    star=0; case "$l" in *+) star=1; l="${l%+}" ;; esac
    for c in $ctxs; do
      name="$m:$l@$c"
      ask "$m:$l" "$c"
      if [ "$PLAN" = 0 ]; then step FAIL "plan:$name" "no plan (exit $PCODE): ${PWHY:-no message}"; continue; fi
      words="$(plan_words "$PJ")"; pl="$(printf '%s' "$PJ" | jq -r '.level')"; pu="$(printf '%s' "$PJ" | jq -r '.users // "unknown"')"
      if [ "$pl" != "$l" ]; then step FAIL "plan:$name" "the plan is for the level $pl, not $l: $words"; continue; fi
      if [ "$pu" != "$USERS" ] && [ "$star" = 1 ]; then step PASS "plan:$name" "$words [the list marks this level *: the planner counts $pu users, not $USERS]"
      elif [ "$pu" != "$USERS" ]; then step FAIL "plan:$name" "the planner counts $pu users, not $USERS ($([ "$pu" = unknown ] && echo "an estimate: the loader did not size it" || echo "the loader's sizing")): $words"; continue
      elif [ "$star" = 1 ]; then step PASS "plan:$name" "$words [the list marks this level *, but the planner counts $USERS users now]"
      else step PASS "plan:$name" "$words"; fi
      if [ "$PLAN_ONLY" = 0 ]; then PLAN_J="$PJ"; serve_64 "$m:$l" "$c" "$name"; fi
    done
    if [ "$PLAN_ONLY" = 0 ]; then logits_vs_ref "$m" "$l"; fi
  done
}

# logits_vs_ref MODEL LEVEL: `glyd-engine usercheck --ref` of --engine on the model at the level, 64 users (with --ref-dir's file of the row, `--ref-from FILE`): PASS or FAIL as
# its last verdict line says, and its exit status agrees
logits_vs_ref() {
  local m="$1" l="$2" log line code src="" from=""
  if [ -z "$ENGINE" ]; then step SKIP "logits-vs-ref:$m:$l" "$NO_REF"; return; fi
  log="$OUT/ref-$(printf '%s' "$m:$l" | tr '/:@' '___').log"
  if [ -n "$REFDIR" ] && [ -f "$REFDIR/$(printf '%s' "$m" | sed 's|/|--|g').$l.ref.json.gz" ]; then
    from="$REFDIR/$(printf '%s' "$m" | sed 's|/|--|g').$l.ref.json.gz"; src=" [the reference's rows from $from]"
    tmo $((LIMIT * 4)) "$ENGINE" usercheck --ref-from "$from" --dir "$m" --level "$l" --users "$USERS" --tokens 5 --prompt-tokens 32 > "$log" 2>&1; code=$?
  else
    tmo $((LIMIT * 4)) "$ENGINE" usercheck --ref --dir "$m" --level "$l" --users "$USERS" --tokens 5 --prompt-tokens 32 > "$log" 2>&1; code=$?
  fi
  line="$(grep -E '^(PASS|FAIL) usercheck --ref' "$log" | tail -1)$src"
  case "$code:$line" in
    0:PASS*) step PASS "logits-vs-ref:$m:$l" "${line#PASS usercheck --ref }" ;;
    *:FAIL*) step FAIL "logits-vs-ref:$m:$l" "${line#FAIL usercheck --ref } (log: $log)" ;;
    124:*) step FAIL "logits-vs-ref:$m:$l" "no verdict within $((LIMIT * 4)) s (log: $log): $(tail -1 "$log" | cut -c1-160)" ;;
    *) step FAIL "logits-vs-ref:$m:$l" "exit $code with no verdict that agrees (log: $log): $(tail -1 "$log" | cut -c1-160)" ;;
  esac
}

# a NONE row: the planner refuses (exit 3, its first line), and the command's refusal is a few lines and a non-zero exit
none_row() { # MODEL THE-LISTS-REFUSAL
  local m="$1" listed="$2" note="" out code n
  ask "$m" ""
  if [ "$PLAN" = 1 ]; then step FAIL "plan:$m" "the list says it does not run here, the planner has a plan: $(plan_words "$PJ")"; return; fi
  if [ "$PCODE" != 3 ]; then step FAIL "plan:$m" "not refused by the planner (exit $PCODE, a refusal is 3): ${PWHY:-no message}; the list: $listed"; return; fi
  # (the figures of the refusal, not of the model's name before its first space: the list's words differ from the planner's, its numbers should not)
  if numbers "${PWHY#* }" | grep -qvxFf <(numbers "$listed"); then note=" [the list says: $listed]"; fi
  step PASS "plan:$m" "refused (exit 3): $PWHY$note"
  if [ "$PLAN_ONLY" = 1 ]; then return; fi
  out="$(tmo 120 "$GLYD" serve "$m" --users "$USERS" --host 127.0.0.1 --port "$PORT" 2>&1)"; code=$?; n="$(printf '%s\n' "$out" | grep -c .)"
  if [ "$code" = 3 ] && [ "$n" -le 3 ]; then step PASS "refusal:$m" "exit $code, $n lines: $(printf '%s\n' "$out" | head -1 | cut -c1-160)"
  else step FAIL "refusal:$m" "exit $code, $n lines (a refusal is exit 3 and a few lines): $(printf '%s\n' "$out" | head -1 | cut -c1-140)"; fi
}

# ---- the tools
if [ -z "$JQ" ]; then step FAIL jq "jq is not installed (apt-get install jq, brew install jq): the gate reads JSON with it, not with Python"; finish; fi
if [ "$PLAN_ONLY" = 0 ] && [ -z "$CURL" ]; then step FAIL curl "curl is not installed: the gate sends its requests with it"; finish; fi

# ---- the machine, and the GPU whose rows are run
HAVE_SMI=0; command -v nvidia-smi > /dev/null 2>&1 && HAVE_SMI=1
if [ "$HAVE_SMI" = 1 ]; then
  SMI="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2> /dev/null | head -1)"
  case "$SMI" in *4080*SUPER*) HERE=rtx4080s ;; *) if printf '%s' "$SMI" | grep -Eq '(^|[^A-Za-z0-9])L4([^A-Za-z0-9]|$)'; then HERE=l4; else HERE=""; fi ;; esac
else SMI=""; HERE=""; fi
if [ "$PLAN_ONLY" = 1 ]; then GPU="${GPU:-$MACHINE}"
else
  GPU="${GPU:-$HERE}"
  if [ "$HAVE_SMI" = 0 ]; then step FAIL machine "no nvidia-smi here: the run needs the GPU (--plan-only --machine PRESET asks the planner alone)"; finish; fi
  if [ -z "$GPU" ]; then step FAIL machine "$SMI is neither an RTX 4080 SUPER nor an L4: --gpu rtx4080s|l4 names the list to run"; finish; fi
  if [ -n "$HERE" ] && [ "$HERE" != "$GPU" ]; then step FAIL machine "$SMI: this is an $HERE machine, and the rows asked for are $GPU's"; finish; fi
  step PASS machine "$SMI; rows of $GPU; $(ram_gib)$(os_name)"
  if ! "$GLYD" --version > "$OUT/version.txt" 2>&1; then step FAIL glyd-version "$GLYD --version failed: the gate stops"; finish; fi
  step PASS glyd-version "$(head -1 "$OUT/version.txt") ($GLYD; GLYD_ENGINE=$GLYD_ENGINE)"
fi

# ---- the rows of this GPU
SEEN=0
while IFS= read -r line <&3; do
  nf="$(printf '%s\n' "$line" | awk -F'|' '{ print NF }')"
  if [ "$nf" != 10 ]; then step FAIL row "a line of the list with $nf fields, not 10: $(printf '%s' "$line" | cut -c1-90)"; continue; fi
  kind="$(field "$line" 1)"; [ "$(field "$line" 2)" = "$GPU" ] || continue
  m="$(field "$line" 3)"
  if [ -n "$ROWS" ] && ! printf '%s %s %s %s\n' "$kind" "$GPU" "$m" "$(field "$line" 4)" | grep -Eq "$ROWS"; then continue; fi
  SEEN=$((SEEN + 1))
  case "$kind" in
    RUN) run_row "$m" "$(field "$line" 7)" "$(field "$line" 8)" ;;
    ALT) if [ "$PLAN_ONLY" = 1 ] || [ -n "$ROWS" ]; then run_row "$m" "$(field "$line" 7)" "$(field "$line" 8)"
         else step INFO "alt:$m" "an ALT row, the owner's choice instead of its family's RUN row: run only where --rows selects it"; fi ;;
    NONE) none_row "$m" "$(field "$line" 9)" ;;
    *) step FAIL row "$m: the kind is $kind, not RUN, ALT or NONE" ;;
  esac
done 3< <(rows)
if [ "$SEEN" = 0 ]; then step FAIL row "no row of $GPU${ROWS:+ matching $ROWS} in $MODELS"; fi
if [ "$PLAN_ONLY" = 1 ]; then finish; fi

# ---- what the other checks measured
if [ -n "$ROWS" ] && [ -z "$EXT" ]; then step INFO ext "a subset run (--rows) without --ext: the required results are not read"; finish; fi
for name in $REQUIRED_EXT; do
  f="$EXT/$name.result"
  if [ "$name" = packs-since-v0.29 ] && [ -n "$ENGINE" ]; then
    if tmo "$LIMIT" "$ENGINE" packcheck --gpu --engine > "$OUT/packcheck.log" 2>&1; then step PASS "ext:$name" "run here, not read: glyd-engine packcheck --gpu --engine: $(tail -1 "$OUT/packcheck.log" | cut -c1-160)"
    else step FAIL "ext:$name" "run here, not read: glyd-engine packcheck --gpu --engine failed (log: $OUT/packcheck.log): $(tail -1 "$OUT/packcheck.log" | cut -c1-160)"; fi
  elif [ -n "$EXT" ] && [ -f "$f" ]; then
    s="$(cut -f1 "$f" | head -1)"; d="$(cut -f3- "$f" | head -1)"
    case "$s" in PASS|FAIL|SKIP) step "$s" "ext:$name" "$d" ;; *) step FAIL "ext:$name" "$f does not start with PASS, FAIL or SKIP and a tab" ;; esac
  else step FAIL "ext:$name" "no result (--ext DIR with $name.result)"; fi
done
finish
