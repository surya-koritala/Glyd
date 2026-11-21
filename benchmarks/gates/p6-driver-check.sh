#!/usr/bin/env bash
# P6 (#158) acceptance on one driver: the v0.34.0 candidate's engine bundle on a machine with a plain NVIDIA driver (no CUDA toolkit, no Python), an Ada or Ampere GPU of 24 GB or more
# on the oldest supported driver (R535, CUDA 12.2), and again on R550 and R570 where the machine can. Writes DIR/driver-rNNN.result for m1-gate.sh --ext.
#
#   gh run download RUN -R surya-koritala/Glyd -n engine-bundle-linux-x86_64 -D rc && mkdir b && tar -xzf rc/glyd-engine-v*-linux-x86_64.tar.gz -C b --strip-components=1
#   benchmarks/gates/p6-driver-check.sh --bundle b [--out DIR] [--model glyd/gemma-4-12B-wren]
# Run the BUNDLE, not the pip wheel, for R535: only the bundle's CUDA 12 library is uncompressed (the wheel's compressed fatbins need R550).
set -u
BUNDLE=""; OUT=""; MODEL="glyd/gemma-4-12B-wren"; PORT=18735
while [ $# -gt 0 ]; do case "$1" in --bundle) BUNDLE="$2"; shift 2 ;; --out) OUT="$2"; shift 2 ;; --model) MODEL="$2"; shift 2 ;; -h|--help) sed -n '2,9p' "$0"; exit 0 ;; *) echo "$1: not an option"; exit 2 ;; esac; done
[ -x "$BUNDLE/glyd" ] || { echo "--bundle DIR holding glyd (the unpacked engine bundle)"; exit 2; }
if [ -z "$OUT" ]; then OUT="$(mktemp -d "${TMPDIR:-/tmp}/p6-driver.XXXXXX")" || exit 2; else mkdir -p "$OUT" || exit 2; fi
G="$(cd "$BUNDLE" && pwd)/glyd"; CURL="$(command -v curl || true)"; FAILS=0; DETAIL=""
note() { printf '%-5s %-26s %s\n' "$1" "$2" "$3"; DETAIL="$DETAIL$1 $2: $3; "; [ "$1" = FAIL ] && FAILS=$((FAILS + 1)); return 0; }

# 1. the driver
read -r NAME DRV < <(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2}')
[ -n "${DRV:-}" ] || { echo "no nvidia-smi: not a machine for this check"; exit 2; }
MAJOR="${DRV%%.*}"; SERIES="r$MAJOR"; note PASS "1 driver" "$NAME, driver $DRV ($(nvidia-smi 2> /dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/CUDA \1/p' | head -1))"

# 2. the version
v="$("$G" --version 2>&1)"; code=$?
if [ "$code" = 0 ] && echo "$v" | grep -q '^glyd ' && echo "$v" | grep -q 'engine: linked'; then note PASS "2 --version" "$(echo "$v" | head -1)"; else note FAIL "2 --version" "exit $code: $(echo "$v" | head -2 | tr '\n' ' ')"; fi

# 3. doctor: the library the driver can run, and nothing marked NO
d="$("$G" doctor 2>&1)"; code=$?; echo "$d" > "$OUT/doctor.txt"
lib="$(echo "$d" | grep -i 'GPU library')"
ok=1
[ "$(echo "$d" | tail -1)" = "Everything glyd run needs is here." ] || ok=0
if [ "$MAJOR" -lt 580 ]; then echo "$lib" | grep -q 'libglyd_gpu_cuda12.so' && ! echo "$lib" | grep -q 'cuda13' || ok=0; fi
echo "$d" | grep -qE '(^|[[:space:]])NO([[:space:]]|$)' && ok=0
if [ "$ok" = 1 ] && [ "$code" = 0 ]; then note PASS "3 doctor" "$(echo "$lib" | head -1 | cut -c1-120)"; else note FAIL "3 doctor" "exit $code; library line: ${lib:-none}; last line: $(echo "$d" | tail -1) (full output in $OUT/doctor.txt)"; fi

# 4. a first answer
"$G" run "$MODEL" --prompt "Say hello in five words." --temp 0 --no-think > "$OUT/run.out" 2> "$OUT/run.err"; code=$?
if [ "$code" = 0 ] && [ -n "$(tr -d '[:space:]' < "$OUT/run.out")" ] && ! cat "$OUT/run.out" "$OUT/run.err" | grep -qiE 'driver cannot run|no kernel image|no code for'; then note PASS "4 run" "$(tr '\n' ' ' < "$OUT/run.out" | cut -c1-100)"
else note FAIL "4 run" "exit $code: $(cat "$OUT/run.out" "$OUT/run.err" | grep -iE 'kernel image|invalid|driver cannot|no code for|error' | head -2 | tr '\n' ' ') (full output in $OUT/run.out, run.err)"; fi

# 5. the same answer with CUDA 12 forced, on a driver of 580 or newer (below it the unforced run is already CUDA 12)
if [ "$MAJOR" -ge 580 ] && [ -f "$(dirname "$G")/libglyd_gpu_cuda12.so" ]; then
  GLYD_ENGINE_CUDA=12 GLYD_GPU_LIB="$(dirname "$G")/libglyd_gpu_cuda12.so" "$G" run "$MODEL" --prompt "Say hello in five words." --temp 0 --no-think > "$OUT/run12.out" 2> "$OUT/run12.err"
  if cmp -s "$OUT/run.out" "$OUT/run12.out"; then note PASS "5 cuda12 = unforced" "the same answer"; else note FAIL "5 cuda12 = unforced" "the answers differ (run12.out)"; fi
else note SKIP "5 cuda12 = unforced" "driver $DRV is below 580: the unforced run is the CUDA 12 one"; fi

# 6. the server
"$G" serve "$MODEL" --host 127.0.0.1 --port "$PORT" > "$OUT/serve.log" 2>&1 & SRV=$!
for _ in $(seq 1 600); do "$CURL" -sf -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null && break; kill -0 $SRV 2> /dev/null || break; sleep 1; done
http="$("$CURL" -s -m 120 -o "$OUT/chat.json" -w '%{http_code}' -H 'content-type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16,"temperature":0}' "http://127.0.0.1:$PORT/v1/chat/completions")"
kill $SRV 2> /dev/null; wait $SRV 2> /dev/null
if [ "$http" = 200 ] && grep -q '"content"' "$OUT/chat.json"; then note PASS "6 serve" "HTTP 200: $(tr -d '\n' < "$OUT/chat.json" | cut -c1-90)"; else note FAIL "6 serve" "HTTP $http (serve.log: $(tail -1 "$OUT/serve.log" | cut -c1-100))"; fi

# 7. a driver below R535 gets one "too old:" line
if [ "$MAJOR" -lt 535 ]; then
  d="$("$G" doctor 2>&1)"; code=$?
  if [ "$code" != 0 ] && echo "$d" | grep -q 'too old:' && echo "$d" | grep -q 'driver 535 or newer'; then note PASS "7 too old" "$(echo "$d" | grep -m1 'too old:' | cut -c1-120)"; else note FAIL "7 too old" "exit $code: $(echo "$d" | head -2 | tr '\n' ' ')"; fi
else note SKIP "7 too old" "no driver below R535 here"; fi

S=PASS; [ "$FAILS" = 0 ] || S=FAIL
printf '%s\tdriver-%s\t%s %s; %s\n' "$S" "$SERIES" "$NAME" "$DRV" "$DETAIL" > "$OUT/driver-$SERIES.result"
echo "---"; echo "$S: $OUT/driver-$SERIES.result"
[ "$FAILS" = 0 ]
