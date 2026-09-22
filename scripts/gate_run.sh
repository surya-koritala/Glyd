#!/usr/bin/env bash
# The store's gate, run on the machine holding the corpus: every object
# put through the store into S3, in name order; zstd -3's bytes for the
# same objects; every object read back and compared byte for byte; the
# metadata directory deleted and rebuilt from the bucket, then verified.
# Wall time and rate of each phase. Results in <results>/gate.txt, the
# per-object put lines in <results>/put.log.
#   scripts/gate_run.sh <corpus dir> <s3://bucket/prefix> <work dir> <results dir>
set -uo pipefail
CORPUS="$1"; S3="$2"; WORK="$3"; RESULTS="$4"
GS="${GLYD_STORE:-glyd-store}"
META="$WORK/meta"; BACK="$WORK/back"
mkdir -p "$RESULTS" "$WORK"
rm -rf "$META" "$BACK"; mkdir -p "$BACK"
OUT="$RESULTS/gate.txt"
now() { date +%s.%N; }
rate() { echo "$1 / ($3 - $2) / 1000000" | bc -l | cut -c1-7; } # bytes t0 t1 -> MB/s
FILES=$(ls -1 "$CORPUS" | grep -v '\.part$' | sort)
N=$(echo "$FILES" | wc -l | tr -d ' ')
size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }
RAW=0; for f in $FILES; do RAW=$((RAW + $(size "$CORPUS/$f"))); done
{
    echo "date: $(date -u +%FT%TZ)"
    echo "machine: $(lscpu | grep -m1 -E 'Model name|BIOS Model name' | cut -d: -f2 | sed 's/^ *//') ($(nproc) vCPU, $(free -g | awk '/Mem:/ {print $2}') GB)"
    echo "glyd-store: $($GS --version 2>&1 | head -1)"
    echo "corpus: $N objects, $RAW bytes"
} > "$OUT"

# zstd -3, every object alone, all cores.
t0=$(now); Z=0
for f in $FILES; do Z=$((Z + $(zstd -3 -T0 -q -c "$CORPUS/$f" | wc -c))); done
t1=$(now)
echo "zstd-3 alone: $Z bytes, $(rate $RAW $t0 $t1) MB/s" >> "$OUT"

# Put.
t0=$(now)
(cd "$CORPUS" && $GS "$META" --s3 "$S3" --put $FILES) > "$RESULTS/put.log" 2>&1
PUT_RC=$?
t1=$(now)
STATS=$($GS "$META" --s3 "$S3" --stats 2>&1 | tail -1)
echo "put: rc $PUT_RC, $(rate $RAW $t0 $t1) MB/s, $(echo "$t1 - $t0" | bc) s; $STATS" >> "$OUT"
DELTAS=$(grep -c 'delta against' "$RESULTS/put.log")
echo "put: $DELTAS of $N objects as deltas" >> "$OUT"

# Get every object back, compared with the original.
t0=$(now); BAD=0; GOT=0
while IFS=$'\t' read -r id name; do
    if $GS "$META" --s3 "$S3" --get "$id" -o "$BACK/obj" 2>>"$RESULTS/get.err" && cmp -s "$BACK/obj" "$CORPUS/$name"; then
        GOT=$((GOT + 1))
    else
        BAD=$((BAD + 1)); echo "MISMATCH $id $name" >> "$RESULTS/get.err"
    fi
done < <(grep -v '^D' "$META/index" | grep -v $'\tpack of ' | cut -f1,7 | sort -n -u)
rm -f "$BACK/obj"
t1=$(now)
echo "get: $GOT byte-exact, $BAD failed, $(rate $RAW $t0 $t1) MB/s, $(echo "$t1 - $t0" | bc) s" >> "$OUT"

# Rebuild from the bucket, then verify.
cp "$META/index" "$RESULTS/index.before"
rm -rf "$META"
t0=$(now)
$GS "$META" --s3 "$S3" --rebuild > "$RESULTS/rebuild.log" 2>&1
t1=$(now)
cmp -s <(sort "$RESULTS/index.before") <(sort "$META/index") && SAME=same || SAME=DIFFERENT
echo "rebuild: $(tail -1 "$RESULTS/rebuild.log"); index $SAME as before; $(echo "$t1 - $t0" | bc) s" >> "$OUT"
t0=$(now)
$GS "$META" --s3 "$S3" --verify > "$RESULTS/verify.log" 2>&1
t1=$(now)
echo "verify: $(tail -1 "$RESULTS/verify.log"); $(echo "$t1 - $t0" | bc) s" >> "$OUT"
cat "$OUT"
