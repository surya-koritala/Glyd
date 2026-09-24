#!/usr/bin/env bash
# The store's gate, run on the machine holding the corpus, with zstd -3
# through the same S3 client as the baseline:
#   zstd put:    every object compressed (zstd -3, all cores) and uploaded
#   store put:   every object through the store into S3, in name order
#   zstd get:    every object downloaded and decompressed, one at a time
#   store get:   every object read back, one process per object (a cold
#                reader: its chain decoded from the bucket)
#   restore:     every object read back by one process (a warm reader)
# Every read is compared byte for byte with its original, outside the
# timing. With GATE_REBUILD=1 the metadata directory is then deleted,
# rebuilt from the bucket and verified. Results in <results>/gate.txt.
#   scripts/gate_run.sh <corpus dir> <s3://bucket/prefix> <work dir> <results dir>
set -uo pipefail
CORPUS="$1"; S3="$2"; WORK="$3"; RESULTS="$4"
GS="${GLYD_STORE:-glyd-store}"
S3CP="${S3CP:-$(dirname "$GS")/examples/s3cp}"
META="$WORK/meta"; BACK="$WORK/back"
mkdir -p "$RESULTS" "$WORK"
rm -rf "$META" "$BACK"; mkdir -p "$BACK"
OUT="$RESULTS/gate.txt"
now() { date +%s.%N; }
rate() { echo "$1 / $2 / 1000000" | bc -l | cut -c1-7; } # bytes seconds -> MB/s
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

# zstd -3: compress and upload, every object.
t=0; Z=0; i=0
for f in $FILES; do
    t0=$(now)
    zstd -3 -T0 -q -c "$CORPUS/$f" > "$BACK/z" && "$S3CP" put "$BACK/z" "$S3/zstd/$i"
    t1=$(now)
    t=$(echo "$t + $t1 - $t0" | bc); Z=$((Z + $(size "$BACK/z"))); i=$((i + 1))
done
rm -f "$BACK/z"
echo "zstd-3 put: $Z bytes stored, $(rate $RAW $t) MB/s, $t s (compress and upload)" >> "$OUT"

# The store: put.
t0=$(now)
(cd "$CORPUS" && $GS "$META" --s3 "$S3/store" --put $FILES) > "$RESULTS/put.log" 2>&1
PUT_RC=$?
t1=$(now)
STATS=$($GS "$META" --s3 "$S3/store" --stats 2>&1 | tail -1)
PUT_BYTES=$(awk '{s+=$2} END {print s+0}' "$RESULTS/put.log")
echo "store put: rc $PUT_RC, $(rate $PUT_BYTES $(echo "$t1 - $t0" | bc)) MB/s, $(echo "$t1 - $t0" | bc) s; $STATS" >> "$OUT"
echo "store put: $(grep -c 'delta against' "$RESULTS/put.log") of $N objects as deltas" >> "$OUT"

# zstd -3: download and decompress, every object.
t=0; BAD=0; i=0
for f in $FILES; do
    t0=$(now)
    "$S3CP" get "$S3/zstd/$i" "$BACK/z" && zstd -d -q -c "$BACK/z" > "$BACK/obj"
    t1=$(now)
    t=$(echo "$t + $t1 - $t0" | bc)
    cmp -s "$BACK/obj" "$CORPUS/$f" || BAD=$((BAD + 1))
    i=$((i + 1))
done
rm -f "$BACK/z" "$BACK/obj"
echo "zstd-3 get: $BAD failed, $(rate $RAW $t) MB/s, $t s (download and decompress, one object at a time)" >> "$OUT"

# The store: every object by a process of its own.
LIST=$(grep -v '^D' "$META/index" | grep -v $'\tpack of ' | awk -F'\t' '{print $1 "\t" $NF}' | sort -n -u)
GET_BYTES=$(grep -v '^D' "$META/index" | grep -v $'\tpack of ' | sort -n -u | awk -F'\t' '{s+=$4} END {print s+0}')
t=0; BAD=0; GOT=0
while IFS=$'\t' read -r id name; do
    t0=$(now)
    $GS "$META" --s3 "$S3/store" --get "$id" -o "$BACK/obj" 2>>"$RESULTS/get.err"
    t1=$(now)
    t=$(echo "$t + $t1 - $t0" | bc)
    if cmp -s "$BACK/obj" "$CORPUS/$name"; then GOT=$((GOT + 1)); else BAD=$((BAD + 1)); echo "MISMATCH $id $name" >> "$RESULTS/get.err"; fi
done <<< "$LIST"
rm -f "$BACK/obj"
echo "store get: $GOT byte-exact, $BAD failed, $(rate $GET_BYTES $t) MB/s, $t s (one process per object)" >> "$OUT"

# The store: every object by one process.
t0=$(now)
$GS "$META" --s3 "$S3/store" --restore "$BACK/all" > "$RESULTS/restore.txt" 2>"$RESULTS/restore.log"
t1=$(now)
BAD=0; GOT=0
while IFS=$'\t' read -r id name; do
    if cmp -s "$BACK/all/$id" "$CORPUS/$name"; then GOT=$((GOT + 1)); else BAD=$((BAD + 1)); echo "RESTORE MISMATCH $id $name" >> "$RESULTS/get.err"; fi
done < "$RESULTS/restore.txt"
rm -rf "$BACK/all"
echo "store restore: $GOT byte-exact, $BAD failed, $(rate $GET_BYTES $(echo "$t1 - $t0" | bc)) MB/s, $(echo "$t1 - $t0" | bc) s (one process)" >> "$OUT"

# Gzip objects opened: this machine's gzip (GNU on Linux) on three
# objects, then glyd on the gzip, decoded and compared.
GLYD="${GLYD:-$(dirname "$GS")/glyd}"
if [ -x "$GLYD" ]; then
    for f in $(echo "$FILES" | grep -m1 gharchive) $(echo "$FILES" | grep -m1 simplewiki.*categorylinks) $(echo "$FILES" | grep -m1 'linux-6.6'); do
        [ -n "$f" ] || continue
        head -c 536870912 "$CORPUS/$f" | gzip -6 > "$BACK/x.gz"
        case "$f" in *.json|*.sql) opt="-r" ;; *) opt="" ;; esac
        t0=$(now); "$GLYD" --max $opt "$BACK/x.gz" -o "$BACK/x.g" 2>/dev/null; t1=$(now)
        "$GLYD" -d "$BACK/x.g" -o "$BACK/x.back" 2>/dev/null
        cmp -s "$BACK/x.back" "$BACK/x.gz" && ok=exact || ok=DIFFERENT
        echo "gzip-inside $f ($(gzip --version | head -1)): gzip-6 $(size "$BACK/x.gz") -> glyd --max $opt $(size "$BACK/x.g") B, $ok, $(echo "$t1 - $t0" | bc | cut -c1-6) s" >> "$OUT"
    done
    rm -f "$BACK/x.gz" "$BACK/x.g" "$BACK/x.back"
fi

if [ "${GATE_REBUILD:-0}" = 1 ]; then
    cp "$META/index" "$RESULTS/index.before"
    rm -rf "$META"
    t0=$(now)
    $GS "$META" --s3 "$S3/store" --rebuild > "$RESULTS/rebuild.log" 2>&1
    t1=$(now)
    cmp -s <(sort "$RESULTS/index.before") <(sort "$META/index") && SAME=same || SAME=DIFFERENT
    echo "rebuild: $(tail -1 "$RESULTS/rebuild.log"); index $SAME as before; $(echo "$t1 - $t0" | bc) s" >> "$OUT"
    t0=$(now)
    $GS "$META" --s3 "$S3/store" --verify > "$RESULTS/verify.log" 2>&1
    t1=$(now)
    echo "verify: $(tail -1 "$RESULTS/verify.log"); $(echo "$t1 - $t0" | bc) s" >> "$OUT"
fi
cat "$OUT"
