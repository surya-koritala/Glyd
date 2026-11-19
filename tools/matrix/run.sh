#!/usr/bin/env bash
# The release matrix (GLY-33): a fresh install of release VERSION (the tag's install.sh) on each GPU class, `glyd run MODEL:LEVEL --no-think` at a 4k and a 32k context
# for every model x level, into benchmarks/matrix/VERSION/README.md (a table by GPU class and model:level) with the raw logs. One pod a class, all its cells in it, in parallel.
#   bash tools/matrix/run.sh VERSION [--models "Qwen/Qwen3.5-9B Qwen/Qwen3.8-27B"] [--levels "penguin kestrel swift"] [--classes "4090 3090 16gb l4"] [--dry-run]
# The tag v<VERSION> must be public (install.sh is fetched from it). --dry-run starts nothing (no runpodctl) and prints the plan with its cost.
set -u
[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
VERSION=$1; shift
MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.8-27B"; LEVELS="penguin kestrel swift"; CLASSES="4090 3090 16gb l4"; DRY=0
while [ "${1:-}" ]; do case $1 in
  --models) MODELS=$2; shift 2;; --levels) LEVELS=$2; shift 2;; --classes) CLASSES=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown $1"; exit 2;; esac; done
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
. tools/runpod/classes.sh
PROMPT="Name the planets of the solar system in order, one line."
MAX_MIN=${MAX_MIN:-75}; JOB=tools/matrix/matrix_job.sh; OUT=benchmarks/matrix/$VERSION; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
for c in $CLASSES; do class_types "$c" > /dev/null || { echo "unknown GPU class $c"; exit 2; }; done
if [ $DRY = 1 ]; then
  total=0; cells=$(( $(echo $MODELS | wc -w) * $(echo $LEVELS | wc -w) * 2 ))
  echo "matrix: glyd $VERSION (install.sh from tag v$VERSION), models: $MODELS; levels: $LEVELS; $cells cells a pod (4k and 32k); prompt: \"$PROMPT\""
  for c in $CLASSES; do
    p=$(class_price "$c"); e=$(awk -v p="$p" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", p*m/60}'); total=$(awk -v t="$total" -v e="$e" 'BEGIN{printf "%.2f", t+e}')
    echo "pod: class $c, try in order: $(class_types "$c" | paste -sd, -)  \$$p/h, cap $MAX_MIN min = \$$e"
  done
  echo "worst case total (every pod to its cap): \$$total   job: $JOB   output: $OUT/README.md + $OUT/<class>/ logs"
  echo "per cell: glyd run MODEL:LEVEL --no-think --context 4096|32768 -p PROMPT   (a cell's cap 25 min; the job's cap $((MAX_MIN - 5)) min)"
  exit 0
fi
mkdir -p "$RUNS" "$OUT"; W=$(mktemp -d "$RUNS/matrix.XXXXXX"); trap 'rm -rf "$W"' EXIT
printf 'VERSION=%s\nMODELS="%s"\nLEVELS="%s"\nPROMPT="%s"\nCAP_MIN=%s\n' "$VERSION" "$MODELS" "$LEVELS" "$PROMPT" $((MAX_MIN - 5)) > "$W/matrix.env"
for c in $CLASSES; do
  ( NAME=matrix-$VERSION-$c MAX_MIN=$MAX_MIN UPLOAD="$W/matrix.env" bash tools/runpod/rp_class.sh "$c" "$JOB" > "$RUNS/matrix-$VERSION-$c.log" 2>&1; echo $? > "$W/rc-$c" ) &
done
wait
for c in $CLASSES; do
  echo "== $c: pod exit $(cat "$W/rc-$c" 2> /dev/null) (log $RUNS/matrix-$VERSION-$c.log)"
  [ -d "$RUNS/results-matrix-$VERSION-$c" ] && { rm -rf "${OUT:?}/$c"; cp -R "$RUNS/results-matrix-$VERSION-$c" "$OUT/$c"; rm -f "$OUT/$c/DONE"; }
done
python3 tools/matrix/table.py "$OUT" "$VERSION" > "$OUT/README.md" && echo "wrote $OUT/README.md"
