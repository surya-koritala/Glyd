#!/usr/bin/env bash
# One command for a model card's numbers (GLY-31): speed, first-token time, GPU memory at a 4k context and the 32k fit on each GPU class, from the engine built
# from this checkout's HEAD, into cards/<model>-<level>.json (merged with what is there: one entry per GPU, replaced on a rerun).
#   bash tools/cards/numbers.sh [--kl] [--dry-run] MODEL REVISION LEVEL CLASS...        e.g.  Qwen/Qwen3.8-27B 1d4bf0f2... linnet a6000 l40 a100
# LEVEL: penguin, kestrel or linnet. CLASS: 4090 3090 16gb l4 a6000 l40 a100 (tools/runpod/classes.sh). One pod per class, in parallel, through rp_class.sh.
# --kl: one more A100 pod scores the level against bf16 with research/lossy-2026/kld_study.py (kestrel int:6:32:mse:bf16, linnet int:5:32:mse:bf16; penguin is lossless: none).
# kl_fp8_ref is only from the table below (measured earlier); null for any other model. --dry-run starts nothing (no runpodctl) and prints the plan.
set -u
KL=0; DRY=0
while [ "${1:-}" ]; do case $1 in --kl) KL=1; shift;; --dry-run) DRY=1; shift;; *) break;; esac; done
[ $# -ge 4 ] || { sed -n 2,8p "$0"; exit 2; }
MODEL=$1; REV=$2; LEVEL=$3; shift 3; CLASSES=("$@")
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
. tools/runpod/classes.sh
case $LEVEL in penguin|kestrel|linnet) ;; *) echo "LEVEL is penguin, kestrel or linnet"; exit 2;; esac
for c in "${CLASSES[@]}" $([ $KL = 1 ] && echo a100); do class_types "$c" > /dev/null || { echo "unknown GPU class $c"; exit 2; }; done
case $LEVEL in linnet) VARIANT=int:5:32:mse:bf16;; kestrel) VARIANT=int:6:32:mse:bf16;; *) VARIANT="";; esac
[ $KL = 1 ] && [ -z "$VARIANT" ] && { echo "--kl: penguin is lossless, nothing to score"; exit 2; }
# the KL of the FP8 reference checkpoint, measured earlier (never guessed: null for a model not here)
# (MODEL may be the published glyd/NAME-LEVEL repo, measured as users download it; the KL is scored on the bf16 model: KL_MODEL, KL_REV)
KLM=${KL_MODEL:-$MODEL}; KLR=${KL_REV:-$REV}
case $KLM in Qwen/Qwen3.8-27B) FP8_REF=0.00465;; *) FP8_REF="";; esac
BASE=$(basename "$MODEL"); BASE=${BASE%-$LEVEL}; CARD=cards/$BASE-$LEVEL.json
NAME=card-$BASE-$LEVEL; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
COMMIT=$(git rev-parse --short HEAD); GLYD=$(sed -n 's/^version = "\(.*\)"/\1/p' glyd-engine/Cargo.toml | head -1)
JOB=tools/cards/numbers_job.sh; KLJOB=tools/cards/kl_job.sh; KLFILES="research/lossy-2026/kld_study.py research/lossy-2026/prepare_quality.py research/lossy-2026/matched_quality.py"
MAX_MIN=${MAX_MIN:-60}

if [ $DRY = 1 ]; then
  total=0
  echo "card: $MODEL @ $REV, level $LEVEL (glyd $GLYD, commit $COMMIT) -> $CARD   job: $JOB (built from git archive HEAD)"
  for c in "${CLASSES[@]}"; do
    p=$(class_price "$c"); e=$(awk -v p="$p" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", p*m/60}'); total=$(awk -v t="$total" -v e="$e" 'BEGIN{printf "%.2f", t+e}')
    echo "pod: class $c, try in order: $(class_types "$c" | paste -sd, -)  \$$p/h, at most $MAX_MIN min = \$$e"
  done
  if [ $KL = 1 ]; then
    p=$(class_price a100); e=$(awk -v p="$p" 'BEGIN{printf "%.2f", p*90/60}'); total=$(awk -v t="$total" -v e="$e" 'BEGIN{printf "%.2f", t+e}')
    echo "pod: class a100 for the KL ($VARIANT) \$$p/h, at most 90 min = \$$e   job: $KLJOB   kl_fp8_ref: ${FP8_REF:-null}"
  fi
  echo "worst case total: \$$total (pods end when the job does; a card is usually a third of that)"
  echo "--- $JOB"; cat "$JOB"
  exit 0
fi

mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/card.XXXXXX"); trap 'rm -rf "$W"' EXIT
git archive --format=tar.gz --prefix=src/ HEAD > "$W/src.tar"   # (the job's tar -xf reads gzip too)
printf 'MODEL=%s\nREV=%s\nLEVEL=%s\n' "$MODEL" "$REV" "$LEVEL" > "$W/numbers.env"
for c in "${CLASSES[@]}"; do
  ( NAME=$NAME-$c MAX_MIN=$MAX_MIN UPLOAD="$W/src.tar $W/numbers.env" bash tools/runpod/rp_class.sh "$c" "$JOB" > "$RUNS/$NAME-$c.log" 2>&1; echo $? > "$W/rc-$c" ) &
done
if [ $KL = 1 ]; then
  printf 'MODEL=%s\nREV=%s\nVARIANT=%s\n' "$KLM" "$KLR" "$VARIANT" > "$W/kl.env"
  ( NAME=$NAME-kl MAX_MIN=90 UPLOAD="$W/kl.env $KLFILES" bash tools/runpod/rp_class.sh a100 "$KLJOB" > "$RUNS/$NAME-kl.log" 2>&1; echo $? > "$W/rc-kl" ) &
fi
wait
KLARG=(); [ $KL = 1 ] && [ -f "$RUNS/results-$NAME-kl/kl.jsonl" ] && KLARG=(--kl-jsonl "$RUNS/results-$NAME-kl/kl.jsonl")
[ -n "$FP8_REF" ] && KLARG+=(--kl-fp8-ref "$FP8_REF")
for c in "${CLASSES[@]}"; do
  echo "== $c: pod exit $(cat "$W/rc-$c" 2> /dev/null) (log $RUNS/$NAME-$c.log)"
  [ "$(cat "$W/rc-$c")" = 0 ] && python3 tools/cards/merge.py "$CARD" --results "$RUNS/results-$NAME-$c" --glyd "$GLYD" --commit "$COMMIT" ${KLARG[@]+"${KLARG[@]}"}
done
[ $KL = 1 ] && echo "== kl: pod exit $(cat "$W/rc-kl" 2> /dev/null)"
[ -f "$CARD" ] && cat "$CARD"
