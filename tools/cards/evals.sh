#!/usr/bin/env bash
# A model card's task benchmarks (GSM8K, MMLU-Pro through `glyd serve`, Inspect AI, non-thinking), into cards/<model>-<level>.json under "evals".
#   bash tools/cards/evals.sh [--dry-run] [--as-bf16-for CARD.json] MODEL REVISION LEVEL CLASS [--shards N] [--limit N]    e.g.  glyd/Qwen3.8-27B-linnet 690a3ff5... linnet a6000 --shards 8
# TASKS=gsm8k (env): one benchmark only. One pod per shard (every N-th question of each task), through rp_class.sh; the shards' results are merged weighted by their question counts. --limit: at most N questions per task per shard.
# The bf16 baseline is the same command on the source model at LEVEL bf16 (stock vLLM serves it: the engine's lossless path runs Qwen3 only so far) with --as-bf16-for cards/X-linnet.json: stored there as "evals_bf16".
# --dry-run starts nothing (no runpodctl) and prints the plan and the worst-case cost.
set -u
DRY=0; SHARDS=1; LIMIT=0; BF16=""; POS=()
while [ $# -gt 0 ]; do case $1 in --dry-run) DRY=1; shift;; --shards) SHARDS=$2; shift 2;; --limit) LIMIT=$2; shift 2;; --as-bf16-for) BF16=$2; shift 2;; *) POS+=("$1"); shift;; esac; done
[ ${#POS[@]} -eq 4 ] || { sed -n 2,6p "$0"; exit 2; }
MODEL=${POS[0]}; REV=${POS[1]}; LEVEL=${POS[2]}; C=${POS[3]}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
. tools/runpod/classes.sh
case $LEVEL in bf16|penguin|kestrel|linnet) ;; *) echo "LEVEL is bf16, penguin, kestrel or linnet"; exit 2;; esac
class_types "$C" > /dev/null || { echo "unknown GPU class $C"; exit 2; }
[[ $SHARDS =~ ^[1-9][0-9]*$ && $LIMIT =~ ^[0-9]+$ ]] || { echo "--shards and --limit are whole numbers"; exit 2; }
BASE=$(basename "$MODEL"); BASE=${BASE%-$LEVEL}; CARD=${BF16:-cards/$BASE-$LEVEL.json}; KEY=evals; [ -n "$BF16" ] && KEY=evals_bf16
NAME=eval-$BASE-$LEVEL${BF16:+-bf16}; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
COMMIT=$(git rev-parse --short HEAD); JOB=tools/cards/eval_job.sh
MAX_MIN=${MAX_MIN:-150}   # (rp_run.sh's cap)

if [ $DRY = 1 ]; then
  p=$(class_price "$C"); e=$(awk -v p="$p" -v m="$MAX_MIN" -v s="$SHARDS" 'BEGIN{printf "%.2f", p*m/60*s}')
  echo "evals: $MODEL @ $REV, level $LEVEL (commit $COMMIT) -> $CARD [$KEY]   job: $JOB (built from git archive HEAD)"
  echo "tasks: gsm8k (1319 questions, openai/gsm8k), mmlu_pro (12032 questions, TIGER-Lab/MMLU-Pro); non-thinking, Inspect AI, glyd-engine serve --users 8"
  if [ "$LIMIT" = 0 ]; then echo "per pod: shard i/$SHARDS = $(( (1319 + SHARDS - 1) / SHARDS )) gsm8k + $(( (12032 + SHARDS - 1) / SHARDS )) mmlu_pro questions"; else echo "per pod: at most $LIMIT questions of each task"; fi
  echo "pods: $SHARDS x class $C (try in order: $(class_types "$C" | paste -sd, -)) \$$p/h, at most $MAX_MIN min each = \$$e"
  echo "worst case total: \$$e (pods end when the job does)"
  exit 0
fi

mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/eval.XXXXXX"); trap 'rm -rf "$W"' EXIT
git archive --format=tar.gz --prefix=src/ HEAD > "$W/src.tar"
for i in $(seq 1 "$SHARDS"); do
  mkdir -p "$W/up$i"   # (rp_run.sh uploads each file under its own name)
  printf 'MODEL=%s\nREV=%s\nLEVEL=%s\nSHARD=%s/%s\nLIMIT=%s\nTASKS=%s\n' "$MODEL" "$REV" "$LEVEL" "$i" "$SHARDS" "$LIMIT" "${TASKS:-}" > "$W/up$i/eval.env"
  ( NAME=$NAME-$i MAX_MIN=$MAX_MIN UPLOAD="$W/src.tar $W/up$i/eval.env" bash tools/runpod/rp_class.sh "$C" "$JOB" > "$RUNS/$NAME-$i.log" 2>&1; echo $? > "$W/rc-$i" ) &
done
wait
OK=(); for i in $(seq 1 "$SHARDS"); do
  echo "== shard $i/$SHARDS: pod exit $(cat "$W/rc-$i" 2> /dev/null) (log $RUNS/$NAME-$i.log)"
  [ "$(cat "$W/rc-$i")" = 0 ] && [ -f "$RUNS/results-$NAME-$i/evals.json" ] && OK+=("$RUNS/results-$NAME-$i")
done
[ ${#OK[@]} = "$SHARDS" ] || { echo "not every shard has an evals.json: the card is not changed"; exit 1; }
python3 tools/cards/merge.py "$CARD" --evals "${OK[@]}" --key "$KEY" && cat "$CARD"
