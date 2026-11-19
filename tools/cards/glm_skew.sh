#!/usr/bin/env bash
# Expert-routing skew of a MoE model (Linear GLY-127): one 4-GPU Runpod pod, forward passes over six text domains (~295k tokens + ~49k held out) with a hook on every
# router, ~20 short decodes, then hit-rate tables (static profile, LRU/LFU replay). Results: $GLYD_RUNS/results-skew-MODEL/ (skew.md, skew.json, steps.txt, counts/, decode.npz, routing/).
#   bash tools/cards/glm_skew.sh [--dry-run] glm|nemotron        CLASS=glmh to try only Hopper cards for glm; MAX_MIN (at most 150: rp_run.sh's cap)
# glm      zai-org/GLM-5.3-Flash, FP8 328 GB, 4 x RTX PRO 6000 96 GB (class glm; fallbacks H100 NVL, H200 NVL, H200). 4 x 80 GB (320 GB) does not hold it.
# nemotron NVIDIA-Nemotron-3-Super-120B-A12B-BF16, 247 GB, 4 x A100 80 GB (class nemo; fallbacks H100 PCIe, H100 SXM).
# Needs ~/.runpod/api_key and runpodctl on this machine (not for --dry-run, which starts nothing). Optional env, passed to the job: CAL_SEQS HELD_SEQS BATCH GEN_NEW JOB_MAX_MIN FLA.
set -u
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
MODEL=${1:-}; case $MODEL in glm|nemotron) ;; *) sed -n 2,6p "$0"; exit 2;; esac
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
PY=research/lossy-2026/glm_skew.py; JOB=tools/cards/glm_skew_job.sh
eval "$(python3 $PY plan "$MODEL")"   # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB (one source: glm_skew.py MODELS)
. tools/runpod/classes.sh
CLASS=${CLASS:-$POD_CLASS}; class_types "$CLASS" > /dev/null || { echo "unknown GPU class $CLASS"; exit 2; }
GPUS=$(class_gpus "$CLASS"); NAME=skew-$MODEL; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
MAX_MIN=${MAX_MIN:-150}; CAPPED=""; [ "$MAX_MIN" -le 150 ] || { CAPPED=" (asked $MAX_MIN: rp_run.sh cuts every pod to 150, the owner's limit)"; MAX_MIN=150; }
export RP_CUDA=${RP_CUDA:-12.8} MIN_DRIVER=${MIN_DRIVER:-570}   # the job installs torch for CUDA 12.8, which runs on any driver >= 570 (more hosts than 13.0/580)
APPROVED=40

if [ $DRY = 1 ]; then
  echo "skew: $MODEL = $REPO @ $REV"
  echo "  $N_MOE MoE layers x $NE experts, top-$K per token; cache sizes $CACHE_PCTS % of the experts (per layer)"
  echo "text: 6 domains (wiki, code, chat, math, multi, reason) x 24 x 2048 = 294,912 calibration tokens; 6 x 4 x 2048 = 49,152 held-out tokens; 20 decodes of up to 128 tokens (thinking off)"
  echo "pod: class $CLASS x $GPUS GPUs, container disk $DISK GB ($WEIGHTS_GB GB of weights + venv + data), RP_CUDA $RP_CUDA, MIN_DRIVER $MIN_DRIVER"
  while IFS= read -r t; do
    p=$(type_price "$t"); h=$(awk -v p="$p" -v g="$GPUS" 'BEGIN{printf "%.2f", p*g}'); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
    echo "  try in order: $t x$GPUS  \$$p each = \$$h/h, worst case \$$w$(awk -v w="$w" -v a="$APPROVED" 'BEGIN{if (w > a) printf "  (over the approved $%d)", a}')"
  done <<< "$(class_types "$CLASS")"
  h=$(class_price "$CLASS"); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
  echo "time: MAX_MIN $MAX_MIN$CAPPED; the job stops new work at JOB_MAX_MIN ${JOB_MAX_MIN:-120} min (a typical run is 60-100 min)"
  echo "worst case with the first choice: \$$h/h x $MAX_MIN min = \$$w (approved \$$APPROVED)"
  echo "job steps (each timed in results/steps.txt): pyenv (torch cu128, transformers 5.19.0, kernels, fla) | selftest | download (hf, hf_xet) + prep (pinned datasets) | trace (load, prefill, decode) | analyze | collect"
  echo "results: $RUNS/results-$NAME/   (skew.md, skew.json, steps.txt, timing.json, counts/, decode.npz, routing/ raw ids, logs)"
  echo "note: a running rp_guard.sh deletes glyd-* pods older than its MAX_AGE (default 60 min): start it with MAX_AGE=160, or stop it, before a launch"
  echo "launch: bash tools/cards/glm_skew.sh $MODEL     (runs: NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK RP_CUDA=$RP_CUDA MIN_DRIVER=$MIN_DRIVER UPLOAD=\"<skew.env> $PY\" bash tools/runpod/rp_class.sh $CLASS $JOB; a dry run starts nothing)"
  exit 0
fi

[ -s ~/.runpod/api_key ] || { echo "no ~/.runpod/api_key"; exit 2; }
pgrep -f rp_guard.sh > /dev/null && echo "WARNING: rp_guard.sh is running; it deletes glyd-* pods older than its MAX_AGE (default 60 min, this pod runs up to $MAX_MIN): it needs MAX_AGE=160"
mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/skew.XXXXXX"); trap 'rm -rf "$W"' EXIT
{ echo "MODEL=$MODEL"; for v in CAL_SEQS HELD_SEQS BATCH GEN_NEW JOB_MAX_MIN FLA; do [ -n "${!v:-}" ] && echo "$v=${!v}"; done; } > "$W/skew.env"
NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK UPLOAD="$W/skew.env $ROOT/$PY" bash tools/runpod/rp_class.sh "$CLASS" "$JOB" 2>&1 | tee "$RUNS/$NAME.log"
rc=${PIPESTATUS[0]}
echo "== pod exit $rc (log $RUNS/$NAME.log, results $RUNS/results-$NAME)"
[ -f "$RUNS/results-$NAME/steps.txt" ] && cat "$RUNS/results-$NAME/steps.txt"
[ -f "$RUNS/results-$NAME/skew.md" ] && cat "$RUNS/results-$NAME/skew.md"
exit "$rc"
