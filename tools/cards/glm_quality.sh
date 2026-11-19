#!/usr/bin/env bash
# Output quality of GLM-5.3-Flash with its routed experts rounded to 2-4 bits (Linear GLY-127): one 4-GPU Runpod pod, the FP8 model's next-token distribution on ~49k held-out tokens (six domains)
# against seven passes of the experts rounded (control, 6 variants): KL divergence, top-1 agreement and perplexity, overall and by domain. Results: $GLYD_RUNS/results-quality-glm/
# (quality.md, quality.json, results.jsonl and the files after each variant, counts/ = the routing counts in glm_skew.py's format, perpos/, steps.txt, logs).
#   bash tools/cards/glm_quality.sh [--dry-run]       CLASS=glmh to try only Hopper cards; MAX_MIN (at most 150: rp_run.sh's cap); VARIANTS="ref,mix13,int4" to run only some, in that order (names in the dry run; keep ref: it is the control that stops the job if the dequantised experts do not reproduce the FP8 model)
# zai-org/GLM-5.3-Flash, FP8 328 GB, 4 x RTX PRO 6000 96 GB (class glm; fallbacks H100 NVL, H200 NVL, H200). 4 x 80 GB (320 GB) does not hold it.
# Needs ~/.runpod/api_key and runpodctl on this machine (not for --dry-run, which starts nothing). Optional env, passed to the job: VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN FLA SELFTEST_GPU.
set -u
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
PY=research/lossy-2026/glm_quality.py; SKEW=research/lossy-2026/glm_skew.py; JOB=tools/cards/glm_quality_job.sh
eval "$(python3 $PY plan)"   # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB ORDER (one source: glm_skew.py MODELS, glm_quality.py VARIANTS)
. tools/runpod/classes.sh
CLASS=${CLASS:-$POD_CLASS}; class_types "$CLASS" > /dev/null || { echo "unknown GPU class $CLASS"; exit 2; }
GPUS=$(class_gpus "$CLASS"); NAME=quality-glm; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
MAX_MIN=${MAX_MIN:-150}; CAPPED=""; [ "$MAX_MIN" -le 150 ] || { CAPPED=" (asked $MAX_MIN: rp_run.sh cuts every pod to 150, the owner's limit)"; MAX_MIN=150; }
export RP_CUDA=${RP_CUDA:-12.8} MIN_DRIVER=${MIN_DRIVER:-570}   # the job installs torch for CUDA 12.8, which runs on any driver >= 570 (more hosts than 13.0/580)
APPROVED=${APPROVED:-40}   # (the skew job's figure: the owner approved this GPU spend, not a number)
JOB_MAX_MIN=${JOB_MAX_MIN:-$((MAX_MIN - 22))}; export JOB_MAX_MIN   # (22 min of the cap are the pod's start and the last pull; for plan --text, which marks the passes that would start too late)
CAL=${CAL_SEQS:-24}; HELD=${HELD_SEQS:-4}; CNT=${COUNT_SEQS:-12}; GSEQ=${GPTQ_SEQS:-24}

if [ $DRY = 1 ]; then
  echo "quality: $REPO @ $REV: KL divergence, top-1 agreement and perplexity of the model with its routed experts rounded to 2-4 bits, against the FP8 original"
  python3 $PY plan --text
  echo "text: glm_skew.py's held-out set, 6 domains (wiki, code, chat, math, multi, reason) x $HELD x 2048 = $((6 * HELD * 2048)) tokens, $((6 * HELD * 2047)) scored positions, KL over the FP8 model's top 128 tokens + a tail bucket; per-domain tables too"
  echo "      routing counts for the most-used experts: $((6 * CNT)) calibration sequences ($((6 * CNT * 2048)) tokens); GPTQ calibration: $((6 * GSEQ)) sequences ($((6 * GSEQ * 2048)) tokens), fewer if the pod's RAM is short; $CAL sequences a domain prepared"
  echo "design: the FP8 model (328 GB) stays on the $GPUS GPUs; a variant is one pass of all held-out sequences through the layers one after another, each MoE layer's experts read from the FP8 weights a chunk at a time,"
  echo "        rounded in bf16, applied to the tokens routed to them and dropped (no packed copy: 328 + 171 GB would not fit 384 GB); each layer sees the layers before it already rounded"
  echo "pod: class $CLASS x $GPUS GPUs, container disk $DISK GB ($WEIGHTS_GB GB of weights + venv + data), RP_CUDA $RP_CUDA, MIN_DRIVER $MIN_DRIVER"
  while IFS= read -r t; do
    p=$(type_price "$t"); h=$(awk -v p="$p" -v g="$GPUS" 'BEGIN{printf "%.2f", p*g}'); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
    echo "  try in order: $t x$GPUS  \$$p each = \$$h/h, worst case \$$w$(awk -v w="$w" -v a="$APPROVED" 'BEGIN{if (w > a) printf "  (over the $%d of the skew job)", a}')"
  done <<< "$(class_types "$CLASS")"
  h=$(class_price "$CLASS"); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
  echo "time: MAX_MIN $MAX_MIN$CAPPED; the job starts no pass that would not end 2 min before JOB_MAX_MIN $JOB_MAX_MIN min (a skipped or cut-off pass is recorded, the rest of the results stay)"
  echo "worst case with the first choice: \$$h/h x $MAX_MIN min = \$$w (approved \$$APPROVED); a run that gets all 7 passes is ~85-110 min of pod with its start, \$$(awk -v h="$h" 'BEGIN{printf "%.0f-%.0f", h*85/60, h*110/60}') (glm_skew.py's whole job was 16 min: download 2.5 min, load 1.8 min)"
  echo "job steps (each timed in results/steps.txt): venv + hub client | download (hf, hf_xet; in the background) | pyenv (torch cu128, transformers 5.19.0, kernels) | selftest (CPU, and the block codes and a tiny model split over the GPUs: SELFTEST_GPU=0 skips the GPU part) | fla | prep (pinned datasets) | run (load, check, FP8 reference, $ORDER) | report"
  echo "results: $RUNS/results-$NAME/   (quality.md and quality.json after every variant, results.jsonl, meta.json, counts/, perpos/ KL of every position, steps.txt, versions.txt, run.log)"
  echo "note: a running rp_guard.sh deletes glyd-* pods older than its MAX_AGE (default 60 min) and this pod runs up to $MAX_MIN: start it with MAX_AGE=160, or stop it, before a launch (the launch itself stops, exit 2, while a guard runs unless GUARD_OK=1)"
  echo "launch: bash tools/cards/glm_quality.sh     (runs: NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK RP_CUDA=$RP_CUDA MIN_DRIVER=$MIN_DRIVER UPLOAD=\"<quality.env> $PY $SKEW\" bash tools/runpod/rp_class.sh $CLASS $JOB; a dry run starts nothing)"
  exit 0
fi

[ -s ~/.runpod/api_key ] || { echo "no ~/.runpod/api_key"; exit 2; }
if pgrep -f rp_guard.sh > /dev/null && [ "${GUARD_OK:-0}" != 1 ]; then
  echo "STOP: rp_guard.sh is running; it deletes glyd-* pods older than its MAX_AGE (default 60 min) and this pod runs up to $MAX_MIN. Restart it with MAX_AGE=160 (or stop it) and run this again; GUARD_OK=1 if it already has MAX_AGE of 160 or more."
  exit 2
fi
mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/quality.XXXXXX"); trap 'rm -rf "$W"' EXIT
{ for v in VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN FLA SELFTEST_GPU; do [ -n "${!v:-}" ] && printf '%s=%q\n' "$v" "${!v}"; done; true; } > "$W/quality.env"
[ -e "$RUNS/results-$NAME" ] && mv "$RUNS/results-$NAME" "$RUNS/results-$NAME.$(date +%Y%m%d-%H%M%S)"   # (rp_run.sh pulls into this directory: an old quality.md must not pass for this run's)
NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK UPLOAD="$W/quality.env $ROOT/$PY $ROOT/$SKEW" bash tools/runpod/rp_class.sh "$CLASS" "$JOB" 2>&1 | tee "$RUNS/$NAME.log"
rc=${PIPESTATUS[0]}
echo "== pod exit $rc (log $RUNS/$NAME.log, results $RUNS/results-$NAME)"
[ -f "$RUNS/results-$NAME/steps.txt" ] && cat "$RUNS/results-$NAME/steps.txt"
[ -f "$RUNS/results-$NAME/quality.md" ] && cat "$RUNS/results-$NAME/quality.md"
exit "$rc"
