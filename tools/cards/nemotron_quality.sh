#!/usr/bin/env bash
# Output quality of Nemotron 3 Super (120B-A12B, bf16) with its routed experts rounded to 2-4 bits (Linear GLY-131, the second half of GLY-127's quality item): one 4-GPU Runpod pod, the bf16 model's
# next-token distribution on ~49k held-out tokens (six domains) against seven passes of the experts rounded (control, 6 variants; glm_quality.sh's, the same recipes): KL divergence, top-1 agreement
# and perplexity, overall and by domain. Results: $GLYD_RUNS/results-quality-nemotron/ (quality.md, quality.json, results.jsonl and the files after each variant, counts/ = the routing counts in
# glm_skew.py's format, perpos/, steps.txt, logs).
#   bash tools/cards/nemotron_quality.sh [--dry-run]       CLASS=... to change the pod class; MAX_MIN (at most 150: rp_run.sh's cap); VARIANTS="ref,mix13,int4" to run only some, in that order (names in the dry run, which plans what VARIANTS says; keep ref: it is the control that stops the job if the unrounded experts do not reproduce the model)
#   The Hummingbird recipes (at most 3 bits/weight on this model's experts: hlm2g, hb40, hb20) are in no default run: VARIANTS="ref,hlm2g,hb40,hb20" bash tools/cards/nemotron_quality.sh
#   The neuron-row recipes (GLY-98): VARIANTS="ref,hbr40,hb40,hbr40p" MAX_MIN=120 bash tools/cards/nemotron_quality.sh. hbr40 in VARIANTS also exports its codes (EXPORT=0: not): the job writes them to ~/results/export as its pass goes and
#   rp_run.sh copies each finished file to $GLYD_RUNS/results-quality-nemotron/export (~42 GB: nemotron_hbr.py's format; the launch stops if the disk there has less free).
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, 247 GB (241 GB without the MTP head, which transformers leaves out), 4 x A100 80 GB (class nemo; fallbacks A100 PCIe, H100 PCIe, H100 SXM): the bf16 weights stay on the cards (the skew run's map: 57, 61, 61 and 45 GiB, so 18-35 GiB stay free for the rounded experts and the streams).
# Needs ~/.runpod/api_key and runpodctl on this machine (not for --dry-run, which starts nothing). Optional env, passed to the job: VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN SELFTEST_GPU OMP_NUM_THREADS.
set -u
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
PY=research/lossy-2026/nemotron_quality.py; GQ=research/lossy-2026/glm_quality.py; SKEW=research/lossy-2026/glm_skew.py; JOB=tools/cards/nemotron_quality_job.sh
eval "$(python3 $PY plan)"   # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB ORDER SETUP_MIN PASSES_MIN (one source: glm_skew.py MODELS, nemotron_quality.py VARIANTS)
. tools/runpod/classes.sh
CLASS=${CLASS:-$POD_CLASS}; class_types "$CLASS" > /dev/null || { echo "unknown GPU class $CLASS"; exit 2; }
GPUS=$(class_gpus "$CLASS"); NAME=quality-nemotron; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
MAX_MIN=${MAX_MIN:-150}; CAPPED=""; [ "$MAX_MIN" -le 150 ] || { CAPPED=" (asked $MAX_MIN: rp_run.sh cuts every pod to 150, the owner's limit)"; MAX_MIN=150; }
export RP_CUDA=${RP_CUDA:-12.8} MIN_DRIVER=${MIN_DRIVER:-570}   # the job installs torch for CUDA 12.8, which runs on any driver >= 570 (more hosts than 13.0/580)
APPROVED=${APPROVED:-40}   # (the skew job's figure: the owner approved this GPU spend, not a number)
JOB_MAX_MIN=${JOB_MAX_MIN:-$((MAX_MIN - 22))}; export JOB_MAX_MIN   # (22 min of the cap are the pod's start and the last pull; for plan --text, which marks the passes that would start too late)
CAL=${CAL_SEQS:-24}; HELD=${HELD_SEQS:-4}; CNT=${COUNT_SEQS:-12}; GSEQ=${GPTQ_SEQS:-24}
case ",${VARIANTS:-}," in *,hbr40,*) EXPORT=${EXPORT:-1};; *) EXPORT=0;; esac; [ "$EXPORT" = 1 ] || EXPORT=   # (EXPORT_GB: from the plan)
FREE_GB=$(df -k "$RUNS" 2> /dev/null | awk 'NR==2 {printf "%.0f", $4 / 1e6}')
DISK_NEED=$(awk -v w="$WEIGHTS_GB" -v x="${EXPORT_GB:-0}" 'BEGIN{printf "%.0f", w + 15 + x}')   # (weights + ~15 GB of venv and data + the export)

if [ $DRY = 1 ]; then
  echo "quality: $REPO @ $REV: KL divergence, top-1 agreement and perplexity of the model with its routed experts rounded to 2-4 bits, against the bf16 original"
  python3 $PY plan --text
  echo "text: glm_skew.py's held-out set, 6 domains (wiki, code, chat, math, multi, reason) x $HELD x 2048 = $((6 * HELD * 2048)) tokens, $((6 * HELD * 2047)) scored positions, KL over the bf16 model's top 128 tokens + a tail bucket; per-domain tables too"
  echo "      routing counts for the most-used experts: $((6 * CNT)) calibration sequences ($((6 * CNT * 2048)) tokens); GPTQ calibration: $((6 * GSEQ)) sequences ($((6 * GSEQ * 2048)) tokens), fewer if the pod's RAM is short; $CAL sequences a domain prepared"
  echo "design: the bf16 model ($WEIGHTS_GB GB) stays on the $GPUS GPUs; a variant is one pass of all held-out sequences through the 88 layers one after another (Mamba-2, attention, latent MoE), each MoE layer's experts read from the bf16 weights"
  echo "        a chunk at a time, rounded in bf16, applied to the latent z = fc1_latent_proj(x) of the tokens routed to them (the router chose on x) and dropped (no packed copy); each layer sees the layers before it already rounded"
  echo "pod: class $CLASS x $GPUS GPUs, container disk $DISK GB ($WEIGHTS_GB GB of weights + venv + data${EXPORT:+ + $EXPORT_GB GB of export = ~$DISK_NEED GB}), RP_CUDA $RP_CUDA, MIN_DRIVER $MIN_DRIVER"
  [ -n "$EXPORT" ] && echo "export: hbr40's codes ($EXPORT_GB GB, 40 layer files + meta.json; format and reader: research/lossy-2026/nemotron_hbr.py) are written to ~/results/export on the pod as its pass goes; rp_run.sh leaves that directory out of its minute's tar and copies each finished file once to $RUNS/results-$NAME/export (this machine has ${FREE_GB:-?} GB free there; a launch needs $EXPORT_GB). The pod is deleted at the end: whatever has not been copied by then is gone"
  while IFS= read -r t; do
    p=$(type_price "$t"); h=$(awk -v p="$p" -v g="$GPUS" 'BEGIN{printf "%.2f", p*g}'); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
    echo "  try in order: $t x$GPUS  \$$p each = \$$h/h, worst case \$$w$(awk -v w="$w" -v a="$APPROVED" 'BEGIN{if (w > a) printf "  (over the $%d of the skew job)", a}')"
  done <<< "$(class_types "$CLASS")"
  h=$(class_price "$CLASS"); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}'); NOM=$((SETUP_MIN + PASSES_MIN))
  echo "time: MAX_MIN $MAX_MIN$CAPPED; the job starts no pass that would not end 2 min before JOB_MAX_MIN $JOB_MAX_MIN min (a skipped or cut-off pass is recorded, the rest of the results stay)"
  echo "worst case with the first choice: \$$h/h x $MAX_MIN min = \$$w (approved \$$APPROVED); a run that gets all $(echo $ORDER | wc -w | tr -d ' ') passes is ~$NOM-$((NOM + 22)) min of pod with its start, \$$(awk -v h="$h" -v a="$NOM" 'BEGIN{printf "%.0f-%.0f", h*a/60, h*(a+22)/60}') (glm_skew.py's whole nemotron job was 16 min: download 3.4 min, load 1 min)"
  echo "job steps (each timed in results/steps.txt): venv + hub client | download (hf, hf_xet; in the background) | pyenv (torch cu128, transformers 5.19.0) | prep (pinned datasets; in the background) | selftest (after the download, so it has the CPUs: CPU, and the block codes and a tiny model split over the GPUs; SELFTEST_GPU=0 skips the GPU part; 15 min at most) | run (load, check, bf16 reference, $ORDER) | report"
  echo "results: $RUNS/results-$NAME/   (quality.md and quality.json after every variant, results.jsonl, meta.json, counts/, perpos/ KL of every position, steps.txt, versions.txt, run.log)"
  echo "note: a running rp_guard.sh deletes glyd-* pods older than its MAX_AGE (default 60 min) and this pod runs up to $MAX_MIN: start it with MAX_AGE=160, or stop it, before a launch (the launch itself stops, exit 2, while a guard runs unless GUARD_OK=1)"
  echo "launch: ${VARIANTS:+VARIANTS=\"$VARIANTS\" }MAX_MIN=$MAX_MIN bash tools/cards/nemotron_quality.sh     (runs: NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK RP_CUDA=$RP_CUDA MIN_DRIVER=$MIN_DRIVER UPLOAD=\"<quality.env> $PY $GQ $SKEW research/lossy-2026/nemotron_hbr.py\" bash tools/runpod/rp_class.sh $CLASS $JOB; a dry run starts nothing)"
  exit 0
fi

[ -s ~/.runpod/api_key ] || { echo "no ~/.runpod/api_key"; exit 2; }
if pgrep -f rp_guard.sh > /dev/null && [ "${GUARD_OK:-0}" != 1 ]; then
  echo "STOP: rp_guard.sh is running; it deletes glyd-* pods older than its MAX_AGE (default 60 min) and this pod runs up to $MAX_MIN. Restart it with MAX_AGE=160 (or stop it) and run this again; GUARD_OK=1 if it already has MAX_AGE of 160 or more."
  exit 2
fi
mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/quality.XXXXXX"); trap 'rm -rf "$W"' EXIT
FREE_GB=$(df -k "$RUNS" | awk 'NR==2 {printf "%.0f", $4 / 1e6}')
# the export to the private bucket (the owner's go, 2026-10-10; S3=0: copied to this machine instead): temporary keys for that bucket alone, 12 hours, made here from the
# sigbound-deploy profile into s3.env, uploaded beside the job and sourced in its upload step alone (never printed)
S3_BUCKET=glyd-models-712416033631
if [ -n "$EXPORT" ] && [ "${S3:-1}" = 1 ]; then
  POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:PutObject","s3:GetObject","s3:ListBucket","s3:AbortMultipartUpload","s3:ListMultipartUploadParts"],"Resource":["arn:aws:s3:::'$S3_BUCKET'","arn:aws:s3:::'$S3_BUCKET'/*"]}]}'
  AWS_PROFILE=sigbound-deploy aws sts get-federation-token --name glyd-pod --duration-seconds 43200 --policy "$POLICY" --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text 2> /dev/null \
    | awk -v b="$S3_BUCKET" '{printf "AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nAWS_SESSION_TOKEN=%s\nAWS_DEFAULT_REGION=us-east-1\nS3_BUCKET=%s\n", $1, $2, $3, b}' > "$W/s3.env"
  grep -q '^AWS_SESSION_TOKEN=.' "$W/s3.env" || { echo "STOP: no temporary keys for the bucket (the sigbound-deploy profile)"; exit 2; }
  S3_ENV="$W/s3.env"; export PULL_EXPORT=0
elif [ -n "$EXPORT" ] && [ "${FREE_GB:-0}" -le "${EXPORT_GB%.*}" ]; then echo "STOP: the export is $EXPORT_GB GB and $RUNS has ${FREE_GB:-?} GB free (EXPORT=0 to run without it)"; exit 2; fi
{ for v in VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN SELFTEST_GPU OMP_NUM_THREADS EXPORT; do [ -n "${!v:-}" ] && printf '%s=%q\n' "$v" "${!v}"; done; true; } > "$W/quality.env"
[ -e "$RUNS/results-$NAME" ] && mv "$RUNS/results-$NAME" "$RUNS/results-$NAME.$(date +%Y%m%d-%H%M%S)"   # (rp_run.sh pulls into this directory: an old quality.md must not pass for this run's)
NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK UPLOAD="$W/quality.env ${S3_ENV:-} $ROOT/$PY $ROOT/$GQ $ROOT/$SKEW $ROOT/research/lossy-2026/nemotron_hbr.py" bash tools/runpod/rp_class.sh "$CLASS" "$JOB" 2>&1 | tee "$RUNS/$NAME.log"
rc=${PIPESTATUS[0]}
echo "== pod exit $rc (log $RUNS/$NAME.log, results $RUNS/results-$NAME)"
[ -f "$RUNS/results-$NAME/steps.txt" ] && cat "$RUNS/results-$NAME/steps.txt"
[ -f "$RUNS/results-$NAME/quality.md" ] && cat "$RUNS/results-$NAME/quality.md"
exit "$rc"
