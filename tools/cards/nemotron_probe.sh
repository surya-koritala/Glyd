#!/usr/bin/env bash
# What the routed experts of Nemotron 3 Super leave to an expert-offload design (Linear GLY-130): one 4-GPU Runpod pod, the bf16 model run over the held-out text of glm_skew.py (24 sequences of 2,048
# tokens, six domains) and over greedy decode steps with nothing rounded, and three measurements taken inside its routed experts: relu2 sparsity (zero neurons, columns of down that must be kept for 1% and
# 0.1% output error, the active neurons named beforehand by their calibration frequency), look-ahead routing (the next MoE layer's router on the stream d blocks earlier) and per-expert importance.
# Results: $GLYD_RUNS/results-probe-nemotron/ (probe.md and probe.json after the pass and after the decodes, meta.json, prefill.npz and decode.npz, steps.txt, logs).
#   bash tools/cards/nemotron_probe.sh [--dry-run]       CLASS=... to change the pod class; MAX_MIN (default 60, at most 150: rp_run.sh's cap)
# Cost: class nemo is 4 x A100-SXM4-80GB at $7.16/h; the job is ~22 min and the pod's start and the last pull add 5-15 (the minutes are guesses until a first run), so ~$3.2-4.4; MAX_MIN 60 caps it at $7.16.
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, 247 GB, 4 x A100 80 GB (class nemo; fallbacks A100 PCIe, H100 PCIe, H100 SXM): the bf16 weights stay on the cards (the skew run's map: 57, 61, 61 and 45 GiB).
# Needs ~/.runpod/api_key and runpodctl on this machine (not for --dry-run, which starts nothing). Optional env, passed to the job: CAL_SEQS HELD_SEQS DECODE_PROMPTS DECODE_NEW CHUNK MEM_FRAC JOB_MAX_MIN
# SELFTEST_GPU OMP_NUM_THREADS.
set -u
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT" || exit 1
PY=research/lossy-2026/nemotron_probe.py; NQ=research/lossy-2026/nemotron_quality.py; GQ=research/lossy-2026/glm_quality.py; SKEW=research/lossy-2026/glm_skew.py; JOB=tools/cards/nemotron_probe_job.sh
eval "$(python3 $PY plan)"   # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB SETUP_MIN PASSES_MIN (one source: glm_skew.py MODELS, nemotron_probe.py)
. tools/runpod/classes.sh
CLASS=${CLASS:-$POD_CLASS}; class_types "$CLASS" > /dev/null || { echo "unknown GPU class $CLASS"; exit 2; }
GPUS=$(class_gpus "$CLASS"); NAME=probe-nemotron; RUNS=${GLYD_RUNS:-$HOME/glyd-runs}
MAX_MIN=${MAX_MIN:-60}; CAPPED=""; [ "$MAX_MIN" -le 150 ] || { CAPPED=" (asked $MAX_MIN: rp_run.sh cuts every pod to 150, the owner's limit)"; MAX_MIN=150; }
export RP_CUDA=${RP_CUDA:-12.8} MIN_DRIVER=${MIN_DRIVER:-570}   # the job installs torch for CUDA 12.8, which runs on any driver >= 570 (more hosts than 13.0/580)
APPROVED=${APPROVED:-5}   # (the owner's figure for this probe: about $5, under 40 min of pod)
JOB_MAX_MIN=${JOB_MAX_MIN:-$((MAX_MIN - 15))}; export JOB_MAX_MIN   # (15 min of the cap are the pod's start and the last pull; for plan --text, and the job's own deadline)
CAL=${CAL_SEQS:-12}; HELD=${HELD_SEQS:-4}

if [ $DRY = 1 ]; then
  echo "probe: $REPO @ $REV: relu2 sparsity inside the routed experts, look-ahead routing, per-expert importance; the bf16 model, nothing rounded"
  python3 $PY plan --text
  echo "text: glm_skew.py's held-out set, 6 domains (wiki, code, chat, math, multi, reason) x $HELD x 2048 = $((6 * HELD * 2048)) tokens measured; $CAL calibration sequences a domain ($((6 * CAL * 2048)) tokens) go through first in the same pass for the neuron profiles and the routing counts"
  echo "design: the bf16 model ($WEIGHTS_GB GB) stays on the $GPUS GPUs; all sequences go through the 88 layers one after another (the streams on the host between layers), each MoE layer's experts read from the bf16 weights"
  echo "        16 at a time (CHUNK) and run for the tokens routed to them; the decodes are the model's own generate() with a hook on every block"
  echo "pod: class $CLASS x $GPUS GPUs, container disk $DISK GB ($WEIGHTS_GB GB of weights + venv + data), RP_CUDA $RP_CUDA, MIN_DRIVER $MIN_DRIVER"
  while IFS= read -r t; do
    p=$(type_price "$t"); h=$(awk -v p="$p" -v g="$GPUS" 'BEGIN{printf "%.2f", p*g}'); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}')
    echo "  try in order: $t x$GPUS  \$$p each = \$$h/h, worst case \$$w$(awk -v w="$w" -v a="$APPROVED" 'BEGIN{if (w > a) printf "  (over the about $%d of the brief)", a}')"
  done <<< "$(class_types "$CLASS")"
  h=$(class_price "$CLASS"); w=$(awk -v h="$h" -v m="$MAX_MIN" 'BEGIN{printf "%.2f", h*m/60}'); NOM=$((SETUP_MIN + PASSES_MIN))
  echo "time: MAX_MIN $MAX_MIN$CAPPED; the job starts no phase that would not end 2 min before JOB_MAX_MIN $JOB_MAX_MIN min (a stopped pass keeps the layers it measured; the decodes are skipped if too late)"
  echo "worst case with the first choice: \$$h/h x $MAX_MIN min = \$$w (brief: about \$$APPROVED); a run that gets everything is ~$NOM min of job, ~$((NOM + 5))-$((NOM + 15)) min of pod with its start, \$$(awk -v h="$h" -v a="$NOM" 'BEGIN{printf "%.1f-%.1f", h*(a+5)/60, h*(a+15)/60}')"
  echo "job steps (each timed in results/steps.txt): CPU gate (under 8 CPUs stops) | venv + hub client | download (hf, hf_xet; in the background) | pyenv (torch cu128, transformers 5.19.0) | prep (pinned datasets; in the background) | download speed gate (8 min in: weights projected past 30 min stop it) | selftest (after the download; CPU, and on the GPUs the hand-made cases on each card and the tiny model split over them; SELFTEST_GPU=0 skips the GPU part; 15 min at most) | run (load, pass, decodes) | report"
  echo "results: $RUNS/results-$NAME/   (probe.md, probe.json, meta.json, prefill.npz, decode.npz, steps.txt, versions.txt, run.log)"
  echo "note: a running rp_guard.sh deletes glyd-* pods older than its MAX_AGE (default 60 min): start it with MAX_AGE=160, or stop it, before a launch (the launch itself stops, exit 2, while a guard runs unless GUARD_OK=1)"
  echo "launch: bash tools/cards/nemotron_probe.sh     (runs: NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK RP_CUDA=$RP_CUDA MIN_DRIVER=$MIN_DRIVER UPLOAD=\"<probe.env> $PY $NQ $GQ $SKEW\" bash tools/runpod/rp_class.sh $CLASS $JOB; a dry run starts nothing)"
  exit 0
fi

[ -s ~/.runpod/api_key ] || { echo "no ~/.runpod/api_key"; exit 2; }
if pgrep -f rp_guard.sh > /dev/null && [ "${GUARD_OK:-0}" != 1 ]; then
  echo "STOP: rp_guard.sh is running; it deletes glyd-* pods older than its MAX_AGE (default 60 min) and this pod runs up to $MAX_MIN. Restart it with MAX_AGE=160 (or stop it) and run this again; GUARD_OK=1 if it already has MAX_AGE of 160 or more."
  exit 2
fi
mkdir -p "$RUNS"; W=$(mktemp -d "$RUNS/probe.XXXXXX"); trap 'rm -rf "$W"' EXIT
{ for v in CAL_SEQS HELD_SEQS DECODE_PROMPTS DECODE_NEW CHUNK MEM_FRAC JOB_MAX_MIN SELFTEST_GPU OMP_NUM_THREADS; do [ -n "${!v:-}" ] && printf '%s=%q\n' "$v" "${!v}"; done; true; } > "$W/probe.env"
[ -e "$RUNS/results-$NAME" ] && mv "$RUNS/results-$NAME" "$RUNS/results-$NAME.$(date +%Y%m%d-%H%M%S)"   # (rp_run.sh pulls into this directory: an old probe.md must not pass for this run's)
NAME=$NAME MAX_MIN=$MAX_MIN RP_DISK=$DISK UPLOAD="$W/probe.env $ROOT/$PY $ROOT/$NQ $ROOT/$GQ $ROOT/$SKEW" bash tools/runpod/rp_class.sh "$CLASS" "$JOB" 2>&1 | tee "$RUNS/$NAME.log"
rc=${PIPESTATUS[0]}
echo "== pod exit $rc (log $RUNS/$NAME.log, results $RUNS/results-$NAME)"
[ -f "$RUNS/results-$NAME/steps.txt" ] && cat "$RUNS/results-$NAME/steps.txt"
[ -f "$RUNS/results-$NAME/probe.md" ] && cat "$RUNS/results-$NAME/probe.md"
exit "$rc"
