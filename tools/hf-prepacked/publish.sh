#!/usr/bin/env bash
# The saves made, checked and uploaded on one Runpod pod: job.sh through rp_run.sh, with the Hugging Face token handed
# to the job over ssh into a FIFO (held in the job's shell, never in a file on the pod or printed). The token is read
# from this Mac's Hugging Face login (hf auth login: ~/.cache/huggingface/token), which needs write access to the org.
# A running rp_guard.sh deletes glyd-* pods older than its MAX_AGE: the seven models take about 45 minutes (2026-10-06).
#   ORG=glyd [PUBLIC=1] [SERVE=1] [MODELS="..."] [GLYD_GPU=0.29.0] bash publish.sh        (PUBLIC=1: public after each repo's files and card check)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
RP=${RP:-$HERE/../runpod}; RP_TMP=${GLYD_RUNS:-$HOME/glyd-runs}  # (tools/runpod/rp_run.sh keeps its run's ssh key in $GLYD_RUNS/rp.*/key)
TOKEN=${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}
[ -s "$TOKEN" ] || { echo "no Hugging Face token at $TOKEN: hf auth login (a write token for ${ORG:?ORG=...})"; exit 1; }
LOG=$(mktemp); JOB=$(mktemp)
{ head -1 "$HERE/job.sh"; echo "export UPLOAD_ORG=$ORG PUBLIC=${PUBLIC:-} SERVE=${SERVE:-} GLYD_GPU=${GLYD_GPU:-}"; [ -n "${MODELS:-}" ] && echo "export MODELS=\"$MODELS\""; tail -n +2 "$HERE/job.sh"; } > "$JOB"  # (a file: rp_run.sh reads it after the pod is up)
GPU=${GPU:-NVIDIA A100-SXM4-80GB} MAX_MIN=${MAX_MIN:-240} RP_DISK=250 NAME=${NAME:-hfpub} UPLOAD="$HERE/check.py $HERE/card.py $HERE/upload.py" \
  bash "$RP/rp_run.sh" "$JOB" > "$LOG" 2>&1 &
until grep -q "job started" "$LOG"; do grep -q "pods left" "$LOG" && { cat "$LOG"; exit 1; }; sleep 10; done
read -r IP PORT < <(sed -n 's/.*ssh ready at \([0-9.]*\):\([0-9]*\).*/\1 \2/p' "$LOG" | head -1)
KEY=$(grep -l "glyd-${NAME:-hfpub}$" "$RP_TMP"/rp.*/key.pub | head -1); KEY=${KEY%.pub}
ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "root@$IP" \
  'for i in $(seq 60); do [ -p ~/hf_token ] && exec cat > ~/hf_token; sleep 5; done; exit 1' < "$TOKEN"
wait; cat "$LOG"
