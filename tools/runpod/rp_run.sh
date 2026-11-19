#!/usr/bin/env bash
# One Runpod GPU pod for one job, through Runpod's CLI (runpodctl, the current API): an ssh key made for this run only,
# the job run over ssh with its files, ~/results pulled every minute into $GLYD_RUNS/results-$NAME, then the pod deleted.
# It cannot outlive MAX_MIN: a watchdog ends the script at the cap and the exit trap deletes the pod, then lists what is left.
# The API key is read from ~/.runpod/api_key into each runpodctl call's environment only; it is never printed or stored.
#   GPU="NVIDIA H100 80GB HBM3" MAX_MIN=50 NAME=rpq38 UPLOAD="file1 file2" bash rp_run.sh JOB.sh   (RP_GPUS=N for an N-GPU pod; RP_DISK GB of disk)
# Exit 3: the host's driver is too old for the job; 4: no capacity or Runpod stopped the pod; both are retried by rp_try.sh.
# Exit 5: refused by the owner's limits (rp_budget.sh wrote $GLYD_RUNS/rp_budget/STOP, or RP_GPUS > 4); not retried.
set -u
RUNS=${GLYD_RUNS:-$HOME/glyd-runs}; mkdir -p "$RUNS"
. "$(dirname "$0")/pull_export.sh"   # pull_export: results/export (a job's big files) one finished file at a time, not in the minute's tar
JOB=$1; NAME=${NAME:-rp}; GPU=${GPU:-NVIDIA H100 80GB HBM3}; MAX_MIN=${MAX_MIN:-50}
B=$RUNS/rp_budget
if [ -f "$B/STOP" ]; then echo "REFUSED by the owner's budget (spent \$$(cat "$B/spent.txt" 2> /dev/null)): no new pod. Do not retry; record this run as not done for budget."; exit 5; fi
if [ "${RP_GPUS:-1}" -gt 4 ]; then echo "REFUSED by the owner: no pods of more than 4 GPUs. Do not retry; record it as dropped by the owner."; exit 5; fi
[ "$MAX_MIN" -le 150 ] || MAX_MIN=150  # (the owner's limit: no pod over 2.5 h)
OUT=$RUNS/results-$NAME; mkdir -p "$OUT"; rm -f "$OUT/DONE"
TMP=$(mktemp -d "$RUNS/rp.XXXXXX"); chmod 700 "$TMP"
rp() { RUNPOD_API_KEY="$(tr -d '[:space:]' < ~/.runpod/api_key)" runpodctl "$@"; }
ID=""; START=$(date +%s)
cleanup() {
  set +e
  if [ -n "$ID" ]; then
    rp pod delete "$ID" > /dev/null 2>&1; echo "terminating $ID"
    for _ in $(seq 1 30); do rp pod list 2> /dev/null | grep -q "\"$ID\"" || break; sleep 5; done
  fi
  echo "pods left: $(rp pod list 2> /dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print([(p.get("id"), p.get("name"), p.get("desiredStatus")) for p in d])' 2> /dev/null)"
  echo "$(( ($(date +%s) - START) / 60 )) min"
  [ -n "${WATCH:-}" ] && pkill -P "$WATCH" 2> /dev/null  # (the watchdog's sleep: left running it holds this script's output open, and a launcher's | tee with it, until the cap)
  rm -rf "$TMP"
}
trap cleanup EXIT
trap 'exit 1' INT TERM HUP
( sleep $((MAX_MIN * 60)) && { echo "time cap reached"; kill -TERM $$; pkill -TERM -P $$; } ) 2> /dev/null & WATCH=$!
command -v caffeinate > /dev/null && caffeinate -i -w $$ &

ssh-keygen -q -t ed25519 -N "" -f "$TMP/key" -C "glyd-$NAME"
ENVJ=$(python3 -c 'import json,sys; print(json.dumps({"PUBLIC_KEY": open(sys.argv[1]).read().strip()}))' "$TMP/key.pub")
# RP_DC: data centers to take the pod from, comma-separated, to keep off a slow host; the API takes one a call, so each is tried in turn until one has the card
for DC in $(echo "${RP_DC:-}" | tr ',' ' ') ""; do
  [ -z "$DC" ] && [ -n "${RP_DC:-}" ] && break
  rp pod create --name "glyd-bench-$NAME" --template-id "${RP_TEMPLATE:-runpod-torch-v280}" --gpu-id "$GPU" --gpu-count "${RP_GPUS:-1}" \
    --cloud-type SECURE --container-disk-in-gb "${RP_DISK:-150}" --ports 22/tcp --env "$ENVJ" \
    --min-cuda-version "${RP_CUDA:-13.0}" ${DC:+--data-center-ids "$DC"} --wait --wait-timeout 15m > "$TMP/create.json" 2> "$TMP/create.err"; CRC=$?
  ID=$(python3 -c 'import sys,json; print(json.load(open(sys.argv[1])).get("id", ""))' "$TMP/create.json" 2> /dev/null)
  [ -n "$ID" ] || ID=$(grep -o '"id" *: *"[a-z0-9]*"' "$TMP/create.err" "$TMP/create.json" 2> /dev/null | head -1 | sed 's/.*"\([a-z0-9]*\)"$/\1/')
  [ -n "$ID" ] && break
  [ -n "$DC" ] && echo "$DC: no pod ($(grep -o 'no longer any instances available' "$TMP/create.err" "$TMP/create.json" | head -1))"
done
[ -n "$ID" ] || { echo "no pod: $(head -c 400 "$TMP/create.err") $(head -c 200 "$TMP/create.json")"; exit 4; }  # (no capacity: nothing to delete)
echo "$(date -u +%H:%M) created $ID ($GPU)"
[ "$CRC" = 0 ] || { echo "create/wait failed (exit $CRC): $(tail -c 400 "$TMP/create.err")"; exit 4; }
tail -1 "$TMP/create.err"
rp ssh info "$ID" > "$TMP/ssh.json" 2>&1
read -r IP PORT < <(python3 -c 'import sys,json; d=json.load(open(sys.argv[1])); print(d.get("ip") or "-", d.get("port") or "-")' "$TMP/ssh.json" 2> /dev/null)
[ "${IP:--}" != "-" ] && [ "${PORT:--}" != "-" ] || { echo "no ssh address for $ID: $(head -c 300 "$TMP/ssh.json")"; exit 4; }
SSH=(ssh -i "$TMP/key" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 -o LogLevel=ERROR "root@$IP")
for _ in $(seq 1 60); do "${SSH[@]}" true 2> /dev/null && break; sleep 10; done
"${SSH[@]}" true 2> /dev/null || { echo "ssh never answered at $IP:$PORT"; exit 1; }
echo "$(date -u +%H:%M) ssh ready at $IP:$PORT after $(( $(date +%s) - START )) s"
"${SSH[@]}" "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" > "$OUT/gpu.txt" 2>&1; cat "$OUT/gpu.txt"
DRV=$(cut -d, -f2 "$OUT/gpu.txt" | head -1 | tr -d " " | cut -d. -f1)
[ "${DRV:-0}" -ge "${MIN_DRIVER:-580}" ] 2> /dev/null || { echo "driver ${DRV:-?} is older than ${MIN_DRIVER:-580}: this host cannot run the job"; exit 3; }
"${SSH[@]}" "cat > job.sh" < "$JOB"
for f in ${UPLOAD:-}; do "${SSH[@]}" "cat > $(basename "$f")" < "$f" || { echo "upload failed: $f"; exit 1; }; done
"${SSH[@]}" "nohup bash job.sh > job.log 2>&1 < /dev/null &"
echo "$(date -u +%H:%M) job started"
while true; do
  sleep 60
  "${SSH[@]}" "tar -C results --exclude=./export -cf - . 2> /dev/null; true" 2> /dev/null | tar -C "$OUT" -xf - 2> /dev/null
  pull_export
  "${SSH[@]}" "tail -c 20000 job.log" > "$OUT/job.log" 2> /dev/null
  [ -f "$OUT/DONE" ] && break
done
echo "$(date -u +%H:%M) job done"
