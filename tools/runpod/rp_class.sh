#!/usr/bin/env bash
# rp_run.sh on a GPU class (classes.sh): each of the class's types in order, ROUNDS times (default 6, waiting a minute longer after each round: ~20 min in all), moving on while a host is too old or there is no capacity (exit 3/4);
# any other exit (0, a failed job, 5 the budget) is final. Same environment as rp_run.sh (NAME, MAX_MIN, UPLOAD, RP_DISK); GPU is set from the class, and RP_GPUS (the pod's GPU count) is the environment's where it is set (a pod of more GPUs has more memory: RP_GPUS=2 bash rp_class.sh 4090 JOB.sh), else the class's.
#   NAME=x UPLOAD="a b" bash rp_class.sh 16gb JOB.sh
. "$(dirname "$0")/classes.sh"
CLASS=$1; shift
TYPES=$(class_types "$CLASS") || { echo "unknown GPU class $CLASS"; exit 2; }
for r in $(seq 1 "${ROUNDS:-6}"); do
  while IFS= read -r t; do
    echo "== $CLASS: $t, round $r $(date -u +%H:%M)"
    GPU=$t RP_GPUS=${RP_GPUS:-$(class_gpus "$CLASS")} bash "$(dirname "$0")/rp_run.sh" "$@"; rc=$?
    case $rc in 3|4) sleep 20;; *) exit $rc;; esac
  done <<< "$TYPES"
  [ "$r" -lt "${ROUNDS:-6}" ] && sleep $((60 * r))   # (capacity comes back in minutes; many pods asked at once drain a class)
done
echo "no capacity in class $CLASS"; exit 4
