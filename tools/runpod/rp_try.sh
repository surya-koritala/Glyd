#!/usr/bin/env bash
# rp_run.sh up to TRIES times while a host is too old (exit 3) or there is no capacity or RunPod stops the pod (exit 4); exit 5 (the budget) is final
for t in $(seq 1 ${TRIES:-4}); do
  echo "== try $t $(date -u +%H:%M)"
  bash "$(dirname "$0")/rp_run.sh" "$@"; rc=$?
  case $rc in 3|4) sleep 20;; *) exit $rc;; esac
done
