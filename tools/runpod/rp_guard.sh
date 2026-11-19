#!/usr/bin/env bash
# Independent Runpod watchdog: every 2 minutes, delete any of our pods (named glyd-*) that has been up longer than
# MAX_AGE minutes since this guard first saw it, or that has stopped/exited (a stopped pod still bills its disk).
# Other pods are never touched. The API key is read into each call's environment only.
RUNS=${GLYD_RUNS:-$HOME/glyd-runs}; mkdir -p "$RUNS"
MAX_AGE=${MAX_AGE:-60}; SEEN=$RUNS/rp_guard.seen; touch "$SEEN"
rp() { RUNPOD_API_KEY="$(tr -d '[:space:]' < ~/.runpod/api_key)" runpodctl "$@"; }
while true; do
  now=$(date +%s)
  rp pod list 2> /dev/null | python3 -c '
import sys, json
for p in json.load(sys.stdin):
    if (p.get("name") or "").startswith("glyd-"):
        print(p.get("id"), p.get("desiredStatus") or "?", p.get("runtimeStatus") or "?")' 2> /dev/null > "$RUNS/rp_guard.now"
  while read -r id ds rs; do
    first=$(awk -v i="$id" '$1 == i {print $2}' "$SEEN"); [ -n "$first" ] || { first=$now; echo "$id $now" >> "$SEEN"; }
    age=$(( (now - first) / 60 ))
    case "$ds/$rs" in
      EXITED/*|TERMINATED/*|*/stopped|*/terminated) echo "$(date -u +%H:%M) deleting stopped pod $id ($ds/$rs)"; rp pod delete "$id" > /dev/null 2>&1;;
      *) [ "$age" -ge "$MAX_AGE" ] && { echo "$(date -u +%H:%M) deleting pod $id: up $age min (cap $MAX_AGE)"; rp pod delete "$id" > /dev/null 2>&1; };;
    esac
  done < "$RUNS/rp_guard.now"
  sleep 120
done
