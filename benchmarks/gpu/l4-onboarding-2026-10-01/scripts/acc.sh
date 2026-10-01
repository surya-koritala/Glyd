#!/bin/bash
# acc.sh NAME WORKDIR ARGS...: gpu/vllm/acceptance.sh (the box's checkout of the onboarding branch) under the GPU queue's lock, in WORKDIR; its output in
# ~/accept-onb/NAME.out, and the run's summary and logs kept in ~/accept-onb/results/NAME
NAME=$1; WORK=$2; shift; shift
mkdir -p ~/accept-onb/results
exec > ~/accept-onb/$NAME.out 2>&1
echo "== queued $(date -u +%T): $*"
exec flock ~/.glyd-box.lock bash -c 'touch ~/.glyd-busy; echo "== started $(date -u +%T)"; git -C ~/onb/src pull -q; cd ~/onb/src/gpu/vllm && git -C ~/onb/src log -1 --format="== checkout %h %s" | cut -c1-90; bash acceptance.sh --work "$0" "${@:2}"; rc=$?; rm -f ~/.glyd-busy; rm -rf ~/accept-onb/results/$1; mkdir -p ~/accept-onb/results/$1; cp -r "$0/summary.txt" "$0/logs" ~/accept-onb/results/$1/ 2>/dev/null; cp -r "$0"/home/.local/state/glyd/logs ~/accept-onb/results/$1/glyd-logs 2>/dev/null; echo "== finished $(date -u +%T) rc=$rc"' "$WORK" "$NAME" "$@"
