#!/bin/bash
# acc.sh NAME WORKDIR ARGS...: gpu/vllm/acceptance.sh of the box's acceptance-028 checkout under the GPU queue's lock, in WORKDIR; its output in ~/accept-028/NAME.out,
# the run's summary and logs kept in ~/accept-028/results/NAME. ~/.glyd-busy is kept while any job holds or waits for the lock; the lock is held by flock alone (-o: no child of the run holds it).
NAME=$1; WORK=$2; shift; shift
unset PYTHONPATH LD_LIBRARY_PATH GLYD_GPU_LIB GLYD_VERSION GLYD_KV
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false
mkdir -p ~/accept-028/results
exec > ~/accept-028/$NAME.out 2>&1
echo "== queued $(date -u +%T): $*"
touch ~/.glyd-busy
flock -o ~/.glyd-box.lock bash -c 'free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); if [ "$free" -lt 30 ]; then echo "== not run: $free GB free on / (30 GB are to stay free)"; exit 3; fi; echo "== started $(date -u +%T), $free GB free on /"; cd ~/rel028/co/gpu/vllm && bash acceptance.sh --work "$0" "${@:2}"; rc=$?; rm -rf ~/accept-028/results/$1; mkdir -p ~/accept-028/results/$1; cp -r "$0/summary.txt" "$0/logs" ~/accept-028/results/$1/ 2>/dev/null; cp -r "$0"/home/.local/state/glyd/logs ~/accept-028/results/$1/glyd-logs 2>/dev/null; echo "== finished $(date -u +%T) rc=$rc"; exit $rc' "$WORK" "$NAME" "$@"
rc=$?
pgrep -f '[f]lock .*glyd-box.lock' > /dev/null || rm -f ~/.glyd-busy
echo "== done rc=$rc"
exit $rc
