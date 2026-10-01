#!/bin/bash
# The box's lock held for one session: session.sh NAME   (the plan is ~/onb/plan_NAME.sh; output in ~/onb/runs/NAME/)
# The plan uses: hog FREE_MIB|stop, dock [-e VAR=VALUE ...] (a container of ubuntu:24.04 with no CUDA toolkit: its /root is ~/onb/ctr-home,
# uv and the tool stay between sessions; ~/onb is /onb there; ~/quick/geforce is /geforce), and mark LABEL (a line in the log).
NAME=$1
R=$HOME/onb/runs/$NAME; rm -rf "$R"; mkdir -p "$R/logs"
exec > >(tee "$R/session.log") 2>&1
echo "== $(date -u +%T) session $NAME"
touch ~/.glyd-busy
HOG=
hog() {
  [ -n "$HOG" ] && { kill $HOG 2>/dev/null; wait $HOG 2>/dev/null; HOG=; sleep 3; }
  [ "$1" = stop ] && return
  $HOME/quick/venv/bin/python $HOME/quick/hog.py "$1" > "$R/hog.txt" 2>&1 &
  HOG=$!
  for i in $(seq 1 90); do grep -q "MiB held" "$R/hog.txt" && break; sleep 1; done
  cat "$R/hog.txt"
}
dock() {
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits -lms 200 > "$R/memwatch-$MARK.csv" &
  local W=$!
  docker run --rm --gpus all --network host --ipc host -v $HOME/hf:/hf -e HF_HOME=/hf -v $HOME/onb/ctr-home:/root -v $HOME/onb:/onb:ro -v $HOME/quick/geforce:/geforce:ro "$@" ubuntu:24.04 bash /onb/inside.sh
  kill $W 2>/dev/null
  python3 - "$R/memwatch-$MARK.csv" <<'PY'
import sys
rows = [l.split(",") for l in open(sys.argv[1]) if l.count(",") == 2]
free = [int(r[2]) for r in rows]; used = [int(r[1]) for r in rows]
print(f"memwatch: {len(rows)} samples; lowest free {min(free)} MiB, most used {max(used)} MiB; samples under 512 MiB free: {sum(f < 512 for f in free)}")
PY
  # the server logs, kept with this session (a label in each name)
  for f in $HOME/onb/ctr-home/.local/state/glyd/logs/*.log; do [ -f "$f" ] && mv "$f" "$R/logs/$MARK-$(basename $f)"; done
}
mark() { MARK=$1; echo; echo "################ $(date -u +%T) $1"; }
cleanup() { hog stop; rm -f ~/.glyd-busy; echo "== $(date -u +%T) session $NAME over"; }
trap cleanup EXIT
MARK=start
source "$HOME/onb/plan_$NAME.sh"
