#!/bin/bash
# The two scoring texts through each build: CKPT PACK FLAGS per line; results in out/NAME-TEXT.json.
B=$HOME/glyd-scratch/qwen-build; Q=$HOME/glyd-scratch/qwen; . $B/env.sh; cd $Q/score; mkdir -p out
for p in $(pgrep -f "^/home/.*/glyd-engine-http qwenserve"); do kill $p; done; sleep 5
while read -r name ckpt pack flags; do
  [ -z "$name" ] && continue
  echo "== $name $(date +%T)"
  ( cat req-loader.json; cat req-game.json ) | flock -w 7200 $HOME/glyd-scratch/gpu.lock $Q/gpurun.sh $B/bin/glyd-engine-score qwenserve $Q/$ckpt $Q/packed/$pack $flags > out/$name.jsonl 2> out/$name.log
  python3 -c "
import json
for t, l in zip(['loader','game'], open('out/$name.jsonl')):
    r = json.loads(l); print('$name', t, {k: r.get(k) for k in ['n','nll','miss','low','error']})"
done <<'CFG'
fastS slim fast_x
fastF flash-next fast_x
precS slim precise_x --precise
precF flash-next precise_x --precise
bestF flash-next h4 --no-side
CFG
echo ALL-DONE
