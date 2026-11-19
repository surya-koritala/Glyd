#!/bin/bash
# After chain6: h86 rounded as today (no calibration: calibration helped wikitext but not ref_long, and it saw wikitext train), with the
# always-on at 6 bits; then WARM decode speed beside today's precise: each server start runs compare.py's speed part twice and the second
# pass counts (the first reads the pack into the page cache: two packs of 38 and 42 GiB don't both stay in 60 GB). h86 rounded has the
# same formats and sizes as h86 calibrated, so the engine's work, and its speed, are the same. The calibrated h86 is deleted first (disk).
set -u
U=$HOME/glyd-scratch/qwen-users
until grep -q CHAIN6-DONE $U/chain6.out 2>/dev/null; do sleep 30; done
source <(sed -n '/^set -u/,/^P=/p' $U/chain3.sh | grep -v "^set -u\|^P=")
source <(sed -n '/^with_server() {/,/^}/p' $U/chain6.sh)
log() { echo "$* $(date +%T)" >> $U/chain7.out; }
SPEED2="python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out cold && python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out ."
rm -rf $U/packs/h86cal
log "pack h86rtn"; QWEN_SNAP=$Q/flash-next $PY $Q/qwen_pack.py pack --plan $U/plan_h86.json --out $U/packs/h86rtn > $U/pack_h86rtn.log 2>&1
log "h86rtn size $(du -sh $U/packs/h86rtn | cut -f1)"
run h86rtn_i6 $Q/slim $U/packs/h86rtn --precise --layout i6 --hi i6
for r in 1 2; do
  PACK=precise_x with_server warm_base_$r "$SPEED2"
  PACKDIR=$U/packs/h86rtn EXTRA="--layout i6 --hi i6" with_server warm_h86_i6_$r "$SPEED2"
done
log CHAIN7-DONE
