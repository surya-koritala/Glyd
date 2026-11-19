#!/bin/bash
# Speed recovery for h86cal (owner, 16:35): warm decode per config (each server start runs compare.py's speed part twice; the second pass
# counts), then the wikitext 32x512 KLD of the two new configs. Configs: today's precise; h86cal with the always-on at i4 (default, attention
# and head i5), i5, i6, and i4 with attention + head at i6.
set -u
U=$HOME/glyd-scratch/qwen-users
until grep -q CHAIN6-DONE $U/chain6.out 2>/dev/null; do sleep 15; done
source <(sed -n '/^set -u/,/^P=/p' $U/chain3.sh | grep -v "^set -u\|^P=")
source <(sed -n '/^with_server() {/,/^}/p' $U/chain6.sh)
log() { echo "$* $(date +%T)" >> $U/chain8.out; }
SPEED2="python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out cold && python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out ."
H=$U/packs/h86cal
PACK=precise_x with_server w_base "$SPEED2"
PACKDIR=$H EXTRA="" with_server w_h86_i4 "$SPEED2"
PACKDIR=$H EXTRA="--layout i5" with_server w_h86_i5 "$SPEED2"
PACKDIR=$H EXTRA="--layout i6 --hi i6" with_server w_h86_i6 "$SPEED2"
PACKDIR=$H EXTRA="--hi i6" with_server w_h86_hi6 "$SPEED2"
wiki() {  # NAME PACK FLAGS...: the wikitext 32x512 KLD only
  n=$1; p=$2; shift 2; log "kld $n"; cd $K
  flock -w 14400 $L $Q/gpurun.sh $ENG --tokens chunks32.json --out s1_${n}_wiki.f32 -- $E qwenserve $Q/slim $p --precise "$@" > s1_${n}_wiki.log 2>&1
  $CMP chunks32.json ref32.f32 s1_${n}_wiki.f32 > s1_${n}_wiki.json 2>/dev/null; rm -f s1_${n}_wiki.f32
}
wiki h86cal_i5 $H --layout i5
wiki h86cal_hi6 $H --hi i6
log CHAIN8-DONE
