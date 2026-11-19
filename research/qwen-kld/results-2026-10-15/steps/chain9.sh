#!/bin/bash
# A second warm pair (today's precise, then h86cal with the always-on at 6 bits) to check chain8's speed gap is not noise.
set -u
U=$HOME/glyd-scratch/qwen-users
until grep -q CHAIN8-DONE $U/chain8.out 2>/dev/null; do sleep 15; done
source <(sed -n '/^set -u/,/^P=/p' $U/chain3.sh | grep -v "^set -u\|^P=")
source <(sed -n '/^with_server() {/,/^}/p' $U/chain6.sh)
log() { echo "$* $(date +%T)" >> $U/chain9.out; }
SPEED2="python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out cold && python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out ."
PACK=precise_x with_server w_base_b "$SPEED2"
PACKDIR=$U/packs/h86cal EXTRA="--layout i6 --hi i6" with_server w_h86_i6_b "$SPEED2"
log CHAIN9-DONE
