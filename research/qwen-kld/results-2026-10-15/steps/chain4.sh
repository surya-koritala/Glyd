#!/bin/bash
# Quality step 2 (after chain3): the same-size h3 plan (60% cb3 / 40% cb2, no cb4, no GPU copies) rounded as today, then calibrated h4 (+ calibrated
# hot copies) and calibrated h3. Each pack measured as in chain3, then deleted.
set -u
U=$HOME/glyd-scratch/qwen-users; K=$U/kld; Q=$HOME/glyd-scratch/qwen; PY=$HOME/glyd-scratch/qwen-build/venv/bin/python
until grep -q CHAIN3-DONE $U/chain3.out; do sleep 30; done
rm -rf $U/packs/b2
source <(sed -n '/^set -u/,/^P=/p' $U/chain3.sh | grep -v "^set -u\|^P=")   # (chain3's variables, log() and run())
log() { echo "$* $(date +%T)" >> $U/chain4.out; }
CAL="$PY $K/research/qwen-calib/qwen_pack_calib.py"
export QWEN_SNAP=$Q/flash-next
cd $Q
log "pack h3rtn"; $PY $Q/qwen_pack.py pack --plan $U/plan_h3.json --out $U/packs/h3rtn > $U/pack_h3rtn.log 2>&1; log "h3rtn size $(du -sh $U/packs/h3rtn | cut -f1)"
run h3rtn $Q/slim $U/packs/h3rtn --precise; rm -rf $U/packs/h3rtn
log "pack h4cal"; $CAL --plan $Q/plan_h4.json --calib $U/calib --out $U/packs/h4cal > $U/pack_h4cal.log 2>&1 && $CAL --side-of $U/packs/h4cal --only cb4 --fmt cb3 --calib $U/calib --out $U/packs/h4cal/hot >> $U/pack_h4cal.log 2>&1
log "h4cal size $(du -sh $U/packs/h4cal | cut -f1)"
run h4cal $Q/slim $U/packs/h4cal --precise
log "pack h3cal"; $CAL --plan $U/plan_h3.json --calib $U/calib --out $U/packs/h3cal > $U/pack_h3cal.log 2>&1; log "h3cal size $(du -sh $U/packs/h3cal | cut -f1)"
run h3cal $Q/slim $U/packs/h3cal --precise
log CHAIN4-DONE
