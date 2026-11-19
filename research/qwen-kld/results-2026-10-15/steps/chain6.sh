#!/bin/bash
# Step 2, part 2 (after chain4): the always-on at 5 bits on today's pack; the calibrated packs with the always-on at 6 bits (each deleted after);
# h86 (86% cb3 / 14% cb2, no cb4 or GPU copies: ~82 GB download, as today) calibrated, with the always-on at 6 bits; then decode speed of today's
# precise and of the h86 candidate with compare.py's speed part (3 prompts, twice each), the server run as a user runs it.
set -u
U=$HOME/glyd-scratch/qwen-users
until grep -q CHAIN4-DONE $U/chain4.out 2>/dev/null; do sleep 30; done
source <(sed -n '/^set -u/,/^P=/p' $U/chain3.sh | grep -v "^set -u\|^P=")
log() { echo "$* $(date +%T)" >> $U/chain6.out; }
with_server() {  # NAME CLIENT-COMMAND, env PACK/HOLD_GIB/RAM_MAX/CTX for the server
  n=$1; shift; log "server $n"
  flock -w 14400 $L bash -c "
    $Q/gpurun.sh true
    $U/limited_inner.sh > $U/srv_$n.log 2>&1 &
    for i in \$(seq 300); do grep -q '^Ready' $U/srv_$n.log && break; sleep 2; done
    mkdir -p $U/runs/$n && cd $U/runs/$n && $* > client.txt 2>&1
    for p in \$(pgrep -f '^/home/.*/qwen-users/bin/glyd-engine qwenserv[e]'); do kill \$p; done
    sleep 6"
}
evict() { $U/venv/bin/python -c "
import os,sys
for root in sys.argv[1:]:
    for d,_,fs in os.walk(root, followlinks=True):
        for f in fs:
            try:
                fd=os.open(os.path.join(d,f),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
            except OSError: pass
" $Q/packed/precise_x $Q/packed/fast_x $Q/slim; }
SPEED="python3 $U/repo/bench/compare.py --url http://127.0.0.1:8093 --model q --speed-only --out ."
[ -d $U/packs/h4cal ] && run h4cal_i6 $Q/slim $U/packs/h4cal --precise --layout i6 --hi i6; rm -rf $U/packs/h4cal
[ -d $U/packs/h3cal ] && run h3cal_i6 $Q/slim $U/packs/h3cal --precise --layout i6 --hi i6; rm -rf $U/packs/h3cal
log "pack h86cal"; QWEN_SNAP=$Q/flash-next $PY $K/research/qwen-calib/qwen_pack_calib.py --plan $U/plan_h86.json --calib $U/calib --out $U/packs/h86cal > $U/pack_h86cal.log 2>&1
log "h86cal size $(du -sh $U/packs/h86cal | cut -f1)"
run h86cal_i6 $Q/slim $U/packs/h86cal --precise --layout i6 --hi i6
for r in 1 2; do
  PACK=precise_x with_server speed_base_$r "$SPEED"
  PACKDIR=$U/packs/h86cal EXTRA="--layout i6 --hi i6" with_server speed_h86cal_i6_$r "$SPEED"
done
run h86cal $Q/slim $U/packs/h86cal --precise
run all5 $Q/slim $Q/packed/precise_x --precise --layout i5
log CHAIN6-DONE
