#!/bin/bash
# Quality step 1: what costs the KLD. Each config: wikitext 32x512 and ref_long 11x512 vs the bf16 reference, then decode speed. GPU lock per run.
set -u
U=$HOME/glyd-scratch/qwen-users; K=$U/kld; Q=$HOME/glyd-scratch/qwen; L=$HOME/glyd-scratch/gpu.lock; PY=$HOME/glyd-scratch/qwen-build/venv/bin/python
. $HOME/glyd-scratch/qwen-build/env.sh
E=$U/bin/glyd-engine; ENG="$U/venv/bin/python $K/research/qwen-kld/kld_engine.py"; CMP="$U/venv/bin/python $K/research/qwen-kld/kld_compare.py"
log() { echo "$* $(date +%T)" >> $U/chain3.out; }
run() {  # NAME MODEL_DIR PACK_DIR FLAGS...
  n=$1; m=$2; p=$3; shift 3; log "kld $n"
  cd $K
  for t in long wiki; do
    tok=chunks32.json; ref=ref32.f32; [ $t = long ] && { tok=long_chunks.json; ref=ref_long.f32; }
    flock -w 14400 $L $Q/gpurun.sh $ENG --tokens $tok --out s1_${n}_$t.f32 -- $E qwenserve $m $p "$@" > s1_${n}_$t.log 2>&1
    $CMP $tok $ref s1_${n}_$t.f32 > s1_${n}_$t.json 2>/dev/null; rm -f s1_${n}_$t.f32
  done
}
P=$Q/packed/precise_x
run base $Q/slim $P --precise
run skip0 $Q/slim $P --precise --skip-rank 0 --skip-rank-later 0
run hi6 $Q/slim $P --precise --hi i6
run all6 $Q/slim $P --precise --layout i6 --hi i6
run ngram16 $Q/flash-next $P --precise
run noside $Q/slim $P --precise --no-side
run fast $Q/slim $Q/packed/fast_x
# the expert tiers: B1 = the 2-bit tier at 3 bits (top 10% cb4 + hot cb3 copies, as precise); B2 = every expert at 4 bits, no copies
$PY - <<'PYE'
import json
p = json.load(open("/home/surya-koritala/glyd-scratch/qwen/plan_h4.json"))
b1 = dict(p, layers={k: ["cb3" if f == "cb2" else f for f in v] for k, v in p["layers"].items()})
b2 = dict(p, layers={k: ["cb4"] * len(v) for k, v in p["layers"].items()})
json.dump(b1, open("/home/surya-koritala/glyd-scratch/qwen-users/plan_b1.json", "w")); json.dump(b2, open("/home/surya-koritala/glyd-scratch/qwen-users/plan_b2.json", "w"))
PYE
mkdir -p $U/packs; cd $Q
log "pack b1"; QWEN_SNAP=$Q/flash-next $PY qwen_pack.py pack --plan $U/plan_b1.json --out $U/packs/b1 > $U/pack_b1.log 2>&1 && QWEN_SNAP=$Q/flash-next $PY qwen_pack.py sidecar --pack $U/packs/b1 --only cb4 --fmt cb3 --out $U/packs/b1/hot >> $U/pack_b1.log 2>&1
log "b1 size $(du -sh $U/packs/b1 | cut -f1) hot $(du -sh $U/packs/b1/hot | cut -f1)"
run b1 $Q/slim $U/packs/b1 --precise
rm -rf $U/packs/b1
cd $Q; log "pack b2"; QWEN_SNAP=$Q/flash-next $PY qwen_pack.py pack --plan $U/plan_b2.json --out $U/packs/b2 > $U/pack_b2.log 2>&1
log "b2 size $(du -sh $U/packs/b2 | cut -f1)"
run b2 $Q/slim $U/packs/b2 --precise --no-side
log CHAIN3-DONE
