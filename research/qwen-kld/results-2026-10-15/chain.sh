#!/bin/bash
# KLD chain: bf16 reference (lock per layer), then our engine precise and fast (lock for each run), then the numbers.
set -u
K=$HOME/glyd-scratch/qwen-users/kld; Q=$HOME/glyd-scratch/qwen; cd $K; . $HOME/glyd-scratch/qwen-build/env.sh
REF="env QWEN_SNAP=$Q/flash-next QWEN_BATCH=16 $HOME/glyd-scratch/qwen-build/venv/bin/python research/qwen-kld/kld_ref.py"
ENG="$HOME/glyd-scratch/qwen-users/venv/bin/python research/qwen-kld/kld_engine.py"
CMP="$HOME/glyd-scratch/qwen-users/venv/bin/python research/qwen-kld/kld_compare.py"
E=$HOME/glyd-scratch/qwen-users/bin/glyd-engine
echo "ref long $(date +%T)"; $REF --tokens long_chunks.json --out ref_long.f32 > ref_long.log 2>&1
echo "ref 32 $(date +%T)"; $REF --tokens chunks32.json --out ref32.f32 > ref32.log 2>&1
for m in precise fast; do
  X=; [ $m = precise ] && X=--precise
  P=$Q/packed/${m}_x
  echo "$m long $(date +%T)"; flock -w 7200 $HOME/glyd-scratch/gpu.lock $Q/gpurun.sh $ENG --tokens long_chunks.json --out ${m}_long.f32 -- $E qwenserve $Q/slim $P $X > ${m}_long.log 2>&1
  $CMP long_chunks.json ref_long.f32 ${m}_long.f32 > kld_${m}_long.json 2>/dev/null
  echo "$m 32 $(date +%T)"; flock -w 7200 $HOME/glyd-scratch/gpu.lock $Q/gpurun.sh $ENG --tokens chunks32.json --out ${m}32.f32 -- $E qwenserve $Q/slim $P $X > ${m}32.log 2>&1
  $CMP chunks32.json ref32.f32 ${m}32.f32 > kld_${m}_wiki32.json 2>/dev/null
done
echo "CHAIN-DONE $(date +%T)"
