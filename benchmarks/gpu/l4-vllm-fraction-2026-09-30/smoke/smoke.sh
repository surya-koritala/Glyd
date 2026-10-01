#!/bin/bash
# A smoke of `fraction` on Qwen3-0.6B, eager, under the box's lock: bf16, then fractions 0.5, 0 and 1 (the tokens and
# logprobs of 0 and 1 against bf16's and Glyd's as they were).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
# the GPU free of others' processes before anything starts (up to 10 minutes)
for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 1500 ] && break; sleep 5; done
echo "gpu used ${used} MiB at $(date -u +%T)"
B=~/budget; S=$B/src; O=$B/smoke; mkdir -p $O $B/tmp
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$HOME/mmoedry/w/lib/libglyd_gpu_cuda13.so
export PYTHONPATH=$S/bindings/python TMPDIR=$B/tmp VLLM_CACHE_ROOT=$B/cache/smoke
PY=~/mmoedry/w/venv/bin/python
M=${1:-Qwen/Qwen3-0.6B}
cd $B   # (not a directory holding gpu/: it would shadow glyd.gpu)
for f in bf16 0.5 0 1; do
  echo "== $(date -u +%T) fraction $f"
  if [ $f = bf16 ]; then $PY $B/smoke.py $M 1 $O/$f.json bf16 > $O/$f.log 2>&1; else $PY $B/smoke.py $M $f $O/$f.json > $O/$f.log 2>&1; fi
  echo "exit $?"; grep -E "^SMOKE|glyd:|Error|error" $O/$f.log | grep -v "^WARNING" | cut -c1-400 | tail -6
done
$PY - <<'PYEOF'
import json
o = "/home/ubuntu/budget/smoke/"
r = {k: json.load(open(o + k + ".json")) for k in ("bf16", "0.5", "0", "1")}
for k in ("0.5", "0", "1"):
    print(k, "tokens == bf16's:", r[k]["tokens"] == r["bf16"]["tokens"], "| logprobs bit for bit:", r[k]["logprobs"] == r["bf16"]["logprobs"], "| KV blocks", r[k]["blocks"], "bf16", r["bf16"]["blocks"])
PYEOF
