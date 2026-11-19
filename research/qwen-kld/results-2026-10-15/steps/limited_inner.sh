#!/bin/bash
# The dev server on port 8093, run inside the GPU lock by its caller. PACK (precise_x | fast_x) or PACKDIR (any pack, precise mode), EXTRA (more
# engine flags), CTX (default 32768), HOLD_GIB: VRAM held
# by another process first (the engine sizes itself from what is left, as on a smaller card), RAM_MAX: a user cgroup memory limit.
U=$HOME/glyd-scratch/qwen-users; Q=$HOME/glyd-scratch/qwen; . $HOME/glyd-scratch/qwen-build/env.sh
P=${PACK:-precise_x}; M=; [ "$P" = precise_x ] && M=--precise
PD=${PACKDIR:-$Q/packed/$P}; [ -n "${PACKDIR:-}" ] && M=--precise
if [ "${HOLD_GIB:-0}" != 0 ]; then
  $HOME/glyd-scratch/qwen-build/venv/bin/python -c "import torch,time; x=torch.empty(int($HOLD_GIB*2**30),dtype=torch.uint8,device='cuda'); print('holding $HOLD_GIB GiB', flush=True); time.sleep(10**7)" &
  H=$!; sleep 10
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
CG=; [ -n "${RAM_MAX:-}" ] && CG="systemd-run --user --scope -q -p MemoryMax=$RAM_MAX -p MemorySwapMax=0"
$CG $U/bin/glyd-engine qwenserve $Q/slim $PD $M ${EXTRA:-} --ctx ${CTX:-32768} --port 8093
[ -n "${H:-}" ] && kill $H
