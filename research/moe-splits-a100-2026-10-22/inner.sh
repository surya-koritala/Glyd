#!/bin/bash
# One H100 (GLY-349): the FP4 experts' one-token product and its waves. moe_splits gives DeepSeek's 384 units (6 experts x 64 row blocks) one split on
# 264 slots (2 blocks an SM x 132): 1.45 waves. GLYD_GPU_MOE_SLOTS forces the slots (so the splits); with 3 blocks an SM (the staged loop) too.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4check $W/rand --tokens 8 > $O/probe.txt 2>&1
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; echo DONE; exit 1; fi
( $E hubslice moonshotai/Kimi-K3 $W/kimi2 --layers 2 --prefix language_model.model. > $O/slice.txt 2>&1; echo "slice exit $?" >> $O/slice.txt ) &
SL=$!
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
run() { local tag=$1; shift; env "$@" $E dsv4prof $W/ds3 --prompt 256 --tokens 64 > $O/ds-$tag.txt 2>&1; }
run base GLYD_X=0; run s528 GLYD_GPU_MOE_SLOTS=528; run s792 GLYD_GPU_MOE_SLOTS=792; run s1056 GLYD_GPU_MOE_SLOTS=1056
run st3 GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_BLOCKS=3; run st3s792 GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_BLOCKS=3 GLYD_GPU_MOE_SLOTS=792; run base2 GLYD_X=0
GLYD_GPU_MOE_SLOTS=792 $E dsv4check $W/rand --tokens 24 > $O/check-s792.txt 2>&1; $E dsv4check $W/rand --tokens 24 > $O/check-base.txt 2>&1
wait $SL
if grep -q "slice exit 0" $O/slice.txt; then
  krun() { local tag=$1; shift; env "$@" $E dsv4prof $W/kimi2 --prompt 64 --tokens 64 --ctx 1024 --layout mma12 > $O/kimi-$tag.txt 2>&1; }
  krun base GLYD_X=0; krun s1848 GLYD_GPU_MOE_SLOTS=1848; krun st3 GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_BLOCKS=3; krun base2 GLYD_X=0
fi
echo DONE
