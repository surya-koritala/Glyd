#!/bin/bash
# One H100 (GLY-334, GLY-330): the penguin level's two lossless layouts at one token, the smallest (tiered, `--layout mma`) against the 12-bit (`mma12`), on Kimi-K3's
# first two real layers and DeepSeek-V4-Flash's widths (random); the depth mix and norm kernels of #223 in the build.
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
for l in mma mma12 mma mma12; do $E dsv4prof $W/ds3 --prompt 256 --tokens 64 --layout $l > $O/ds-$l-$RANDOM.txt 2>&1; done
wait $SL
if grep -q "slice exit 0" $O/slice.txt; then
  for l in mma mma12 mma mma12; do $E dsv4prof $W/kimi2 --prompt 64 --tokens 64 --ctx 1024 --layout $l > $O/kimi-$l-$RANDOM.txt 2>&1; done
fi
echo DONE
