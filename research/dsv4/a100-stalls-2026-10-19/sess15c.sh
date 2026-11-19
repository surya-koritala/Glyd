#!/bin/bash
# The rebuilt library (FP4R's step / ks as a multiply-high, lib80d): checks first (moecheck's kinds, dsv4check's digits on the random model: 11 of 24, 2.4437),
# then the experts at one and two steps ahead, twice each, and the gate/up product's instruction count. Results in ~/w/out3; DONE at the end.
W=$HOME/w; O=$W/out3; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
E=$HOME/stall/glyd-engine; L=$HOME/lib80d/libglyd_gpu_cuda13.so
GLYD_GPU_LIB=$L $E moecheck > $O/moecheck.txt 2>&1
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
GLYD_GPU_LIB=$L $E dsv4check $W/rand --tokens 24 > $O/check-new.txt 2>&1
GLYD_GPU_LIB=$L GLYD_GPU_DEPTH=2 $E dsv4check $W/rand --tokens 24 > $O/check-new-d2.txt 2>&1
for r in 1 2; do for lib in old new; do for d in 1 2; do
  [ $lib = old ] && export GLYD_GPU_LIB=$HOME/stall/libglyd_gpu_cuda13.so || export GLYD_GPU_LIB=$L
  GLYD_GPU_DEPTH=$d $E moecheck dsv4 --speed --no-check > $O/moe-$lib-d$d-r$r.txt 2>&1
done; done; done
export GLYD_GPU_LIB=$L
N=$(ls /opt/nvidia/nsight-compute/*/ncu | head -1)
for d in 1 2; do sudo -E env GLYD_GPU_DEPTH=$d $N --section SpeedOfLight --section WarpStateStats --section InstructionStats --section LaunchStats -k regex:mma_moe_kernel -c 1 --target-processes all $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-gu-d$d.txt 2>&1; done
for r in 1 2; do for d in 1 2; do GLYD_GPU_DEPTH=$d $E dsv4prof $W/ds3 --prompt 256 --tokens 32 > $O/ds-new-d$d-r$r.txt 2>&1; done; done
touch $O/DONE
