#!/bin/bash
# A100: the FP8 layouts, H (GLYD_F8_LAYOUT=h, the default on HBM GPUs) against the 6.75-bit one (c): DeepSeek's digits with each, the GLM experts' product's
# instructions and time (Nsight), GLM's 4-layer and DeepSeek's 3-layer steps twice each. Results in ~/w/out; DONE at the end.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/f8h/libglyd_gpu_cuda13.so; E=$HOME/f8h/glyd-engine; chmod +x $E
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv > $O/machine.txt
(sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)) > $O/apt.txt 2>&1 &
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
$E glmrandom $W/glm4 --real --fp8 --layers 4 --experts 32 > $O/gen-glm.txt 2>&1
for L in h c; do GLYD_F8_LAYOUT=$L $E dsv4check $W/rand --tokens 24 > $O/check-$L.txt 2>&1; done
for r in 1 2; do for L in h c; do
  GLYD_F8_LAYOUT=$L $E dsv4prof $W/glm4 --prompt 256 --tokens 32 > $O/glm-$L-r$r.txt 2>&1
  GLYD_F8_LAYOUT=$L $E dsv4prof $W/ds3 --prompt 256 --tokens 32 > $O/ds-$L-r$r.txt 2>&1
done; done
wait
N=$(ls /opt/nvidia/nsight-compute/*/ncu | head -1)
for L in h c; do sudo -E env GLYD_F8_LAYOUT=$L $N --section SpeedOfLight --section InstructionStats --section LaunchStats --section WarpStateStats -k regex:f8_linear_kernel -c 3 --target-processes all $E dsv4prof $W/glm4 --prompt 1 --tokens 1 > $O/ncu-glm-$L.txt 2>&1; done
for L in h c; do sudo -E env GLYD_F8_LAYOUT=$L $N --section SpeedOfLight --section InstructionStats -k regex:f8_linear_kernel -c 2 --target-processes all $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds-$L.txt 2>&1; done
touch $O/DONE
