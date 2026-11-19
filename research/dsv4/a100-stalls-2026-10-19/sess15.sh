#!/bin/bash
# A100: per-instruction stall samples (Nsight Compute source counters) of the one-token products: DeepSeek's FP4R experts (mma_moe_kernel, gate/up then down),
# GLM's FP8 products (f8_linear_kernel), Gemma 4 12B penguin's dense step_kernel. Results in ~/w/out; DONE at the end.
W=$HOME/w; O=$W/out; mkdir -p $O $W/hub; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/stall/libglyd_gpu_cuda13.so; E=$HOME/stall/glyd-engine; chmod +x $E
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv > $O/machine.txt
(sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)) > $O/apt.txt 2>&1 &
(HF_HUB_ENABLE_HF_TRANSFER=1 $HOME/gpuenv/bin/hf download glyd/gemma-4-12B-penguin --cache-dir $W/hub > $O/dl.txt 2>&1) &
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
$E glmrandom $W/glm4 --real --fp8 --layers 4 --experts 32 > $O/gen-glm.txt 2>&1
wait
G=$(ls -d $W/hub/models--glyd--gemma-4-12B-penguin/snapshots/* | head -1)
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu | head -1)
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section InstructionStats --section SourceCounters --warp-sampling-interval 0 --import-source no"
prof() {  # tag, kernel regex, launches to skip, command...
  local tag=$1 k=$2 s=$3; shift 3
  sudo -E $NCU $SEC -k regex:$k --launch-skip $s -c 1 --target-processes all -f -o $O/$tag "$@" > $O/ncu-$tag.txt 2>&1
  $NCU -i $O/$tag.ncu-rep --page details > $O/$tag-details.txt 2>&1
  $NCU -i $O/$tag.ncu-rep --page source --csv --print-source sass > $O/$tag-sass.csv 2>&1
}
prof fp4-gu mma_moe_kernel 0 $E dsv4prof $W/ds3 --prompt 1 --tokens 1
prof fp4-dn mma_moe_kernel 1 $E dsv4prof $W/ds3 --prompt 1 --tokens 1
prof f8-ds f8_linear_kernel 0 $E dsv4prof $W/ds3 --prompt 1 --tokens 1
prof f8-glm f8_linear_kernel 0 $E dsv4prof $W/glm4 --prompt 1 --tokens 1
prof f8-glm2 f8_linear_kernel 1 $E dsv4prof $W/glm4 --prompt 1 --tokens 1
prof step step_kernel 0 $E prodbench $G
prof step2 step_kernel 8 $E prodbench $G
touch $O/DONE
