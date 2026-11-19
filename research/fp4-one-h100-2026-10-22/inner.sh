#!/bin/bash
# Inside the nvidia/cuda 13.0 devel container on one H100 (GLY-349 step 2, GLY-318): where a one-token FP4 experts' product goes on the H100 (31% of the
# memory's rate in October's records). DeepSeek-V4-Flash at the release's widths on random weights (dsv4random --real: 3 layers, 32 experts), dsv4prof with the
# experts' steps; Kimi-K3's first two real layers (hubslice, 24 GB: its experts all on the GPU, then 318 places as on 8x A100); then Nsight Compute on the
# experts' kernels (instructions, time, bytes, issue) and one SourceCounters report.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
# the forward-compatibility driver library, copied aside: installing Nsight Compute later replaces /usr/local/cuda/compat with one the host's kernel module refuses
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,memory.total,driver_version --format=csv > $O/machine.txt
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4check $W/rand --tokens 8 > $O/probe.txt 2>&1
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; echo DONE; exit 1; fi
( $E hubslice moonshotai/Kimi-K3 $W/kimi2 --layers 2 --prefix language_model.model. > $O/slice.txt 2>&1; echo "slice exit $?" >> $O/slice.txt ) &
SL=$!
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
for r in 1 2; do $E dsv4prof $W/ds3 --prompt 256 --tokens 64 > $O/ds-prof-r$r.txt 2>&1; done
GLYD_MOE_BESIDE=0 $E dsv4prof $W/ds3 --prompt 256 --tokens 64 > $O/ds-prof-serial.txt 2>&1
# (the library's one-token variants not yet measured on an H100, each the same mma order: two steps in flight for FP4R, the steps through shared memory in 4 or 8
# slots, 3 blocks an SM, the L2 prefetch)
for v in "GLYD_GPU_DEPTH=2" "GLYD_GPU_STAGES=4" "GLYD_GPU_STAGES=8" "GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_BLOCKS=3" "GLYD_GPU_PF=2"; do
  env $v $E dsv4prof $W/ds3 --prompt 256 --tokens 64 > "$O/ds-prof-$(echo $v | tr ' =' '_-').txt" 2>&1
done
wait $SL
if grep -q "slice exit 0" $O/slice.txt; then
  for r in 1 2; do $E dsv4prof $W/kimi2 --prompt 64 --tokens 64 --ctx 1024 > $O/kimi-prof-r$r.txt 2>&1; done
  GLYD_KIMI_PLACES=318 GLYD_KIMI_STATS=1 $E dsv4prof $W/kimi2 --prompt 64 --tokens 200 --ctx 1024 > $O/kimi-prof-318.txt 2>&1
fi
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025' | sort | tail -1 | cut -d' ' -f1); [ -z "$NCU_PKG" ] && NCU_PKG=$(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NCU_PKG) > $O/apt.txt 2>&1
N=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1); [ -z "$N" ] && N=$(command -v ncu)
$E dsv4check $W/rand --tokens 8 > $O/probe2.txt 2>&1  # (the GPU still there after the install)
M=smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active
# (one decode step's kernels: the experts' (mma_moe), the dense FP8 products (f8), the shared expert's; a prompt of one token, one step)
$N --metrics $M -k regex:"mma_moe|f8h_lin1|f8b_lin1|f8_linear" -c 40 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds.txt 2>&1
[ -d $W/kimi2 ] && $N --metrics $M -k regex:"mma_moe|mma_gemm|step_kernel|tiered" -c 40 $E dsv4prof $W/kimi2 --prompt 1 --tokens 1 > $O/ncu-kimi.txt 2>&1
$N --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section SourceCounters --warp-sampling-interval 0 --import-source no -k regex:"mma_moe" -c 2 -f -o $O/moe $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-moe.txt 2>&1
$N -i $O/moe.ncu-rep --page details > $O/moe-details.txt 2>&1
$N -i $O/moe.ncu-rep --page source --csv --print-source sass > $O/moe-sass.csv 2>&1
# (every kernel of one decode step, its GPU time alone: the prompt of one token's launches skipped)
$N --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size -s 110 -c 110 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-all-ds.txt 2>&1
[ -d $W/kimi2 ] && $N --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size -s 60 -c 70 $E dsv4prof $W/kimi2 --prompt 1 --tokens 1 > $O/ncu-all-kimi.txt 2>&1
echo DONE
