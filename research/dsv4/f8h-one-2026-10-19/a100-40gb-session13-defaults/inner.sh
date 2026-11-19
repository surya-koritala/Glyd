#!/bin/bash
# A100, session 13: the new defaults on an HBM GPU (plain packs, f8b_lin1_kernel) with no environment set, the coded layout as the opt-out, and the FP4 experts'
# ring with four steps a stage (GLYD_GPU_STAGE_STEPS=4) against one step a stage: digits, steps, ncu per kernel.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
for i in $(seq 1 12); do $E f8moecheck --tokens 33 > $O/probe.txt 2>&1; grep -qiE "CUDA error (802|100)" $O/probe.txt || break; sleep 20; done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
run() { local out=${@: -1}; local n=0; while :; do "${@:1:$#-1}" > $out 2>&1; grep -qiE "cuInit: CUDA error|CUDA error 100" $out || return 0; n=$((n + 1)); [ $n -ge 8 ] && return 1; sleep 20; done; }
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
setv() { unset GLYD_F8H_PLAIN GLYD_F8H_ONE GLYD_GPU_STAGES GLYD_GPU_STAGE_STEPS
  case $1 in def) ;; coded) export GLYD_F8H_PLAIN=0;; s8) export GLYD_GPU_STAGES=8;; s8x4) export GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_STEPS=4;; s4x4) export GLYD_GPU_STAGES=4 GLYD_GPU_STAGE_STEPS=4;; esac; }
for v in def coded s8 s8x4 s4x4; do setv $v
  run $E dsv4check $W/rand --tokens 24 $O/check-$v.txt
  run $E f8moecheck --rows 2048 --cols 4096 --experts 16 --k 4 --tokens 33 --peaked $O/f8moeb-$v.txt
done
for r in 1 2; do for v in def coded s8 s8x4 s4x4; do setv $v; run $E dsv4prof $W/ds3 --prompt 256 --tokens 32 $O/ds-$v-r$r.txt; done; done
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025' | sort | tail -1 | cut -d' ' -f1); [ -z "$NCU_PKG" ] && NCU_PKG=$(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NCU_PKG) > $O/apt.txt 2>&1
N=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1); [ -z "$N" ] && N=$(command -v ncu)
M=smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
for v in def coded s8 s8x4 s4x4; do setv $v
  $N --metrics $M -k regex:"f8_linear_kernel|f8h_lin|f8b_lin|mma_moe_kernel" -c 16 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds-$v.txt 2>&1
done
touch $O/DONE
