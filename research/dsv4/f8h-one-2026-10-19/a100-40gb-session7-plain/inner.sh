#!/bin/bash
# Inside the nvidia/cuda 13.0 devel container (glibc 2.39, the forward-compatibility libcuda for the host's older driver), session 6: f8h_lin2_kernel (the copies two
# steps ahead) against f8h_lin1_kernel and f8_linear_kernel on an A100: the same digits, the time and instructions of each, then where the warps wait (stall reasons,
# by source line) for the dense product and the experts' product, and every kernel's time in one DeepSeek step and one GLM step.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so GLYD_F8_LAYOUT=h; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
ls -l /dev/nvidia* > $O/devices.txt 2>&1; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $O/devices.txt
$E f8moecheck --tokens 33 > $O/probe.txt 2>&1
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
run() {  # run "$@" > the last argument's file, again after 20 s where CUDA would not start, 8 times at most
  local out=${@: -1}; local n=0
  while :; do
    "${@:1:$#-1}" > $out 2>&1
    grep -qiE "cuInit: CUDA error|CUDA error 100" $out || return 0
    n=$((n + 1)); echo "$(date +%T) retry $n: $out" >> $O/retries.txt; [ $n -ge 8 ] && return 1; sleep 20
  done
}
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
$E glmrandom $W/glm4 --real --fp8 --layers 4 --experts 32 > $O/gen-glm.txt 2>&1
# the digits, every variant (ONE: 0 f8_linear_kernel, 1 f8h_lin1_kernel, 2 f8h_lin2_kernel; CTAS 3 or 2 an SM; PLAIN 1: every patch its e4m3 bytes, 8 bits a value)
for v in 1:3:0 2:3:0 2:3:1 1:3:1 0:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  run $E dsv4check $W/rand --tokens 24 $O/check-$v.txt
  run $E f8moecheck --tokens 33 --peaked $O/f8moe-$v.txt
  run $E f8moecheck --rows 2048 --cols 4096 --experts 16 --k 4 --tokens 33 --peaked $O/f8moeb-$v.txt
done
# the steps (graphs), two repeats each, the variants interleaved
for r in 1 2; do for v in 1:3:0 2:3:0 2:3:1 1:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  run $E dsv4prof $W/ds3 --prompt 256 --tokens 32 $O/ds-$v-r$r.txt
  run $E dsv4prof $W/glm4 --prompt 256 --tokens 32 $O/glm-$v-r$r.txt
done; done
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025' | sort | tail -1 | cut -d' ' -f1); [ -z "$NCU_PKG" ] && NCU_PKG=$(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NCU_PKG) > $O/apt.txt 2>&1
N=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1); [ -z "$N" ] && N=$(command -v ncu)
$E f8moecheck --tokens 33 > $O/probe2.txt 2>&1
M=smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio,smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
for v in 1:3:0 2:3:0 2:3:1 1:3:1 0:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  $N --metrics $M -k regex:"f8_linear_kernel|f8h_lin" -c 12 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds-$v.txt 2>&1
  $N --metrics $M -k regex:"f8_linear_kernel|f8h_lin" -c 12 $E dsv4prof $W/glm4 --prompt 1 --tokens 1 > $O/ncu-glm-$v.txt 2>&1
done
# where the warps wait, by instruction and source line: the dense product (DeepSeek) and the experts' product (GLM), coded (H) and plain (the bytes)
for v in 2:3:0 2:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  $N --set full --import-source yes -k regex:"f8h_lin" -c 2 -f -o $O/src-ds-$v $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-src-ds-$v.txt 2>&1
  $N -i $O/src-ds-$v.ncu-rep --page details > $O/details-ds-$v.txt 2>&1
  $N -i $O/src-ds-$v.ncu-rep --page source --csv --print-source sass > $O/sass-ds-$v.csv 2>&1
  $N -i $O/src-ds-$v.ncu-rep --page source --csv --print-source cuda > $O/cuda-ds-$v.csv 2>&1
  $N --set full --import-source yes -k regex:"f8h_lin" -c 2 -f -o $O/src-glm-$v $E dsv4prof $W/glm4 --prompt 1 --tokens 1 > $O/ncu-src-glm-$v.txt 2>&1
  $N -i $O/src-glm-$v.ncu-rep --page details > $O/details-glm-$v.txt 2>&1
  $N -i $O/src-glm-$v.ncu-rep --page source --csv --print-source sass > $O/sass-glm-$v.csv 2>&1
  $N -i $O/src-glm-$v.ncu-rep --page source --csv --print-source cuda > $O/cuda-glm-$v.csv 2>&1
done
rm -f $O/*.ncu-rep
touch $O/DONE
