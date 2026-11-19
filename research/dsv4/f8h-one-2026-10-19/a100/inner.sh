#!/bin/bash
# Inside the nvidia/cuda 13.0 devel container (glibc 2.39, the forward-compatibility libcuda for the host's older driver): f8h_lin1_kernel (one token, layout H)
# against f8_linear_kernel on an A100: the same digits, then time and instructions.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so GLYD_F8_LAYOUT=h; E=$HOME/dc/glyd-engine; chmod +x $E
# the forward-compatibility driver library (a host driver older than CUDA 13's), copied aside: installing Nsight Compute later replaces /usr/local/cuda/compat with one the host's kernel module refuses (cuInit error 100)
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
ls -l /dev/nvidia* > $O/devices.txt 2>&1; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $O/devices.txt; ls -l $HOME/compat >> $O/devices.txt 2>&1
$E f8moecheck --tokens 33 > $O/probe.txt 2>&1
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
run() {  # run "$@" > the last argument's file, again after 20 s where CUDA would not start (the GPU went away for minutes in the earlier sessions), 8 times at most
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
for v in 0:3 1:3 1:2; do one=${v%%:*}; c=${v#*:}
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c
  run $E dsv4check $W/rand --tokens 24 $O/check-$one-$c.txt
  run $E f8moecheck --tokens 33 --peaked $O/f8moe-$one-$c.txt
  run $E f8moecheck --rows 2048 --cols 4096 --experts 16 --k 4 --tokens 33 --peaked $O/f8moeb-$one-$c.txt
  for r in 1 2; do
    run $E dsv4prof $W/ds3 --prompt 256 --tokens 32 $O/ds-$one-$c-r$r.txt
    run $E dsv4prof $W/glm4 --prompt 256 --tokens 32 $O/glm-$one-$c-r$r.txt
  done
done
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025' | sort | tail -1 | cut -d' ' -f1); [ -z "$NCU_PKG" ] && NCU_PKG=$(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NCU_PKG) > $O/apt.txt 2>&1
N=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1); [ -z "$N" ] && N=$(command -v ncu)
$E f8moecheck --tokens 33 > $O/probe2.txt 2>&1  # (the GPU still there after the install)
for v in 0:3 1:3 1:2; do one=${v%%:*}; c=${v#*:}
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c
  $N --metrics smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active -k regex:"f8_linear_kernel|f8h_lin1" -c 12 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds-$one-$c.txt 2>&1
  $N --metrics smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active -k regex:"f8_linear_kernel|f8h_lin1" -c 12 $E dsv4prof $W/glm4 --prompt 1 --tokens 1 > $O/ncu-glm-$one-$c.txt 2>&1
done
export GLYD_F8H_ONE=1 GLYD_F8H_CTAS=3
$N --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section SourceCounters --warp-sampling-interval 0 --import-source no -k regex:"f8h_lin1" -c 2 -f -o $O/l1 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-l1.txt 2>&1
$N -i $O/l1.ncu-rep --page details > $O/l1-details.txt 2>&1
$N -i $O/l1.ncu-rep --page source --csv --print-source sass > $O/l1-sass.csv 2>&1
touch $O/DONE
