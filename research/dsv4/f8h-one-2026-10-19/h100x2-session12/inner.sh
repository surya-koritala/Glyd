#!/bin/bash
# In the CUDA 13.0 container on the 2x H100: today's kernel (ONE=1), the lean plain kernel (ONE=3, PLAIN=1) and lin3 on plain packs (ONE=4, PLAIN=1): digits,
# steps, ncu per kernel; then the real DeepSeek-V4-Flash on the two GPUs with layout H and with raw FP8 bytes (PLAIN=1): text, ms a token, score.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so GLYD_F8_LAYOUT=h; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
for i in $(seq 1 12); do  # (error 802: the fabric still coming up; 100: the GPU not yet in the container)
  $E f8moecheck --tokens 33 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  echo "$(date +%T) probe retry $i: $(grep -i 'CUDA error' $O/probe.txt | head -1)" >> $O/retries.txt; sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
run() { local out=${@: -1}; local n=0; while :; do "${@:1:$#-1}" > $out 2>&1; grep -qiE "cuInit: CUDA error|CUDA error 100" $out || return 0; n=$((n + 1)); echo "$(date +%T) retry $n: $out" >> $O/retries.txt; [ $n -ge 8 ] && return 1; sleep 20; done; }
$E dsv4random $W/rand > $O/gen-rand.txt 2>&1
$E dsv4random $W/ds3 --real --layers 3 --experts 32 > $O/gen-ds.txt 2>&1
$E glmrandom $W/glm4 --real --fp8 --layers 4 --experts 32 > $O/gen-glm.txt 2>&1
for v in 1:3:0 3:3:1 4:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  run $E dsv4check $W/rand --tokens 24 $O/check-$v.txt
  run $E f8moecheck --rows 2048 --cols 4096 --experts 16 --k 4 --tokens 33 --peaked $O/f8moeb-$v.txt
done
for r in 1 2; do for v in 1:3:0 3:3:1 4:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  run $E dsv4prof $W/ds3 --prompt 256 --tokens 32 $O/ds-$v-r$r.txt
  run $E dsv4prof $W/glm4 --prompt 256 --tokens 32 $O/glm-$v-r$r.txt
done; done
NCU_PKG=$(apt-cache search --names-only '^nsight-compute-2025' | sort | tail -1 | cut -d' ' -f1); [ -z "$NCU_PKG" ] && NCU_PKG=$(apt-cache search --names-only '^nsight-compute-20' | sort | tail -1 | cut -d' ' -f1)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NCU_PKG) > $O/apt.txt 2>&1
N=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1); [ -z "$N" ] && N=$(command -v ncu)
M=smsp__inst_executed.sum,gpu__time_duration.sum,dram__bytes_read.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
for v in 1:3:0 3:3:1 4:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
  export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  $N --metrics $M -k regex:"f8_linear_kernel|f8h_lin|f8b_lin|mma_moe_kernel" -c 16 $E dsv4prof $W/ds3 --prompt 1 --tokens 1 > $O/ncu-ds-$v.txt 2>&1
  $N --metrics $M -k regex:"f8_linear_kernel|f8h_lin|f8b_lin|mma_moe_kernel" -c 16 $E dsv4prof $W/glm4 --prompt 1 --tokens 1 > $O/ncu-glm-$v.txt 2>&1
done
# the real model: wait for the host's download
for i in $(seq 1 90); do grep -q "download exit" $O/download.txt 2>/dev/null && break; sleep 20; done
cat $O/download.txt
if grep -q "download exit 0" $O/download.txt; then
  for v in 1:3:0 3:3:1; do IFS=: read -r one c pl <<< "$v"; v=$one-$c-$pl
    export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
    nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/real-mem-before-$v.txt
    run $E dsv4gen $W/ds-real --prompt "The capital of France is" --tokens 32 --split 22 --devices 0,1 $O/real-paris-$v.txt
    run $E dsv4gen $W/ds-real --score $HOME/dc/score-text.txt --split 22 --devices 0,1 $O/real-score-$v.txt
  done
else
  echo "no real model" > $O/real-FAIL
fi
touch $O/DONE
