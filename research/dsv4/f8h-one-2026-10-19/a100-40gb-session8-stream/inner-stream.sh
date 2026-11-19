#!/bin/bash
# Inside the nvidia/cuda 13.0 devel container on the A100: the streaming micro-benchmark (stream.cu) for the one-token product's byte stream, two matrix sizes.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,memory.total,driver_version --format=csv > $O/machine.txt
nvcc --version | tail -2
nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90a,code=sm_90a -o $W/stream $HOME/dc/stream.cu > $O/build.txt 2>&1 || { cat $O/build.txt; echo FAIL-BUILD > $O/FAIL; touch $O/DONE; exit 1; }
for r in 1 2; do
  $W/stream 8192 4096 15 > $O/stream-8192x4096-r$r.txt 2>&1
  $W/stream 29696 4096 9 > $O/stream-29696x4096-r$r.txt 2>&1
  $W/stream 4096 7168 15 > $O/stream-4096x7168-r$r.txt 2>&1
done
nvidia-smi -q -d CLOCK | grep -A3 -E "^\s+Clocks$|Max Clocks" | head -12 > $O/clocks.txt
touch $O/DONE
