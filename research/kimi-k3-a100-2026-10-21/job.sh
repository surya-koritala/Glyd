#!/bin/bash
# Host side on a Lambda 1x A100 (Kimi-K3's real weights before the 8x H100 session): the CUDA 13 container (the engine needs glibc 2.39) runs inner.sh.
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; df -h ~ | tail -1; free -g | head -2
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) $(free -g | awk '/Mem/{print $3"G used"}')"; sleep 30; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG > /dev/null
$D run --gpus all --rm --privileged --network host --ipc host -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/dc/inner.sh
kill $WD 2>/dev/null; ls ~/w/out
