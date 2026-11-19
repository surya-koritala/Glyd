#!/bin/bash
# The host side on the Lambda A100: run inner.sh in the CUDA 13.0 devel container (the host's Lambda Stack has driver 570 and glibc 2.35; the engine needs 2.39 and CUDA 13).
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version --format=csv; docker --version
sudo nvidia-modprobe -u -c=0; ls -l /dev/nvidia-uvm /dev/nvidia0 /dev/nvidiactl
sudo nvidia-smi -pm 1; systemctl is-active nvidia-fabricmanager nvidia-persistenced unattended-upgrades; uptime
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=pstate,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1) fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null) uu=$(systemctl is-active unattended-upgrades 2>/dev/null) apt=$(pgrep -c -f 'apt|dpkg')"; sudo dmesg | grep -iE "nvrm|xid|nvidia" | tail -n 3; sleep 20; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG
$D run --gpus all --rm --cap-add=SYS_ADMIN -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/inner.sh
kill $WD 2>/dev/null; sudo dmesg | tail -n 40 > ~/w/out/dmesg-tail.txt 2>&1; ls ~/w/out
