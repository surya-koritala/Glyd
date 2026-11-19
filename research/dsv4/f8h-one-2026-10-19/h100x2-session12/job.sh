#!/bin/bash
# Host side on the Lambda 2x H100: the real DeepSeek-V4-Flash download on the host (python3 is here), the container for the rest (the engine needs glibc 2.39, CUDA 13).
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi topo -m | head -5; docker --version
sudo nvidia-modprobe -u -c=0; ls -l /dev/nvidia-uvm /dev/nvidia0 /dev/nvidia1 /dev/nvidiactl
sudo nvidia-smi -pm 1
# NVLink/NVSwitch boxes: CUDA says 802 (system not yet initialized) until the fabric manager has brought the fabric up; wait for it (up to 6 min)
for i in $(seq 1 72); do
  st=$(nvidia-smi -q 2>/dev/null | grep -A3 -i "Fabric" | grep -i "State" | head -1 | awk -F: '{print $2}' | tr -d ' ')
  fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
  echo "$(date +%T) fabric: state=$st fabricmanager=$fm"
  [ "$st" = Completed ] && break
  [ -z "$st" ] && [ "$fm" != activating ] && [ $i -ge 6 ] && break   # (no fabric state at all: not an NVSwitch box)
  sleep 5
done
nvidia-smi -q | grep -A4 -i "Fabric" | head -6
(python3 -m pip install -q --user huggingface_hub hf_transfer && HF_HUB_ENABLE_HF_TRANSFER=1 ~/.local/bin/hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/w/ds-real; echo "download exit $?"; du -sh ~/w/ds-real) > ~/w/out/download.txt 2>&1 &
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 30; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG > /dev/null
$D run --gpus all --rm --privileged -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/inner.sh
kill $WD 2>/dev/null; ls ~/w/out
