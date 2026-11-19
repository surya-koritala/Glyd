#!/bin/bash
# Host side on the Lambda 4x H100 (GLY-354, GLY-351, GLY-318): the four models downloaded into the Hub cache on the host (python3 is here; DeepSeek first,
# GLM last, so the checks start while the rest comes), the CUDA 13 container for the checks (the engine needs glibc 2.39).
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi topo -m | head -6; df -h ~ | tail -1
sudo nvidia-modprobe -u -c=0; sudo nvidia-smi -pm 1
# NVSwitch boxes: CUDA says 802 until the fabric manager has the fabric up (up to 6 min)
for i in $(seq 1 72); do
  st=$(nvidia-smi -q 2>/dev/null | grep -A3 -i "Fabric" | grep -i "State" | head -1 | awk -F: '{print $2}' | tr -d ' ')
  fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
  echo "$(date +%T) fabric: state=$st fabricmanager=$fm"
  [ "$st" = Completed ] && break
  [ -z "$st" ] && [ "$fm" != activating ] && [ $i -ge 6 ] && break
  sleep 5
done
python3 -m pip install -q --user huggingface_hub hf_transfer
( for r in deepseek-ai/DeepSeek-V4-Flash nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 glyd/nemotron-3-hummingbird zai-org/GLM-5.3-Flash; do
    t=$(date +%s); HF_HUB_ENABLE_HF_TRANSFER=1 ~/.local/bin/hf download $r > /dev/null; echo "$r exit $? in $(( $(date +%s) - t )) s"
  done; du -sh ~/.cache/huggingface/hub/*; echo "downloads done" ) > ~/w/out/download.txt 2>&1 &
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 30; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG > /dev/null
$D run --gpus all --rm --privileged --network host --ipc host -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/dc/inner.sh
kill $WD 2>/dev/null; ls ~/w/out
