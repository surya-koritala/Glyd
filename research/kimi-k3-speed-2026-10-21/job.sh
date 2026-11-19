#!/bin/bash
# Host side on a Lambda 8x H100 (GLY-330): Kimi-K3 downloaded into the Hub cache with hf (Xet, about 1.7 GB/s; GLY-356), the CUDA 13 container (the engine needs
# glibc 2.39) runs inner.sh: `glyd serve` and `glyd run` on the dynamic engine with the expert cache (1.45 TB of experts in host memory, places on the GPUs).
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi topo -m | head -10; df -h ~ | tail -1; free -g | head -2
sudo nvidia-modprobe -u -c=0; sudo nvidia-smi -pm 1
# NVSwitch boxes: CUDA says 802 until the fabric manager has the fabric up (up to 6 min)
for i in $(seq 1 72); do
  st=$(nvidia-smi -q 2>/dev/null | grep -A3 -i "Fabric" | grep -i "State" | head -1 | awk -F: '{print $2}' | tr -d ' ')
  fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
  echo "$(date +%T) fabric: state=$st fabricmanager=$fm"
  [ "$st" = Completed ] && break
  { [ -z "$st" ] || [ "$st" = N/A ]; } && [ "$fm" != activating ] && [ $i -ge 6 ] && break  # (N/A: an A100 HGX board, no fabric state)
  sleep 5
done
python3 -m pip install -q --user -U huggingface_hub hf_xet
( for r in moonshotai/Kimi-K3 deepseek-ai/DeepSeek-V4-Flash; do
    t=$(date +%s); HF_XET_HIGH_PERFORMANCE=1 ~/.local/bin/hf download $r > /dev/null; echo "$r exit $? in $(( $(date +%s) - t )) s"
  done; du -sh ~/.cache/huggingface/hub/models--*; echo "downloads done" ) > ~/w/out/download.txt 2>&1 &
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ') $(free -g | awk '/Mem/{print $3"G used, "$6"G cache"}')"; sleep 30; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG > /dev/null
$D run --gpus all --rm --privileged --network host --ipc host -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/dc/inner.sh
kill $WD 2>/dev/null; ls ~/w/out
