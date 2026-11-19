#!/bin/bash
# Host side on a Lambda 8x H100 or A100 80 GB: Kimi-K3 into the Hub cache with hf (Xet), then the CUDA 13 container (the engine needs glibc 2.39) runs
# inner.sh: DeepSeek-V4-Flash and Kimi-K3 with every layer across the GPUs (GLYD_TP=1, branch gly-tp) and split by layers as main has them, the same requests.
TAG=${TAG:-13.0.3-devel-ubuntu24.04}
D=docker; docker ps > /dev/null 2>&1 || D="sudo docker"
mkdir -p ~/w/out; exec > ~/w/out/host.log 2>&1; set -x
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi topo -m > ~/w/out/topo.txt 2>&1; lscpu > ~/w/out/lscpu.txt; df -h ~ | tail -1; free -g | head -2
sudo nvidia-modprobe -u -c=0; sudo nvidia-smi -pm 1
for i in $(seq 1 72); do
  st=$(nvidia-smi -q 2>/dev/null | grep -A3 -i "Fabric" | grep -i "State" | head -1 | awk -F: '{print $2}' | tr -d ' ')
  fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
  echo "$(date +%T) fabric: state=$st fabricmanager=$fm"
  [ "$st" = Completed ] && break
  { [ -z "$st" ] || [ "$st" = N/A ]; } && [ "$fm" != activating ] && [ $i -ge 6 ] && break
  sleep 5
done
python3 -m pip install -q --user -U huggingface_hub hf_xet
( for r in deepseek-ai/DeepSeek-V4-Flash moonshotai/Kimi-K3; do
    t=$(date +%s); HF_XET_HIGH_PERFORMANCE=1 ~/.local/bin/hf download $r > /dev/null; echo "$r exit $? in $(( $(date +%s) - t )) s"
  done; du -sh ~/.cache/huggingface/hub/models--*; echo "downloads done" ) > ~/w/out/download.txt 2>&1 &
( while true; do echo "== $(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ') $(free -g | awk '/Mem/{print $3"G used, "$6"G cache"}')"; sleep 30; done ) > ~/w/out/host-watch.log 2>&1 &
WD=$!
$D pull nvidia/cuda:$TAG > /dev/null
$D run --gpus all --rm --privileged --network host --ipc host -e HOME=/home/ubuntu -v $HOME:/home/ubuntu -w /home/ubuntu nvidia/cuda:$TAG bash /home/ubuntu/dc/inner.sh
kill $WD 2>/dev/null; ls ~/w/out
