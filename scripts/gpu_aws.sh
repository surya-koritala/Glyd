#!/usr/bin/env bash
# Large models with their weights compressed in GPU memory, on AWS: one
# g6e.12xlarge (4x NVIDIA L40S, 48 GB each) runs gpu/e2e.py on
# Qwen2.5-32B-Instruct and Qwen2.5-72B-Instruct, each in bf16 across the
# GPUs it needs and packed on fewer. Everything created (instance, key,
# security group) is tagged glyd-bench and deleted on exit.
# Results in benchmarks/gpu/<instance type>/.
#   AWS_PROFILE=... scripts/gpu_aws.sh [git-ref]
# Env: REGION (us-east-1), TYPE (g6e.12xlarge, $10.49/hour on demand).
# About an hour.
set -euo pipefail
REF="${1:-HEAD}"
REGION="${REGION:-us-east-1}"
TYPE="${TYPE:-g6e.12xlarge}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/gpu/$TYPE"
RUN_ID="glyd-gpu-$(date +%Y%m%d-%H%M%S)"
KEY="$RUN_ID"
TMP="$(mktemp -d)"
KEYFILE="$TMP/$KEY.pem"
TARBALL="$TMP/glyd.tar"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
ID=""; SG=""
aws() { command aws --region "$REGION" --output text "$@"; }
cleanup() {
    set +e
    echo "cleanup: terminating ${ID:-nothing}"
    [ -n "$ID" ] && aws ec2 terminate-instances --instance-ids "$ID" >/dev/null && aws ec2 wait instance-terminated --instance-ids "$ID"
    [ -n "$SG" ] && aws ec2 delete-security-group --group-id "$SG"
    aws ec2 delete-key-pair --key-name "$KEY"
    rm -rf "$TMP"
}
trap cleanup EXIT
COMMIT="$(git -C "$ROOT" rev-parse --short "$REF")"
git -C "$ROOT" archive --format=tar -o "$TARBALL" "$REF" gpu
aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd gpu" --vpc-id "$VPC" --query GroupId)"
aws ec2 create-tags --resources "$SG" --tags "Key=glyd-bench,Value=$RUN_ID"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null
# The instance's NVMe for the models (210 GB of bf16 weights), else the root volume.
USERDATA='#!/bin/bash
for d in /dev/nvme*n1; do
  if ! lsblk -n "$d" | grep -q part && ! findmnt -S "$d" >/dev/null; then mkfs.ext4 -q -F "$d" && mkdir -p /data && mount "$d" /data && break; fi
done
mkdir -p /data && chown ubuntu /data
touch /home/ubuntu/READY
'
RUN='set -x
mkdir -p ~/results
echo "commit: COMMIT_PLACEHOLDER" > ~/results/machine.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> ~/results/machine.txt
free -g | head -2 >> ~/results/machine.txt
bash ~/glyd/gpu/setup_env.sh ~/gpuenv > ~/results/setup.txt 2>&1 || { echo "SETUP FAILED" > ~/results/FAILED; touch ~/results/DONE; exit 1; }
source ~/gpuenv/cuda.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct; do
  (hf download Qwen/$m --local-dir /data/$m || huggingface-cli download Qwen/$m --local-dir /data/$m) > ~/results/download-$m.txt 2>&1 &
done
wait
du -sh /data/Qwen2.5-* >> ~/results/machine.txt
cd ~/glyd/gpu
python -c "import glyd_gpu" > ~/results/build.txt 2>&1
E="python e2e.py --fused --tokens 64 --batch 1,4 --prefill 128,512"
# 32B: bf16 on 2 GPUs (it needs them), then packed on 1.
$E /data/Qwen2.5-32B-Instruct --format huffman --baseline --gpus 2 > ~/results/32b-bf16-2gpu-huffman-2gpu.txt 2>&1
$E /data/Qwen2.5-32B-Instruct --format huffman --gpus 1 > ~/results/32b-huffman-1gpu.txt 2>&1
$E /data/Qwen2.5-32B-Instruct --format fast --gpus 1 > ~/results/32b-fast-1gpu.txt 2>&1
# 72B: bf16 on 4 GPUs, then packed on 3.
$E /data/Qwen2.5-72B-Instruct --format huffman --baseline --gpus 4 > ~/results/72b-bf16-4gpu-huffman-4gpu.txt 2>&1
$E /data/Qwen2.5-72B-Instruct --format huffman --gpus 3 > ~/results/72b-huffman-3gpu.txt 2>&1
$E /data/Qwen2.5-72B-Instruct --format fast --gpus 3 > ~/results/72b-fast-3gpu.txt 2>&1
touch ~/results/DONE
'
RUN="${RUN//COMMIT_PLACEHOLDER/$COMMIT}"
AMI="$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id --query Parameter.Value)"
ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$SG" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,Throughput=1000,Iops=16000}' \
    --user-data "$USERDATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID},{Key=glyd-bench,Value=$RUN_ID}]" \
    --query 'Instances[0].InstanceId')"
echo "launched $TYPE: $ID ($AMI)"
aws ec2 wait instance-running --instance-ids "$ID"
IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"
SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 "ubuntu@$IP")
echo "waiting for $IP"
until "${SSH[@]}" test -f READY 2>/dev/null; do sleep 15; done
"${SSH[@]}" "mkdir -p glyd && tar -C glyd -xf -" < "$TARBALL"
LIVE="$TMP/results"; mkdir -p "$LIVE"
"${SSH[@]}" "cat > run.sh" <<< "$RUN"
"${SSH[@]}" "nohup bash run.sh > run.log 2>&1 < /dev/null &"
waited=0
while true; do
    sleep 120; waited=$((waited + 2))
    "${SSH[@]}" "tar -C results -cf - . 2>/dev/null; true" | tar -C "$LIVE" -xf - 2>/dev/null || true
    "${SSH[@]}" cat run.log > "$LIVE/run.log" 2>/dev/null || true
    [ -f "$LIVE/DONE" ] && break
    if [ "$waited" -ge 180 ]; then echo "giving up after 3 hours"; break; fi
    echo "$waited min: $(ls "$LIVE" | tr '\n' ' ')"
done
rm -rf "$OUT"; mkdir -p "$OUT"; cp -R "$LIVE"/. "$OUT"/
grep -h -E "tokens/s|prefill|packed|OutOfMemory" "$OUT"/*.txt 2>/dev/null
echo "done: $RUN_ID"
