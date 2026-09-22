#!/usr/bin/env bash
# The store's gate on AWS: one instance with NVMe next to the bucket
# downloads the terabyte corpus (scripts/gate_corpus.sh), builds
# glyd-store from this tree, and runs scripts/gate_run.sh against
# s3://<bucket>/gate-<run id>. Everything created (instance, key,
# security group, IAM role and profile) is tagged glyd-bench and deleted
# on exit; the objects put in the bucket are deleted at the end too.
# Results in benchmarks/gate/<instance type>/.
#   AWS_PROFILE=... scripts/gate_aws.sh <bucket> [git-ref]
# Env: REGION (us-east-1), TYPE (im4gn.4xlarge: 16 vCPU, 7.5 TB NVMe).
# About 6 hours; ~$10 of instance time and under $1 of S3.
set -euo pipefail
BUCKET="$1"
REF="${2:-main}"
REGION="${REGION:-us-east-1}"
TYPE="${TYPE:-im4gn.4xlarge}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/gate/$TYPE"
RUN_ID="glyd-gate-$(date +%Y%m%d-%H%M%S)"
KEY="$RUN_ID"; ROLE="$RUN_ID"
TMP="$(mktemp -d)"
KEYFILE="$TMP/$KEY.pem"
TARBALL="$TMP/glyd.tar"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
ID=""; SG=""
aws() { command aws --region "$REGION" --output text "$@"; }
cleanup() {
    set +e
    echo "cleanup: terminating ${ID:-nothing}, deleting s3://$BUCKET/$RUN_ID/"
    [ -n "$ID" ] && aws ec2 terminate-instances --instance-ids "$ID" >/dev/null && aws ec2 wait instance-terminated --instance-ids "$ID"
    command aws --region "$REGION" s3 rm --quiet --recursive "s3://$BUCKET/$RUN_ID/"
    [ -n "$SG" ] && aws ec2 delete-security-group --group-id "$SG"
    aws ec2 delete-key-pair --key-name "$KEY"
    aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null
    aws iam delete-instance-profile --instance-profile-name "$ROLE" 2>/dev/null
    aws iam delete-role-policy --role-name "$ROLE" --policy-name bucket 2>/dev/null
    aws iam delete-role --role-name "$ROLE" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
COMMIT="$(git -C "$ROOT" rev-parse --short "$REF")"
git -C "$ROOT" archive --format=tar -o "$TARBALL" "$REF"
aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd gate" --vpc-id "$VPC" --query GroupId)"
aws ec2 create-tags --resources "$SG" --tags "Key=glyd-bench,Value=$RUN_ID"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null
aws iam create-role --role-name "$ROLE" --tags Key=glyd-bench,Value="$RUN_ID" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name bucket --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:DeleteObject\",\"s3:ListBucket\",\"s3:AbortMultipartUpload\",\"s3:ListMultipartUploadParts\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
sleep 15   # the profile takes a moment to become usable
USERDATA='#!/bin/bash
apt-get update -y && apt-get install -y build-essential clang curl git xz-utils zstd pigz bc
for d in /dev/nvme*n1; do
  if ! lsblk -n "$d" | grep -q part; then mkfs.ext4 -q -F "$d" && mkdir -p /data && mount "$d" /data && break; fi
done
chown ubuntu /data
sudo -u ubuntu bash -c "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
touch /home/ubuntu/READY
'
RUN='set -x
source ~/.cargo/env
cd ~/glyd
export RUSTFLAGS="-C target-cpu=native"
mkdir -p ~/results
echo "commit: COMMIT_PLACEHOLDER" > ~/results/machine.txt
cargo build --release --workspace > ~/results/build.txt 2>&1 || { echo "BUILD FAILED" > ~/results/FAILED; touch ~/results/DONE; exit 1; }
export GLYD_STORE=$PWD/target/release/glyd-store AWS_REGION=REGION_PLACEHOLDER
bash scripts/gate_corpus.sh /data/corpus > ~/results/corpus.txt 2>&1
bash scripts/gate_run.sh /data/corpus s3://BUCKET_PLACEHOLDER/RUN_PLACEHOLDER /data/work ~/results > ~/results/gate_run.log 2>&1
touch ~/results/DONE
'
RUN="${RUN//COMMIT_PLACEHOLDER/$COMMIT}"; RUN="${RUN//REGION_PLACEHOLDER/$REGION}"; RUN="${RUN//BUCKET_PLACEHOLDER/$BUCKET}"; RUN="${RUN//RUN_PLACEHOLDER/$RUN_ID}"
AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$SG" \
    --iam-instance-profile "Name=$ROLE" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
    --user-data "$USERDATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID},{Key=glyd-bench,Value=$RUN_ID}]" \
    --query 'Instances[0].InstanceId')"
echo "launched $TYPE: $ID ($AMI); objects go to s3://$BUCKET/$RUN_ID/"
aws ec2 wait instance-running --instance-ids "$ID"
IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"
SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 "ubuntu@$IP")
echo "waiting for the toolchain on $IP"
until "${SSH[@]}" test -f READY 2>/dev/null; do sleep 15; done
"${SSH[@]}" "mkdir -p glyd && tar -C glyd -xf -" < "$TARBALL"
rm -rf "$OUT"; mkdir -p "$OUT"
"${SSH[@]}" "cat > run.sh" <<< "$RUN"
"${SSH[@]}" "nohup bash run.sh > run.log 2>&1 < /dev/null &"
waited=0
while true; do
    sleep 600; waited=$((waited + 10))
    "${SSH[@]}" "tar -C results -cf - . 2>/dev/null; true" | tar -C "$OUT" -xf - 2>/dev/null || true
    "${SSH[@]}" cat run.log > "$OUT/run.log" 2>/dev/null || true
    "${SSH[@]}" "df -h /data | tail -1; ls /data/corpus 2>/dev/null | wc -l" > "$OUT/progress.txt" 2>/dev/null || true
    [ -f "$OUT/DONE" ] && break
    if [ "$waited" -ge 600 ]; then echo "giving up after 10 hours"; break; fi
    echo "$waited min: $(cat "$OUT/progress.txt" | tr '\n' ' ') $(ls "$OUT" | tr '\n' ' ')"
done
cat "$OUT/gate.txt" 2>/dev/null || cat "$OUT/run.log"
echo "done: $RUN_ID"
