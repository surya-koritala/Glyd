#!/usr/bin/env bash
# The full verification and benchmark program on AWS, one on-demand
# instance per type (default Graviton3 and Sapphire Rapids), each with
# the benchmark corpus (scripts/download_bench_corpus.sh, ~9 GB), the
# reference CLIs (zstd, lz4) from Ubuntu's packages, and an instance
# role that may read and write one S3 bucket:
#   1. scripts/verify_roundtrip.sh on three files (CLI, every level,
#      single- and multi-core, corrupted copies);
#   2. examples/bench_suite: every codec at all threads on the large
#      files, the small objects (one thread), and the fast codecs at one
#      thread (every decode checked against the input; the single-thread
#      passes of --ultra and zstd -19 over 9 GB would take an hour each
#      and are left out);
#   3. scripts/s3_workflow.sh over the corpus (compress, upload,
#      download, decompress, verify; costs).
# Results land in benchmarks/suite/<instance-type>/. Everything created
# (instances, key, security group, IAM role and profile) is tagged
# glyd-bench and deleted on exit; the bucket is the caller's.
#
#   AWS_PROFILE=... scripts/bench_aws_suite.sh <bucket> [git-ref]
# Env: REGION (us-east-1), TYPES ("c7g.2xlarge c7i.2xlarge").
# A run takes about 2 hours per instance (in parallel); ~$0.70 per instance.
set -euo pipefail

BUCKET="$1"
REF="${2:-main}"
REGION="${REGION:-us-east-1}"
TYPES="${TYPES:-c7g.2xlarge c7i.2xlarge}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/suite"
RUN_ID="glyd-bench-$(date +%Y%m%d-%H%M%S)"
KEY="$RUN_ID"
TMP="$(mktemp -d)"
KEYFILE="$TMP/$KEY.pem"
TARBALL="$TMP/glyd.tar"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
declare -a INSTANCES=()
SG=""
ROLE="$RUN_ID"

aws() { command aws --region "$REGION" --output text "$@"; }

cleanup() {
    set +e
    echo "cleanup: terminating ${INSTANCES[*]:-nothing}"
    if [ "${#INSTANCES[@]}" -gt 0 ]; then
        aws ec2 terminate-instances --instance-ids "${INSTANCES[@]}" >/dev/null
        aws ec2 wait instance-terminated --instance-ids "${INSTANCES[@]}"
    fi
    [ -n "$SG" ] && aws ec2 delete-security-group --group-id "$SG"
    aws ec2 delete-key-pair --key-name "$KEY"
    aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null
    aws iam delete-instance-profile --instance-profile-name "$ROLE" 2>/dev/null
    aws iam delete-role-policy --role-name "$ROLE" --policy-name bucket 2>/dev/null
    aws iam delete-role --role-name "$ROLE" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

git -C "$ROOT" archive --format=tar -o "$TARBALL" "$REF"
COMMIT="$(git -C "$ROOT" rev-parse "$REF")"
echo "run $RUN_ID: $REF ($COMMIT), types: $TYPES, region $REGION, bucket $BUCKET"

# An instance role for the one bucket.
aws iam create-role --role-name "$ROLE" --tags Key=glyd-bench,Value="$RUN_ID" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name bucket --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
sleep 12 # the profile takes a moment to become usable by EC2

aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd benchmark, temporary" --vpc-id "$VPC" --query GroupId)"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null

USERDATA='#!/bin/bash
for i in 1 2 3; do apt-get update -y && break; sleep 20; done
apt-get install -y build-essential clang curl git unzip xz-utils pkg-config
apt-get install -y zstd lz4 bc python3
cd /tmp && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o awscli.zip && unzip -q awscli.zip && ./aws/install
sudo -u ubuntu bash -c "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
touch /home/ubuntu/READY
'

BENCH='set -x
source ~/.cargo/env
cd ~/glyd
export RUSTFLAGS="-C target-cpu=native"
export AWS_DEFAULT_REGION=REGION_PLACEHOLDER
mkdir -p ~/results
TOK=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE=$(curl -s -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-type)
{
  echo "commit: COMMIT_PLACEHOLDER"
  echo "instance: $INSTANCE"
  echo "cpu: $(lscpu | grep -m1 -E "Model name|BIOS Model name" | cut -d: -f2 | sed "s/^ *//") ($(nproc) vCPU)"
  echo "memory: $(free -g | awk "/Mem:/ {print \$2}") GB"
  echo "rustc: $(rustc --version)"
  echo "zstd: $(zstd --version 2>&1 | head -1)"
  echo "lz4: $(lz4 --version 2>&1 | head -1)"
  echo "kernel: $(uname -sr)"
  echo "date: $(date -u +%FT%TZ)"
} > ~/results/machine.txt
cp /var/log/cloud-init-output.log ~/results/cloud-init.log 2>/dev/null || true
cargo build --release --examples --bins > ~/results/build.txt 2>&1
tail -3 ~/results/build.txt
if [ ! -x target/release/examples/bench_suite ] || [ ! -x target/release/glyd ]; then echo "BUILD FAILED" > ~/results/FAILED; touch ~/results/DONE; exit 1; fi
bash scripts/download_bench_corpus.sh > ~/results/corpus.txt 2>&1
export PATH=$PWD/target/release:$PATH
which zstd lz4 aws glyd >> ~/results/machine.txt
scripts/verify_roundtrip.sh corpus/bench/nasa-access-jul95.log corpus/bench/gharchive-2024-01-16-12.json corpus/bench/yellow_tripdata_2024-02.parquet > ~/results/verify.txt 2>&1
./target/release/examples/bench_suite --large --threads $(nproc) --repeats 3 --slow-repeats 2 --out ~/results/bench_suite_allthreads.jsonl > ~/results/bench_suite_allthreads.txt 2>&1
./target/release/examples/bench_suite --small --threads 1 --repeats 3 --out ~/results/bench_suite_small.jsonl > ~/results/bench_suite_small.txt 2>&1
./target/release/examples/bench_suite --large --threads 1 --repeats 3 --codecs glyd-default,glyd-max,zstd-3,lz4 --out ~/results/bench_suite_1thread.jsonl > ~/results/bench_suite_1thread.txt 2>&1
S3WF_OUT=~/results/s3_workflow.jsonl scripts/s3_workflow.sh BUCKET_PLACEHOLDER corpus/bench > ~/results/s3_workflow.txt 2>&1
touch ~/results/DONE
'
BENCH="${BENCH//COMMIT_PLACEHOLDER/$COMMIT}"
BENCH="${BENCH//BUCKET_PLACEHOLDER/$BUCKET}"
BENCH="${BENCH//REGION_PLACEHOLDER/$REGION}"

for T in $TYPES; do
    case "$T" in *g.*|*gd.*|*gn.*|a1.*) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
    AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
    ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$ROLE" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,Throughput=500}' \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID-$T},{Key=glyd-bench,Value=$RUN_ID}]" \
        --query 'Instances[0].InstanceId')"
    INSTANCES+=("$ID")
    echo "launched $T: $ID ($AMI)"
done

aws ec2 wait instance-running --instance-ids "${INSTANCES[@]}"

bench_one() {
    local T="$1" ID="$2"
    local IP
    IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"
    local SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 "ubuntu@$IP")
    echo "[$T] waiting for the toolchain on $IP"
    until "${SSH[@]}" test -f READY 2>/dev/null; do sleep 15; done
    "${SSH[@]}" "mkdir -p glyd && tar -C glyd -xf -" < "$TARBALL"
    echo "[$T] running (corpus download, build, verification, benchmarks, S3 workflow: 2-3 hours)"
    # A previous run's results (its DONE above all) would end the poll
    # below at once; they live in git.
    rm -rf "$OUT/$T"
    mkdir -p "$OUT/$T"
    "${SSH[@]}" "cat > bench.sh" <<< "$BENCH"
    "${SSH[@]}" "nohup bash bench.sh > run.log 2>&1 < /dev/null &"
    # Poll: fetch results every 10 minutes until DONE, so a failure or
    # a timeout still leaves the partial results here.
    local waited=0
    while true; do
        sleep 600; waited=$((waited + 10))
        "${SSH[@]}" "tar -C results -cf - . 2>/dev/null; true" | tar -C "$OUT/$T" -xf - 2>/dev/null || true
        "${SSH[@]}" cat run.log > "$OUT/$T/run.log" 2>/dev/null || true
        if [ -f "$OUT/$T/DONE" ]; then break; fi
        if [ "$waited" -ge 420 ]; then echo "[$T] giving up after 7 hours"; break; fi
        echo "[$T] ${waited} min: $(ls "$OUT/$T" | tr '\n' ' ')"
    done
    echo "[$T] results in $OUT/$T"
}

i=0
for T in $TYPES; do
    bench_one "$T" "${INSTANCES[$i]}" &
    i=$((i + 1))
done
wait
for T in $TYPES; do
    echo "== $T"; cat "$OUT/$T/machine.txt" 2>/dev/null
    tail -3 "$OUT/$T/verify.txt" 2>/dev/null
    cat "$OUT/$T/s3_workflow.txt" 2>/dev/null
done
echo "done: $RUN_ID"
