#!/usr/bin/env bash
# Where the max level's write time goes on a server core, and what the
# finder's table size is worth there: examples/max_split on three files
# (an hour of GitHub events, a Wikipedia table dump, Silesia's mozilla)
# with the full tables and with zstd -3's sizes, against zstd -3 on one
# thread. One on-demand instance per type, tagged glyd-bench, deleted on
# exit. Results in benchmarks/max/<instance-type>/.
#   AWS_PROFILE=... scripts/bench_aws_max.sh [git-ref]
# Env: REGION (us-east-1), TYPES ("c7g.2xlarge c7i.2xlarge"). ~$0.15 per instance.
set -euo pipefail
REF="${1:-main}"
REGION="${REGION:-us-east-1}"
TYPES="${TYPES:-c7g.2xlarge c7i.2xlarge}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/max"
RUN_ID="glyd-max-$(date +%Y%m%d-%H%M%S)"
KEY="$RUN_ID"
TMP="$(mktemp -d)"
KEYFILE="$TMP/$KEY.pem"
TARBALL="$TMP/glyd.tar"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
declare -a INSTANCES=()
SG=""
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
    rm -rf "$TMP"
}
trap cleanup EXIT
COMMIT="$(git -C "$ROOT" rev-parse --short "$REF")"
git -C "$ROOT" archive --format=tar -o "$TARBALL" "$REF"
aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd max bench" --vpc-id "$VPC" --query GroupId)"
aws ec2 create-tags --resources "$SG" --tags "Key=glyd-bench,Value=$RUN_ID"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null
USERDATA='#!/bin/bash
apt-get update -y && apt-get install -y build-essential clang curl git unzip xz-utils zstd
sudo -u ubuntu bash -c "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
touch /home/ubuntu/READY
'
BENCH='set -ex
source ~/.cargo/env
cd ~/glyd
export RUSTFLAGS="-C target-cpu=native"
mkdir -p ~/results ~/data
curl -sL https://data.gharchive.org/2024-01-15-12.json.gz | gzip -dc | head -c 536870912 > ~/data/gharchive.json
curl -sL https://dumps.wikimedia.org/simplewiki/20260901/simplewiki-20260901-categorylinks.sql.gz | gzip -dc > ~/data/categorylinks.sql
curl -sL https://cloud-images.ubuntu.com/noble/20260911/noble-server-cloudimg-amd64-root.tar.xz | xz -dc | head -c 536870912 > ~/data/root.tar
cargo build --release --example max_split --example scale 2>&1 | tail -1
{
  echo "commit: COMMIT_PLACEHOLDER"
  echo "cpu: $(lscpu | grep -m1 -E "Model name|BIOS Model name" | cut -d: -f2 | sed "s/^ *//") ($(nproc) vCPU)"
  lscpu | grep -E "L2|L3" || true
  for f in ~/data/gharchive.json ~/data/categorylinks.sql ~/data/root.tar; do
    n=$(stat -c %s $f)
    t0=$(date +%s.%N); zstd -3 -T1 -q -c $f > /tmp/z.zst; t1=$(date +%s.%N)
    echo "$(basename $f): zstd -3 one thread $(echo "$n / ($t1 - $t0) / 1000000" | bc -l | cut -c1-6) MB/s, $(stat -c %s /tmp/z.zst) B"
    t0=$(date +%s.%N); zstd -3 -T8 -q -c $f > /tmp/z.zst; t1=$(date +%s.%N)
    echo "$(basename $f): zstd -3 eight threads $(echo "$n / ($t1 - $t0) / 1000000" | bc -l | cut -c1-6) MB/s"
    for bits in 18,18 17,16 16,15; do
      echo -n "tables $bits: "; GLYD_BITS=$bits ./target/release/examples/max_split $f
    done
    ./target/release/examples/scale $f
  done
} > ~/results/max.txt 2>&1
touch ~/results/DONE
'
BENCH="${BENCH//COMMIT_PLACEHOLDER/$COMMIT}"
for T in $TYPES; do
    case "$T" in *g.*|*gd.*|*gn.*|a1.*) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
    AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
    ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" --security-group-ids "$SG" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID-$T},{Key=glyd-bench,Value=$RUN_ID}]" \
        --query 'Instances[0].InstanceId')"
    INSTANCES+=("$ID")
    echo "launched $T: $ID"
done
aws ec2 wait instance-running --instance-ids "${INSTANCES[@]}"
bench_one() {
    local T="$1" ID="$2" IP
    IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"
    local SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "ubuntu@$IP")
    until "${SSH[@]}" test -f READY 2>/dev/null; do sleep 15; done
    "${SSH[@]}" "mkdir -p glyd && tar -C glyd -xf -" < "$TARBALL"
    mkdir -p "$OUT/$T"
    "${SSH[@]}" bash -s <<< "$BENCH" > "$OUT/$T/run.log" 2>&1 || echo "[$T] non-zero exit, see $OUT/$T/run.log"
    "${SSH[@]}" tar -C results -cf - . | tar -C "$OUT/$T" -xf - || true
    echo "== $T"; cat "$OUT/$T/max.txt" 2>/dev/null
}
i=0
for T in $TYPES; do bench_one "$T" "${INSTANCES[$i]}" & i=$((i + 1)); done
wait
echo "done: $RUN_ID"
