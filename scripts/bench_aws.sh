#!/usr/bin/env bash
# Reproducible cross-platform benchmark on AWS EC2.
#
# Launches one on-demand instance per type (default: Graviton3 and Sapphire
# Rapids), ships the tree at a git ref as a tarball (so private checkouts
# work too), downloads the Silesia corpus, runs the same-run harnesses
# (quick3: v6 levels vs liblz4; v7_bench: max level vs zstd; ultra_bench:
# ultra level vs zstd -16/-19; field_survey: the whole field; mc: multi-core), copies the results into
# benchmarks/<instance-type>/ and terminates everything it created.
#
# Usage:
#   AWS_PROFILE=... scripts/bench_aws.sh [git-ref]
# Env: REGION (us-east-1), TYPES ("c7g.2xlarge c7i.2xlarge"). Cost:
# ~$0.35/hour per instance; a run takes 40-60 minutes. Everything created
# is tagged glyd-bench and deleted on exit (also on Ctrl-C).
set -euo pipefail

REF="${1:-main}"
REGION="${REGION:-us-east-1}"
TYPES="${TYPES:-c7g.2xlarge c7i.2xlarge}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks"
RUN_ID="glyd-bench-$(date +%Y%m%d-%H%M%S)"
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

git -C "$ROOT" archive --format=tar -o "$TARBALL" "$REF"
COMMIT="$(git -C "$ROOT" rev-parse "$REF")"
echo "run $RUN_ID: $REF ($COMMIT), types: $TYPES, region $REGION"

aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd benchmark, temporary" --vpc-id "$VPC" --query GroupId)"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null

# Boot: toolchain only. The benchmark runs over SSH once the tree is up.
USERDATA='#!/bin/bash
apt-get update -y && apt-get install -y build-essential clang curl git unzip xz-utils pkg-config
sudo -u ubuntu bash -c "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
touch /home/ubuntu/READY
'

# Runs on the instance as ubuntu, inside the extracted tree.
BENCH='set -ex
source ~/.cargo/env
cd ~/glyd
bash scripts/download_corpus.sh
export RUSTFLAGS="-C target-cpu=native"
mkdir -p ~/results
{
  echo "commit: COMMIT_PLACEHOLDER"
  TOK=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  echo "instance: $(curl -s -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-type)"
  echo "cpu: $(lscpu | grep -m1 -E "Model name|BIOS Model name" | cut -d: -f2 | sed "s/^ *//") ($(nproc) vCPU)"
  echo "simd: $(grep -m1 -o "avx512bw\|avx2\|asimd" /proc/cpuinfo | head -1)"
  echo "rustc: $(rustc --version)"
  echo "kernel: $(uname -sr)"
  echo "date: $(date -u +%FT%TZ)"
} > ~/results/machine.txt
cargo build --release --examples 2>&1 | tail -2
./target/release/examples/quick3 aws 3 0.3 -v > ~/results/quick3.txt 2>&1
./target/release/examples/quick3 aws-turbo 3 0.3 --turbo -v > ~/results/quick3_turbo.txt 2>&1 || true
./target/release/examples/v7_bench > ~/results/v7_bench.txt 2>&1
./target/release/examples/ultra_bench > ~/results/ultra_bench.txt 2>&1
./target/release/examples/field_survey 3 0.3 > ~/results/field_survey.txt 2>&1
./target/release/examples/mc > ~/results/multicore.txt 2>&1
touch ~/results/DONE
'
BENCH="${BENCH//COMMIT_PLACEHOLDER/$COMMIT}"

for T in $TYPES; do
    case "$T" in *g.*|*gd.*|*gn.*|a1.*) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
    AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
    ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" --security-group-ids "$SG" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
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
    local SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "ubuntu@$IP")
    echo "[$T] waiting for the toolchain on $IP"
    until "${SSH[@]}" test -f READY 2>/dev/null; do sleep 15; done
    "${SSH[@]}" "mkdir -p glyd && tar -C glyd -xf -" < "$TARBALL"
    echo "[$T] benchmarking (corpus download + build + harnesses, 20-40 min)"
    mkdir -p "$OUT/$T"
    if ! "${SSH[@]}" bash -s <<< "$BENCH" > "$OUT/$T/run.log" 2>&1; then
        echo "[$T] bench script exited non-zero, see $OUT/$T/run.log"
    fi
    "${SSH[@]}" tar -C results -cf - . | tar -C "$OUT/$T" -xf - || true
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
    grep -E "^v7 total" "$OUT/$T/v7_bench.txt" 2>/dev/null
    tail -1 "$OUT/$T/quick3.txt" 2>/dev/null
done
echo "done: $RUN_ID"
