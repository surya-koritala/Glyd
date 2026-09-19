#!/usr/bin/env bash
# Reproducible cross-platform benchmark on AWS EC2.
#
# Launches one on-demand instance per type (default: Graviton3 and Sapphire
# Rapids), builds Glyd from a git ref, downloads the Silesia corpus, runs the
# three same-run harnesses (quick3: v6 levels vs liblz4; v7_bench: max level
# vs zstd; field_survey: the whole field), copies the results into
# benchmarks/<instance-type>/ and terminates everything it created.
#
# Usage:
#   AWS_PROFILE=... scripts/bench_aws.sh [git-ref]
# Env: REGION (us-east-1), TYPES ("c7g.2xlarge c7i.2xlarge"), REPO
# (https://github.com/surya-koritala/Glyd.git). Cost: ~$0.35/hour per
# instance; a run takes 30-50 minutes. Everything created is tagged
# glyd-bench and deleted on exit (also on Ctrl-C).
set -euo pipefail

REF="${1:-main}"
REGION="${REGION:-us-east-1}"
TYPES="${TYPES:-c7g.2xlarge c7i.2xlarge}"
REPO="${REPO:-https://github.com/surya-koritala/Glyd.git}"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmarks"
RUN_ID="glyd-bench-$(date +%Y%m%d-%H%M%S)"
KEY="$RUN_ID"
KEYFILE="$(mktemp -d)/$KEY.pem"
MYIP="$(curl -s https://checkip.amazonaws.com)/32"
declare -a INSTANCES=()
SG=""

aws() { command aws --region "$REGION" --output text "$@"; }

cleanup() {
    set +e
    echo "cleanup: terminating ${INSTANCES[*]:-nothing}"
    [ "${#INSTANCES[@]}" -gt 0 ] && aws ec2 terminate-instances --instance-ids "${INSTANCES[@]}" >/dev/null
    [ "${#INSTANCES[@]}" -gt 0 ] && aws ec2 wait instance-terminated --instance-ids "${INSTANCES[@]}"
    [ -n "$SG" ] && aws ec2 delete-security-group --group-id "$SG"
    aws ec2 delete-key-pair --key-name "$KEY"
    rm -f "$KEYFILE"
}
trap cleanup EXIT

echo "run $RUN_ID: ref $REF, types: $TYPES, region $REGION"
aws ec2 create-key-pair --key-name "$KEY" --query KeyMaterial > "$KEYFILE"
chmod 600 "$KEYFILE"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId')"
SG="$(aws ec2 create-security-group --group-name "$RUN_ID" --description "glyd benchmark, temporary" --vpc-id "$VPC" --query GroupId)"
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null

# The instance does all the work at boot and writes a marker when finished.
userdata() {
    cat <<EOF
#!/bin/bash
set -x
export HOME=/root
apt-get update -y && apt-get install -y build-essential clang curl git unzip xz-utils pkg-config
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source /root/.cargo/env
git clone "$REPO" /opt/glyd && cd /opt/glyd && git checkout "$REF"
bash scripts/download_corpus.sh
export RUSTFLAGS="-C target-cpu=native"
mkdir -p /opt/results
{
  echo "commit: \$(git rev-parse HEAD)"; echo "instance: \$(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
  echo "cpu: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"; grep -m1 -o 'avx2\|asimd' /proc/cpuinfo | head -1
  echo "rustc: \$(rustc --version)"; echo "kernel: \$(uname -sr)"; echo "date: \$(date -u +%FT%TZ)"
} > /opt/results/machine.txt
cargo build --release --examples 2>&1 | tail -3
./target/release/examples/quick3 aws 3 0.3 -v > /opt/results/quick3.txt 2>&1
cargo run --release --example v7_bench > /opt/results/v7_bench.txt 2>&1
cargo run --release --example field_survey 3 0.3 > /opt/results/field_survey.txt 2>&1
cargo run --release --example mc > /opt/results/multicore.txt 2>&1
touch /opt/results/DONE
EOF
}

for T in $TYPES; do
    case "$T" in *g.*|*gd.*|*gn.*|a1.*) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
    AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
    ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$KEY" --security-group-ids "$SG" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
        --user-data "$(userdata)" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_ID-$T},{Key=glyd-bench,Value=$RUN_ID}]" \
        --query 'Instances[0].InstanceId')"
    INSTANCES+=("$ID")
    echo "launched $T: $ID ($AMI)"
done

aws ec2 wait instance-running --instance-ids "${INSTANCES[@]}"
i=0
for T in $TYPES; do
    ID="${INSTANCES[$i]}"; i=$((i + 1))
    IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"
    SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "ubuntu@$IP")
    echo "waiting for $T ($IP) to finish (build + Silesia download + three harnesses)"
    until "${SSH[@]}" sudo test -f /opt/results/DONE 2>/dev/null; do sleep 30; done
    mkdir -p "$OUT/$T"
    "${SSH[@]}" sudo tar -C /opt/results -cf - . | tar -C "$OUT/$T" -xf -
    echo "results in $OUT/$T:"; cat "$OUT/$T/machine.txt"
    grep -E "^v7 total|ratio .* comp .* decomp" "$OUT/$T/v7_bench.txt" | tail -1
    tail -1 "$OUT/$T/quick3.txt"
done
echo "done: $RUN_ID"
