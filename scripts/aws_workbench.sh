#!/usr/bin/env bash
# A workbench on Graviton3 and Sapphire Rapids for measuring a change on
# server cores: `up` launches one instance of each (tagged glyd-bench),
# downloads the files below into ~/data and builds the working tree;
# `sync` ships and builds the tree again; `run <script>` runs a bash
# script on both and prints the outputs; `down` terminates everything.
# ~$0.70 an hour for the pair while up: `down` when done.
#   AWS_PROFILE=... scripts/aws_workbench.sh up | sync | run <script> | down
# State (key, security group, instances) in $GLYD_WORKBENCH (default
# ~/.glyd-workbench).
# A machine of your own instead: GLYD_HOST=user@address (and GLYD_KEY=
# its key file) makes `sync` and `run` target it; `up` and `down` do
# nothing. It needs build-essential, perf and zstd installed and the
# files above in ~/data.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ST=${GLYD_WORKBENCH:-$HOME/.glyd-workbench}
mkdir -p "$ST"
if [ -n "${GLYD_HOST:-}" ]; then
  TYPES="$(hostname -s 2>/dev/null || echo host)-box"
  sshto() { local ip=$1; shift; ssh ${GLYD_KEY:+-i "$GLYD_KEY"} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o LogLevel=ERROR "$ip" "$@"; }
  echo "$TYPES $GLYD_HOST" > "$ST/ips"
  case "${1:-}" in
    up|down) echo "GLYD_HOST=$GLYD_HOST: nothing to $1"; exit 0 ;;
  esac
else
  : "${AWS_PROFILE:?set AWS_PROFILE}"
  REGION=us-east-1
  TYPES="c7g.2xlarge c7i.2xlarge"
  aws() { command aws --region "$REGION" --output text "$@"; }
  sshto() { local ip=$1; shift; ssh -i "$ST/key.pem" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR "ubuntu@$ip" "$@"; }
fi
case "${1:-}" in
up)
  RUN="glyd-bench-$(date +%Y%m%d-%H%M%S)"; echo "$RUN" > $ST/run
  aws ec2 create-key-pair --key-name "$RUN" --query KeyMaterial > $ST/key.pem; chmod 600 $ST/key.pem
  VPC="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId')"
  SG="$(aws ec2 create-security-group --group-name "$RUN" --description "glyd workbench" --vpc-id "$VPC" --query GroupId)"; echo "$SG" > $ST/sg
  aws ec2 create-tags --resources "$SG" --tags "Key=glyd-bench,Value=$RUN"
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32" >/dev/null
  USERDATA='#!/bin/bash
apt-get update -y && apt-get install -y build-essential clang curl git unzip xz-utils zstd bc
sudo -u ubuntu bash -c "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
sudo -u ubuntu bash -c "mkdir -p ~/data && cd ~/data && (curl -sL https://data.gharchive.org/2024-01-15-12.json.gz | gzip -dc | head -c 200000000 > gh200.json) && (curl -sL ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz | gzip -dc > nasa.log) && (curl -sL https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pagelinks.sql.gz | gzip -dc | head -c 200000000 > sql200.sql) && (curl -sL https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip -o silesia.zip && unzip -q silesia.zip mozilla && rm silesia.zip) && (curl -sL http://mattmahoney.net/dc/enwik8.zip -o e.zip && unzip -q e.zip && rm e.zip)"
touch /home/ubuntu/READY
'
  : > $ST/instances
  for T in $TYPES; do
    case "$T" in *g.*) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
    AMI="$(aws ssm get-parameter --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/$ARCH/hvm/ebs-gp3/ami-id" --query Parameter.Value)"
    ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$T" --key-name "$RUN" --security-group-ids "$SG" \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' --user-data "$USERDATA" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN-$T},{Key=glyd-bench,Value=$RUN}]" --query 'Instances[0].InstanceId')"
    echo "$T $ID" >> $ST/instances; echo "launched $T $ID"
  done
  aws ec2 wait instance-running --instance-ids $(awk '{print $2}' $ST/instances)
  : > $ST/ips
  while read T ID; do IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress')"; echo "$T $IP" >> $ST/ips; done < $ST/instances
  while read T IP; do until sshto $IP test -f READY 2>/dev/null; do sleep 20; done; echo "$T ready"; done < $ST/ips
  "$0" sync
  ;;
sync)
  # the working tree, tracked files at their current state
  (cd $ROOT && git ls-files | tar -cf $ST/tree.tar -T -)
  while read T IP; do
    ( sshto $IP "rm -rf glyd && mkdir -p glyd && tar -C glyd -xf - && cd glyd && source ~/.cargo/env && RUSTFLAGS='-C target-cpu=native' cargo build --release -q 2>&1 | grep -E '^error' -A5; echo '$T built'" < $ST/tree.tar ) &
  done < $ST/ips; wait
  ;;
run)
  SCRIPT="$2"
  while read T IP; do
    ( echo "=== $T"; sshto $IP bash -s < "$SCRIPT" 2>&1 ) > $ST/out.$T &
  done < $ST/ips; wait
  for T in $TYPES; do cat $ST/out.$T; done
  ;;
down)
  aws ec2 terminate-instances --instance-ids $(awk '{print $2}' $ST/instances) >/dev/null
  aws ec2 wait instance-terminated --instance-ids $(awk '{print $2}' $ST/instances)
  aws ec2 delete-security-group --group-id "$(cat $ST/sg)"
  aws ec2 delete-key-pair --key-name "$(cat $ST/run)"
  echo "terminated"
  ;;
esac
