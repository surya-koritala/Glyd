#!/bin/bash
# Move the stage 1 invitation (base64, last line of invite.txt) to pod G in small pieces, then prepare.
set -euo pipefail
cd "$(dirname "$0")"
B=$(tail -1 invite.txt | tr -d '\r\n ')
G=qwp9lan9p376y8-64410aaa
./pod.sh $G "mkdir -p /workspace/m4run && : > /workspace/m4run/inv.b64"
for ((i = 0; i < ${#B}; i += 1500)); do
    ./pod.sh $G "printf %s '${B:i:1500}' >> /workspace/m4run/inv.b64"
done
./pod.sh $G '/opt/m4/m4.sh g-prepare "$(cat /workspace/m4run/inv.b64)" 2>&1'
