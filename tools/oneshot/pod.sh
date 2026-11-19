#!/bin/bash
# Run a command on a Runpod pod through its SSH gateway (which needs a terminal): pod.sh USER@ssh.runpod.io 'COMMAND'
# Prints the command's output between two markers.
printf 'echo @@START; %s; echo @@END; exit\n' "$2" | ssh -tt -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 "$1" 2>&1 | tr -d '\r' | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | awk '/^@@END/{p=0} p{print} /^@@START/{p=1}'
