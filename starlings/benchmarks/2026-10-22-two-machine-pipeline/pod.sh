#!/bin/bash
# Run commands on a Runpod pod through the PTY-only SSH proxy; print what is between the markers.
u=$1; shift
ssh -tt -o LogLevel=QUIET -o StrictHostKeyChecking=accept-new "$u@ssh.runpod.io" < <(sleep 3; printf 'stty -echo; echo "<<<""BEGIN"; %s; echo "END"">>>"; exit\n' "$*"; sleep 3600) 2>&1 | tr -d '\r' | sed -n '/<<<BEGIN$/,/^END>>>/p' | sed '1d;$d'
